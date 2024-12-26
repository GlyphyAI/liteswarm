# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import AsyncGenerator

from typing_extensions import Protocol, override

from liteswarm.types.events import SwarmEvent


class SwarmEventHandler(Protocol):
    r"""Protocol for handlers that process swarm events.

    This protocol defines the interface for handling events from a swarm.
    Event handlers can be used to:
    - Process events in real-time
    - Filter and transform events
    - Forward events to other systems
    - Collect events for analysis
    - Drive UI updates

    Example:
        ```python
        class ConsoleEventHandler(SwarmEventHandler):
            async def on_event(self, event: SwarmEvent) -> None:
                if event.type == "agent_response_chunk":
                    if event.chunk.completion.delta.content:
                        print(event.chunk.completion.delta.content, end="", flush=True)
                elif event.type == "agent_switch":
                    print(f"\nSwitching to {event.current.id}")
                elif event.type == "error":
                    print(f"\nError: {event.error}")
                elif event.type == "complete":
                    print("\nComplete!")


        # Use with execute() method
        result = await swarm.execute(
            agent=agent,
            prompt="Hello!",
            event_handler=ConsoleEventHandler(),
        )

        # Or use streaming API directly
        stream = swarm.stream(agent, prompt="Hello!")
        async for event in stream:
            if event.type == "agent_response_chunk":
                print(event.chunk.completion.delta.content, end="", flush=True)
        ```

    Notes:
        - Events are handled asynchronously
        - Handlers should process events quickly
        - Events are immutable
        - Events carry full context
    """

    async def on_event(self, event: SwarmEvent) -> None:
        """Handle an event from the swarm.

        This method is called by the swarm when an event occurs.
        The handler should process the event quickly to avoid blocking the swarm.

        Args:
            event: The event to process.

        Example:
            ```python
            async def on_event(self, event: SwarmEvent) -> None:
                if event.type == "agent_response_chunk":
                    await self.process_response(event.chunk.completion)
                elif event.type == "error":
                    await self.handle_error(event.error)
                else:
                    await self.process_other(event)
            ```

        Notes:
            - Should not raise exceptions
            - Should process events quickly
            - Can filter or transform events
            - Can forward events to other handlers
        """
        ...


class LiteSwarmEventHandler(SwarmEventHandler):
    """Default no-op implementation of the event handler protocol.

    Provides an empty implementation of the event handler protocol.
    This class serves multiple purposes:
    - Base class for custom handlers that only need some events
    - Default handler when no custom handling is needed
    - Example of minimal protocol implementation

    Example:
        ```python
        class LoggingEventHandler(LiteSwarmEventHandler):
            async def on_event(self, event: SwarmEvent) -> None:
                if event.type == "agent_response_chunk":
                    print(f"Response: {event.chunk.completion.delta.content}")
                elif event.type == "error":
                    print(f"Error: {event.error}")


        # Use with execute() method
        result = await swarm.execute(
            agent=agent,
            prompt="Hello!",
            event_handler=LoggingEventHandler(),
        )
        ```

    Notes:
        - All methods are implemented as no-ops
        - Safe to use as a base class
        - Provides protocol compliance by default
        - Suitable for testing and development
    """

    @override
    async def on_event(self, event: SwarmEvent) -> None:
        """No-op implementation of event handling.

        Args:
            event: Event to process (ignored).
        """
        pass


class SwarmEventGenerator(SwarmEventHandler):
    """Generator for swarm events with buffering and filtering.

    This class provides an async generator interface for consuming swarm events.
    It can be used to:
    - Stream events in real-time
    - Filter events by type
    - Buffer events for batch processing
    - Transform events for specific use cases

    Example:
        ```python
        generator = SwarmEventGenerator()
        result = await swarm.execute(
            agent=agent,
            prompt="Hello!",
            event_handler=generator,
        )
        ```

    Notes:
        - Events are buffered internally
        - Generator can be consumed multiple times
        - Events are delivered in order
        - Backpressure is handled automatically
    """

    def __init__(self, buffer_size: int = 100) -> None:
        """Initialize the event generator.

        Args:
            buffer_size: Maximum number of events to buffer.
                Defaults to 100. Set to 0 for unbounded buffer.
        """
        self._buffer_size = buffer_size
        self._events: list[SwarmEvent] = []
        self._closed: bool = False

    def __aiter__(self) -> AsyncGenerator[SwarmEvent, None]:
        """Get async iterator for events.

        Returns:
            Async generator yielding events.
        """
        return self.stream()

    @override
    async def on_event(self, event: SwarmEvent) -> None:
        """Handle an event from the swarm.

        Args:
            event: Event to handle.

        Notes:
            - Events are buffered if buffer_size > 0
            - Oldest events are dropped if buffer is full
            - No-op if generator is closed
        """
        if self._closed:
            return

        if self._buffer_size > 0 and len(self._events) >= self._buffer_size:
            self._events.pop(0)

        self._events.append(event)

    async def stream(self) -> AsyncGenerator[SwarmEvent, None]:
        """Stream events from the generator.

        Yields:
            Events in order of arrival.

        Notes:
            - Can be called multiple times
            - Events are not removed from buffer
            - Stops when generator is closed
        """
        current_index = 0
        while not self._closed or current_index < len(self._events):
            if current_index < len(self._events):
                yield self._events[current_index]
                current_index += 1

    def close(self) -> None:
        """Close the generator.

        After closing:
        - No new events will be accepted
        - Existing events can still be consumed
        - Generator will stop after yielding buffered events
        """
        self._closed = True
