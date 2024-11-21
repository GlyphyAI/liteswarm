# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.types.swarm import Agent, Delta, Message


class StreamHandler(Protocol):
    """Protocol for handlers that process streaming events from agents.

    Stream handlers receive and process various events during agent interactions:
    - Content streaming updates
    - Tool call events
    - Agent switching
    - Errors and completion

    Example:
    ```python
    class CustomStreamHandler(StreamHandler):
        async def on_stream(self, delta: Delta, agent: Agent) -> None:
            if delta.content:
                print(f"[{agent.id}]: {delta.content}")

        async def on_tool_call(
            self,
            tool_call: ChatCompletionDeltaToolCall,
            agent: Agent
        ) -> None:
            print(f"[{agent.id}] calling {tool_call.function.name}")

        async def on_agent_switch(
            self,
            previous: Agent | None,
            current: Agent
        ) -> None:
            print(f"Switching from {previous.id} to {current.id}")

        async def on_error(self, error: Exception, agent: Agent) -> None:
            print(f"Error from {agent.id}: {error}")

        async def on_complete(
            self,
            messages: list[Message],
            agent: Agent | None
        ) -> None:
            print("Conversation complete")

    # Use in Swarm
    swarm = Swarm(
        stream_handler=CustomStreamHandler(),
        include_usage=True
    )
    ```
    """

    async def on_stream(self, delta: Delta, agent: Agent) -> None:
        """Handle streaming content updates from an agent.

        Called for each chunk of content or tool call update
        received from the agent.

        Args:
            delta: The content or tool call update
            agent: The agent generating the content

        Example:
        ```python
        async def on_stream(self, delta: Delta, agent: Agent) -> None:
            # Handle content updates
            if delta.content:
                print(f"Content: {delta.content}")

            # Handle tool calls
            if delta.tool_calls:
                for call in delta.tool_calls:
                    print(f"Tool call: {call.function.name}")
        ```
        """
        ...

    async def on_tool_call(self, tool_call: ChatCompletionDeltaToolCall, agent: Agent) -> None:
        """Handle a tool call from an agent.

        Called when an agent initiates a tool call, before
        the tool is executed.

        Args:
            tool_call: Details of the tool being called
            agent: The agent making the call

        Example:
        ```python
        async def on_tool_call(
            self,
            tool_call: ChatCompletionDeltaToolCall,
            agent: Agent
        ) -> None:
            print(
                f"Agent {agent.id} calling {tool_call.function.name}"
                f" with args: {tool_call.function.arguments}"
            )
        ```
        """
        ...

    async def on_agent_switch(self, previous_agent: Agent | None, current_agent: Agent) -> None:
        """Handle an agent switch event.

        Called when the conversation switches from one agent
        to another, typically due to a tool call result.

        Args:
            previous_agent: The agent being switched from (None if first agent)
            current_agent: The agent being switched to

        Example:
        ```python
        async def on_agent_switch(
            self,
            previous: Agent | None,
            current: Agent
        ) -> None:
            if previous:
                print(f"Switching from {previous.id} to {current.id}")
            else:
                print(f"Starting with agent {current.id}")
        ```
        """
        ...

    async def on_error(self, error: Exception, agent: Agent | None) -> None:
        """Handle an error during agent execution.

        Called when an error occurs during agent execution,
        tool calls, or response processing.

        Args:
            error: The exception that occurred
            agent: The agent that encountered the error (None if no active agent)

        Example:
        ```python
        async def on_error(self, error: Exception, agent: Agent) -> None:
            print(f"Error in agent {agent.id}:")
            print(f"- Type: {type(error).__name__}")
            print(f"- Message: {str(error)}")
        ```
        """
        ...

    async def on_complete(self, messages: list[Message], agent: Agent | None) -> None:
        """Handle completion of a conversation.

        Called when a conversation is complete, providing access
        to the full message history and final agent.

        Args:
            messages: Complete conversation history
            agent: The final agent (None if no active agent)

        Example:
        ```python
        async def on_complete(
            self,
            messages: list[Message],
            agent: Agent | None
        ) -> None:
            print('Conversation summary:')
            print(f"- Messages: {len(messages)}")
            print(f"- Final agent: {agent.id if agent else 'None'}")

            # Log final response
            if messages and messages[-1].content:
                print(f"Final response: {messages[-1].content}")
        ```
        """
        ...


class LiteStreamHandler(StreamHandler):
    """Default no-op implementation of the StreamHandler protocol.

    Provides empty implementations of all event handlers.
    Useful as a base class for custom handlers that only need
    to implement specific events.

    Example:
    ```python
    class LoggingHandler(LiteStreamHandler):
        # Only override the events we care about
        async def on_stream(self, delta: Delta, agent: Agent) -> None:
            if delta.content:
                print(f"[{agent.id}]: {delta.content}")

        async def on_error(self, error: Exception, agent: Agent) -> None:
            print(f"Error in {agent.id}: {error}")
    ```
    """

    async def on_stream(self, delta: Delta, agent: Agent) -> None:
        """Handle streaming content updates."""
        pass

    async def on_tool_call(self, tool_call: ChatCompletionDeltaToolCall, agent: Agent) -> None:
        """Handle tool call events."""
        pass

    async def on_agent_switch(self, previous_agent: Agent | None, current_agent: Agent) -> None:
        """Handle agent switch events."""
        pass

    async def on_error(self, error: Exception, agent: Agent | None) -> None:
        """Handle error events."""
        pass

    async def on_complete(self, messages: list[Message], agent: Agent | None) -> None:
        """Handle conversation completion."""
        pass
