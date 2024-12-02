# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.types.swarm import Agent, Delta, Message


class SwarmStreamHandler(Protocol):
    """Protocol for handlers that process streaming events from agents.

    This protocol defines the interface for handling various events during agent
    interactions in a streaming context. Handlers receive and process events for:
    - Content streaming updates from agents
    - Tool call invocations and results
    - Agent switching operations
    - Error conditions and recovery
    - Conversation completion

    The protocol enables real-time monitoring and response to agent activities,
    making it suitable for logging, debugging, and user interface updates.

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

    Notes:
        - All methods are asynchronous to support non-blocking operations
        - Handlers should be lightweight to avoid blocking the event stream
        - Errors in handlers should be caught and handled internally
        - Agent may be None in error and complete events if no agent is active
    """

    async def on_stream(self, delta: Delta, agent: Agent) -> None:
        """Handle streaming content updates from an agent.

        Called each time new content is received from an agent, including both
        text content and tool call updates. This method should process updates
        quickly to avoid blocking the stream.

        Args:
            delta: The content or tool call update, containing:
                - content: Optional new text content
                - tool_calls: Optional list of tool call updates
            agent: The agent generating the content, providing:
                - id: Agent identifier
                - model: LLM model information
                - other agent-specific attributes

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

        Notes:
            - May be called frequently with small content updates
            - Should avoid expensive operations
            - Can receive both content and tool calls in same delta
        """
        ...

    async def on_tool_call(self, tool_call: ChatCompletionDeltaToolCall, agent: Agent) -> None:
        """Handle a tool call from an agent.

        Called when an agent initiates a tool call, before the tool is executed.
        This provides an opportunity to log or monitor tool usage, validate calls,
        or prepare for tool execution.

        Args:
            tool_call: Details of the tool being called, including:
                - id: Unique tool call identifier
                - function: Function name and arguments
                - other tool-specific metadata
            agent: The agent making the call, providing:
                - id: Agent identifier
                - tools: Available tool configurations
                - other agent-specific attributes

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

        Notes:
            - Called before tool execution begins
            - Can be used for validation or logging
            - Should not modify the tool call
        """
        ...

    async def on_agent_switch(self, previous_agent: Agent | None, current_agent: Agent) -> None:
        """Handle an agent switch event.

        Called when the conversation transitions from one agent to another,
        typically due to a tool call result or explicit switch request.
        The first agent in a conversation will have previous_agent as None.

        Args:
            previous_agent: The agent being switched from, or None if this
                is the first agent in the conversation.
            current_agent: The agent being switched to, never None.

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

        Notes:
            - First agent switch has previous_agent as None
            - Can be used to track agent transitions
            - May occur multiple times in a conversation
        """
        ...

    async def on_error(self, error: Exception, agent: Agent | None) -> None:
        """Handle an error during agent execution.

        Called when an error occurs during any phase of agent operation,
        including content generation, tool calls, or response processing.
        The agent may be None if the error occurred outside agent context.

        Args:
            error: The exception that occurred, providing:
                - Error type and message
                - Stack trace if available
                - Additional error context
            agent: The agent that encountered the error, or None if no
                agent was active when the error occurred.

        Example:
            ```python
            async def on_error(self, error: Exception, agent: Agent) -> None:
                print(f"Error in agent {agent.id}:")
                print(f"- Type: {type(error).__name__}")
                print(f"- Message: {str(error)}")
            ```

        Notes:
            - Called for all types of errors
            - Agent may be None for system-level errors
            - Can be used for error logging and monitoring
            - Should not raise exceptions
        """
        ...

    async def on_complete(self, messages: list[Message], agent: Agent | None) -> None:
        """Handle completion of a conversation.

        Called when a conversation reaches its natural conclusion or is
        terminated. Provides access to the complete message history and
        final agent state for analysis or cleanup.

        Args:
            messages: Complete conversation history, including:
                - All agent messages and responses
                - Tool calls and results
                - System messages if present
            agent: The final agent in the conversation, or None if no
                agent was active at completion.

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

        Notes:
            - Called once per conversation
            - Good place for conversation analysis
            - Can be used for cleanup operations
            - Should handle missing agent gracefully
        """
        ...


class LiteSwarmStreamHandler(SwarmStreamHandler):
    """Default no-op implementation of the StreamHandler protocol.

    Provides empty implementations of all event handlers defined in the
    StreamHandler protocol. This class serves multiple purposes:
    - Base class for custom handlers that only need some events
    - Default handler when no custom handling is needed
    - Example of minimal protocol implementation

    The class implements all protocol methods with pass statements,
    allowing subclasses to override only the events they care about.

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

    Notes:
        - All methods are implemented as no-ops
        - Safe to use as a base class
        - Provides protocol compliance by default
        - Suitable for testing and development
    """

    async def on_stream(self, delta: Delta, agent: Agent) -> None:
        """Handle streaming content updates.

        Args:
            delta: Content or tool call update.
            agent: Agent generating the content.
        """
        pass

    async def on_tool_call(self, tool_call: ChatCompletionDeltaToolCall, agent: Agent) -> None:
        """Handle tool call events.

        Args:
            tool_call: Details of the tool call.
            agent: Agent making the call.
        """
        pass

    async def on_agent_switch(self, previous_agent: Agent | None, current_agent: Agent) -> None:
        """Handle agent switch events.

        Args:
            previous_agent: Agent being switched from.
            current_agent: Agent being switched to.
        """
        pass

    async def on_error(self, error: Exception, agent: Agent | None) -> None:
        """Handle error events.

        Args:
            error: Exception that occurred.
            agent: Agent that encountered the error.
        """
        pass

    async def on_complete(self, messages: list[Message], agent: Agent | None) -> None:
        """Handle conversation completion.

        Args:
            messages: Complete conversation history.
            agent: Final agent in the conversation.
        """
        pass
