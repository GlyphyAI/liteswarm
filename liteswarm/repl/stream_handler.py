# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.core.stream_handler import SwarmStreamHandler
from liteswarm.types.swarm import (
    Agent,
    Delta,
    Message,
    ToolCallAgentResult,
    ToolCallMessageResult,
    ToolCallResult,
)


class ReplStreamHandler(SwarmStreamHandler):
    """Stream handler for REPL interface with better formatting.

    Implements a specialized stream handler for the REPL environment that provides:
    - Formatted output with agent identification
    - Visual indicators for different event types
    - Real-time status updates
    - Error handling with clear feedback
    - Tool usage tracking and reporting

    The handler uses various emoji indicators to make different types of events
    visually distinct and easy to follow:
    - 🔄 Agent switches
    - 🔧 Tool calls
    - 📎 Tool results
    - ❌ Errors
    - ✅ Completion

    Example:
        ```python
        handler = ReplStreamHandler()
        swarm = Swarm(stream_handler=handler)

        # Handler will automatically format output:
        # [agent_id] This is a response...
        # 🔧 [agent_id] Using tool_name [tool_id]
        # 📎 [agent_id] Got result: tool result
        # ✅ [agent_id] Completed
        ```

    Notes:
        - Maintains agent context between messages
        - Handles continuation indicators for long responses
        - Provides clear error feedback
        - Ensures consistent formatting across all events
    """

    def __init__(self) -> None:
        """Initialize the stream handler with usage tracking.

        Sets up the handler with initial state for tracking the last active
        agent to manage message continuity and formatting.
        """
        self._last_agent: Agent | None = None

    async def on_stream(
        self,
        chunk: Delta,
        agent: Agent | None,
    ) -> None:
        """Handle streaming content from agents.

        Manages the real-time display of agent responses with:
        - Agent identification prefixes
        - Continuation indicators for long responses
        - Proper formatting and flushing
        - Message continuity tracking

        Args:
            chunk: Content update containing:
                - content: Optional text content
                - finish_reason: Optional completion status
            agent: The agent generating the content, or None if system message.

        Notes:
            - Only shows agent prefix for first chunk of new messages
            - Handles length-limited responses with [...continuing...] indicator
            - Maintains visual continuity for multi-chunk responses
            - Ensures immediate output through flushing
        """
        if chunk.content:
            # Show a continuation indicator if the response ended due to a length limit
            if getattr(chunk, "finish_reason", None) == "length":
                print("\n[...continuing...]", end="", flush=True)

            # Only print agent ID prefix for the first character of a new message
            if not hasattr(self, "_last_agent") or self._last_agent != agent:
                agent_id = agent.id if agent else "unknown"
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = agent

            print(f"{chunk.content}", end="", flush=True)

    async def on_error(
        self,
        error: Exception,
        agent: Agent | None,
    ) -> None:
        """Handle and display errors.

        Provides clear error feedback with:
        - Visual error indicator (❌)
        - Agent identification
        - Error message details
        - Proper error stream routing

        Args:
            error: The exception that occurred.
            agent: The agent that encountered the error, or None if system error.

        Notes:
            - Outputs to stderr for proper error handling
            - Resets message continuity tracking
            - Uses consistent error formatting
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n❌ [{agent_id}] Error: {str(error)}", file=sys.stderr)
        self._last_agent = None

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None:
        """Display agent switching information.

        Provides visual feedback for agent transitions with:
        - Switch indicator (🔄)
        - Previous agent identification
        - Next agent identification
        - Clear transition messaging

        Args:
            previous_agent: The agent being switched from, or None if first agent.
            next_agent: The agent being switched to.

        Notes:
            - Resets message continuity tracking
            - Handles initial agent assignment (previous=None)
            - Ensures clear transition visibility
        """
        print(
            f"\n🔄 Switching from {previous_agent.id if previous_agent else 'none'} to {next_agent.id}..."
        )
        self._last_agent = None

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None:
        """Handle completion of agent tasks.

        Provides completion feedback with:
        - Completion indicator (✅)
        - Final agent identification
        - Clear completion status

        Args:
            messages: The complete conversation history.
            agent: The final agent, or None if no agent active.

        Notes:
            - Resets message continuity tracking
            - Provides clear task completion signal
            - Maintains consistent formatting
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n✅ [{agent_id}] Completed")
        self._last_agent = None

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None:
        """Display tool call information.

        Provides tool usage feedback with:
        - Tool call indicator (🔧)
        - Agent identification
        - Tool name and ID
        - Clear action indication

        Args:
            tool_call: Details of the tool being called, including:
                - id: Unique tool call identifier
                - function: Tool name and parameters
            agent: The agent making the call, or None if system call.

        Notes:
            - Resets message continuity tracking
            - Shows tool call initiation clearly
            - Maintains consistent formatting
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n🔧 [{agent_id}] Using {tool_call.function.name} [{tool_call.id}]")
        self._last_agent = None

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None:
        """Display tool call results.

        Provides comprehensive tool result feedback with:
        - Result indicator (📎 or 🔧)
        - Agent identification
        - Tool name and ID
        - Result content or agent switch information

        Handles different result types:
        - ToolCallMessageResult: Shows function return value
        - ToolCallAgentResult: Shows agent switch information
        - Other results: Shows basic completion status

        Args:
            tool_call_result: The result of the tool call, which can be:
                - ToolCallMessageResult: Contains function result
                - ToolCallAgentResult: Contains new agent info
                - Other ToolCallResult types
            agent: The agent that made the call, or None if system call.

        Notes:
            - Uses different indicators for different result types
            - Resets message continuity tracking
            - Maintains consistent formatting
            - Handles all result types appropriately
        """
        agent_id = agent.id if agent else "unknown"

        match tool_call_result:
            case ToolCallMessageResult() as tool_call_message_result:
                print(
                    f"\n📎 [{agent_id}] Got result for {tool_call_message_result.tool_call.function.name} [{tool_call_message_result.tool_call.id}]: {tool_call_message_result.message.content}"
                )
            case ToolCallAgentResult() as tool_call_agent_result:
                print(
                    f"\n🔧 [{agent_id}] Switching to: {tool_call_agent_result.agent.id} [{tool_call_agent_result.tool_call.id}]"
                )
            case _:
                print(
                    f"\n📎 [{agent_id}] Got result for: {tool_call_result.tool_call.function.name} [{tool_call_result.tool_call.id}]"
                )

        self._last_agent = None
