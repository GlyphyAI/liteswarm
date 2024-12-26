import sys
from typing import TYPE_CHECKING

from typing_extensions import override

from liteswarm.core.console_handler import ConsoleEventHandler
from liteswarm.types.events import (
    AgentResponseChunkEvent,
    AgentSwitchEvent,
    CompleteEvent,
    ErrorEvent,
    ToolCallResultEvent,
)

if TYPE_CHECKING:
    from liteswarm.types.swarm import Agent


class ReplEventHandler(ConsoleEventHandler):
    """REPL event handler with formatted output.

    Implements an event handler for the REPL environment that provides:
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
        handler = ReplEventHandler()
        result = await swarm.execute(
            agent=agent,
            prompt="Hello!",
            event_handler=handler,
        )

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
        """Initialize the event handler with usage tracking.

        Sets up the handler with initial state for tracking the last active
        agent to manage message continuity and formatting.
        """
        super().__init__()
        self._last_agent: Agent | None = None

    @override
    async def _handle_response(self, event: AgentResponseChunkEvent) -> None:
        """Handle agent response events.

        Args:
            event: Response event to handle.
        """
        completion = event.chunk.completion
        if completion.finish_reason == "length":
            print("\n[...continuing...]", end="", flush=True)

        if content := completion.delta.content:
            # Only print agent ID prefix for the first character of a new message
            if self._last_agent != event.chunk.agent:
                agent_id = event.chunk.agent.id
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = event.chunk.agent

            print(content, end="", flush=True)

        # Always ensure a newline at the end of a complete response
        if completion.finish_reason:
            print("", flush=True)

    @override
    async def _handle_error(self, event: ErrorEvent) -> None:
        """Handle error events.

        Args:
            event: Error event to handle.
        """
        agent_id = event.agent.id if event.agent else "unknown"
        print(f"\n❌ [{agent_id}] Error: {str(event.error)}", file=sys.stderr)
        self._last_agent = None

    @override
    async def _handle_agent_switch(self, event: AgentSwitchEvent) -> None:
        """Handle agent switch events.

        Args:
            event: Agent switch event to handle.
        """
        prev_id = event.previous.id if event.previous else "none"
        curr_id = event.current.id
        print(f"\n🔄 Switching from {prev_id} to {curr_id}...")

    @override
    async def _handle_tool_call_result(self, event: ToolCallResultEvent) -> None:
        """Handle tool result events.

        Args:
            event: Tool result event to handle.
        """
        agent_id = event.agent.id
        tool_call = event.tool_call_result.tool_call
        tool_name = tool_call.function.name
        tool_id = tool_call.id
        print(f"\n📎 [{agent_id}] Tool '{tool_name}' [{tool_id}] called")

    @override
    async def _handle_complete(self, event: CompleteEvent) -> None:
        """Handle completion events.

        Args:
            event: Completion event to handle.
        """
        agent_id = event.agent.id if event.agent else "unknown"
        print(f"\n✅ [{agent_id}] Completed", flush=True)
        self._last_agent = None
