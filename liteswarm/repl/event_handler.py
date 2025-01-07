# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from typing import TYPE_CHECKING

from liteswarm.types.events import (
    AgentCompleteEvent,
    AgentResponseChunkEvent,
    AgentSwitchEvent,
    ErrorEvent,
    ExecutionCompleteEvent,
    PlanCreateEvent,
    PlanExecutionCompleteEvent,
    SwarmEvent,
    TaskCompleteEvent,
    TaskStartEvent,
    ToolCallResultEvent,
)

if TYPE_CHECKING:
    from liteswarm.types.swarm import Agent


class ConsoleEventHandler:
    """Console event handler providing formatted output for REPL interactions.

    Processes and displays Swarm events with distinct visual indicators for
    different event types. Maintains message continuity and provides clear
    feedback for each event type:

    Event Types:
        - Agent responses with message continuity
        - Agent switches with transition indicators
        - Tool calls with function details
        - Task and plan status updates
        - Error messages and completion states

    Example:
        ```python
        handler = ConsoleEventHandler()
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
    """

    def __init__(self) -> None:
        """Initialize event handler with message continuity tracking."""
        super().__init__()
        self._last_agent: Agent | None = None

    def on_event(self, event: SwarmEvent) -> None:
        """Process and display Swarm events with appropriate formatting.

        Formats each event type with distinct visual indicators and context:
        - Agent responses include agent ID and content
        - Tool calls show function name and arguments
        - Error messages provide clear feedback
        - Status updates show progress indicators

        Args:
            event: Swarm event to process and display.
        """
        match event:
            # Swarm Events
            case AgentResponseChunkEvent():
                self._handle_response_chunk(event)
            case ToolCallResultEvent():
                self._handle_tool_call_result(event)
            case AgentSwitchEvent():
                self._handle_agent_switch(event)
            case AgentCompleteEvent():
                self._handle_agent_complete(event)
            case ErrorEvent():
                self._handle_error(event)
            case ExecutionCompleteEvent():
                self._handle_complete(event)

            # Swarm Team Events
            case PlanCreateEvent():
                self._handle_team_plan_created(event)
            case TaskStartEvent():
                self._handle_team_task_started(event)
            case TaskCompleteEvent():
                self._handle_team_task_completed(event)
            case PlanExecutionCompleteEvent():
                self._handle_team_plan_completed(event)

    def _handle_response_chunk(self, event: AgentResponseChunkEvent) -> None:
        """Process response chunk from agent.

        Args:
            event: Response chunk event.
        """
        completion = event.response_chunk.completion
        if completion.finish_reason == "length":
            print("\n[...continuing...]", end="", flush=True)

        if content := completion.delta.content:
            # Only print agent ID prefix for the first character of a new message
            if self._last_agent != event.agent:
                agent_id = event.agent.id
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = event.agent

            print(content, end="", flush=True)

        # Always ensure a newline at the end of a complete response
        if completion.finish_reason:
            print("", flush=True)

    def _handle_tool_call_result(self, event: ToolCallResultEvent) -> None:
        """Process tool call result.

        Args:
            event: Tool call result event.
        """
        agent_id = event.agent.id
        tool_call = event.tool_call_result.tool_call
        tool_name = tool_call.function.name
        tool_id = tool_call.id
        print(f"\n📎 [{agent_id}] Tool '{tool_name}' [{tool_id}] called")

    def _handle_agent_switch(self, event: AgentSwitchEvent) -> None:
        """Process agent switch.

        Args:
            event: Agent switch event.
        """
        prev_id = event.prev_agent.id if event.prev_agent else "none"
        curr_id = event.next_agent.id
        print(f"\n🔄 Switching from {prev_id} to {curr_id}...")

    def _handle_agent_complete(self, event: AgentCompleteEvent) -> None:
        """Process agent completion.

        Args:
            event: Agent completion event.
        """
        agent_id = event.agent.id
        print(f"\n✅ [{agent_id}] Completed", flush=True)
        self._last_agent = None

    def _handle_error(self, event: ErrorEvent) -> None:
        """Process error.

        Args:
            event: Error event.
        """
        agent_id = event.agent.id if event.agent else "unknown"
        print(f"\n❌ [{agent_id}] Error: {str(event.error)}", file=sys.stderr)
        self._last_agent = None

    def _handle_complete(self, event: ExecutionCompleteEvent) -> None:
        """Process completion.

        Args:
            event: Completion event.
        """
        self._last_agent_id = None
        print("\n\n✅ Completed\n", flush=True)

    def _handle_team_plan_created(self, event: PlanCreateEvent) -> None:
        """Process team plan creation.

        Args:
            event: Plan creation event.
        """
        plan_id = event.plan.id
        task_count = len(event.plan.tasks)
        print(f"\n\n🔧 Plan created (task count: {task_count}): {plan_id}\n", flush=True)

    def _handle_team_task_started(self, event: TaskStartEvent) -> None:
        """Process team task start.

        Args:
            event: Task start event.
        """
        task_id = event.task.id
        assignee_id = event.task.assignee if event.task.assignee else "unknown"
        print(f"\n\n🔧 Task started: {task_id} by {assignee_id}\n", flush=True)

    def _handle_team_task_completed(self, event: TaskCompleteEvent) -> None:
        """Process team task completion.

        Args:
            event: Task completion event.
        """
        task_id = event.task.id
        assignee_id = event.task.assignee if event.task.assignee else "unknown"
        task_status = event.task.status
        print(
            f"\n\n✅ Task completed: {task_id} by {assignee_id} (status: {task_status})\n",
            flush=True,
        )

    def _handle_team_plan_completed(self, event: PlanExecutionCompleteEvent) -> None:
        """Process team plan completion.

        Args:
            event: Plan completion event.
        """
        plan_id = event.plan.id
        task_count = len(event.plan.tasks)
        print(f"\n\n✅ Plan completed (task count: {task_count}): {plan_id}\n", flush=True)
