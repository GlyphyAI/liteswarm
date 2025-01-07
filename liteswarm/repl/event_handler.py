# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from typing import TYPE_CHECKING

from liteswarm.types.events import (
    AgentCompleteEvent,
    AgentExecutionCompleteEvent,
    AgentExecutionStartEvent,
    AgentResponseChunkEvent,
    AgentSwitchEvent,
    ErrorEvent,
    PlanCreateEvent,
    PlanExecutionCompleteEvent,
    PlanExecutionStartEvent,
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
    feedback for each event type.

    Examples:
        Handling events from the agent execution stream:
            ```python
            handler = ConsoleEventHandler()

            async for event in swarm.execute(agent, messages):
                handler.on_event(event)
            ```

    Note:
        This is an internal event handler and is not intended to be used by
        end-users. You should handle events on your own when receiving them
        from the agent execution stream.
    """

    def __init__(self) -> None:
        """Initialize event handler with message continuity tracking."""
        super().__init__()
        self._last_agent: Agent | None = None

    def on_event(self, event: SwarmEvent) -> None:
        """Process and display a Swarm event with appropriate formatting."""
        match event:
            # Agent Events
            case AgentExecutionStartEvent():
                self._handle_agent_execution_start(event)
            case AgentResponseChunkEvent():
                self._handle_response_chunk(event)
            case ToolCallResultEvent():
                self._handle_tool_call_result(event)
            case AgentSwitchEvent():
                self._handle_agent_switch(event)
            case AgentCompleteEvent():
                self._handle_agent_complete(event)
            case AgentExecutionCompleteEvent():
                self._handle_agent_execution_complete(event)

            # Team Events
            case PlanCreateEvent():
                self._handle_plan_create(event)
            case PlanExecutionStartEvent():
                self._handle_plan_execution_start(event)
            case TaskStartEvent():
                self._handle_task_start(event)
            case TaskCompleteEvent():
                self._handle_task_complete(event)
            case PlanExecutionCompleteEvent():
                self._handle_plan_execution_complete(event)

            # System Events
            case ErrorEvent():
                self._handle_error(event)

    # ================================================
    # MARK: Agent Events
    # ================================================

    def _handle_agent_execution_start(self, event: AgentExecutionStartEvent) -> None:
        """Display agent execution start message."""
        agent_id = event.agent.id
        print(f"\n\n🔧 [{agent_id}] Agent execution started\n", flush=True)

    def _handle_response_chunk(self, event: AgentResponseChunkEvent) -> None:
        """Display streaming response chunk with agent context."""
        completion = event.response_chunk.completion
        if completion.finish_reason == "length":
            print("\n[...continuing...]", end="", flush=True)

        if content := completion.delta.content:
            if self._last_agent != event.agent:
                agent_id = event.agent.id
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = event.agent

            print(content, end="", flush=True)

        if completion.finish_reason:
            print("", flush=True)

    def _handle_tool_call_result(self, event: ToolCallResultEvent) -> None:
        """Display tool call result with function details."""
        agent_id = event.agent.id
        tool_call = event.tool_call_result.tool_call
        tool_name = tool_call.function.name
        tool_id = tool_call.id
        print(f"\n📎 [{agent_id}] Tool '{tool_name}' [{tool_id}] called")

    def _handle_agent_switch(self, event: AgentSwitchEvent) -> None:
        """Display agent switch notification."""
        prev_id = event.prev_agent.id if event.prev_agent else "none"
        curr_id = event.next_agent.id
        print(f"\n🔄 Switching from {prev_id} to {curr_id}...")

    def _handle_agent_complete(self, event: AgentCompleteEvent) -> None:
        """Display agent completion status."""
        agent_id = event.agent.id
        print(f"\n✅ [{agent_id}] Completed", flush=True)
        self._last_agent = None

    def _handle_agent_execution_complete(self, event: AgentExecutionCompleteEvent) -> None:
        """Display execution completion status."""
        self._last_agent_id = None
        print("\n\n✅ Completed\n", flush=True)

    # ================================================
    # MARK: Team Events
    # ================================================

    def _handle_plan_create(self, event: PlanCreateEvent) -> None:
        """Display plan creation details."""
        plan_id = event.plan.id
        task_count = len(event.plan.tasks)
        print(f"\n\n🔧 Plan created (task count: {task_count}): {plan_id}\n", flush=True)

    def _handle_plan_execution_start(self, event: PlanExecutionStartEvent) -> None:
        """Display plan execution start notification."""
        plan_id = event.plan.id
        print(f"\n\n🔧 Plan execution started: {plan_id}\n", flush=True)

    def _handle_task_start(self, event: TaskStartEvent) -> None:
        """Display task start details."""
        task_id = event.task.id
        assignee_id = event.task.assignee if event.task.assignee else "unknown"
        print(f"\n\n🔧 Task started: {task_id} by {assignee_id}\n", flush=True)

    def _handle_task_complete(self, event: TaskCompleteEvent) -> None:
        """Display task completion status."""
        task_id = event.task.id
        assignee_id = event.task.assignee if event.task.assignee else "unknown"
        task_status = event.task.status
        print(
            f"\n\n✅ Task completed: {task_id} by {assignee_id} (status: {task_status})\n",
            flush=True,
        )

    def _handle_plan_execution_complete(self, event: PlanExecutionCompleteEvent) -> None:
        """Display plan completion summary."""
        plan_id = event.plan.id
        task_count = len(event.plan.tasks)
        print(f"\n\n✅ Plan completed (task count: {task_count}): {plan_id}\n", flush=True)

    # ================================================
    # MARK: System Events
    # ================================================

    def _handle_error(self, event: ErrorEvent) -> None:
        """Display error message with agent context."""
        agent_id = event.agent.id if event.agent else "unknown"
        print(f"\n❌ [{agent_id}] Error: {str(event.error)}", file=sys.stderr)
        self._last_agent = None
