# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from collections.abc import Sequence

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.core import LiteSwarmStreamHandler
from liteswarm.experimental import LiteSwarmTeamStreamHandler
from liteswarm.types import (
    Agent,
    AgentResponse,
    ContextVariables,
    Message,
    Plan,
    PlanFeedbackHandler,
    Task,
    ToolCallResult,
)


class SwarmStreamHandler(LiteSwarmStreamHandler):
    """Stream handler for software team with real-time progress updates."""

    def __init__(self) -> None:
        """Initialize the stream handler."""
        self._last_agent: Agent | None = None
        self._current_content = ""

    async def on_stream(self, agent_response: AgentResponse) -> None:
        """Handle streaming content from agents.

        Args:
            agent_response: The agent response
        """
        if agent_response.finish_reason == "length":
            print("\n[...continuing...]", end="", flush=True)

        if content := agent_response.delta.content:
            # Only print agent ID prefix for the first character of a new message
            if self._last_agent != agent_response.agent:
                agent_id = agent_response.agent.id if agent_response.agent else "unknown"
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = agent_response.agent

            print(content, end="", flush=True)

        # Always ensure a newline at the end of a complete response
        if agent_response.finish_reason:
            print("", flush=True)

    async def on_error(self, error: Exception, agent: Agent | None) -> None:
        """Handle and display errors.

        Args:
            error: The error that occurred
            agent: The agent that encountered the error
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n\n❌ Error from {agent_id}: {str(error)}", file=sys.stderr, flush=True)
        self._last_agent = None

    async def on_agent_switch(self, previous_agent: Agent | None, next_agent: Agent) -> None:
        """Display agent switching information.

        Args:
            previous_agent: The agent being switched from
            next_agent: The agent being switched to
        """
        prev_id = previous_agent.id if previous_agent else "none"
        print(
            f"\n\n🔄 Switching from {prev_id} to {next_agent.id}...",
            flush=True,
        )
        self._last_agent = None

    async def on_tool_call(
        self, tool_call: ChatCompletionDeltaToolCall, agent: Agent | None
    ) -> None:
        """Display tool call information.

        Args:
            tool_call: The tool being called
            agent: The agent making the call
        """
        agent_id = agent.id if agent else "unknown"
        print(
            f"\n\n🔧 [{agent_id}] Calling: {tool_call.function.name}",
            flush=True,
        )
        self._last_agent = None

    async def on_tool_call_result(self, result: ToolCallResult, agent: Agent | None) -> None:
        """Display tool call results.

        Args:
            result: The result of the tool call
            agent: The agent that made the call
        """
        agent_id = agent.id if agent else "unknown"
        print(
            f"\n\n📎 [{agent_id}] Tool result received",
            flush=True,
        )
        self._last_agent = None

    async def on_complete(self, messages: Sequence[Message], agent: Agent | None) -> None:
        """Handle completion of agent tasks.

        Args:
            messages: The complete message history
            agent: The agent completing its task
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n\n✅ [{agent_id}] Task completed\n", flush=True)
        self._last_agent = None


class SwarmTeamStreamHandler(LiteSwarmTeamStreamHandler):
    async def on_plan_created(self, plan: Plan) -> None:
        """Print the created development plan."""
        print("\nDevelopment Plan Created:")
        print("-------------------------")
        for task in plan.tasks:
            print(f"\nTask: {task.title}")
            print(f"Task Type: {task.type}")
            print(f"Description: {task.description}")
            if task.dependencies:
                print(f"Dependencies: {', '.join(task.dependencies)}")
            if task.metadata:
                print("Metadata:")
                for key, value in task.metadata.items():
                    print(f"- {key}: {value}")
        print("-------------------------")

    async def on_task_started(self, task: Task) -> None:
        """Print when a task is started."""
        print(f"\nStarting Task: {task.title}")
        print(f"Assigned to: {task.assignee}")

    async def on_task_completed(self, task: Task) -> None:
        """Print when a task is completed."""
        print(f"\nCompleted Task: {task.title}")

    async def on_plan_completed(self, plan: Plan) -> None:
        """Print when the plan is completed."""
        print("\nPlan Completed!")
        print("All tasks have been executed successfully.")


class InteractivePlanFeedbackHandler(PlanFeedbackHandler):
    """Interactive feedback handler for plan review and refinement."""

    async def handle(
        self,
        plan: Plan,
        prompt: str,
        context: ContextVariables | None,
    ) -> tuple[str, ContextVariables | None] | None:
        """Handle plan feedback interactively.

        Args:
            plan: The current plan to review.
            prompt: The current prompt used to generate the plan.
            context: The current context variables.

        Returns:
            None if the plan is approved, or a tuple of (new_prompt, new_context)
            to create a new plan with the updated inputs.
        """
        print("\nProposed Plan:")
        print("-" * 30)
        for i, task in enumerate(plan.tasks, 1):
            print(f"{i}. {task.title}")
            if task.description:
                print(f"   {task.description}")
        print("-" * 30)

        choice = input("\n1. Approve and execute\n2. Provide feedback\n3. Exit\n\nYour choice (1-3): ")  # fmt: skip

        match choice:
            case "1":
                return None
            case "2":
                feedback = input("\nEnter your feedback: ")
                new_prompt = f"Original request: {prompt}\n\nPrevious attempt wasn't quite right because: {feedback}\n\nPlease try again with these adjustments."
                return new_prompt, context
            case "3":
                raise KeyboardInterrupt("User chose to exit")
            case _:
                print("Invalid choice. Please try again.")
                return await self.handle(plan, prompt, context)
