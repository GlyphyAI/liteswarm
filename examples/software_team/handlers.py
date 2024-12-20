# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing_extensions import override

from liteswarm.core.console_handler import ConsoleEventHandler
from liteswarm.types import ContextVariables, Plan, PlanFeedbackHandler
from liteswarm.types.events import (
    SwarmTeamPlanCompletedEvent,
    SwarmTeamPlanCreatedEvent,
    SwarmTeamTaskCompletedEvent,
    SwarmTeamTaskStartedEvent,
)


class SwarmEventHandler(ConsoleEventHandler):
    """Software team event handler with detailed task and plan tracking."""

    @override
    async def _handle_team_plan_created(self, event: SwarmTeamPlanCreatedEvent) -> None:
        """Handle team plan created events with detailed task breakdown.

        Args:
            event: Team plan created event to handle.
        """
        print("\nDevelopment Plan Created:")
        print("-------------------------")
        print(f"Plan ID: {event.plan.id}")
        for task in event.plan.tasks:
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

    @override
    async def _handle_team_task_started(self, event: SwarmTeamTaskStartedEvent) -> None:
        """Handle team task started events with assignment details.

        Args:
            event: Team task started event to handle.
        """
        assignee = event.task.assignee or "unassigned"
        print(f"\n🔵 Starting Task: {event.task.title}")
        print(f"Task ID: {event.task.id}")
        print(f"Assigned to: {assignee}")
        if event.task.dependencies:
            print(f"Dependencies: {', '.join(event.task.dependencies)}")

    @override
    async def _handle_team_task_completed(self, event: SwarmTeamTaskCompletedEvent) -> None:
        """Handle team task completed events with results.

        Args:
            event: Team task completed event to handle.
        """
        print(f"\n✅ Completed Task: {event.task.title}")
        print(f"Task ID: {event.task.id}")
        print(f"Assignee: {event.task.assignee}")
        if event.task_result.output:
            print("\nOutput:")
            print(event.task_result.output.model_dump_json(indent=2))

    @override
    async def _handle_team_plan_completed(self, event: SwarmTeamPlanCompletedEvent) -> None:
        """Handle team plan completed events with execution summary.

        Args:
            event: Team plan completed event to handle.
        """
        print(f"\n✨ Plan Completed: {event.plan.id}")
        print("\nExecution Summary:")
        print(f"Total Tasks: {len(event.plan.tasks)}")
        print(f"Results: {len(event.artifact.task_results)} tasks completed")
        if event.artifact.error:
            print(f"Error: {event.artifact.error}")
        print("\nAll tasks have been executed successfully.")


class InteractivePlanFeedbackHandler(PlanFeedbackHandler):
    """Interactive feedback handler for plan review and refinement."""

    @override
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
        print(f"Plan ID: {plan.id}")
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
