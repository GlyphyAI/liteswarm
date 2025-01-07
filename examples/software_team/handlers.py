# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing_extensions import override

from liteswarm.repl.event_handler import ConsoleEventHandler
from liteswarm.types.events import (
    PlanCreateEvent,
    PlanExecutionCompleteEvent,
    TaskCompleteEvent,
    TaskStartEvent,
)


class EventHandler(ConsoleEventHandler):
    """Software team event handler with detailed task and plan tracking."""

    @override
    def _handle_plan_create(self, event: PlanCreateEvent) -> None:
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
    def _handle_task_start(self, event: TaskStartEvent) -> None:
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
    def _handle_task_complete(self, event: TaskCompleteEvent) -> None:
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
    def _handle_plan_execution_complete(self, event: PlanExecutionCompleteEvent) -> None:
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
