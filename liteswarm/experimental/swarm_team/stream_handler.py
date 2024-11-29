# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from liteswarm.types.swarm_team import Plan, Task


class SwarmTeamStreamHandler(Protocol):
    """Protocol for handling task and plan execution events.

    Defines methods for handling lifecycle events during task and plan execution,
    allowing custom handling of progress updates and state changes.

    Examples:
        Create a custom stream handler:
            ```python
            class LoggingStreamHandler(SwarmTeamStreamHandler):
                async def on_task_started(self, task: Task) -> None:
                    print(f"Starting task {task.id}: {task.title}")

                async def on_task_completed(self, task: Task) -> None:
                    print(f"Completed task {task.id} with status: {task.status}")

                async def on_plan_created(self, plan: Plan) -> None:
                    print(f"Created plan with {len(plan.tasks)} tasks")

                async def on_plan_completed(self, plan: Plan) -> None:
                    print(f"Plan completed with status: {plan.status}")

            # Use with SwarmTeam
            team = SwarmTeam(
                swarm=swarm,
                members=members,
                task_definitions=task_defs,
                stream_handler=LoggingStreamHandler()
            )
            ```
    """

    async def on_task_started(self, task: Task) -> None:
        """Handle task started event.

        Called when a task begins execution. Use this to track task progress
        or update external systems.

        Args:
            task: Task that is starting execution.
        """
        ...

    async def on_plan_created(self, plan: Plan) -> None:
        """Handle plan created event.

        Called when a new plan is successfully created. Use this to analyze
        the plan or prepare resources.

        Args:
            plan: Newly created execution plan.
        """
        ...

    async def on_plan_completed(self, plan: Plan) -> None:
        """Handle plan completed event.

        Called when all tasks in a plan are completed. Use this to perform
        cleanup or trigger follow-up actions.

        Args:
            plan: Plan that has completed execution.
        """
        ...

    async def on_task_completed(self, task: Task) -> None:
        """Handle task completed event.

        Called when a task finishes execution. Use this to process results
        or update progress tracking.

        Args:
            task: Task that has completed execution.
        """
        ...


class LiteSwarmTeamStreamHandler(SwarmTeamStreamHandler):
    """Default no-op implementation of the stream handler protocol.

    Provides empty implementations of all event handlers. Use as a base class
    for custom handlers that only need to implement specific events.

    Examples:
        Create a task-focused handler:
            ```python
            class TaskProgressHandler(LiteSwarmTeamStreamHandler):
                def __init__(self):
                    self.total_tasks = 0
                    self.completed_tasks = 0

                async def on_plan_created(self, plan: Plan) -> None:
                    self.total_tasks = len(plan.tasks)
                    self.completed_tasks = 0
                    print(f"Starting execution of {self.total_tasks} tasks")

                async def on_task_completed(self, task: Task) -> None:
                    self.completed_tasks += 1
                    progress = (self.completed_tasks / self.total_tasks) * 100
                    print(f"Progress: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks})")
            ```
    """

    async def on_task_started(self, task: Task) -> None:
        """Handle task started event.

        Args:
            task: Task that is starting execution.
        """
        pass

    async def on_plan_created(self, plan: Plan) -> None:
        """Handle plan created event.

        Args:
            plan: Newly created execution plan.
        """
        pass

    async def on_plan_completed(self, plan: Plan) -> None:
        """Handle plan completed event.

        Args:
            plan: Plan that has completed execution.
        """
        pass

    async def on_task_completed(self, task: Task) -> None:
        """Handle task completed event.

        Args:
            task: Task that has completed execution.
        """
        pass
