# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from liteswarm.types.swarm_team import Plan, Task


class SwarmTeamStreamHandler(Protocol):
    """Protocol for stream handlers that handle task execution events.

    Defines the interface for handling various events during task and plan execution:
    - Task lifecycle events (start/completion)
    - Plan lifecycle events (creation/completion)

    Example:
    ```python
    class CustomStreamHandler(SwarmTeamStreamHandler):
        async def on_task_started(self, task: Task) -> None:
            print(f"Starting task: {task.title}")

        async def on_plan_created(self, plan: Plan) -> None:
            print(f"Created plan with {len(plan.tasks)} tasks")

        async def on_plan_completed(self, plan: Plan) -> None:
            print("Plan execution completed")

        async def on_task_completed(self, task: Task) -> None:
            print(f"Completed task: {task.title}")

    # Use in SwarmTeam
    team = SwarmTeam(
        swarm=swarm,
        members=members,
        task_definitions=task_defs,
        stream_handler=CustomStreamHandler()
    )
    ```
    """

    async def on_task_started(self, task: Task) -> None:
        """Handle task started event.

        Called when a task begins execution.

        Args:
            task: The task that is starting
        """
        ...

    async def on_plan_created(self, plan: Plan) -> None:
        """Handle plan created event.

        Called when a new plan is successfully created.

        Args:
            plan: The newly created plan
        """
        ...

    async def on_plan_completed(self, plan: Plan) -> None:
        """Handle plan completed event.

        Called when all tasks in a plan are completed.

        Args:
            plan: The completed plan
        """
        ...

    async def on_task_completed(self, task: Task) -> None:
        """Handle task completed event.

        Called when a task finishes execution.

        Args:
            task: The completed task
        """
        ...


class LiteSwarmTeamStreamHandler(SwarmTeamStreamHandler):
    """Default no-op implementation of the SwarmTeamStreamHandler protocol.

    Provides empty implementations of all event handlers.
    Useful as a base class for custom handlers that only need
    to implement specific events.

    Example:
    ```python
    class TaskLogHandler(LiteSwarmTeamStreamHandler):
        # Only override the events we care about
        async def on_task_started(self, task: Task) -> None:
            print(f"Starting: {task.title}")

        async def on_task_completed(self, task: Task) -> None:
            print(f"Completed: {task.title}")
    ```
    """

    async def on_task_started(self, task: Task) -> None:
        """Handle task started event."""
        pass

    async def on_plan_created(self, plan: Plan) -> None:
        """Handle plan created event."""
        pass

    async def on_plan_completed(self, plan: Plan) -> None:
        """Handle plan completed event."""
        pass

    async def on_task_completed(self, task: Task) -> None:
        """Handle task completed event."""
        pass
