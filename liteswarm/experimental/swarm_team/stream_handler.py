# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from liteswarm.types.swarm_team import Plan, Task


class SwarmTeamStreamHandler(Protocol):
    """Protocol for stream handlers that handle task execution."""

    async def on_task_started(self, task: Task) -> None:
        """Handle task started event."""
        ...

    async def on_plan_created(self, plan: Plan) -> None:
        """Handle plan created event."""
        ...

    async def on_plan_completed(self, plan: Plan) -> None:
        """Handle plan completed event."""
        ...

    async def on_task_completed(self, task: Task) -> None:
        """Handle task completed event."""
        ...


class LiteSwarmTeamStreamHandler(SwarmTeamStreamHandler):
    """Lite implementation of the SwarmTeamStreamHandler protocol."""

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
