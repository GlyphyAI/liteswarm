# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, ContextVariables, Result

TTask = TypeVar("TTask", bound="Task")

TaskStatus = Literal["pending", "in_progress", "completed"]
PlanStatus = Literal["draft", "approved", "in_progress", "completed"]
TaskInstructions = str | Callable[[TTask, ContextVariables], str]


class Task(BaseModel):
    """Base class for tasks in a plan."""

    id: str
    title: str
    description: str | None = None
    status: TaskStatus = "pending"
    assignee: str | None = None
    metadata: dict[str, Any] = {}


class Plan(BaseModel, Generic[TTask]):
    """Base class for development plans."""

    tasks: list[TTask]
    status: PlanStatus = "draft"
    metadata: dict[str, Any] = {}


class PlannerAgent(Protocol, Generic[TTask]):
    """Protocol for planner agents that create task plans."""

    async def create_plan(self, prompt: str, context: dict[str, Any]) -> Result[Plan[TTask]]:
        """Create a plan from the given prompt and context."""
        ...


class SwarmTeamStreamHandler(Protocol[TTask]):
    """Protocol for stream handlers that handle task execution."""

    async def on_task_started(self, task: TTask) -> None:
        """Handle task started event."""
        ...

    async def on_plan_created(self, plan: Plan[TTask]) -> None:
        """Handle plan created event."""
        ...

    async def on_plan_completed(self, plan: Plan[TTask]) -> None:
        """Handle plan completed event."""
        ...

    async def on_task_completed(self, task: TTask) -> None:
        """Handle task completed event."""
        ...


@dataclass
class TeamMember:
    """Represents a team member that can execute tasks."""

    agent: Agent
    task_types: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def default_instructions(task: Task, context: ContextVariables) -> str:
    """Default task instructions builder."""
    return f"Complete the following task: {task.title}\n\n{task.description}"


class SwarmTeam(Generic[TTask]):
    """Orchestrates a team of agents working on tasks according to a plan."""

    def __init__(
        self,
        planner: PlannerAgent[TTask],
        members: list[TeamMember],
        swarm: Swarm | None = None,
        instructions: TaskInstructions | None = None,
        stream_handler: SwarmTeamStreamHandler[TTask] | None = None,
    ) -> None:
        """Initialize a new SwarmTeam.

        Args:
            planner: Agent responsible for creating plans
            members: List of team members that can execute tasks
            swarm: Optional Swarm instance to use (creates new one if not provided)
            instructions: Optional instructions for tasks
            stream_handler: Optional stream handler for handling events
        """
        self.planner = planner
        self.members = {member.agent.id: member for member in members}
        self.available_types = {task_type for member in members for task_type in member.task_types}
        self.swarm = swarm or Swarm()
        self.instructions: TaskInstructions = instructions or default_instructions
        self.stream_handler = stream_handler

        # Internal state
        self._current_plan: Plan[TTask] | None = None
        self._context: dict[str, Any] = {}

    async def create_plan(
        self,
        prompt: str,
        feedback: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Plan[TTask]:
        """Create a new plan from the prompt and optional feedback.

        Args:
            prompt: User prompt describing what needs to be done
            feedback: Optional feedback on previous plan iteration
            context: Optional context variables for the planner

        Returns:
            The created plan

        Raises:
            ValueError: If plan creation fails
        """
        if context:
            self._context.update(context)

        # Add available engineer types to context
        self._context["available_engineer_types"] = list(self.available_types)

        if feedback:
            full_prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"
        else:
            full_prompt = prompt

        result = await self.planner.create_plan(full_prompt, self._context)
        if not result.value:
            raise ValueError("Failed to create plan")

        self._current_plan = result.value

        if self.stream_handler:
            await self.stream_handler.on_plan_created(self._current_plan)

        return self._current_plan

    async def execute_plan(self) -> None:
        """Execute all tasks in the current plan in order.

        Raises:
            ValueError: If no approved plan exists
        """
        if not self._current_plan:
            raise ValueError("No plan to execute")

        self._current_plan.status = "in_progress"

        for task in self._current_plan.tasks:
            if task.status != "pending":
                continue

            instructions = self.instructions
            if callable(instructions):
                instructions = instructions(task, self._context)

            await self.execute_task(task, instructions)

        self._current_plan.status = "completed"

        if self.stream_handler:
            await self.stream_handler.on_plan_completed(self._current_plan)

    async def execute_task(self, task: TTask, prompt: str) -> Result[Any]:
        """Execute a single task using the appropriate team member."""
        engineer_type = getattr(task, "engineer_type", None) or task.metadata.get("type")
        if not engineer_type:
            raise ValueError(f"No engineer type specified for task: {task.id}")

        assignee = None
        for member in self.members.values():
            if engineer_type in member.task_types:
                assignee = member
                break

        if not assignee:
            raise ValueError(
                f"No team member found for engineer type '{engineer_type}' in task: {task.id}"
            )

        if self.stream_handler:
            await self.stream_handler.on_task_started(task)

        task.status = "in_progress"
        task.assignee = assignee.agent.id

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=prompt,
            context_variables={
                "task": task.model_dump(),
                **self._context,
            },
        )

        task.status = "completed"

        if self.stream_handler:
            await self.stream_handler.on_task_completed(task)

        return Result(value=result.content)
