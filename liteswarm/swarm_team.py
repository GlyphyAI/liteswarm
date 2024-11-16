# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, ContextVariables, Result
from liteswarm.utils import extract_json

# ================================================
# MARK: Swarm Team Types
# ================================================

TaskInstructions = str | Callable[["Task", ContextVariables], str]


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Task(BaseModel):
    """Base class for tasks in a plan."""

    id: str
    title: str
    description: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    assignee: str | None = None
    task_type: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Base class for development plans."""

    tasks: list[Task]
    status: PlanStatus = PlanStatus.DRAFT
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_dependencies(self) -> list[str]:
        """Validate that all task dependencies exist."""
        task_ids = {task.id for task in self.tasks}
        errors = []

        for task in self.tasks:
            invalid_deps = [dep for dep in task.dependencies if dep not in task_ids]
            if invalid_deps:
                errors.append(f"Task {task.id} has invalid dependencies: {invalid_deps}")

        return errors

    def get_next_tasks(self) -> list[Task]:
        """Get tasks that are ready to be executed (all dependencies completed)."""
        completed_tasks = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        return [
            task
            for task in self.tasks
            if task.status == TaskStatus.PENDING
            and all(dep in completed_tasks for dep in task.dependencies)
        ]


class TeamMember(BaseModel):
    """Represents a team member that can execute tasks."""

    agent: Agent
    task_types: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ================================================
# MARK: Task Planner
# ================================================


class PromptTemplate(Protocol):
    """Protocol for prompt templates."""

    @property
    def template(self) -> str:
        """Return the template string."""
        ...

    def format_context(self, prompt: str, context: dict[str, Any]) -> str:
        """Format the prompt with the given context."""
        ...


class Planner:
    """Protocol for planner agents that create task plans."""

    def __init__(
        self,
        agent: Agent,
        template: PromptTemplate,
        swarm: Swarm | None = None,
    ) -> None:
        self.agent = agent
        self.template = template
        self.swarm = swarm or Swarm()

    async def create_plan(self, prompt: str, context: dict[str, Any]) -> Result[Plan]:
        """Create a plan from the given prompt and context."""
        formatted_prompt = self.template.format_context(prompt, context)
        result = await self.swarm.execute(
            agent=self.agent,
            prompt=formatted_prompt,
            context_variables=context,
        )

        if not result.content:
            return Result(error=ValueError("Failed to create plan"))

        try:
            plan_json = extract_json(result.content)
            plan = Plan.model_validate(plan_json)

            if errors := plan.validate_dependencies():
                return Result(error=ValueError("\n".join(errors)))

            return Result(value=plan)

        except Exception as e:
            return Result(error=e)


# ================================================
# MARK: Stream Handler Protocol
# ================================================


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


# ================================================
# MARK: Swarm Team
# ================================================


def default_instructions(task: Task, context: ContextVariables) -> str:
    """Default task instructions builder."""
    prompt = f"""
    Execute the following task:

    Task Details:
    - ID: {task.id}
    - Title: {task.title}
    - Description: {task.description}
    """

    return prompt


class SwarmTeam:
    """Orchestrates a team of agents working on tasks according to a plan."""

    def __init__(
        self,
        planner: Planner,
        members: list[TeamMember],
        swarm: Swarm | None = None,
        instructions: TaskInstructions | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
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
        self.swarm = swarm or Swarm()
        self.instructions: TaskInstructions = instructions or default_instructions
        self.stream_handler = stream_handler

        # Internal state
        self._current_plan: Plan | None = None
        self._context: dict[str, Any] = {
            "task_types": {task_type for member in members for task_type in member.task_types}
        }

    async def create_plan(
        self,
        prompt: str,
        feedback: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Plan:
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

        self._current_plan.status = PlanStatus.IN_PROGRESS

        for task in self._current_plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            instructions = self.instructions
            if callable(instructions):
                instructions = instructions(task, self._context)

            await self.execute_task(task, instructions)

        self._current_plan.status = PlanStatus.COMPLETED

        if self.stream_handler:
            await self.stream_handler.on_plan_completed(self._current_plan)

    async def execute_task(self, task: Task, prompt: str) -> Result[Any]:
        """Execute a single task using the appropriate team member."""
        assignee = None
        for member in self.members.values():
            if task.task_type in member.task_types:
                assignee = member
                break

        if not assignee:
            raise ValueError(
                f"No team member found for task type '{task.task_type}' in task: {task.id}"
            )

        if self.stream_handler:
            await self.stream_handler.on_task_started(task)

        task.status = TaskStatus.IN_PROGRESS
        task.assignee = assignee.agent.id

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=prompt,
            context_variables={
                "task": task.model_dump(),
                **self._context,
            },
        )

        task.status = TaskStatus.COMPLETED

        if self.stream_handler:
            await self.stream_handler.on_task_completed(task)

        return Result(value=result.content)
