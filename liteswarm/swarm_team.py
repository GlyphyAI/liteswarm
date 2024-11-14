from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, Result

TaskStatus = Literal["pending", "in_progress", "completed"]
PlanStatus = Literal["draft", "approved", "in_progress", "completed"]

T = TypeVar("T", bound="Task")
P = TypeVar("P", bound="Plan")


class Task(BaseModel):
    """Base class for tasks in a plan."""

    id: str
    title: str
    description: str
    status: TaskStatus = "pending"
    assignee: str | None = None
    metadata: dict[str, Any] = {}


class Plan(BaseModel, Generic[T]):
    """Base class for development plans."""

    tasks: list[T]
    current_task_id: str | None = None
    status: PlanStatus = "draft"
    metadata: dict[str, Any] = {}


class PlannerAgent(Protocol[P]):
    """Protocol for planner agents that create task plans."""

    async def create_plan(self, prompt: str, context: dict[str, Any]) -> Result[P]:
        """Create a plan from the given prompt and context."""
        ...


@dataclass
class TeamMember:
    """Represents a team member that can execute tasks."""

    agent: Agent
    task_types: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class SwarmTeam(Generic[T, P]):
    """Orchestrates a team of agents working on tasks according to a plan."""

    def __init__(  # noqa: PLR0913
        self,
        planner: PlannerAgent[P],
        members: list[TeamMember],
        swarm: Swarm | None = None,
        on_plan_created: Callable[[P], None] | None = None,
        on_task_started: Callable[[T], None] | None = None,
        on_task_completed: Callable[[T], None] | None = None,
        on_plan_completed: Callable[[P], None] | None = None,
    ) -> None:
        """Initialize a new SwarmTeam.

        Args:
            planner: Agent responsible for creating plans
            members: List of team members that can execute tasks
            swarm: Optional Swarm instance to use (creates new one if not provided)
            on_plan_created: Optional callback when a plan is created
            on_task_started: Optional callback when a task is started
            on_task_completed: Optional callback when a task is completed
            on_plan_completed: Optional callback when the plan is completed
        """
        self.planner = planner
        self.members = {member.agent.id: member for member in members}

        # Track available engineer types
        self.available_types: set[str] = set()
        for member in members:
            self.available_types.update(member.task_types)

        self.swarm = swarm or Swarm()

        # Callbacks
        self.on_plan_created = on_plan_created
        self.on_task_started = on_task_started
        self.on_task_completed = on_task_completed
        self.on_plan_completed = on_plan_completed

        # State
        self.current_plan: P | None = None
        self.context: dict[str, Any] = {}

    async def create_plan(
        self,
        prompt: str,
        feedback: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> P:
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
            self.context.update(context)

        # Add available engineer types to context
        self.context["available_engineer_types"] = list(self.available_types)

        if feedback:
            full_prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"
        else:
            full_prompt = prompt

        result = await self.planner.create_plan(full_prompt, self.context)
        if not result.value:
            raise ValueError("Failed to create plan")

        self.current_plan = result.value

        if self.on_plan_created:
            self.on_plan_created(self.current_plan)

        return self.current_plan

    async def execute_plan(self) -> None:
        """Execute all tasks in the current plan in order.

        Raises:
            ValueError: If no approved plan exists
        """
        if not self.current_plan:
            raise ValueError("No plan to execute")

        self.current_plan.status = "in_progress"

        for task in self.current_plan.tasks:
            if task.status != "pending":
                continue

            await self.execute_task(task)

        self.current_plan.status = "completed"

        if self.on_plan_completed:
            self.on_plan_completed(self.current_plan)

    async def execute_task(self, task: T) -> Result[Any]:
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

        if self.on_task_started:
            self.on_task_started(task)

        task.status = "in_progress"
        task.assignee = assignee.agent.id

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=f"Complete this task: {task.description}",
            context_variables={
                "task": task.model_dump(),
                **self.context,
            },
        )

        task.status = "completed"

        if self.on_task_completed:
            self.on_task_completed(task)

        return Result(value=result.content)
