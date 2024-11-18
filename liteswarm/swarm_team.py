# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from datetime import datetime
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


class ExecutionResult(BaseModel):
    """Represents the result of an executed task."""

    task: Task
    content: str | None = None
    assignee: TeamMember | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ================================================
# MARK: Planning Strategy
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


class PlanningStrategy(Protocol):
    """Protocol for planning strategies that create task plans."""

    async def create_plan(
        self,
        prompt: str,
        context: dict[str, Any],
        feedback: str | None = None,
    ) -> Result[Plan]:
        """Create a plan from the given prompt and context."""
        ...


class AgentPlanner(PlanningStrategy):
    """Default implementation of the planning strategy."""

    def __init__(
        self,
        swarm: Swarm,
        agent: Agent,
        template: PromptTemplate,
    ) -> None:
        self.swarm = swarm
        self.agent = agent
        self.template = template

    async def create_plan(
        self,
        prompt: str,
        context: dict[str, Any],
        feedback: str | None = None,
    ) -> Result[Plan]:
        """Create a plan using the configured agent."""
        if feedback:
            full_prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"
        else:
            full_prompt = prompt

        formatted_prompt = self.template.format_context(full_prompt, context)
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
    # Get relevant execution history for this task type
    relevant_history = [
        result
        for result in context.get("execution_history", [])
        if result.task.task_type == task.task_type
    ]

    # Get team capabilities
    team_capabilities = context.get("team_capabilities", {})
    task_specialists = team_capabilities.get(task.task_type, [])

    # Build task context section
    task_context = f"""
    Task Details:
    - ID: {task.id}
    - Title: {task.title}
    - Description: {task.description or 'No description provided'}
    - Type: {task.task_type}

    Team Context:
    - Specialists for this task type: {', '.join(task_specialists)}
    - Previous similar tasks: {len(relevant_history)}
    """

    # Add dependency context if any
    if task.dependencies:
        dependency_results = [
            result
            for result in context.get("execution_history", [])
            if result.task.id in task.dependencies
        ]
        task_context += "\nDependency Context:"
        for result in dependency_results:
            task_context += f"\n- Task {result.task.id} ({result.task.title}) completed by {result.assignee.agent.id}"

    # Add project context if available
    if project := context.get("project"):
        task_context += "\nProject Context:"
        if dirs := project.get("directories"):
            task_context += f"\n- Project structure: {', '.join(dirs)}"
        if files := project.get("files"):
            task_context += f"\n- Available files: {', '.join(f['filepath'] for f in files)}"

    return f"""
    Execute the following task with the provided context:

    {task_context}

    Important Guidelines:
    1. Consider the project structure and existing files
    2. Build upon previous work from the execution history
    3. Follow team coding standards and patterns
    4. Provide clear documentation of your changes
    5. Consider impact on dependent tasks

    Your response must follow this exact format:

    <thoughts>
    Explain your implementation approach here. Include:
    - Your understanding of the task
    - Key implementation decisions
    - Considerations and trade-offs
    - Impact on the overall project
    </thoughts>

    <files>
    Provide complete file contents in JSON format:
    [
        {{
            "filepath": "path/to/file",
            "content": "// Complete file contents here"
        }}
    ]
    </files>
    """


class SwarmTeam:
    """Orchestrates a team of agents working on tasks according to a plan."""

    def __init__(
        self,
        swarm: Swarm,
        members: list[TeamMember],
        planning_strategy: PlanningStrategy,
        instructions: TaskInstructions | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
    ) -> None:
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.planning_strategy = planning_strategy
        self.instructions: TaskInstructions = instructions or default_instructions
        self.stream_handler = stream_handler

        # Internal state
        self._current_plan: Plan | None = None
        self._context: dict[str, Any] = {
            "task_types": {task_type for member in members for task_type in member.task_types},
            "team_capabilities": self._get_team_capabilities(),
        }
        self._execution_history: list[ExecutionResult] = []

    def _get_team_capabilities(self) -> dict[str, list[str]]:
        """Get a mapping of task types to team member capabilities."""
        capabilities: dict[str, list[str]] = {}
        for member in self.members.values():
            for task_type in member.task_types:
                if task_type not in capabilities:
                    capabilities[task_type] = []
                capabilities[task_type].append(member.agent.id)
        return capabilities

    async def execute_task(self, task: Task, prompt: str) -> Result[Any]:
        """Execute a single task using the appropriate team member."""
        assignee = await self._select_matching_member(task)
        if not assignee:
            raise ValueError(
                f"No team member found for task type '{task.task_type}' in task: {task.id}"
            )

        if self.stream_handler:
            await self.stream_handler.on_task_started(task)

        task.status = TaskStatus.IN_PROGRESS
        task.assignee = assignee.agent.id
        task_context = {
            "task": task.model_dump(),
            "execution_history": self._execution_history,
            **self._context,
        }

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=prompt,
            context_variables=task_context,
        )

        self._execution_history.append(
            ExecutionResult(
                task=task,
                content=result.content,
                assignee=assignee,
                timestamp=datetime.now(),
            )
        )

        task.status = TaskStatus.COMPLETED

        if self.stream_handler:
            await self.stream_handler.on_task_completed(task)

        return Result(value=result.content)

    async def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select the best matching team member for a task based on capabilities and workload."""
        eligible_members = [
            member for member in self.members.values() if task.task_type in member.task_types
        ]

        if not eligible_members:
            return None

        # For now, just return the first eligible member
        # This could be enhanced with workload balancing, specialization matching, etc.
        return eligible_members[0]

    async def create_plan(
        self,
        prompt: str,
        feedback: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Plan:
        """Create a new plan from the prompt and optional feedback."""
        if context:
            self._context.update(context)

        result = await self.planning_strategy.create_plan(
            prompt=prompt,
            context=self._context,
            feedback=feedback,
        )

        if not result.value:
            raise ValueError("Failed to create plan")

        self._current_plan = result.value

        if self.stream_handler:
            await self.stream_handler.on_plan_created(self._current_plan)

        return self._current_plan

    async def execute_plan(self) -> None:
        """Execute all tasks in the current plan in order."""
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
