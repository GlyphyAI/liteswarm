# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import operator
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from functools import reduce
from textwrap import dedent
from typing import Any, Generic, Protocol, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, ContextVariables, Result
from liteswarm.utils import extract_json

# ================================================
# MARK: Utility Functions
# ================================================

T = TypeVar("T")


def create_union_type(types: list[T]) -> T:
    if not types:
        raise ValueError("No types provided for Union.")
    elif len(types) == 1:
        return types[0]
    else:
        return reduce(operator.or_, types)


def generate_plan_json_schema(task_definitions: list["TaskDefinition"]) -> dict[str, Any]:
    task_schemas = [td.task_schema for td in task_definitions]
    task_schemas_union = create_union_type(task_schemas)
    return Plan[task_schemas_union].model_json_schema()  # type: ignore [valid-type]


def get_output_schema_type(output_schema: "TaskOutput") -> type[BaseModel]:
    """Generalizes the unpacking of TaskOutput objects to return their JSON schema."""
    if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        return output_schema

    try:
        dummy_output = output_schema("", {})  # type: ignore [call-arg]
        if isinstance(dummy_output, BaseModel):
            return dummy_output.__class__
        else:
            raise TypeError("Callable did not return a BaseModel instance.")
    except Exception as e:
        raise TypeError(f"Error invoking callable TaskOutput: {e}") from e


def dedent_prompt(prompt: str) -> str:
    return dedent(prompt).strip()


# ================================================
# MARK: Swarm Team Types
# ================================================

TTask = TypeVar("TTask", bound="Task")
TaskInstructions = str | Callable[[TTask, ContextVariables], str]
TaskOutput = type[BaseModel] | Callable[[str, ContextVariables], BaseModel]


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Instance of a task created based on its TaskDefinition."""

    id: str = Field(
        description="Unique identifier for the task. REQUIRED: Must be provided.",
    )
    title: str = Field(
        description="Short descriptive title of the task. REQUIRED: Must be provided.",
    )
    task_type: str = Field(
        description="Type of the task. USE_DEFAULT: Do not modify, use default value.",
    )
    description: str | None = Field(
        default=None,
        description="Detailed description of the task. OPTIONAL: Provide if needed, otherwise use default value.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current status of the task. USE_DEFAULT: Do not modify, use default value.",
    )
    assignee: str | None = Field(
        default=None,
        description="ID of the assigned team member. USE_DEFAULT: Do not modify, use default value.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must be completed before this task. OPTIONAL: Provide if needed, otherwise use default value.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata. USE_DEFAULT: Do not modify, use default value.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, **kwargs: Any) -> type[Self]:
        return create_model(
            cls.__name__,
            __base__=cls,
            **kwargs,
        )


class TaskDefinition(BaseModel):
    """Definition of a task type, including how to create tasks of this type."""

    task_type: str
    task_schema: type[Task]
    task_instructions: TaskInstructions
    task_output: TaskOutput | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(
        cls,
        task_type: str,
        task_schema: type[Task],
        task_instructions: TaskInstructions,
        task_output: TaskOutput | None = None,
    ) -> "TaskDefinition":
        task_type_field = Field(
            default=task_type,
            description="Type of the task. USE_DEFAULT: Do not modify, use default value.",
        )

        task_schema = task_schema.create(
            task_type=(str, task_type_field),
        )

        return cls(
            task_type=task_type,
            task_schema=task_schema,
            task_instructions=task_instructions,
            task_output=task_output,
        )

    @model_validator(mode="after")
    def inject_task_type_into_schema(self) -> Self:
        """Ensures that the task_schema includes the task_type field with the correct default value."""
        task_type = self.task_schema.model_fields.get("task_type")
        if not task_type or task_type.default != self.task_type:
            self.task_schema = self.task_schema.create(
                task_type=(str, Field(default=self.task_type)),
            )

        return self


class TaskRegistry:
    """Registry for managing task types and their definitions."""

    def __init__(self, task_definitions: list[TaskDefinition] | None = None) -> None:
        self._registry: dict[str, TaskDefinition] = {}
        if task_definitions:
            self.register_tasks(task_definitions)

    def register_task(self, task_definition: TaskDefinition) -> None:
        self._registry[task_definition.task_type] = task_definition

    def register_tasks(self, task_definitions: list[TaskDefinition]) -> None:
        for task_definition in task_definitions:
            self.register_task(task_definition)

    def get_task_definition(self, task_type: str) -> TaskDefinition:
        return self._registry[task_type]

    def list_task_types(self) -> list[str]:
        return list(self._registry.keys())


class Plan(BaseModel, Generic[TTask]):
    """Base class for development plans."""

    tasks: list[TTask]
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

    def get_next_tasks(self) -> list[TTask]:
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

    id: str
    agent: Agent
    task_types: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Represents the result of an executed task."""

    task: Task
    content: str | None = None
    output: BaseModel | None = None
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


class PlanningAgent(Protocol):
    """Protocol for planning agents that create task plans."""

    async def create_plan(
        self,
        prompt: str,
        context: dict[str, Any],
        feedback: str | None = None,
    ) -> Result[Plan[Task]]:
        """Create a plan from the given prompt and context."""
        ...


class AgentPlanner(PlanningAgent):
    """Default implementation of the planning strategy."""

    def __init__(
        self,
        swarm: Swarm,
        agent: Agent,
        template: PromptTemplate,
        task_definitions: list[TaskDefinition],
    ) -> None:
        self.swarm = swarm
        self.agent = agent
        self.template = template
        self.task_definitions = {td.task_type: td for td in task_definitions}

    async def create_plan(
        self,
        prompt: str,
        context: dict[str, Any],
        feedback: str | None = None,
    ) -> Result[Plan[Task]]:
        """Create a plan using the configured agent."""
        if feedback:
            full_prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"
        else:
            full_prompt = prompt

        task_definitions = list(self.task_definitions.values())
        plan_json_schema = generate_plan_json_schema(task_definitions)
        context.update(plan_json_schema=plan_json_schema)

        formatted_prompt = self.template.format_context(full_prompt, context)

        result = await self.swarm.execute(
            agent=self.agent,
            prompt=formatted_prompt,
            context_variables=context,
        )

        if not result.content:
            return Result(error=ValueError("Failed to create plan"))

        try:
            json_tasks: dict[str, Any] = extract_json(result.content)
            raw_tasks: list[dict[str, Any]] = json_tasks.get("tasks", [])
            tasks: list[Task] = []

            for task_data in raw_tasks:
                task_type: str = task_data.get("task_type", "")
                if not task_type:
                    return Result(error=ValueError("Task type is missing in task data"))

                task_definition = self.task_definitions[task_type]
                if not task_definition:
                    return Result(error=ValueError(f"Unknown task type: {task_type}"))

                # Validate and create Task instance based on TaskDefinition's schema
                task_instance = task_definition.task_schema.model_validate(task_data)
                tasks.append(task_instance)

            plan = Plan(tasks=tasks)

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


def default_instructions(
    task: Task,
    context: ContextVariables,
    task_definition: TaskDefinition,
) -> str:
    """Default task instructions builder based on TaskDefinition."""
    execution_history: list[ExecutionResult] = context.get("execution_history", [])

    task_context = f"""
    Task Details:
    - ID: {task.id}
    - Title: {task.title}
    - Description: {task.description or 'No description provided'}
    - Type: {task.task_type}
    """

    # Add dependency context if any
    if task.dependencies:
        dependency_results = [
            result for result in execution_history if result.task.id in task.dependencies
        ]

        task_context += "\nDependency Context:"
        for result in dependency_results:
            if result.assignee:
                task_context += f"\n- Task {result.task.id} ({result.task.title}) completed by {result.assignee.agent.id}"

    # Add project context if available
    if project := context.get("project", {}):
        task_context += "\nProject Context:"
        if dirs := project.get("directories", []):
            task_context += f"\n- Project structure: {', '.join(dirs)}"
        if files := project.get("files", []):
            task_context += f"\n- Available files: {', '.join(f['filepath'] for f in files)}"

    # Obtain instructions from TaskDefinition or use default guidelines
    instructions = task_definition.task_instructions or "<No instructions provided>"
    if callable(instructions):
        instructions = instructions(task, context)

    return f"""
    Execute the following task with the provided context:

    {task_context}

    {instructions}

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
        task_definitions: list[TaskDefinition],
        planning_agent: PlanningAgent | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
    ) -> None:
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.task_definitions = task_definitions
        self.stream_handler = stream_handler
        self.planning_agent = planning_agent or AgentPlanner(
            swarm=self.swarm,
            agent=self._default_agent(),
            template=self._default_prompt_template(),
            task_definitions=task_definitions,
        )

        # Private state
        self._task_registry = TaskRegistry(task_definitions)
        self._execution_history: list[ExecutionResult] = []
        self._context: dict[str, Any] = {
            "team_capabilities": self._get_team_capabilities(),
        }

    def _default_agent(self) -> Agent:
        """Create a default agent if none is provided."""
        return Agent.create(
            id="agent-planner",
            model="gpt-4o",
            instructions="""<TODO: Add instructions for the agent planner>""",
        )

    def _default_prompt_template(self) -> PromptTemplate:
        """Create a default prompt template."""

        class DefaultPromptTemplate:
            @property
            def template(self) -> str:
                return "Default template"

            def format_context(self, prompt: str, context: dict[str, Any]) -> str:
                return prompt

        return DefaultPromptTemplate()

    def _get_team_capabilities(self) -> dict[str, list[str]]:
        """Get a mapping of task types to team member capabilities."""
        capabilities: dict[str, list[str]] = {}
        for member in self.members.values():
            for task_type in member.task_types:
                if task_type not in capabilities:
                    capabilities[task_type] = []
                capabilities[task_type].append(member.agent.id)

        return capabilities

    async def create_plan(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> Result[Plan]:
        """Create a new plan from the prompt and context by delegating to AgentPlanner."""
        if context:
            self._context.update(context)

        result = await self.planning_agent.create_plan(
            prompt=prompt,
            context=self._context,
        )

        if not result.value:
            return result

        plan = result.value

        if self.stream_handler:
            await self.stream_handler.on_plan_created(plan)

        return Result(value=plan)

    async def execute_plan(self, plan: Plan) -> Result[list[ExecutionResult]]:
        """Execute all tasks in the given plan."""
        plan.status = PlanStatus.IN_PROGRESS
        results: list[ExecutionResult] = []

        try:
            while next_tasks := plan.get_next_tasks():
                for task in next_tasks:
                    result = await self.execute_task(task)
                    if result.error:
                        return Result(error=result.error)

                    if result.value:
                        results.append(result.value)
                    else:
                        return Result(error=ValueError(f"Failed to execute task {task.id}"))

            plan.status = PlanStatus.COMPLETED

            if self.stream_handler:
                await self.stream_handler.on_plan_completed(plan)

            return Result(value=results)

        except Exception as e:
            return Result(error=e)

    async def execute_task(self, task: Task) -> Result[ExecutionResult]:
        """Execute a single task using the appropriate team member."""
        assignee = await self._select_matching_member(task)
        if not assignee:
            return Result(
                error=ValueError(f"No team member found for task type '{task.task_type}'")
            )

        if self.stream_handler:
            await self.stream_handler.on_task_started(task)

        task.status = TaskStatus.IN_PROGRESS
        task.assignee = assignee.agent.id

        task_definition = self._task_registry.get_task_definition(task.task_type)
        if not task_definition:
            return Result(
                error=ValueError(f"No TaskDefinition found for task type '{task.task_type}'")
            )

        instructions = task_definition.task_instructions or default_instructions(
            task,
            self._context,
            task_definition,
        )

        if callable(instructions):
            instructions = instructions(task, self._context)

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=instructions,
            context_variables={
                "task": task.model_dump(),
                "execution_history": self._execution_history,
                **self._context,
            },
        )

        if not result.content:
            return Result(error=ValueError("The agent did not return any content"))

        if task_definition.task_output and result.content:
            try:
                task_output = task_definition.task_output
                if isinstance(task_output, type) and issubclass(task_output, BaseModel):
                    output: BaseModel = task_output.model_validate_json(result.content)
                elif callable(task_output):
                    output: BaseModel = task_output(result.content, self._context)  # type: ignore
                else:
                    raise TypeError("Invalid task output schema")

                execution_result = ExecutionResult(
                    task=task,
                    content=result.content,
                    output=output,
                    assignee=assignee,
                )

                self._execution_history.append(execution_result)
                task.status = TaskStatus.COMPLETED

                if self.stream_handler:
                    await self.stream_handler.on_task_completed(task)

                return Result(value=execution_result)

            except Exception as e:
                return Result(error=ValueError(f"Invalid task output: {e}"))

        execution_result = ExecutionResult(
            task=task,
            content=result.content,
            assignee=assignee,
            timestamp=datetime.now(),
        )

        self._execution_history.append(execution_result)
        task.status = TaskStatus.COMPLETED

        if self.stream_handler:
            await self.stream_handler.on_task_completed(task)

        return Result(value=execution_result)

    async def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select the best matching team member for a task."""
        if task.assignee and task.assignee in self.members:
            return self.members[task.assignee]

        eligible_members = [
            member for member in self.members.values() if task.task_type in member.task_types
        ]

        if not eligible_members:
            return None

        # TODO: Implement more sophisticated selection logic
        # Could consider:
        # - Member workload
        # - Task type specialization scores
        # - Previous task performance
        # - Agent polling/voting
        return eligible_members[0]
