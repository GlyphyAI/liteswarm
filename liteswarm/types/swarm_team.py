# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Discriminator, Field, field_serializer

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent, Message

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)
"""Type variable representing a Pydantic model or its subclass."""

TaskInstructions: TypeAlias = str | Callable[["Task", ContextVariables], str]
"""Instructions for executing a task.

Can be either a static template string or a function that generates instructions
dynamically based on task and context.

Examples:
    Static template:
        ```python
        instructions: TaskInstructions = f"Process {task.title} using {tools}"
        ```

    Dynamic generator:
        ```python
        def generate_instructions(task: Task, context: ContextVariables) -> str:
            tools = context.get('available_tools', [])
            return f"Process {task.title} using these tools: {tools}"

        instructions: TaskInstructions = generate_instructions
        ```
"""

PydanticResponseFormat: TypeAlias = (
    type[PydanticModel] | Callable[[str, ContextVariables], PydanticModel]
)
"""Generic format specification for Pydantic model responses.

Can be either a Pydantic model for direct validation or a function that parses
output with context. This format is used anywhere a Pydantic model is expected
as a response format.

Examples:
    Static format using a Pydantic model:
        ```python
        class ReviewOutput(BaseModel):
            issues: list[str]
            approved: bool

        response_format: PydanticResponseFormat = ReviewOutput
        ```

    Dynamic format using a parser function:
        ```python
        def parse_review(content: str, context: ContextVariables) -> ReviewOutput:
            data = json.loads(content)
            return ReviewOutput.model_validate(data)

        response_format: PydanticResponseFormat = parse_review
        ```
"""

TaskResponseFormat: TypeAlias = PydanticResponseFormat[BaseModel]
"""Format specification for task responses.

Can be either a Pydantic model for direct validation or a function that parses
output with context.

Examples:
    Using model:
        ```python
        class ProcessingOutput(BaseModel):
            items_processed: int
            success_rate: float
            errors: list[str] = []

        response_format: TaskResponseFormat = ProcessingOutput
        ```

    Using parser:
        ```python
        def parse_output(content: str, context: ContextVariables) -> BaseModel:
            data = json.loads(content)
            return ProcessingOutput.model_validate(data)

        response_format: TaskResponseFormat = parse_output
        ```
"""

PromptTemplate: TypeAlias = str | Callable[[str, ContextVariables], str]
"""Template for formatting prompts with context.

Can be either a static template string or a function that generates prompts
dynamically based on context.

Examples:
    Static template:
        ```python
        template: PromptTemplate = "Process {prompt} using {tools}"
        ```

    Dynamic template:
        ```python
        def generate_prompt(prompt: str, context: ContextVariables) -> str:
            tools = context.get("tools", [])
            return f"Process {prompt} using available tools: {tools}"

        template: PromptTemplate = generate_prompt
        ```
"""

PlanResponseFormat: TypeAlias = PydanticResponseFormat["Plan"]
"""Format specification for plan responses.

Can be either a Plan subclass or a function that parses responses into Plan objects.

Examples:
    Static format using a Plan subclass:
        ```python
        class CustomPlan(Plan):
            tasks: list[ReviewTask | TestTask]
            metadata: dict[str, str]

        response_format: PlanResponseFormat = CustomPlan
        ```

    Dynamic format using a parser function:
        ```python
        def parse_plan_response(response: str, context: ContextVariables) -> Plan:
            # Parse response and create plan
            tasks = extract_tasks(response)
            return Plan(tasks=tasks)

        response_format: PlanResponseFormat = parse_plan_response
        ```
"""


class TaskStatus(str, Enum):
    """Status of a task in its execution lifecycle.

    Tracks the progression of a task from creation through execution
    to completion or failure. Status transitions occur automatically
    during task processing.
    """

    PENDING = "pending"
    """Task is created but not yet started."""

    IN_PROGRESS = "in_progress"
    """Task is currently being executed."""

    COMPLETED = "completed"
    """Task has finished successfully."""

    FAILED = "failed"
    """Task execution has failed."""


class Task(BaseModel):
    """Base schema for structured task execution.

    Defines the core structure for tasks in a SwarmTeam workflow.
    Each task represents a unit of work with specific requirements,
    dependencies, and execution tracking.

    Note:
        When subclassing, ensure OpenAI compatibility:
        - Use required fields without defaults
        - Use Literal types for discriminators
        - Use simple JSON-serializable types
        - Avoid complex Pydantic features

    Examples:
        Code review task:
            ```python
            class ReviewTask(Task):
                type: Literal["code_review"] = "code_review"
                diff: str
                focus_areas: list[str]
            ```
    """

    type: str
    """Type identifier for task matching."""

    id: str
    """Unique task identifier."""

    title: str
    """Short descriptive title."""

    description: str | None
    """Optional detailed description."""

    status: TaskStatus
    """Current execution status."""

    assignee: str | None
    """ID of assigned team member."""

    dependencies: list[str]
    """IDs of prerequisite tasks."""

    metadata: dict[str, Any] | None
    """Optional task metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @classmethod
    def get_task_type(cls) -> str:
        """Get the type identifier for this task class.

        Returns:
            Task type string defined in the schema.

        Raises:
            ValueError: If task type is not defined as a Literal.
        """
        type_field = cls.model_fields["type"]
        type_field_annotation = type_field.annotation

        if type_field_annotation and get_origin(type_field_annotation) is Literal:
            return get_args(type_field_annotation)[0]

        raise ValueError("Task type is not defined as a Literal in the task schema")


class TaskDefinition(BaseModel):
    """Blueprint for task creation and execution.

    Defines how a specific type of task should be created, executed,
    and validated. Supports both static and dynamic instruction
    generation and response parsing.

    Examples:
        Static definition:
            ```python
            task_def = TaskDefinition(
                task_type=ReviewTask,
                instructions="Review the provided diff",
                response_format=ReviewOutput,
            )
            ```

        Dynamic definition:
            ```python
            def get_instructions(task: ReviewTask, context: ContextVariables) -> str:
                return f"Review {task.diff} focusing on {task.focus_areas}"


            task_def = TaskDefinition(
                task_type=ReviewTask,
                instructions=get_instructions,
                response_format=ReviewOutput,
            )
            ```
    """

    task_type: type[Task]
    """Task schema for validation."""

    instructions: TaskInstructions
    """Static template or dynamic builder."""

    response_format: TaskResponseFormat | None = None
    """Optional output format specification."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class Plan(BaseModel):
    """Execution plan with ordered tasks.

    Represents a structured workflow of dependent tasks. Manages task
    ordering, dependencies, and execution tracking. Designed for
    direct LLM output parsing.

    Note:
        When subclassing, ensure OpenAI compatibility:
        - Use required fields without defaults
        - Use Literal types for discriminators
        - Use simple JSON-serializable types
        - Avoid complex Pydantic features

    Example:
        ```python
        class CustomPlan(Plan):
            tasks: list[ReviewTask | TestTask]
        ```
    """

    id: str
    """Unique plan identifier."""

    tasks: Sequence[Task]
    """Tasks in execution order."""

    metadata: dict[str, Any] | None
    """Optional plan metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    def validate_dependencies(self) -> list[str]:
        """Validate that all task dependencies exist.

        Returns:
            List of error messages for invalid dependencies.
        """
        task_ids = {task.id for task in self.tasks}
        errors: list[str] = []

        for task in self.tasks:
            invalid_deps = [dep for dep in task.dependencies if dep not in task_ids]
            if invalid_deps:
                errors.append(f"Task {task.id} has invalid dependencies: {invalid_deps}")

        return errors

    def get_next_tasks(self) -> list[Task]:
        """Get tasks that are ready for execution.

        A task is ready when it's pending and all its dependencies have completed.

        Returns:
            List of tasks ready for execution.
        """
        completed_tasks = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        return [
            task
            for task in self.tasks
            if task.status == TaskStatus.PENDING
            and all(dep in completed_tasks for dep in task.dependencies)
        ]


class PlanResult(BaseModel):
    """Result of plan creation.

    Contains the generated plan along with new messages,
    complete message history, and final context state.
    """

    plan: Plan
    """Generated execution plan."""

    new_messages: list[Message]
    """Output messages generated during planning."""

    all_messages: list[Message]
    """Complete conversation history of planning."""

    context_variables: ContextVariables | None = None
    """Final context variables of planning."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class TeamMember(BaseModel):
    """Specialized agent for executing particular task types.

    Represents a team participant with specific capabilities for
    handling particular task types. Each member has an agent
    configuration and a list of supported task types.

    Examples:
        Code reviewer:
            ```python
            reviewer = TeamMember(
                id="code-reviewer",
                agent=Agent(
                    id="review-gpt",
                    instructions="You are a code reviewer.",
                    llm=LLM(model="gpt-4o"),
                ),
                task_types=[ReviewTask],
            )
            ```

        Test writer:
            ```python
            tester = TeamMember(
                id="test-writer",
                agent=Agent(
                    id="test-gpt",
                    instructions="You write comprehensive tests.",
                    llm=LLM(model="gpt-4o"),
                ),
                task_types=[TestTask],
            )
            ```
    """

    id: str
    """Unique member identifier."""

    agent: Agent
    """Agent configuration."""

    task_types: list[type[Task]]
    """Supported task types."""

    metadata: dict[str, Any] | None = None
    """Optional member metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @classmethod
    def from_agent(cls, agent: Agent, /, task_types: list[type[Task]]) -> "TeamMember":
        """Create TeamMember from existing Agent configuration.

        Simplifies TeamMember creation by reusing an Agent's configuration
        and ID. This is useful when converting standalone agents into team
        members with specific task capabilities.

        Args:
            agent: Existing Agent configuration to use.
            task_types: List of Task types this member can handle.

        Returns:
            TeamMember instance with the agent's configuration.
        """
        return cls(
            id=agent.id,
            agent=agent,
            task_types=task_types,
        )

    @field_serializer("task_types")
    def serialize_task_types(self, task_types: list[type[Task]]) -> list[str]:
        """Serialize task type classes to their string identifiers.

        Converts task type classes to their string identifiers for JSON
        serialization. Uses each task type's get_task_type() method to
        extract the identifier.

        Args:
            task_types: List of Task subclass types.

        Returns:
            List of task type string identifiers.
        """
        return [task_type.get_task_type() for task_type in task_types]


class TaskResult(BaseModel):
    """Result of task execution.

    Contains all outputs and execution details from a task,
    including raw content, structured output, messages, and
    context updates.
    """

    task: Task
    """Task that was executed."""

    content: str | None = None
    """Raw output content."""

    output: BaseModel | None = None
    """Structured output data."""

    new_messages: list[Message] | None = None
    """Output messages generated during task execution."""

    all_messages: list[Message] | None = None
    """Complete conversation history of task execution."""

    context_variables: ContextVariables | None = None
    """Final context variables of task execution."""

    assignee: TeamMember | None = None
    """Member who executed the task."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """Task execution completion time."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ArtifactStatus(str, Enum):
    """Status of an execution artifact.

    Tracks the progression of a planning and execution cycle
    from creation through completion or failure. Status
    transitions occur automatically during processing.
    """

    CREATED = "created"
    """Artifact is initialized."""

    PLANNING = "planning"
    """Plan is being created."""

    EXECUTING = "executing"
    """Plan is being executed."""

    COMPLETED = "completed"
    """Execution completed successfully."""

    FAILED = "failed"
    """Execution failed."""


class Artifact(BaseModel):
    """Record of plan execution.

    Contains the complete execution record including the plan,
    task results, messages, and any errors that occurred during
    the process.
    """

    id: str
    """Unique artifact identifier."""

    plan: Plan | None = None
    """Executed plan."""

    task_results: list[TaskResult] = Field(default_factory=list)
    """Results from executed tasks."""

    error: Exception | None = None
    """Execution error if failed."""

    status: ArtifactStatus = ArtifactStatus.CREATED
    """Current execution status."""

    created_at: datetime = datetime.now()
    """Artifact creation time."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ApprovePlan(BaseModel):
    """Plan approval feedback.

    Indicates that the generated plan is acceptable and
    execution should proceed.
    """

    type: Literal["approve"]
    """Feedback type identifier."""


class RejectPlan(BaseModel):
    """Plan rejection feedback.

    Indicates that the plan needs revision and includes
    feedback for regeneration.
    """

    type: Literal["reject"]
    """Feedback type identifier."""

    feedback: str
    """Feedback for plan revision."""


PlanFeedback: TypeAlias = Annotated[ApprovePlan | RejectPlan, Discriminator("type")]
"""Feedback on a generated plan."""

PlanFeedbackCallback: TypeAlias = Callable[[Plan], Awaitable[PlanFeedback]]
"""Callback for plan feedback.

Callback function that receives a plan and returns feedback
indicating whether the plan is acceptable or needs revision.

Examples:
    Approve the plan:
        ```python
        def feedback_callback(plan: Plan) -> PlanFeedback:
            return ApprovePlan()
        ```

    Reject the plan:
        ```python
        def feedback_callback(plan: Plan) -> PlanFeedback:
            return RejectPlan(feedback="The plan is incomplete.")
        ```
"""
