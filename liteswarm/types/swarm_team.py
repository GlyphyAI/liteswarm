# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable, Sequence
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypeAlias, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)
"""Type variable representing a Pydantic model or its subclass."""

TaskType = TypeVar("TaskType", bound="Task")
"""Type variable representing a Task or its subclass."""

TaskInstructions: TypeAlias = str | Callable[[TaskType, ContextVariables], str]
"""Instructions for executing a task.

Can be either a static template string or a function that generates instructions
dynamically based on task and context.

Examples:
    Static template:
        ```python
        instructions: TaskInstructions = '''
            Process data in {task.input_file}.
            Use batch size of {task.batch_size}.
            Output format: {context.get('output_format')}.
        '''
        ```

    Dynamic generator:
        ```python
        def generate_instructions(task: Task, context: ContextVariables) -> str:
            tools = context.get('available_tools', [])
            return f'''
                Process {task.title} using these tools: {', '.join(tools)}.
                Priority: {context.get('priority', 'normal')}.
            '''
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
            return ReviewOutput(**data)

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
            return ProcessingOutput(
                items_processed=data['processed'],
                success_rate=data['success'] * 100,
                errors=data.get('errors', [])
            )
        ```
"""

PromptTemplate: TypeAlias = str | Callable[[str, ContextVariables], str]
"""Template for formatting prompts with context.

Can be either a static template string or a function that generates prompts
dynamically based on context.

Examples:
    Static template:
        ```python
        template: PromptTemplate = "Process {prompt} using {context.get('tools')}"
        ```

    Dynamic template:
        ```python
        def generate_prompt(prompt: str, context: ContextVariables) -> str:
            tools = context.get("tools", [])
            return f"Process {prompt} using available tools: {', '.join(tools)}"
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
    to completion or failure.
    """

    PENDING = "pending"
    """Task is created but not yet started."""

    IN_PROGRESS = "in_progress"
    """Task is currently being executed."""

    COMPLETED = "completed"
    """Task has finished successfully."""

    FAILED = "failed"
    """Task execution has failed."""


class PlanStatus(str, Enum):
    """Status of a plan in its lifecycle.

    Tracks the progression of a plan from creation through approval
    to execution and completion.
    """

    DRAFT = "draft"
    """Plan is created but not yet approved."""

    APPROVED = "approved"
    """Plan is approved and ready for execution."""

    IN_PROGRESS = "in_progress"
    """Plan is currently being executed."""

    COMPLETED = "completed"
    """All tasks in plan have completed successfully."""

    FAILED = "failed"
    """Plan execution has failed."""


class Task(BaseModel):
    """Base class for defining task schemas in a SwarmTeam workflow.

    Tasks are the fundamental units of work in a SwarmTeam. Each task has a type
    that determines its schema and execution requirements.

    Examples:
        Define a custom task type:
            ```python
            class DataProcessingTask(Task):
                type: Literal["data_processing"]
                input_file: str
                batch_size: int = 100
                output_format: str = "json"
            ```

        Create a task instance:
            ```python
            task = DataProcessingTask(
                id="process-1",
                title="Process customer data",
                description="Process Q1 customer data",
                input_file="data/customers_q1.csv",
                metadata={"priority": "high"}
            )
            ```
    """

    type: str
    """Type identifier for matching with team members."""

    id: str
    """Unique identifier for the task."""

    title: str
    """Short descriptive title of the task."""

    description: str | None = None
    """Optional detailed description."""

    status: TaskStatus = TaskStatus.PENDING
    """Current execution status."""

    assignee: str | None = None
    """ID of the assigned team member."""

    dependencies: list[str] = Field(default_factory=list)
    """IDs of tasks that must complete first."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional task-specific data."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )

    @classmethod
    def get_task_type(cls) -> str:
        """Get the type identifier for this task class.

        Returns:
            Task type string defined in the schema.

        Raises:
            ValueError: If task type is not defined as a Literal.

        Examples:
            Get task type:
                ```python
                class ReviewTask(Task):
                    type: Literal["code_review"]

                task_type = ReviewTask.get_task_type()  # Returns "code_review"
                ```
        """
        type_field = cls.model_fields["type"]
        type_field_annotation = type_field.annotation

        if type_field_annotation and get_origin(type_field_annotation) is Literal:
            return get_args(type_field_annotation)[0]

        raise ValueError("Task type is not defined as a Literal in the task schema")


class TaskDefinition(BaseModel):
    """Definition of a task type and its execution requirements.

    Provides the blueprint for creating and executing tasks, including their
    schema, instructions, and output format.

    Examples:
        Simple task definition:
            ```python
            task_def = TaskDefinition(
                task_schema=ReviewTask,
                task_instructions="Review code at {task.pr_url}",
                task_response_format=ReviewOutput
            )
            ```

        Dynamic task definition:
            ```python
            def generate_instructions(task: Task, context: ContextVariables) -> str:
                return f"Review {task.pr_url} focusing on {context.get('focus_areas')}"

            def parse_output(content: str, context: ContextVariables) -> BaseModel:
                data = json.loads(content)
                return ReviewOutput(
                    approved=data['approved'],
                    comments=data['comments'],
                    suggestions=data['suggestions']
                )

            task_def = TaskDefinition(
                task_schema=ReviewTask,
                task_instructions=generate_instructions,
                task_response_format=parse_output
            )
            ```
    """

    task_schema: type[Task]
    """Pydantic model for validating tasks."""

    task_instructions: TaskInstructions
    """Template or function for generating instructions."""

    task_response_format: TaskResponseFormat | None = None
    """Optional schema for parsing responses."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class Plan(BaseModel):
    """Plan consisting of ordered tasks with dependencies.

    Organizes tasks into a workflow, managing their dependencies and
    tracking execution status.

    Examples:
        Create a simple plan:
            ```python
            plan = Plan(
                tasks=[
                    ReviewTask(
                        id="review-1",
                        title="Review PR #123",
                        pr_url="github.com/org/repo/123"
                    ),
                    TestTask(
                        id="test-1",
                        title="Test changes",
                        dependencies=["review-1"]
                    )
                ],
                metadata={"priority": "high"}
            )
            ```
    """

    tasks: Sequence[Task]
    """Tasks in this plan."""

    status: PlanStatus = PlanStatus.DRAFT
    """Current plan status."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional plan metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )

    def validate_dependencies(self) -> list[str]:
        """Validate that all task dependencies exist.

        Returns:
            List of error messages for invalid dependencies.

        Examples:
            Check dependencies:
                ```python
                errors = plan.validate_dependencies()
                if errors:
                    print("Invalid dependencies found:")
                    for error in errors:
                        print(f"- {error}")
                ```
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

        Examples:
            Process ready tasks:
                ```python
                while next_tasks := plan.get_next_tasks():
                    for task in next_tasks:
                        print(f"Executing: {task.title}")
                        # Execute task
                ```
        """
        completed_tasks = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        return [
            task
            for task in self.tasks
            if task.status == TaskStatus.PENDING
            and all(dep in completed_tasks for dep in task.dependencies)
        ]


class TeamMember(BaseModel):
    """Team member that can execute specific types of tasks.

    Represents an agent with specialized capabilities and configuration
    for handling particular task types.

    Examples:
        Create specialized team members:
            ```python
            # Code review specialist
            reviewer = TeamMember(
                id="code-reviewer",
                agent=Agent(
                    id="review-gpt",
                    instructions="You are a code reviewer.",
                    llm=LLM(model="gpt-4o")
                ),
                task_types=[ReviewTask],
                metadata={"specialty": "security"}
            )

            # Testing specialist
            tester = TeamMember(
                id="tester",
                agent=Agent(
                    id="test-gpt",
                    instructions="You are a testing expert.",
                    llm=LLM(model="gpt-4o")
                ),
                task_types=[TestTask],
                metadata={"coverage_target": 0.9}
            )
            ```
    """

    id: str
    """Unique identifier for the member."""

    agent: Agent
    """Agent configuration for task execution."""

    task_types: list[type[Task]]
    """Task types this member can handle."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional member metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ExecutionResult(BaseModel):
    """Result of a task execution.

    Captures all outputs and metadata from a task execution, including
    raw content, structured output, and execution details.

    Examples:
        Process execution result:
            ```python
            result = ExecutionResult(
                task=review_task,
                content="Code review completed.",
                output=ReviewOutput(
                    approved=True,
                    comments=["Good error handling", "Well documented"],
                    suggestions=[]
                ),
                assignee=reviewer,
                timestamp=datetime.now()
            )

            if result.output and result.output.approved:
                print("Review passed!")
                for comment in result.output.comments:
                    print(f"- {comment}")
            ```
    """

    task: Task
    """Task that was executed."""

    content: str | None = None
    """Raw output content."""

    output: BaseModel | None = None
    """Structured output data."""

    assignee: TeamMember | None = None
    """Member who executed the task."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When execution completed."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
