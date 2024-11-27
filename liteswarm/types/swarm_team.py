# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable, Sequence
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent

TaskType = TypeVar("TaskType", bound="Task")
"""Type variable representing a Task or its subclass."""

TaskInstructions: TypeAlias = str | Callable[[TaskType, ContextVariables], str]
"""Instructions for executing a task.

Can be either a string template with static instructions or a function that generates
dynamic instructions based on the task and context.

Example:
    ```python
    # Static template
    instructions: TaskInstructions = "Process data in {task.input_file}"

    # Dynamic generator
    def generate_instructions(task: Task, context: ContextVariables) -> str:
        return f"Process {task.title} with {context.get('parameters')}"
    ```
"""

TaskResponseFormat: TypeAlias = type[BaseModel] | Callable[[str, ContextVariables], BaseModel]
"""Schema or parser for parsing the task output.

Can be either a Pydantic model class for direct parsing or a function that parses
output with context.

Example:
    ```python
    # Direct model
    response_format: TaskResponseFormat = DataOutput

    # Custom parser
    def parse_output(content: str, context: ContextVariables) -> BaseModel:
        return DataOutput.model_validate_json(content)
    ```
"""


class TaskStatus(str, Enum):
    """Status of a task in the execution lifecycle."""

    PENDING = "pending"
    """Task is created but not started"""

    IN_PROGRESS = "in_progress"
    """Task is currently being executed"""

    COMPLETED = "completed"
    """Task finished successfully"""

    FAILED = "failed"
    """Task execution failed"""


class PlanStatus(str, Enum):
    """Status of a plan in its lifecycle."""

    DRAFT = "draft"
    """Plan is created but not approved"""

    APPROVED = "approved"
    """Plan is approved and ready for execution"""

    IN_PROGRESS = "in_progress"
    """Plan is currently being executed"""

    COMPLETED = "completed"
    """All tasks in plan completed successfully"""

    FAILED = "failed"
    """Plan execution failed"""


class Task(BaseModel):
    """Base class for defining task schemas in a SwarmTeam workflow.

    Tasks are the fundamental units of work in a SwarmTeam. Each task has a type,
    which determines its schema and how it should be executed.

    Example:
        ```python
        class DataProcessingTask(Task):
            type: Literal["data_processing"]
            input_file: str
            batch_size: int = 100

        task = DataProcessingTask(
            type="data_processing",
            id="task-1",
            title="Process customer data",
            input_file="data.csv"
        )
        ```
    """

    type: str
    """Type identifier for task matching with team members"""

    id: str
    """Unique identifier for the task"""

    title: str
    """Short descriptive title of the task"""

    description: str | None = None
    """Optional detailed description of the task"""

    status: TaskStatus = TaskStatus.PENDING
    """Current execution status of the task"""

    assignee: str | None = None
    """ID of the assigned team member executing the task"""

    dependencies: list[str] = Field(default_factory=list)
    """IDs of tasks that must complete first"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional task-specific data"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )

    @classmethod
    def get_task_type(cls) -> str:
        """Get the type identifier for this task class.

        Returns:
            The task type string defined in the schema.

        Raises:
            ValueError: If task type is not defined as a Literal.
        """
        type_field = cls.model_fields["type"]
        type_field_annotation = type_field.annotation

        if type_field_annotation and get_origin(type_field_annotation) is Literal:
            return get_args(type_field_annotation)[0]

        raise ValueError("Task type is not defined as a Literal in the task schema")


class TaskDefinition(BaseModel):
    """Definition of a task type and its execution requirements.

    TaskDefinition serves as a blueprint for creating and executing tasks, specifying
    their schema, instructions, and output format.

    Example:
        ```python
        task_def = TaskDefinition(
            task_schema=DataProcessingTask,
            task_instructions="Process {task.input_file}",
            task_response_format=ProcessingOutput
        )
        ```
    """

    task_schema: type[Task]
    """Pydantic model for task validation"""

    task_instructions: TaskInstructions
    """Template or function for task instructions"""

    task_response_format: TaskResponseFormat | None = None
    """Optional schema or parser for parsing the task output"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class Plan(BaseModel):
    """A plan consisting of ordered tasks with dependencies.

    Plans organize tasks into a workflow, tracking their dependencies and execution
    status.

    Example:
        ```python
        plan = Plan(tasks=[
            Task(id="task-1", title="First step"),
            Task(id="task-2", title="Second step", dependencies=["task-1"])
        ])
        ```
    """

    tasks: Sequence[Task]
    """List of tasks in this plan"""

    status: PlanStatus = PlanStatus.DRAFT
    """Current plan status"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional plan metadata"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
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
        """Get tasks that are ready to be executed.

        A task is ready when it's pending and all its dependencies are completed.

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


class TeamMember(BaseModel):
    """Represents a team member that can execute tasks.

    Team members are agents with specific capabilities for handling certain types
    of tasks.

    Example:
        ```python
        member = TeamMember(
            id="data-processor",
            agent=Agent(id="processor-gpt"),
            task_types=[DataProcessingTask]
        )
        ```
    """

    id: str
    """Unique identifier for the team member"""

    agent: Agent
    """Agent configuration used to execute tasks"""

    task_types: list[type[Task]]
    """Task types this team member can handle"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional team member metadata"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ExecutionResult(BaseModel):
    """Result of an executed task.

    Captures the output and metadata from a task execution.

    Example:
        ```python
        result = ExecutionResult(
            task=task,
            content="Processing complete",
            output=ProcessingOutput(items=100)
        )
        ```
    """

    task: Task
    """The executed task"""

    content: str | None = None
    """Raw output content"""

    output: BaseModel | None = None
    """Structured output data"""

    assignee: TeamMember | None = None
    """Member who executed the task"""

    timestamp: datetime = Field(default_factory=datetime.now)
    """Execution timestamp"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
