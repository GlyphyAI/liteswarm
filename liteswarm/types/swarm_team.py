# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Self, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent

TaskType = TypeVar("TaskType", bound="Task")
TaskInstructions: TypeAlias = str | Callable[[TaskType, ContextVariables], str]
TaskOutput: TypeAlias = type[BaseModel] | Callable[[str, ContextVariables], BaseModel]


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
    def create(cls, model_name: str | None = None, **kwargs: Any) -> type[Self]:
        return create_model(
            model_name or cls.__name__,
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


class Plan(BaseModel, Generic[TaskType]):
    """Base class for development plans."""

    tasks: list[TaskType]
    status: PlanStatus = PlanStatus.DRAFT
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, model_name: str | None = None, **kwargs: Any) -> type[Self]:
        return create_model(
            model_name or cls.__name__,
            __base__=cls,
            **kwargs,
        )

    def validate_dependencies(self) -> list[str]:
        """Validate that all task dependencies exist."""
        task_ids = {task.id for task in self.tasks}
        errors = []

        for task in self.tasks:
            invalid_deps = [dep for dep in task.dependencies if dep not in task_ids]
            if invalid_deps:
                errors.append(f"Task {task.id} has invalid dependencies: {invalid_deps}")

        return errors

    def get_next_tasks(self) -> list[TaskType]:
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
