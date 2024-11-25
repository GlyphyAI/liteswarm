# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Self, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent
from liteswarm.utils.misc import change_field_type

TaskType = TypeVar("TaskType", bound="Task")
"""Type variable representing a Task or its subclass."""

TaskInstructions: TypeAlias = str | Callable[[TaskType, ContextVariables], str]
"""Instructions for executing a task.

Can be either:
- A string template with the task instructions
- A function that generates instructions based on the task and context

Example:
```python
# String template
instructions: TaskInstructions = f"Process the data in {task.metadata['file_path']}"

# Function generator
def generate_instructions(task: Task, context: ContextVariables) -> str:
    return f"Process {task.title} using context: {context.get('parameters')}"
```
"""

TaskOutput: TypeAlias = type[BaseModel] | Callable[[str, ContextVariables], BaseModel]
"""Schema or parser for structured task output.

Can be either:
- A Pydantic model class for direct JSON parsing
- A function that parses the output using context

Example:
```python
class DataOutput(BaseModel):
    processed_items: int
    success_rate: float

# Using model directly
output_schema: TaskOutput = DataOutput

# Using custom parser
def parse_output(content: str, context: ContextVariables) -> BaseModel:
    # Custom parsing logic
    return DataOutput(...)
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


class Task(BaseModel):
    """Instance of a task created based on its `TaskDefinition`.

    A task represents a single unit of work in a plan, with identification,
    status tracking, dependencies, and metadata. You can define your own
    custom fields by subclassing this model.

    Example:
    ```python
    class DataProcessingTask(Task):
        input_file: str
        batch_size: int = 100

    task_def = TaskDefinition.create(
        task_type="process_data",
        task_schema=DataProcessingTask,
        task_instructions="Process {task.input_file} in batches of {task.batch_size}",
        task_output=DataOutput  # From previous example
    )

    task = DataProcessingTask(
        id="task-1",
        title="Process customer data",
        input_file="data.csv",
        dependencies=["task-0"]
    )
    ```
    """

    id: str
    """Unique identifier for the task. REQUIRED: Must be provided."""

    title: str
    """Short descriptive title of the task. REQUIRED: Must be provided."""

    task_type: str
    """Type of the task. USE_DEFAULT: Do not modify, use default value."""

    description: str | None = None
    """Detailed description of the task. OPTIONAL: Provide if needed."""

    status: TaskStatus = TaskStatus.PENDING
    """Current status of the task. USE_DEFAULT: Do not modify, use default value."""

    assignee: str | None = None
    """ID of the assigned team member. USE_DEFAULT: Do not modify, use default value."""

    dependencies: list[str] = Field(default_factory=list)
    """List of task IDs that must be completed before this task. OPTIONAL: Provide if needed."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional task metadata. USE_DEFAULT: Do not modify, use default value."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class TaskDefinition(BaseModel):
    """Definition of a task type, including how to create tasks of this type.

    A TaskDefinition serves as a template for creating and executing tasks, defining:
    - The task type identifier
    - The schema for tasks of this type (validation)
    - Instructions for executing the task (prompt)
    - Optional output format for structured responses

    Example:
    ```python
    # Define custom task type with required fields
    class CodeReviewTask(Task):
        pr_url: str
        repository: str
        review_type: Literal["security", "style", "functionality"]

    # Define output schema
    class ReviewOutput(BaseModel):
        approved: bool
        issues: list[str]

    # Create task definition with dynamic instructions
    def generate_review_instructions(task: CodeReviewTask, context: ContextVariables) -> str:
        return f'''Review the PR at {task.pr_url} focusing on {task.review_type} aspects.
                  Use the team's coding standards from {context.get('standards_doc')}.'''

    review_task_def = TaskDefinition.create(
        task_type="code_review",
        task_schema=CodeReviewTask,
        task_instructions=generate_review_instructions,
        task_output=ReviewOutput,
    )
    ```
    """

    task_type: str
    """Unique identifier for this type of task"""

    task_schema: type[Task]
    """Pydantic model defining the structure of tasks of this type"""

    task_instructions: TaskInstructions
    """Instructions template or function to generate task instructions"""

    task_output: TaskOutput | None = None
    """Optional schema or function for parsing structured task output"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )

    @classmethod
    def create(
        cls,
        task_type: str,
        task_schema: type[Task],
        task_instructions: TaskInstructions,
        task_output: TaskOutput | None = None,
    ) -> "TaskDefinition":
        """Create a new `TaskDefinition` instance with the `task_type` field injected into the `task_schema`.

        Returns:
            The created `TaskDefinition` instance.
        """
        task_schema = change_field_type(
            model_type=task_schema,
            field_name="task_type",
            new_type=str,
            new_model_name=task_schema.__name__,
            default=task_type,
        )

        return cls(
            task_type=task_type,
            task_schema=task_schema,
            task_instructions=task_instructions,
            task_output=task_output,
        )

    @model_validator(mode="after")
    def inject_task_type_into_schema(self) -> Self:
        """Ensures that the `task_schema` includes the `task_type` field with the correct default value.

        Returns:
            The updated `TaskDefinition` instance.
        """
        task_type = self.task_schema.model_fields.get("task_type")
        if not task_type or task_type.default != self.task_type:
            self.task_schema = change_field_type(
                model_type=self.task_schema,
                field_name="task_type",
                new_type=str,
                new_model_name=self.task_schema.__name__,
                default=self.task_type,
            )

        return self


class Plan(BaseModel):
    """A plan consisting of ordered tasks with dependencies.

    Plans organize tasks into a workflow, tracking:
    - The list of tasks to be executed
    - Dependencies between tasks
    - Overall execution status
    - Additional metadata

    Example:
    ```python
    # Create tasks with dependencies
    tasks = [
        CodeReviewTask(
            id="review-1",
            title="Review authentication changes",
            pr_url="https://github.com/org/repo/pull/1",
            repository="auth-service",
            review_type="security"
        ),
        CodeReviewTask(
            id="review-2",
            title="Review API changes",
            pr_url="https://github.com/org/repo/pull/2",
            repository="api-service",
            review_type="functionality",
            dependencies=["review-1"]  # Must complete security review first
        )
    ]

    plan = Plan(tasks=tasks)
    next_tasks = plan.get_next_tasks()  # Returns [review-1]
    ```
    """

    tasks: list[Task]
    """List of tasks that make up this plan"""

    status: PlanStatus = PlanStatus.DRAFT
    """Current status of the plan"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional plan metadata"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )

    def validate_dependencies(self) -> list[str]:
        """Validate that all task dependencies exist.

        Returns:
            List of error messages if any dependencies are invalid, otherwise an empty list.
        """
        task_ids = {task.id for task in self.tasks}
        errors: list[str] = []

        for task in self.tasks:
            invalid_deps = [dep for dep in task.dependencies if dep not in task_ids]
            if invalid_deps:
                errors.append(f"Task {task.id} has invalid dependencies: {invalid_deps}")

        return errors

    def get_next_tasks(self) -> list[Task]:
        """Get tasks that are ready to be executed (all dependencies completed).

        Returns:
            List of tasks that are ready to be executed.
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

    Team members are agents with specific capabilities:
    - Unique identifier within the team
    - Associated agent configuration
    - List of task types they can handle
    - Additional metadata about the team member

    Example:
    ```python
    security_reviewer = TeamMember(
        id="security-bot",
        agent=Agent(
            id="security-gpt",
            instructions="You are a security-focused code reviewer...",
            llm=LLM(model="gpt-4o")
        ),
        task_types=["code_review"],
        metadata={
            "specialization": "security",
            "experience_level": "senior"
        }
    )
    ```
    """

    id: str
    """Unique identifier for the team member"""

    agent: Agent
    """Agent configuration for this team member"""

    task_types: list[str]
    """List of task types this member can execute"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional metadata about the team member"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ExecutionResult(BaseModel):
    """Represents the result of an executed task.

    Captures all relevant information about a task execution:
    - The original task
    - Raw content from the agent
    - Structured output (if applicable)
    - Team member who executed the task
    - Timestamp of execution

    Example:
    ```python
    result = ExecutionResult(
        task=code_review_task,
        content="The PR has several security issues...",
        output=ReviewOutput(
            issues=[
                Issue(severity="high", description="Unvalidated user input"),
                Issue(severity="medium", description="Weak password policy")
            ],
            approved=False
        ),
        assignee=security_reviewer,
    )
    ```
    """

    task: Task
    """The task that was executed"""

    content: str | None = None
    """Raw content returned by the agent"""

    output: BaseModel | None = None
    """Structured output parsed according to task schema"""

    assignee: TeamMember | None = None
    """Team member who executed the task"""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When the task was executed"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
