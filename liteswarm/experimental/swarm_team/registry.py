# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.types import TaskDefinition


class TaskRegistry:
    """Registry for managing task types and their definitions.

    Provides a central store for task definitions with:
    - Registration of single or multiple task types
    - Type-safe task definition lookup
    - Task type enumeration

    Example:
    ```python
    # Create task definitions
    review_def = TaskDefinition.create(
        task_type="code_review",
        task_schema=CodeReviewTask,
        task_instructions="Review {task.pr_url}..."
    )

    test_def = TaskDefinition.create(
        task_type="testing",
        task_schema=TestingTask,
        task_instructions="Test {task.test_path}..."
    )

    # Initialize registry
    registry = TaskRegistry([review_def, test_def])

    # Add another task type
    deploy_def = TaskDefinition.create(
        task_type="deployment",
        task_schema=DeploymentTask,
        task_instructions="Deploy to {task.environment}..."
    )
    registry.register_task(deploy_def)

    # Get task definition
    review_task_def = registry.get_task_definition("code_review")

    # List available task types
    task_types = registry.list_task_types()  # ["code_review", "testing", "deployment"]
    ```
    """

    def __init__(self, task_definitions: list[TaskDefinition] | None = None) -> None:
        """Initialize a new TaskRegistry.

        Args:
            task_definitions: Optional list of task definitions to register initially
        """
        self._registry: dict[str, TaskDefinition] = {}
        if task_definitions:
            self.register_tasks(task_definitions)

    def register_task(self, task_definition: TaskDefinition) -> None:
        """Register a single task definition.

        Args:
            task_definition: The task definition to register

        Example:
        ```python
        registry.register_task(
            TaskDefinition.create(
                task_type="analysis",
                task_schema=AnalysisTask,
                task_instructions="Analyze {task.data_path}..."
            )
        )
        ```
        """
        self._registry[task_definition.task_type] = task_definition

    def register_tasks(self, task_definitions: list[TaskDefinition]) -> None:
        """Register multiple task definitions at once.

        Args:
            task_definitions: List of task definitions to register

        Example:
        ```python
        registry.register_tasks([
            review_def,
            test_def,
            deploy_def
        ])
        ```
        """
        for task_definition in task_definitions:
            self.register_task(task_definition)

    def get_task_definition(self, task_type: str) -> TaskDefinition:
        """Get a task definition by its type.

        Args:
            task_type: The type identifier of the task

        Returns:
            The corresponding task definition

        Raises:
            KeyError: If task_type is not registered

        Example:
        ```python
        review_def = registry.get_task_definition("code_review")
        task = review_def.task_schema(
            id="review-1",
            title="Review PR #123",
            pr_url="..."
        )
        ```
        """
        return self._registry[task_type]

    def list_task_types(self) -> list[str]:
        """Get a list of all registered task types.

        Returns:
            List of task type identifiers

        Example:
        ```python
        task_types = registry.list_task_types()
        print("Available tasks:")
        for task_type in task_types:
            print(f"- {task_type}")
        ```
        """
        return list(self._registry.keys())
