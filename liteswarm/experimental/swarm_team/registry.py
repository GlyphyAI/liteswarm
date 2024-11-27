# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.types import TaskDefinition


class TaskRegistry:
    """Registry for managing task definitions.

    Maintains a mapping of task types to their definitions, providing methods for
    registration and lookup.

    Example:
        ```python
        # Create registry with initial tasks
        registry = TaskRegistry([
            TaskDefinition(
                task_schema=ReviewTask,
                task_instructions="Review {task.pr_url}"
            ),
            TaskDefinition(
                task_schema=TestTask,
                task_instructions="Test {task.path}"
            )
        ])

        # Add another task type
        registry.register_task(
            TaskDefinition(
                task_schema=DeployTask,
                task_instructions="Deploy to {task.env}"
            )
        )
        ```
    """

    def __init__(self, task_definitions: list[TaskDefinition] | None = None) -> None:
        """Initialize a new registry.

        Args:
            task_definitions: Optional initial task definitions
        """
        self._registry: dict[str, TaskDefinition] = {}
        if task_definitions:
            self.register_tasks(task_definitions)

    def register_task(self, task_definition: TaskDefinition) -> None:
        """Register a single task definition.

        Args:
            task_definition: Task definition to register

        Example:
            ```python
            registry.register_task(
                TaskDefinition(
                    task_schema=AnalysisTask,
                    task_instructions="Analyze {task.data}"
                )
            )
            ```
        """
        self._registry[task_definition.task_schema.get_task_type()] = task_definition

    def register_tasks(self, task_definitions: list[TaskDefinition]) -> None:
        """Register multiple task definitions.

        Args:
            task_definitions: List of definitions to register

        Example:
            ```python
            registry.register_tasks([review_def, test_def])
            ```
        """
        for task_definition in task_definitions:
            self.register_task(task_definition)

    def get_task_definition(self, task_type: str) -> TaskDefinition:
        """Get a task definition by type.

        Args:
            task_type: Type identifier to look up

        Returns:
            Corresponding task definition

        Raises:
            KeyError: If task type is not registered

        Example:
            ```python
            review_def = registry.get_task_definition("review")
            task = review_def.task_schema(
                id="review-1",
                title="Review PR #123"
            )
            ```
        """
        return self._registry[task_type]

    def get_task_definitions(self) -> list[TaskDefinition]:
        """Get all registered task definitions.

        Returns:
            List of all task definitions
        """
        return list(self._registry.values())

    def list_task_types(self) -> list[str]:
        """Get all registered task types.

        Returns:
            List of task type identifiers

        Example:
            ```python
            types = registry.list_task_types()  # ["review", "test", "deploy"]
            ```
        """
        return list(self._registry.keys())

    def contains_task_type(self, task_type: str) -> bool:
        """Check if a task type exists.

        Args:
            task_type: Type identifier to check

        Returns:
            True if type is registered, False otherwise
        """
        return task_type in self._registry
