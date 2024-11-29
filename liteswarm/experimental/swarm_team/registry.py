# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.types import TaskDefinition


class TaskRegistry:
    """Registry for managing task definitions.

    Maintains a central store of task definitions, providing methods for registration,
    lookup, and type validation.

    Examples:
        Create a registry with initial tasks:
            ```python
            # Define task types
            class ReviewTask(Task):
                pr_url: str
                review_type: str

            class TestTask(Task):
                path: str
                coverage: float

            # Create and initialize registry
            registry = TaskRegistry([
                TaskDefinition(
                    task_schema=ReviewTask,
                    task_instructions="Review {task.pr_url}"
                ),
                TaskDefinition(
                    task_schema=TestTask,
                    task_instructions="Test {task.path} with {task.coverage}% coverage"
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
            task_definitions: Optional list of initial task definitions.
        """
        self._registry: dict[str, TaskDefinition] = {}
        if task_definitions:
            self.register_tasks(task_definitions)

    def register_task(self, task_definition: TaskDefinition) -> None:
        """Register a single task definition.

        Args:
            task_definition: Task definition to register.

        Examples:
            Register a new task type:
                ```python
                registry.register_task(
                    TaskDefinition(
                        task_schema=AnalysisTask,
                        task_instructions="Analyze data in {task.path}",
                        task_response_format=AnalysisOutput
                    )
                )
                ```
        """
        self._registry[task_definition.task_schema.get_task_type()] = task_definition

    def register_tasks(self, task_definitions: list[TaskDefinition]) -> None:
        """Register multiple task definitions.

        Args:
            task_definitions: List of task definitions to register.

        Examples:
            Register multiple task types:
                ```python
                registry.register_tasks([
                    TaskDefinition(
                        task_schema=ReviewTask,
                        task_instructions="Review {task.pr_url}"
                    ),
                    TaskDefinition(
                        task_schema=TestTask,
                        task_instructions="Test {task.path}"
                    )
                ])
                ```
        """
        for task_definition in task_definitions:
            self.register_task(task_definition)

    def get_task_definition(self, task_type: str) -> TaskDefinition:
        """Get a task definition by type.

        Args:
            task_type: Type identifier to look up.

        Returns:
            Corresponding task definition.

        Raises:
            KeyError: If task type is not registered.

        Examples:
            Look up and use a task definition:
                ```python
                review_def = registry.get_task_definition("review")
                task = review_def.task_schema(
                    id="review-1",
                    title="Review PR #123",
                    pr_url="github.com/org/repo/123"
                )
                ```
        """
        return self._registry[task_type]

    def get_task_definitions(self) -> list[TaskDefinition]:
        """Get all registered task definitions.

        Returns:
            List of all registered task definitions.
        """
        return list(self._registry.values())

    def list_task_types(self) -> list[str]:
        """Get all registered task types.

        Returns:
            List of registered task type identifiers.

        Examples:
            List available task types:
                ```python
                types = registry.list_task_types()  # ["review", "test", "deploy"]
                for task_type in types:
                    print(f"Registered task type: {task_type}")
                ```
        """
        return list(self._registry.keys())

    def contains_task_type(self, task_type: str) -> bool:
        """Check if a task type exists.

        Args:
            task_type: Type identifier to check.

        Returns:
            True if type is registered, False otherwise.

        Examples:
            Check task type availability:
                ```python
                if registry.contains_task_type("review"):
                    review_def = registry.get_task_definition("review")
                ```
        """
        return task_type in self._registry
