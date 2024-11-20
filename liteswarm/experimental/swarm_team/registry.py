# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.types import TaskDefinition


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
