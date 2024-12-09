# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from liteswarm.types.swarm_team import Plan, Task
from liteswarm.utils.pydantic import change_field_type
from liteswarm.utils.typing import union


def create_plan_with_tasks(task_types: list[type[Task]]) -> type[Plan]:
    """Create a plan schema with the given task types.

    Creates a new union type for the "tasks" field and uses that type to create a
    new Plan schema with the same name, replacing the "tasks" field with the new
    type.

    Args:
        task_types: List of task types to include in the plan schema.

    Returns:
        Plan schema with tasks replaced by the union of the given task types.

    Examples:
        Create a plan schema with tasks:
            ```python
            plan_schema = create_plan_with_tasks([ReviewTask, TestTask])
            # type of plan_schema.tasks is now list[ReviewTask | TestTask]
            ```
    """
    plan_schema_name = Plan.__name__
    task_schemas: Any = union(task_types)

    return change_field_type(
        model_type=Plan,
        field_name="tasks",
        new_type=list[task_schemas],
        new_model_name=plan_schema_name,
    )
