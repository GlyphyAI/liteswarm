# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from liteswarm.types import ContextVariables, TaskDefinition
from liteswarm.utils import dedent_prompt

from .types import DebugOutput, DebugTask, FlutterOutput, FlutterTask
from .utils import dump_json


def build_flutter_instructions(task: FlutterTask, context: ContextVariables) -> str:
    """Build the instructions for a Flutter task."""
    project: dict[str, Any] = context.get("project", {})
    output_format: dict[str, Any] = context.get_reserved("output_format", {})

    return dedent_prompt(f"""
    Implement the following Flutter feature:
    - Feature Type: {task.feature_type}
    - Title: {task.title}
    - Description: {task.description}

    Project Context:
    - Framework: {project.get('framework', 'Flutter')}
    - Structure: {', '.join(project.get('directories', []))}

    Your response MUST follow this output format:
    {dump_json(output_format)}

    Do not format the output in any other way than the output format.
    Do not use backticks to format the output.
    """)


def build_debug_instructions(task: DebugTask, context: ContextVariables) -> str:
    """Build the instructions for a Debug task."""
    project: dict[str, Any] = context.get("project", {})
    output_format: dict[str, Any] = context.get_reserved("output_format", {})

    return dedent_prompt(f"""
    Debug the following issue:
    - Error Type: {task.error_type}
    - Description: {task.description}

    Stack Trace:
    {task.stack_trace or 'No stack trace provided'}

    Project Context:
    - Framework: {project.get('framework', 'Flutter')}
    - Structure: {', '.join(project.get('directories', []))}

    Analyze the issue and provide:
    1. Root cause analysis
    2. Solution implementation
    3. Prevention recommendations

    Your response MUST follow this output format:
    {dump_json(output_format)}

    Do not format the output in any other way than the output format.
    Do not use backticks to format the output.
    """)


def create_task_definitions() -> list[TaskDefinition]:
    return [
        TaskDefinition.create(
            task_type="flutter_feature",
            task_schema=FlutterTask,
            task_instructions=build_flutter_instructions,
            task_output=FlutterOutput,
        ),
        TaskDefinition.create(
            task_type="flutter_debug",
            task_schema=DebugTask,
            task_instructions=build_debug_instructions,
            task_output=DebugOutput,
        ),
    ]
