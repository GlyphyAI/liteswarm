from typing import Any

import orjson

from liteswarm.swarm_team import ExecutionResult, TaskDefinition
from liteswarm.types import ContextVariables

from .types import DebugOutput, DebugTask, FlutterOutput, FlutterTask
from .utils import load_partial_json


def get_flutter_instructions(task: FlutterTask, context: ContextVariables) -> str:
    project: dict[str, Any] = context.get("project", {})
    project_context = f"""
    - Framework: {project.get('framework', 'Flutter')}
    - Structure: {', '.join(project.get('directories', []))}
    """

    history: list[ExecutionResult] = context.get("execution_history", [])
    history_context = "\n".join(
        f"- {result.task.title} by {result.assignee.agent.id}"
        for result in history[-3:]
        if result.assignee and result.assignee.agent
    )

    return f"""
    Implement the following Flutter feature:

    Feature Type: {task.feature_type}
    Title: {task.title}
    Description: {task.description}

    Project Context:
    {project_context or "null"}

    Previous Implementation History:
    {history_context or "null"}

    Provide your implementation in the standard format with <thoughts> and <files> sections.
    """


def get_debug_instructions(task: DebugTask, context: ContextVariables) -> str:
    return f"""
    Debug the following issue:

    Error Type: {task.error_type}
    Description: {task.description}

    Stack Trace:
    {task.stack_trace or 'No stack trace provided'}

    Analyze the issue and provide:
    1. Root cause analysis
    2. Solution implementation
    3. Prevention recommendations

    Provide your analysis in the standard format with <thoughts> and <files> sections.
    """


def get_flutter_output(content: str, context: ContextVariables) -> FlutterOutput:
    """Parse the agent's response into structured data."""
    try:
        thoughts_start = content.index("<thoughts>") + len("<thoughts>")
        thoughts_end = content.index("</thoughts>")
        thoughts = content[thoughts_start:thoughts_end].strip()

        files_start = content.index("<files>") + len("<files>")
        files_end = content.index("</files>")
        files_json = content[files_start:files_end].strip()
        files = load_partial_json(files_json)

        result = {"thoughts": thoughts, "files": files}

        return FlutterOutput.model_validate(result)

    except (ValueError, orjson.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse agent response: {e}") from e


def get_debug_output(content: str, context: ContextVariables) -> DebugOutput:
    """Parse the agent's response into structured data."""
    try:
        thoughts_start = content.index("<thoughts>") + len("<thoughts>")
        thoughts_end = content.index("</thoughts>")
        thoughts = content[thoughts_start:thoughts_end].strip()

        root_cause_start = content.index("<root_cause>") + len("<root_cause>")
        root_cause_end = content.index("</root_cause>")
        root_cause = content[root_cause_start:root_cause_end].strip()

        files_start = content.index("<files>") + len("<files>")
        files_end = content.index("</files>")
        files_json = content[files_start:files_end].strip()
        files = load_partial_json(files_json)

        result = {"thoughts": thoughts, "root_cause": root_cause, "files": files}

        return DebugOutput.model_validate(result)

    except (ValueError, orjson.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse agent response: {e}") from e


def create_task_definitions() -> list[TaskDefinition]:
    return [
        TaskDefinition.create(
            task_type="flutter",
            task_schema=FlutterTask,
            task_instructions=get_flutter_instructions,
            task_output=get_flutter_output,
        ),
        TaskDefinition.create(
            task_type="debug",
            task_schema=DebugTask,
            task_instructions=get_debug_instructions,
            task_output=get_debug_output,
        ),
    ]
