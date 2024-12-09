# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re

import orjson
from json_repair import repair_json

from liteswarm.types import JSON, Artifact, ArtifactStatus, ContextVariables
from liteswarm.utils.misc import find_tag_content

from .types import CodeBlock, DebugOutput, FileContent, FlutterOutput, Project


def extract_code_block(content: str) -> CodeBlock:
    r"""Extract the content from a code block.

    Args:
        content: The content to extract the code block from.

    Returns:
        CodeBlock containing the content and optional language.

    Example:
        >>> extract_code_block('```json\\n{"key": "value"}\\n```')
        CodeBlock(content='{\"key\": \"value\"}', language='json')
    """
    pattern = r"```(\w+)?\n?(.*?)```"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return CodeBlock(content=content.strip())

    language = match.group(1)
    content = match.group(2).strip()

    return CodeBlock(
        content=content,
        language=language,
    )


def dump_json(data: JSON, indent: bool = False) -> str:
    option = orjson.OPT_INDENT_2 if indent else None
    return orjson.dumps(data, option=option).decode()


def load_json(data: str) -> JSON:
    return orjson.loads(data)


def find_json_tag(text: str, tag: str) -> JSON:
    tag_content = find_tag_content(text, tag)
    if not tag_content:
        return None

    return repair_json(tag_content, return_objects=True)  # type: ignore


def print_artifact(artifact: Artifact) -> None:
    """Print a detailed, well-formatted view of an execution artifact."""
    print("\n" + "=" * 50)
    print(f"🏷  Artifact ID: {artifact.id}")
    print(f"📊 Status: {get_status_emoji(artifact.status.value)} {artifact.status}")
    print("=" * 50 + "\n")

    if artifact.error:
        print("❌ Execution Failed")
        print(f"Error: {artifact.error}")
        return

    print(f"✅ Successfully completed {len(artifact.task_results)} tasks\n")

    for i, task_result in enumerate(artifact.task_results, 1):
        agent_id = task_result.assignee.agent.id if task_result.assignee else "unknown"
        print(f"Task {i}/{len(artifact.task_results)}")
        print(f"├─ ID: {task_result.task.id}")
        print(f"├─ Type: {task_result.task.type}")
        print(f"├─ Title: {task_result.task.title}")
        print(f"├─ Status: {get_status_emoji(task_result.task.status.value)} {task_result.task.status}")  # fmt: off
        print(f"└─ Executed by: 🤖 {agent_id}\n")

        if task_result.output:
            print("   Output:")
            for key, value in task_result.output.model_dump().items():
                print(f"   ├─ {key}: {value}")
            print()

    print("=" * 50)


def get_status_emoji(status: str) -> str:
    """Get an appropriate emoji for a status."""
    emojis: dict[str, str] = {
        "COMPLETED": "✅",
        "FAILED": "❌",
        "IN_PROGRESS": "⏳",
        "PENDING": "⏳",
        "EXECUTING": "🔄",
    }

    return emojis.get(str(status).upper(), "❓")


def extract_project_from_artifact(
    artifact: Artifact,
    current_project: Project | None = None,
) -> Project:
    """Extract project state from an execution artifact.

    Processes task results to build a complete picture of the project state,
    including all files and directories.

    Args:
        artifact: The execution artifact to process.
        current_project: Optional current project state to update.

    Returns:
        A new Project instance with the updated state.

    Examples:
        From scratch:
            ```python
            project = extract_project_from_artifact(artifact)
            ```

        Update existing:
            ```python
            updated = extract_project_from_artifact(artifact, current_project)
            ```
    """
    project = Project()
    if current_project:
        project.tech_stack = current_project.tech_stack
        project.directories = current_project.directories.copy()
        project.files = [FileContent.model_validate(f.model_dump()) for f in current_project.files]

    # Only process successful executions
    if artifact.status != ArtifactStatus.COMPLETED:
        return project

    # Process each task result
    for result in artifact.task_results:
        if not result.output:
            continue

        # Extract files based on task type
        output_data = result.output.model_dump()
        if isinstance(result.output, FlutterOutput | DebugOutput):
            project.update_from_files([FileContent.model_validate(f) for f in output_data["files"]])

    return project


def create_context_from_project(project: Project) -> ContextVariables:
    """Create context dictionary from project state.

    Args:
        project: The project state to convert.

    Returns:
        A dictionary suitable for ContextVariables.

    Examples:
        ```python
        context = create_context_from_project(project)
        ```
    """
    return ContextVariables(**project.model_dump())
