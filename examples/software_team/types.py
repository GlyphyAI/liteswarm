from pydantic import BaseModel, Field

from liteswarm.swarm_team import Task


class FileContent(BaseModel):
    """Represents a file's content and metadata."""

    filepath: str = Field(
        description="The full path to the file. REQUIRED: Must be provided.",
    )
    content: str = Field(
        description="The full content of the file. REQUIRED: Must be provided.",
    )


class CodeBlock(BaseModel):
    """Represents a code block."""

    content: str = Field(
        description="The content of the code block. REQUIRED: Must be provided.",
    )
    language: str | None = Field(
        default=None,
        description="The language of the code block. OPTIONAL: Use if the code is in a specific language.",
    )


class FlutterTask(Task):
    """Task for implementing Flutter features."""

    feature_type: str = Field(
        description="Type of Flutter feature (e.g., 'widget', 'screen', 'service'). REQUIRED: Must be provided.",
    )


class FlutterOutput(BaseModel):
    """Output schema for Flutter implementation tasks."""

    thoughts: str = Field(
        description="Your analysis and implementation approach for the Flutter feature. REQUIRED: Must be provided.",
    )
    files: list[FileContent] = Field(
        description="List of files to be created or modified for this feature. REQUIRED: Must be provided. Must include all file contents.",
    )


class DebugTask(Task):
    """Task for debugging issues."""

    error_type: str = Field(
        description="Type of error being debugged (e.g., 'runtime', 'build', 'state'). REQUIRED: Must be provided.",
    )
    stack_trace: str | None = Field(
        default=None,
        description="Error stack trace if available. OPTIONAL: Provide if available, otherwise use default value.",
    )


class DebugOutput(BaseModel):
    """Output schema for debug tasks."""

    thoughts: str = Field(
        description="Your analysis of the error and debugging process. REQUIRED: Must be provided."
    )
    root_cause: str = Field(
        description="The identified root cause of the error. REQUIRED: Must be provided."
    )
    files: list[FileContent] = Field(
        description="List of files that need to be modified to fix the error. REQUIRED: Must be provided. Must include all file contents.",
    )
