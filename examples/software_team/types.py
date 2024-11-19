from pydantic import BaseModel

from liteswarm.swarm_team import Task


class FileContent(BaseModel):
    """Represents a file's content and metadata."""

    filepath: str
    content: str


class CodeBlock(BaseModel):
    """Represents a code block."""

    content: str
    language: str | None = None


class FlutterTask(Task):
    """Task for implementing Flutter features."""

    feature_type: str


class FlutterOutput(BaseModel):
    """Output schema for Flutter implementation tasks."""

    thoughts: str
    files: list[FileContent]


class DebugTask(Task):
    """Task for debugging issues."""

    error_type: str
    stack_trace: str | None = None


class DebugOutput(BaseModel):
    """Output schema for debug tasks."""

    thoughts: str
    root_cause: str
    files: list[FileContent]
