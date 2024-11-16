from pydantic import BaseModel, Field

from liteswarm.swarm_team import Task


class FileContent(BaseModel):
    """Represents a file's content and metadata."""

    filepath: str
    content: str


class ProjectContext(BaseModel):
    """Project context containing directory structure and file contents."""

    directories: list[str] = Field(default_factory=list)
    files: list[FileContent] = Field(default_factory=list)


class CodeBlock(BaseModel):
    """Represents a code block."""

    content: str
    language: str | None = None


class SoftwareTask(Task):
    """Represents a development task in the plan."""

    engineer_type: str = Field(
        description="Type of engineer needed",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of tasks this depends on",
    )
    deliverables: list[str] = Field(
        default_factory=list,
        description="Expected outputs from this task",
    )
