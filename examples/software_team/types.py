from pydantic import BaseModel, Field

from liteswarm.swarm_team import Plan, Task


class CodeBlock(BaseModel):
    """Represents a code block."""

    content: str
    language: str | None = None
    filepath: str | None = None


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


class SoftwarePlan(Plan):
    """Represents a software development plan."""

    tasks: list[SoftwareTask]
    repository: str | None = None
    tech_stack: dict[str, str] = Field(default_factory=dict)
