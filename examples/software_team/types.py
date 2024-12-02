# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Literal

from pydantic import BaseModel, ConfigDict

from liteswarm.types import Plan, Task


class FileContent(BaseModel):
    """Represents a file's content and metadata."""

    filepath: str
    """The full path to the file."""

    content: str
    """The full content of the file."""

    model_config = ConfigDict(extra="forbid")


class CodeBlock(BaseModel):
    """Represents a code block."""

    content: str
    """The content of the code block."""

    language: str | None = None
    """The language of the code block."""

    model_config = ConfigDict(extra="forbid")


class FlutterTask(Task):
    """Task for implementing Flutter features."""

    type: Literal["flutter_feature"]
    """The type of the task."""

    feature_type: str
    """Type of Flutter feature (e.g., 'widget', 'screen', 'service', etc.)."""

    model_config = ConfigDict(extra="forbid")


class FlutterOutput(BaseModel):
    """Output schema for Flutter implementation tasks."""

    files: list[FileContent]
    """List of files to be created or modified for this feature."""

    model_config = ConfigDict(extra="forbid")


class DebugTask(Task):
    """Task for debugging issues."""

    type: Literal["flutter_debug"]
    """The type of the task."""

    error_type: str
    """Type of error being debugged (e.g., 'runtime', 'build', 'state')."""

    stack_trace: str | None = None
    """Error stack trace if available."""

    model_config = ConfigDict(extra="forbid")


class DebugOutput(BaseModel):
    """Output schema for debug tasks."""

    root_cause: str
    """The identified root cause of the error."""

    files: list[FileContent]
    """List of files that need to be modified to fix the error."""

    model_config = ConfigDict(extra="forbid")


class SoftwarePlan(Plan):
    """Plan for a software project."""

    tasks: list[FlutterTask | DebugTask]
    """List of tasks to be completed."""

    model_config = ConfigDict(extra="forbid")
