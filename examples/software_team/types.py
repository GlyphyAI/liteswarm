# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict

from liteswarm.types import Plan, Task


class FileContent(BaseModel):
    """Represents a file's content and metadata."""

    filepath: str
    """The full path to the file."""

    content: str
    """The full content of the file."""

    model_config = ConfigDict(
        extra="forbid",
    )


class TechStack(BaseModel):
    """Technical stack configuration."""

    platform: str
    """The platform the project is built for."""

    languages: list[str]
    """The languages used in the project."""

    frameworks: list[str]
    """The frameworks used in the project."""

    model_config = ConfigDict(
        extra="forbid",
    )


class Project(BaseModel):
    """Represents the current state of a software project.

    Tracks all files and directories in the project, maintaining the complete
    project structure and technical configuration.

    Examples:
        Create a project:
            ```python
            project = Project(
                tech_stack=TechStack(
                    platform="mobile",
                    languages=["Dart"],
                    frameworks=["Flutter"],
                },
                directories=["lib", "lib/features"],
                files=[FileContent(filepath="lib/main.dart", content="void main() {...}")],
            )
            ```

        Update from artifact:
            ```python
            new_project = extract_project_from_artifact(artifact, current_project)
            ```
    """

    tech_stack: TechStack | None = None
    """Technical stack configuration."""

    directories: list[str] = []
    """List of all directories in the project."""

    files: list[FileContent] = []
    """List of all files in the project."""

    model_config = ConfigDict(
        extra="forbid",
    )

    def get_file(self, filepath: str) -> FileContent | None:
        """Get a file by its path.

        Args:
            filepath: The path of the file to find.

        Returns:
            The file content if found, None otherwise.
        """
        return next((f for f in self.files if f.filepath == filepath), None)

    def update_from_files(self, new_files: list[FileContent]) -> None:
        """Update project state with new files.

        Updates existing files and adds new ones, maintaining directory structure.

        Args:
            new_files: List of files to add or update.
        """
        for new_file in new_files:
            existing_file = self.get_file(new_file.filepath)
            if existing_file:
                existing_file.content = new_file.content
            else:
                self.files.append(new_file)

        dirs = {os.path.dirname(f.filepath) for f in self.files if os.path.dirname(f.filepath)}
        self.directories = sorted(dirs)


class CodeBlock(BaseModel):
    """Represents a code block."""

    content: str
    """The content of the code block."""

    language: str | None = None
    """The language of the code block."""

    model_config = ConfigDict(
        extra="forbid",
    )


class FlutterTask(Task):
    """Task for implementing Flutter features."""

    type: Literal["flutter_feature"]
    """The type of the task."""

    feature_type: str
    """Type of Flutter feature (e.g., 'widget', 'screen', 'service', etc.)."""

    model_config = ConfigDict(
        extra="forbid",
    )


class FlutterOutput(BaseModel):
    """Output schema for Flutter implementation tasks."""

    files: list[FileContent]
    """List of files to be created or modified for this feature."""

    model_config = ConfigDict(
        extra="forbid",
    )


class DebugTask(Task):
    """Task for debugging issues."""

    type: Literal["flutter_debug"]
    """The type of the task."""

    error_type: str
    """Type of error being debugged (e.g., 'runtime', 'build', 'state')."""

    stack_trace: str | None = None
    """Error stack trace if available."""

    model_config = ConfigDict(
        extra="forbid",
    )


class DebugOutput(BaseModel):
    """Output schema for debug tasks."""

    root_cause: str
    """The identified root cause of the error."""

    files: list[FileContent]
    """List of files that need to be modified to fix the error."""

    model_config = ConfigDict(
        extra="forbid",
    )


class SoftwarePlan(Plan):
    """Plan for a software project."""

    tasks: list[FlutterTask | DebugTask]
    """List of tasks to be completed."""

    model_config = ConfigDict(
        extra="forbid",
    )
