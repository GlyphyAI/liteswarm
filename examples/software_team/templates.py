# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from pydantic import BaseModel

from liteswarm.types import ContextVariables, Plan, Task
from liteswarm.utils import dedent_prompt

from .utils import dump_json


class SoftwarePlanTemplate(BaseModel):
    """Template for software development plans."""

    template: str = dedent_prompt("""
    Create a detailed development plan for the following project:
    {prompt}

    Project Context:
    {project_context}

    The plan should:
    1. Break down the work into clear, actionable tasks
    2. Each task must be one of the types specified in the plan JSON schema
    3. List dependencies between tasks
    4. Work within the existing project structure if provided

    Your response MUST follow this output format:
    {output_format}

    Example output:
    {output_example}

    Do not format the output in any other way than the output format.
    Do not use backticks to format the output.
    """)

    def format_context(self, prompt: str, context: ContextVariables) -> str:
        project_context: dict[str, Any] = context.get("project", {})
        output_format: dict[str, Any] = context.get_reserved("output_format", {})
        output_example: Plan = Plan(
            tasks=[
                Task(id="<id_0>", title="<title_0>", task_type="<task_type_0>"),
            ]
        )

        return self.template.format(
            prompt=prompt,
            project_context=dump_json(project_context),
            output_format=dump_json(output_format),
            output_example=output_example.model_dump_json(),
        )
