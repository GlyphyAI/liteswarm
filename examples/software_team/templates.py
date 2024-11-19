from typing import Any

from pydantic import BaseModel

from liteswarm.swarm_team import Plan, Task

from .utils import dump_json


class SoftwarePlanTemplate(BaseModel):
    """Template for software development plans."""

    template: str = """
    Create a detailed development plan for the following project:
    {prompt}

    Project Context:
    {context_section}

    The plan should:
    1. Break down the work into clear, actionable tasks
    2. Each task must be one of the types specified in the plan JSON schema
    3. List dependencies between tasks
    4. Work within the existing project structure if provided

    You MUST return the plan as a JSON object with the following schema:
    {plan_json_schema}

    Example output:
    ```json
    {plan_json_example}
    ```
    """

    def format_context(self, prompt: str, context: dict[str, Any]) -> str:
        project_context = context.get("project", "No project context provided")
        plan_json_schema = dump_json(context.get("plan_json_schema", {}))
        plan_json_example = Plan(
            tasks=[
                Task(id="<id_0>", title="<title_0>", task_type="<task_type_0>"),
            ]
        )

        return self.template.format(
            prompt=prompt,
            context_section=project_context,
            plan_json_schema=plan_json_schema,
            plan_json_example=plan_json_example.model_dump_json(),
        )
