from typing import Any

from liteswarm.swarm_team import PromptTemplate


class SoftwarePlanTemplate(PromptTemplate):
    """Template for software development plans."""

    @property
    def template(self) -> str:
        """Return the template string for generating plans."""
        return """
        Create a detailed development plan for the following project:
        {prompt}

        Available task types: {task_types}

        Project Context:
        {context_section}

        The plan should:
        1. Break down the work into clear, actionable tasks
        2. Specify the type of engineer needed for each task
        3. List dependencies between tasks
        4. Define clear deliverables (files to be created/modified)
        5. Consider the tech stack and existing codebase
        6. Work within the existing project structure if provided

        Return the plan as a JSON object with:
        - tasks: list of tasks, each with:
            - id: unique string identifier
            - title: clear task title
            - description: detailed task description
            - task_type: type of task (must be one of: {task_types})
            - dependencies: list of task IDs this depends on
            - metadata: dictionary of additional information
                - deliverables: list of files to be created/modified

        Example output:
        ```json
        {{
            "tasks": [
                {{
                    "id": "task1",
                    "title": "Implement Todo Model",
                    "description": "Create the Todo model class with required fields",
                    "task_type": "flutter",
                    "dependencies": [],
                    "metadata": {{
                        "deliverables": ["lib/models/todo.dart"]
                    }}
                }}
            ]
        }}
        ```
        """

    def format_context(self, prompt: str, context: dict[str, Any]) -> str:
        task_types = context.get("available_types", [])
        project_context = context.get("project", "No project context provided")

        return self.template.format(
            prompt=prompt,
            task_types=", ".join(task_types),
            context_section=project_context,
        )
