from typing import Any

from liteswarm.swarm import Swarm
from liteswarm.swarm_team import Planner, PlanTemplate
from liteswarm.types import Agent


class SoftwarePlanTemplate(PlanTemplate):
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


def create_planner(swarm: Swarm | None = None) -> Planner:
    """Create a software planner agent."""
    agent = Agent.create(
        id="planner",
        model="gpt-4o",
        instructions="""You are a technical project planner specializing in Flutter app development.
        Create detailed development plans, ensuring:
        1. Clear task breakdown with dependencies
        2. Appropriate engineer assignments
        3. Well-defined deliverables (files to be created/modified)
        4. Technical feasibility
        5. Efficient execution order

        Consider the provided context files when planning tasks.""",
    )

    return Planner(
        agent=agent,
        template=SoftwarePlanTemplate(),
        swarm=swarm,
    )


def create_flutter_engineer() -> Agent:
    """Create a Flutter engineer agent."""
    return Agent.create(
        id="flutter_engineer",
        model="gpt-4o",
        instructions="""You are a Flutter engineer specializing in mobile app development.
        When implementing features:
        1. Write clean, maintainable Flutter code
        2. Follow Flutter best practices and patterns
        3. Consider performance and user experience
        4. Always provide complete file contents

        Your response must use two root XML tags to separate thoughts and implementation:

        <thoughts>
        Explain your implementation approach here. This section should include:
        - What changes you're making and why
        - Key implementation decisions
        - Any important considerations
        The content can be free-form text, formatted for readability.
        </thoughts>

        <files>
        Provide complete file contents in JSON format:
        [
            {
                "filepath": "path/to/file.dart",
                "content": "// Complete file contents here"
            }
        ]
        </files>

        Example response:
        <thoughts>
        I'll implement the Todo model class with these fields:
        - id: unique identifier
        - title: task title
        - completed: completion status
        - createdAt: timestamp

        I'll also add JSON serialization for persistence.
        </thoughts>

        <files>
        [
            {
                "filepath": "lib/models/todo.dart",
                "content": "import 'package:flutter/foundation.dart';\n\nclass Todo {\n    final String id;\n    final String title;\n    final bool completed;\n    final DateTime createdAt;\n\n    Todo({\n        required this.id,\n        required this.title,\n        this.completed = false,\n        DateTime? createdAt,\n    }) : this.createdAt = createdAt ?? DateTime.now();\n}"
            }
        ]
        </files>""",
    )


def create_debug_engineer() -> Agent:
    """Create a debug engineer agent."""
    return Agent.create(
        id="debug_engineer",
        model="gpt-4o",
        instructions="""You are a debugging specialist for Flutter applications.
        Help resolve:
        1. Runtime errors and exceptions
        2. Build and compilation issues
        3. Linting and static analysis problems
        4. Performance bottlenecks

        Always analyze the error context and relevant files before suggesting fixes.
        Provide solutions in the same format as the Flutter engineer.""",
    )
