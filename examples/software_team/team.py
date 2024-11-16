from typing import Any, ClassVar

from liteswarm.swarm import Swarm
from liteswarm.swarm_team import Plan, PlannerAgent
from liteswarm.types import Agent, Result

from .types import ProjectContext, SoftwareTask
from .utils import extract_code_block


class SoftwarePlanner(PlannerAgent[SoftwareTask]):
    """Agent responsible for creating software development plans."""

    PLAN_TEMPLATE: ClassVar[str] = """
    Create a detailed development plan for the following project:
    {prompt}

    Available engineer types: {engineer_types}
    Each task must be assigned to one of these engineer types.

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
        - engineer_type: type of engineer needed (must be one of: {engineer_types})
        - dependencies: list of task IDs this depends on
        - deliverables: list of files to be created/modified
    - tech_stack: dictionary of technology choices

    Example output:
    ```json
    {{
        "tasks": [
            {{
                "id": "task1",
                "title": "Implement Todo Model",
                "description": "Create the Todo model class with required fields",
                "engineer_type": "flutter",
                "dependencies": [],
                "deliverables": ["lib/models/todo.dart"]
            }}
        ],
        "tech_stack": {{
            "framework": "flutter",
            "storage": "shared_preferences"
        }}
    }}
    ```
    """

    def __init__(self, agent: Agent, swarm: Swarm | None = None) -> None:
        """Initialize the software planner."""
        self.agent = agent
        self.swarm = swarm or Swarm()

    async def create_plan(self, prompt: str, context: dict[str, Any]) -> Result[Plan[SoftwareTask]]:
        """Create a software development plan from the prompt."""
        engineer_types = context.get("available_engineer_types", [])
        project_context = (
            ProjectContext.model_validate(context["project"]) if "project" in context else None
        )

        context_section = (
            project_context.model_dump_json(indent=2)
            if project_context
            else "No existing project context provided."
        )

        result = await self.swarm.execute(
            agent=self.agent,
            prompt=self.PLAN_TEMPLATE.format(
                prompt=prompt,
                engineer_types=", ".join(engineer_types),
                context_section=context_section,
            ),
            context_variables=context,
        )

        try:
            if not result.content:
                return Result(error=ValueError("No plan result"))

            plan_code_block = extract_code_block(result.content)
            plan_json = plan_code_block.content
            plan = Plan[SoftwareTask].model_validate_json(plan_json)

            return Result(value=plan)

        except Exception as e:
            return Result(error=e)


def create_planner(swarm: Swarm | None = None) -> SoftwarePlanner:
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
    return SoftwarePlanner(agent=agent, swarm=swarm)


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
