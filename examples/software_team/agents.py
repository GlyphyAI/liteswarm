from liteswarm.swarm import Swarm
from liteswarm.swarm_team import AgentPlanner, TaskDefinition, TeamMember
from liteswarm.types import Agent

from .templates import SoftwarePlanTemplate


def create_agent_planner(swarm: Swarm, task_definitions: list[TaskDefinition]) -> AgentPlanner:
    """Create a software planning agent."""
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

        Consider:
        - Team capabilities and specialties
        - Previous execution history
        - Project context and requirements

        When creating plans:
        - Break down complex tasks into smaller, manageable units
        - Assign tasks to engineers based on their specialties
        - Include clear success criteria for each task
        - Consider dependencies and optimal execution order""",
    )

    return AgentPlanner(
        swarm=swarm,
        agent=agent,
        template=SoftwarePlanTemplate(),
        task_definitions=task_definitions,
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

        Your response must use three root XML tags to separate thoughts, root cause, and implementation:

        <thoughts>
        Explain your implementation approach here. This section should include:
        - What changes you're making and why
        - Key implementation decisions
        - Any important considerations
        The content can be free-form text, formatted for readability.
        </thoughts>

        <root_cause>
        Explain the root cause of the error. This section should include:
        - What is the root cause of the error
        - Key implementation decisions
        - Any important considerations
        The content can be free-form text, formatted for readability.
        </root_cause>

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
        I'll analyze the error and suggest a solution.
        </thoughts>
        <files>
        []
        </files>""",
    )


def create_team_members() -> list[TeamMember]:
    """Create a list of team members."""
    return [
        TeamMember(
            agent=create_flutter_engineer(),
            task_types=["flutter"],
            metadata={"specialty": "mobile"},
        ),
        TeamMember(
            agent=create_debug_engineer(),
            task_types=["debug"],
            metadata={"specialty": "troubleshooting"},
        ),
    ]
