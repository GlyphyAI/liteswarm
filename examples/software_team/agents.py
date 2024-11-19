from liteswarm.swarm import Swarm
from liteswarm.swarm_team import AgentPlanner, TaskDefinition, TeamMember, dedent_prompt
from liteswarm.types import Agent

from .templates import SoftwarePlanTemplate


def create_agent_planner(swarm: Swarm, task_definitions: list[TaskDefinition]) -> AgentPlanner:
    """Create a software planning agent."""
    agent = Agent.create(
        id="planner",
        model="gpt-4o",
        instructions=dedent_prompt("""
        You are a technical project planner specializing in Flutter app development.

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
        - Consider dependencies and optimal execution order
        """),
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
        instructions=dedent_prompt("""
        You are a Flutter software engineer specializing in mobile app development.

        When implementing features:
        1. Write clean, maintainable Flutter code
        2. Follow Flutter best practices and patterns
        3. Consider performance and user experience
        4. Always provide complete file contents
        5. Follow the output format specified in each task
        """),
    )


def create_team_members() -> list[TeamMember]:
    flutter_engineer = create_flutter_engineer()

    return [
        TeamMember(
            id=flutter_engineer.id,
            agent=flutter_engineer,
            task_types=["flutter_feature", "flutter_debug"],
            metadata={"specialty": "mobile"},
        ),
    ]
