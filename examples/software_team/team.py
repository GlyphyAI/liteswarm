from typing import Any, ClassVar

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, Result

from .types import SoftwarePlan
from .utils import extract_code_block


class SoftwarePlanner:
    """Agent responsible for creating software development plans."""

    PLAN_TEMPLATE: ClassVar[str] = """
    Create a detailed development plan for the following project:
    {prompt}

    Available engineer types: {engineer_types}
    Each task must be assigned to one of these engineer types.

    Context files:
    {context_files}

    The plan should:
    1. Break down the work into clear, actionable tasks
    2. Specify the type of engineer needed for each task
    3. List dependencies between tasks
    4. Define clear deliverables (files to be created/modified)
    5. Consider the tech stack and existing codebase

    Return the plan as a JSON object with:
    - tasks: list of tasks, each with:
        - id: unique string identifier
        - title: clear task title
        - description: detailed task description
        - engineer_type: type of engineer needed (must be one of: {engineer_types})
        - dependencies: list of task IDs this depends on
        - deliverables: list of files to be created/modified
    - tech_stack: dictionary of technology choices
    """

    def __init__(self, agent: Agent, swarm: Swarm | None = None) -> None:
        """Initialize the software planner."""
        self.agent = agent
        self.swarm = swarm or Swarm()

    async def create_plan(self, prompt: str, context: dict[str, Any]) -> Result[SoftwarePlan]:
        """Create a software development plan from the prompt."""
        engineer_types = context.get("available_engineer_types", [])
        context_files = context.get("files", [])

        result = await self.swarm.execute(
            agent=self.agent,
            prompt=self.PLAN_TEMPLATE.format(
                prompt=prompt,
                engineer_types=", ".join(engineer_types),
                context_files="\n".join(f"- {f}" for f in context_files),
            ),
            context_variables=context,
        )

        try:
            if not result.content:
                return Result(error=ValueError("No plan result"))

            plan_code_block = extract_code_block(result.content)
            plan_json = plan_code_block.content
            plan = SoftwarePlan.model_validate_json(plan_json)

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
        4. Provide complete file contents or git-style diffs

        Output format for new files:
        ```dart:path/to/file.dart
        // Complete file contents here
        ```

        Output format for file changes:
        ```diff:path/to/file.dart
        - // Old code to remove
        + // New code to add
        ```""",
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
