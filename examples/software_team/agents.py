# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.core import Swarm
from liteswarm.experimental import AgentPlanner
from liteswarm.types import JSON, Agent, ContextVariables, Plan, Task, TaskDefinition, TeamMember
from liteswarm.utils import dedent_prompt

from .utils import dump_json

AGENT_PLANNER_PROMPT = """
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
""".strip()


def build_planner_prompt(prompt: str, context: ContextVariables) -> str:
    """Build prompt for the plan agent."""
    project_context: JSON = context.get("project", {})
    output_format: JSON = context.get_reserved("output_format", {})
    output_example: Plan = Plan(
        tasks=[
            Task(
                id="<id_0>",
                title="<title_0>",
                task_type="<task_type_0>",
            ),
        ]
    )

    return dedent_prompt(
        AGENT_PLANNER_PROMPT.format(
            prompt=prompt,
            project_context=dump_json(project_context),
            output_format=dump_json(output_format),
            output_example=output_example.model_dump_json(),
        )
    )


AGENT_PLANNER_INSTRUCTIONS = """
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

You MUST follow the output format specified in each task.
If output format is a JSON schema, you MUST return a valid JSON object.
""".strip()


def create_agent_planner(swarm: Swarm, task_definitions: list[TaskDefinition]) -> AgentPlanner:
    """Create a software planning agent."""
    agent = Agent.create(
        id="planner",
        model="gpt-4o",
        response_format={"type": "json_object"},
        instructions=AGENT_PLANNER_INSTRUCTIONS,
    )

    return AgentPlanner(
        swarm=swarm,
        agent=agent,
        template=build_planner_prompt,
        task_definitions=task_definitions,
    )


AGENT_FLUTTER_ENGINEER_INSTRUCTIONS = """
You are a Flutter software engineer specializing in mobile app development.

When implementing features:
1. Write clean, maintainable Flutter code
2. Follow Flutter best practices and patterns
3. Consider performance and user experience
4. Always provide complete file contents
5. Follow the output format specified in each task

You MUST follow the output format specified in each task.
If output format is a JSON schema, you MUST return a valid JSON object.
""".strip()


def create_flutter_engineer() -> Agent:
    """Create a Flutter engineer agent."""
    return Agent.create(
        id="flutter_engineer",
        model="gpt-4o",
        response_format={"type": "json_object"},
        instructions=AGENT_FLUTTER_ENGINEER_INSTRUCTIONS,
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
