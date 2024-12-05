# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.core import Swarm
from liteswarm.experimental import AgentPlanner, LiteAgentPlanner
from liteswarm.types import JSON, LLM, Agent, ContextVariables, TaskDefinition

from .types import FlutterTask, SoftwarePlan
from .utils import dump_json, find_json_tag

AGENT_PLANNER_SYSTEM_PROMPT = """
You are an advanced AI Software Planning Agent designed to assist software engineering teams in project planning and task management. Your primary function is to analyze user queries, decompose them into actionable tasks, and generate a structured plan that can be executed by development teams across various technologies and frameworks.

### Project Context

<project_directories>
{PROJECT_DIRECTORIES}
</project_directories>

<project_files>
{PROJECT_FILES}
</project_files>

<tech_stack>
{TECH_STACK}
</tech_stack>

### Response Requirements

Your response MUST include two sections:

1. Project Analysis (wrapped in <project_analysis> tags):
   A concise analysis of how you plan to implement the user's request, focusing on key architectural decisions and implementation strategy.

2. JSON Response (wrapped in <json_response> tags):
   A valid JSON object that strictly follows the schema provided in <response_format> tag.

<response_format>
{RESPONSE_FORMAT}
</response_format>

IMPORTANT: The JSON Response must:
- Be a valid JSON object (not a string or any other type)
- Strictly follow the schema provided in <response_format> tag
- Be properly formatted with correct quotes, commas, and brackets
- Not contain any Python string formatting or escape characters
- Be wrapped in <json_response> tags

### Task Planning Guidelines

1. Task Count and Scope:
   - Create between 1 to 8 core tasks maximum
   - Focus on essential, high-impact tasks that deliver complete functionality
   - Consolidate related features into single tasks where appropriate
   - It's perfectly fine to have fewer tasks (1-4) for simpler projects

2. Task Structure:
   - Each task must include ONLY the fields defined in the schema
   - Do not add extra fields or custom attributes
   - Do not fill metadata fields as they are reserved for system use

3. Task Organization:
   - Ensure logical task ordering through dependencies
   - Optimize for parallel execution where possible
   - Group related functionality into single tasks to minimize redundant work

4. Task Detail Level:
   - Keep task descriptions comprehensive but focused
   - Include key implementation details and requirements
   - Avoid splitting obviously related features into separate tasks

Example Response Structure:

<example_response>
<project_analysis>
{PROJECT_ANALYSIS_EXAMPLE}
</project_analysis>

<json_response>
{JSON_RESPONSE_EXAMPLE}
</json_response>
</example_response>
""".strip()

AGENT_PLANNER_USER_PROMPT = """
Please create a development plan for this request:

<user_request>
{USER_PROMPT}
</user_request>

Here's the relevant context to consider:

<context>
{CONTEXT}
</context>

Now proceed to create a development plan for the user's request.
""".strip()


def build_planner_system_prompt(context: ContextVariables) -> str:
    """Build system prompt for the plan agent."""
    project_directories: JSON = context.get("directories", [])
    project_files: JSON = context.get("files", [])
    tech_stack: JSON = context.get("tech_stack", {})

    response_format = SoftwarePlan
    response_example = SoftwarePlan(
        tasks=[
            FlutterTask(
                type="flutter_feature",
                id="<task_id>",
                title="<task_title>",
                feature_type="<feature_type>",
            ),
        ]
    )

    return AGENT_PLANNER_SYSTEM_PROMPT.format(
        PROJECT_DIRECTORIES=dump_json(project_directories),
        PROJECT_FILES=dump_json(project_files),
        TECH_STACK=dump_json(tech_stack),
        RESPONSE_FORMAT=dump_json(response_format.model_json_schema()),
        PROJECT_ANALYSIS_EXAMPLE="Provide your detailed analysis of the project context and tech stack here.",
        JSON_RESPONSE_EXAMPLE=response_example.model_dump_json(),
    )


def build_planner_user_prompt(prompt: str, context: ContextVariables) -> str:
    """Build user prompt for the plan agent."""
    project_context: JSON = context.get("project", {})

    return AGENT_PLANNER_USER_PROMPT.format(
        USER_PROMPT=prompt,
        CONTEXT=dump_json(project_context),
    )


def parse_planner_response(response: str, context: ContextVariables) -> SoftwarePlan:
    """Parse the response from the planner agent."""
    json_response = find_json_tag(response, "json_response")
    if not json_response:
        raise ValueError("No JSON response found")

    return SoftwarePlan.model_validate(json_response)


def create_agent_planner(swarm: Swarm, task_definitions: list[TaskDefinition]) -> AgentPlanner:
    """Create a software planning agent."""
    agent = Agent(
        id="planner",
        instructions=build_planner_system_prompt,
        llm=LLM(model="claude-3-5-sonnet-20241022"),
    )

    return LiteAgentPlanner(
        swarm=swarm,
        agent=agent,
        prompt_template=build_planner_user_prompt,
        response_format=parse_planner_response,
        task_definitions=task_definitions,
    )
