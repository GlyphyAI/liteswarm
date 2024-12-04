# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.core import Swarm
from liteswarm.experimental import AgentPlanner, LiteAgentPlanner
from liteswarm.types import JSON, LLM, Agent, ContextVariables, TaskDefinition

from .types import FlutterTask, SoftwarePlan
from .utils import dump_json, find_tag

AGENT_PLANNER_SYSTEM_PROMPT = """
You are an advanced AI Software Planning Agent designed to assist software engineering teams in project planning and task management. Your primary function is to analyze user queries, decompose them into actionable tasks, and generate a structured plan that can be executed by development teams across various technologies and frameworks.

### Project Context and Technology Stack

<project_context>
{PROJECT_CONTEXT}
</project_context>

<tech_stack>
{TECH_STACK}
</tech_stack>

### Response Format

Your output must be a JSON object adhering to the following schema:

<response_format>
{RESPONSE_FORMAT}
</response_format>

### Instructions for Plan Generation

1. Understand the Request:
   - Thoroughly analyze the user's query in the context of the provided project details and technology stack.

2. Identify Key Components:
   - Extract all relevant components, technologies, and requirements from the project context and tech stack.

3. Decompose into Tasks:
   - Break down the request into clear, actionable tasks.
   - Each task should align with the attributes defined in the provided Plan JSON schema.

4. Define Task Attributes:
   - Type: Assign the appropriate type as defined in the Plan JSON schema.
   - ID: Generate a unique identifier for each task.
   - Title: Provide a concise title for the task.
   - Description: Offer a detailed description if necessary.
   - Dependencies: List any task IDs that must be completed prior to this task.

5. Establish Dependencies:
   - Clearly define dependencies to ensure logical execution order.

6. Optimize Execution Order:
   - Arrange tasks for maximum efficiency, considering dependencies and resource availability.

7. Align with Existing Structures:
   - Ensure new tasks integrate seamlessly with the current project structure to maintain consistency.

### Guidelines for Task Granularity and Feasibility

- Granularity: Ensure tasks are concrete and manageable, preferring medium-sized tasks. Avoid unnecessary splitting if tasks are already clear and actionable.
- Technical Feasibility: Verify that all proposed tasks are technically achievable within the given tech stack.

### Formatting Rules

1. JSON Output:
   - Must be a VALID JSON object without additional explanations or formatting.
   - Use proper indentation and line breaks for readability.

2. Project Analysis:
   - Wrap your detailed analysis in `<project_analysis>` tags.
   - Format the content as plain text without additional explanations.

### Example Structure

<project_analysis>
{PROJECT_ANALYSIS_EXAMPLE}
</project_analysis>

<json_response>
{JSON_RESPONSE_EXAMPLE}
</json_response>

Please proceed to analyze the user query and generate a comprehensive development plan based on the provided project context and technology stack.
""".strip()

AGENT_PLANNER_USER_PROMPT = """
Please proceed to create a detailed development plan for the following user request:

<user_request>
{USER_PROMPT}
</user_request>

Here is the additional context you can use to create the plan:

<additional_context>
{CONTEXT}
</additional_context>

Please ensure that your response adheres to the system prompt rules, using XML tags and conforming to the specified JSON schema.
""".strip()


def build_planner_system_prompt(context: ContextVariables) -> str:
    """Build system prompt for the plan agent."""
    project_context: JSON = context.get("project", {})
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
        PROJECT_CONTEXT=dump_json(project_context),
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
    json_response = find_tag(response, "json_response")
    if not json_response:
        raise ValueError("No JSON response found")

    return SoftwarePlan.model_validate_json(json_response)


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
