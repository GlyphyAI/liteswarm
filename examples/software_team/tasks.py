# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.types import ContextVariables, TaskDefinition

from .types import DebugOutput, DebugTask, FileContent, FlutterOutput, FlutterTask
from .utils import dump_json, find_json_tag

FLUTTER_TASK_PROMPT = """
Implement the following Flutter feature:

<task_details>
Feature Type: {TASK.feature_type}
Title: {TASK.title}
Description: {TASK.description}
</task_details>

Project Context:

<project_context>
Tech Stack: {PROJECT_TECH_STACK}
Structure: {PROJECT_DIRECTORIES}
</project_context>

### Task Execution Guidelines

As a Flutter software engineer, ensure that you:

1. Understand the Feature:
   - Analyze the feature type, title, and description.
   - Assess how the feature integrates with the existing project framework and structure.

2. Plan the Implementation:
   - Identify necessary components, widgets, and services required for the feature.
   - Outline the steps to implement the feature effectively, ensuring scalability and maintainability.

3. Adhere to Coding Best Practices:
   - Write clean, readable, and well-documented Flutter code.
   - Follow Flutter's best practices and coding standards.
   - Ensure proper state management and efficient widget usage.
   - Optimize for performance to provide a smooth user experience.
   - Design intuitive and user-friendly interfaces with consistency across the app.

4. Provide Complete Implementations:
   - Include full code for each file involved in the feature implementation.
   - Avoid partial snippets; ensure every file is complete and functional.
   - Ensure seamless integration with the existing codebase.

### Response Format Requirements

Your response MUST include two sections:

1. Task Analysis (wrapped in <task_analysis> tags):
   A detailed analysis of how you plan to implement the feature.

2. JSON Response (wrapped in <json_response> tags):
   A valid JSON object that strictly follows the response format provided in <response_format> tag.

<response_format>
{RESPONSE_FORMAT}
</response_format>

IMPORTANT: The JSON Response must:
- Be a valid JSON object (not a string or any other type)
- Strictly follow the schema provided in <response_format> tag
- Be properly formatted with correct quotes, commas, and brackets
- Not contain any Python string formatting or escape characters
- Be wrapped in <json_response> tags

Example Response Structure:

<example_response>
<task_analysis>
{TASK_ANALYSIS_EXAMPLE}
</task_analysis>

<json_response>
{JSON_RESPONSE_EXAMPLE}
</json_response>
</example_response>

Now proceed to execute the task.
""".strip()

DEBUG_TASK_PROMPT = """
Debug the following issue:

<task_details>
Error Type: {TASK.error_type}
Description: {TASK.description}
</task_details>

Stack Trace:

<stack_trace>
{STACK_TRACE}
</stack_trace>

Project Context:

<project_context>
Tech Stack: {PROJECT_TECH_STACK}
Structure: {PROJECT_DIRECTORIES}
</project_context>

### Task Execution Guidelines

As a Flutter software engineer, ensure that you:

1. Understand the Issue:
   - Analyze the error type, description, and stack trace
   - Assess how the issue relates to the existing project framework and structure

2. Plan the Solution:
   - Identify the root cause of the error
   - Outline the steps needed to fix the issue effectively
   - Consider potential side effects and dependencies

3. Adhere to Coding Best Practices:
   - Write clean, readable, and well-documented Flutter code
   - Follow Flutter's best practices and coding standards
   - Ensure proper error handling and logging
   - Maintain consistency with the existing codebase
   - Add safeguards to prevent similar issues

4. Provide Complete Implementations:
   - Include full code for each file that needs modification
   - Avoid partial snippets; ensure every file is complete and functional
   - Ensure seamless integration with the existing codebase

### Response Format Requirements

Your response MUST include two sections:

1. Task Analysis (wrapped in <task_analysis> tags):
   A detailed analysis of the issue and your solution approach.

2. JSON Response (wrapped in <json_response> tags):
   A valid JSON object that strictly follows the response format provided in <response_format> tag.

<response_format>
{RESPONSE_FORMAT}
</response_format>

IMPORTANT: The JSON Response must:
- Be a valid JSON object (not a string or any other type)
- Strictly follow the schema provided in <response_format> tag
- Be properly formatted with correct quotes, commas, and brackets
- Not contain any Python string formatting or escape characters
- Be wrapped in <json_response> tags

Example Response Structure:

<example_response>
<task_analysis>
{TASK_ANALYSIS_EXAMPLE}
</task_analysis>

<json_response>
{JSON_RESPONSE_EXAMPLE}
</json_response>
</example_response>

Now proceed to execute the task.
""".strip()


def build_flutter_task_prompt(task: FlutterTask, context: ContextVariables) -> str:
    """Build the instructions for a Flutter task."""
    project_tech_stack = context.get("tech_stack", {})
    project_directories = context.get("directories", [])

    response_example = FlutterOutput(
        files=[
            FileContent(
                filepath="lib/features/<feature_name>/<file>.dart",
                content="// Complete Dart code for <file>",
            ),
        ],
    )

    return FLUTTER_TASK_PROMPT.format(
        TASK=task,
        PROJECT_TECH_STACK=project_tech_stack,
        PROJECT_DIRECTORIES=dump_json(project_directories),
        RESPONSE_FORMAT=dump_json(FlutterOutput.model_json_schema()),
        TASK_ANALYSIS_EXAMPLE="Provide your detailed analysis of the feature implementation here.",
        JSON_RESPONSE_EXAMPLE=response_example.model_dump_json(),
    )


def build_debug_task_prompt(task: DebugTask, context: ContextVariables) -> str:
    """Build the instructions for a Debug task."""
    project_tech_stack = context.get("tech_stack", {})
    project_directories = context.get("directories", [])

    stack_trace = task.stack_trace or "No stack trace provided"

    response_example = DebugOutput(
        root_cause="The error is caused by a null pointer exception in <file>.dart.",
        files=[
            FileContent(
                filepath="lib/<affected_file>.dart",
                content="// Complete Dart code for <affected_file>.dart with fixes",
            ),
        ],
    )

    return DEBUG_TASK_PROMPT.format(
        TASK=task,
        STACK_TRACE=stack_trace,
        PROJECT_TECH_STACK=project_tech_stack,
        PROJECT_DIRECTORIES=dump_json(project_directories),
        RESPONSE_FORMAT=dump_json(DebugOutput.model_json_schema()),
        TASK_ANALYSIS_EXAMPLE="Provide your detailed analysis of the debugging process here.",
        JSON_RESPONSE_EXAMPLE=response_example.model_dump_json(),
    )


def parse_flutter_task_response(content: str, context: ContextVariables) -> FlutterOutput:
    """Parse the output of a Flutter task."""
    json_response = find_json_tag(content, "json_response")
    if not json_response:
        raise ValueError("No JSON response found")

    return FlutterOutput.model_validate(json_response)


def parse_debug_task_response(content: str, context: ContextVariables) -> DebugOutput:
    """Parse the output of a Debug task."""
    json_response = find_json_tag(content, "json_response")
    if not json_response:
        raise ValueError("No JSON response found")

    return DebugOutput.model_validate(json_response)


def create_flutter_task_definition() -> TaskDefinition:
    return TaskDefinition(
        task_type=FlutterTask,
        instructions=build_flutter_task_prompt,
        response_format=parse_flutter_task_response,
    )


def create_debug_task_definition() -> TaskDefinition:
    return TaskDefinition(
        task_type=DebugTask,
        instructions=build_debug_task_prompt,
        response_format=parse_debug_task_response,
    )


def create_task_definitions() -> list[TaskDefinition]:
    return [
        create_flutter_task_definition(),
        create_debug_task_definition(),
    ]
