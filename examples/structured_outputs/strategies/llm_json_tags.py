# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import re
from collections.abc import Callable

from liteswarm.types import LLM, Agent, ContextVariables

from ..strategy import Strategy
from ..types import CodingTask, InnerMonologue, Plan

PROMPT_TEMPLATE = """
You are a planning assistant AI designed to help users break down their queries into manageable tasks and create execution plans. Your goal is to analyze the user's query, identify the necessary steps to complete it, and generate a structured plan using the provided JSON schema.

To complete this task, follow these steps:

1. Carefully read and analyze the user's query.
2. Break down the query into a set of distinct, manageable tasks.
3. Generate an execution plan that outlines how to complete these tasks.
4. Ensure that your response strictly follows the provided response format in the <response_format> tag below.

<response_format>
{RESPONSE_FORMAT}
</response_format>

Structure your response in an inner monologue using the <thoughts> and <response> tags.
Note that the <response> tag must contain a valid JSON object that adheres to the response format in the <response_format> tag.

Here is an example of how you should respond:

<thoughts>
{THOUGHTS_EXAMPLE}
</thoughts>

<response>
{RESPONSE_EXAMPLE}
</response>
""".strip()


def find_tag(text: str, tag: str) -> str | None:
    """Find and extract content from a tagged section.

    Args:
        text: Text containing tagged sections.
        tag: Name of the tag to find.

    Returns:
        Content between the specified tags, or None if not found.
    """
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else None


def create_response_parser() -> Callable[[str, ContextVariables], InnerMonologue]:
    """Create a response parser for tagged JSON responses.

    Returns:
        A callable that parses tagged responses into InnerMonologue objects.

    Notes:
        The parser expects responses to contain <thoughts> and <response> tags,
        with the response section containing valid JSON matching the Plan schema.
    """

    def response_parser(response: str, _: ContextVariables) -> InnerMonologue:
        thoughts = find_tag(response, "thoughts")
        response_json = find_tag(response, "response")
        if not thoughts or not response_json:
            raise ValueError("Response is missing the 'thoughts' or 'response' tag")

        return InnerMonologue(
            thoughts=thoughts,
            response=Plan.model_validate_json(response_json),
        )

    return response_parser


def create_strategy(model: str) -> Strategy[InnerMonologue]:
    """Create a structured output strategy using JSON tags parsing.

    Args:
        model: LLM model identifier to use.

    Returns:
        Strategy configured for JSON tags parsing.

    Notes:
        The strategy instructs the model to format responses using XML-style tags
        with JSON content. This approach provides clear structure while being
        more flexible than pure JSON responses.
    """
    thoughts_example = "First, I'll analyze the user's query..."
    response_example = Plan(
        tasks=[
            CodingTask(
                task_type="coding",
                title="Write a hello world program in Python",
                filepath="main.py",
                code="print('Hello, world!')",
            )
        ],
    )

    agent = Agent(
        id="planner",
        instructions=PROMPT_TEMPLATE.format(
            RESPONSE_FORMAT=json.dumps(Plan.model_json_schema()),
            THOUGHTS_EXAMPLE=thoughts_example,
            RESPONSE_EXAMPLE=response_example.model_dump_json(),
        ),
        llm=LLM(model=model),
    )

    response_parser = create_response_parser()

    return Strategy(agent=agent, response_parser=response_parser)
