# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
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
4. Ensure that your response strictly follows the provided JSON schema.

You must provide the response in the JSON format specified in the <response_format> tag below.
DO NOT include the <response_format> tag or any other tags or text outside the JSON in your response.

<response_format>
{RESPONSE_FORMAT}
</response_format>

In the <response_example> tag below is an example of how you should respond.

<response_example>
{RESPONSE_EXAMPLE}
</response_example>
""".strip()


def create_response_parser() -> Callable[[str, ContextVariables], InnerMonologue]:
    """Create a response parser for JSON object responses.

    Returns:
        A callable that parses JSON responses into InnerMonologue objects.

    Notes:
        The parser expects responses to be valid JSON objects that match the
        InnerMonologue schema exactly.
    """

    def response_parser(response: str, _: ContextVariables) -> InnerMonologue:
        return InnerMonologue.model_validate_json(response)

    return response_parser


def create_strategy(model: str) -> Strategy[InnerMonologue]:
    """Create a structured output strategy using JSON object parsing.

    Args:
        model: LLM model identifier to use.

    Returns:
        Strategy configured for JSON object parsing.

    Notes:
        The strategy provides the full JSON schema to the model and expects
        responses to match it exactly. This approach works across different
        LLM providers but may be less reliable than provider-specific methods.
    """
    response_example = InnerMonologue(
        thoughts="First, I'll analyze the user's query...",
        response=Plan(
            tasks=[
                CodingTask(
                    task_type="coding",
                    title="Write a hello world program in Python",
                    filepath="main.py",
                    code="print('Hello, world!')",
                )
            ],
        ),
    )

    agent = Agent(
        id="planner",
        instructions=PROMPT_TEMPLATE.format(
            RESPONSE_FORMAT=json.dumps(InnerMonologue.model_json_schema()),
            RESPONSE_EXAMPLE=response_example.model_dump_json(),
        ),
        llm=LLM(model=model),
    )

    response_parser = create_response_parser()

    return Strategy(agent=agent, response_parser=response_parser)
