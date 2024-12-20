# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
from collections.abc import Callable
from typing import Literal, TypeAlias, get_args

from liteswarm.types import LLM, Agent, ContextVariables

from ..strategy import Strategy
from ..types import CodingTask, InnerMonologue, Plan

ANTHROPIC_MODELS: TypeAlias = Literal["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

PROMPT_TEMPLATE = """
You are a planning assistant AI designed to help users break down their queries into manageable tasks and create execution plans. Your goal is to analyze the user's query, identify the necessary steps to complete it, and generate a structured plan using the provided response format.

To complete this task, follow these steps:

1. Carefully read and analyze the user's query.
2. Break down the query into a set of distinct, manageable tasks.
3. Generate an execution plan that outlines how to complete these tasks.
4. Ensure that your response strictly follows the provided response format.

In the <response_example> tag below is an example of how you should respond.
DO NOT include the <response_example> tag or any other tags or text outside the JSON in your response.

<response_example>
{RESPONSE_EXAMPLE}
</response_example>
""".strip()


def is_valid_model(model: str) -> bool:
    """Check if the model is a valid Anthropic model.

    Args:
        model: Model identifier to validate.

    Returns:
        True if the model is a valid Anthropic model, False otherwise.
    """
    models = get_args(ANTHROPIC_MODELS)
    return model in models


def create_response_parser() -> Callable[[str, ContextVariables], InnerMonologue]:
    """Create a response parser for Anthropic models.

    Returns:
        A callable that parses Anthropic responses into InnerMonologue objects.

    Notes:
        The parser expects responses to be JSON objects with a single top-level key
        containing the InnerMonologue data.
    """

    def response_parser(response: str, _: ContextVariables) -> InnerMonologue:
        """Parse the response from Anthropic models."""
        response_json = json.loads(response)
        if not isinstance(response_json, dict):
            raise TypeError("Response is not a valid JSON object")

        return InnerMonologue.model_validate(response_json["values"])

    return response_parser


def create_strategy(model: str) -> Strategy[InnerMonologue]:
    """Create a structured output strategy for Anthropic models.

    Args:
        model: Anthropic model identifier to use.

    Returns:
        Strategy configured for the specified Anthropic model.

    Raises:
        ValueError: If the specified model is not supported.

    Notes:
        The strategy uses a template that instructs the model to format responses
        as JSON objects matching the InnerMonologue schema.
    """
    if not is_valid_model(model):
        raise ValueError(f"Invalid model: {model}. Supported models: {get_args(ANTHROPIC_MODELS)}")

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
            RESPONSE_EXAMPLE=response_example.model_dump_json(),
        ),
        llm=LLM(
            model=model,
            response_format=InnerMonologue,
        ),
    )

    response_parser = create_response_parser()

    return Strategy(agent=agent, response_parser=response_parser)
