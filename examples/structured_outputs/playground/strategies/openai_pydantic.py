# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Literal, TypeAlias, TypeVar, get_args

from pydantic import BaseModel

from liteswarm.types import LLM, Agent, ContextVariables
from liteswarm.utils.pydantic import (
    remove_default_values,
    replace_default_values,
    restore_default_values,
)

from ..strategy import Strategy
from ..types import CodingTask, InnerMonologue, Plan

OPENAI_MODELS: TypeAlias = Literal["gpt-4o", "gpt-4o-mini"]

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

T = TypeVar("T", bound=BaseModel)


def is_valid_model(model: str) -> bool:
    """Check if the model is a valid OpenAI model."""
    models = get_args(OPENAI_MODELS)
    return model in models


def create_response_parser(
    response_format: type[T],
    patched_response_format: type[BaseModel],
) -> Callable[[str, ContextVariables], T]:
    """Create a response parser for OpenAI models.

    Args:
        response_format: Original response format type.
        patched_response_format: Modified response format with defaults removed.

    Returns:
        A callable that parses OpenAI responses into the specified format.

    Notes:
        The parser handles OpenAI's response format and restores default values
        that were removed to improve response accuracy.
    """

    def response_parser(response: str, _: ContextVariables) -> T:
        patched_response_content = patched_response_format.model_validate_json(response)
        return restore_default_values(patched_response_content, response_format)

    return response_parser


def create_strategy(model: str) -> Strategy[InnerMonologue]:
    """Create a structured output strategy for OpenAI models.

    Args:
        model: OpenAI model identifier to use.

    Returns:
        Strategy configured for the specified OpenAI model.

    Raises:
        ValueError: If the specified model is not supported.

    Notes:
        The strategy modifies the schema to remove default values before sending
        to OpenAI, then restores them during parsing.
    """
    if not is_valid_model(model):
        raise ValueError(f"Invalid model: {model}. Supported models: {get_args(OPENAI_MODELS)}")

    response_format = InnerMonologue
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

    patched_response_format = remove_default_values(response_format)
    patched_response_example = replace_default_values(response_example)

    agent = Agent(
        id="planner",
        instructions=PROMPT_TEMPLATE.format(
            RESPONSE_EXAMPLE=patched_response_example.model_dump_json(),
        ),
        llm=LLM(
            model=model,
            response_format=patched_response_format,
        ),
    )

    parser = create_response_parser(response_format, patched_response_format)

    return Strategy(agent=agent, response_parser=parser)
