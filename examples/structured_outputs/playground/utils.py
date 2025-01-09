# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.types import Agent, ContextVariables, Message
from liteswarm.types.typing import is_callable

T = TypeVar("T", bound=BaseModel)

ResponseFormat: TypeAlias = type[T] | Callable[[str, ContextVariables], T]


def parse_response(
    response: str,
    response_format: ResponseFormat[T],
    context: ContextVariables | None = None,
) -> T:
    """Parse a raw response string into a structured format.

    Args:
        response: Raw response string from the LLM.
        response_format: Type or callable for parsing/validating the response.
        context: Optional context variables for response parsing.

    Returns:
        Parsed and validated response object.

    Raises:
        ValidationError: If response doesn't match expected format.
    """
    if is_callable(response_format):
        return response_format(response, context or ContextVariables())

    return response_format.model_validate_json(response)


async def generate_structured_response_typed(
    user_prompt: str,
    agent: Agent,
    response_format: ResponseFormat[T],
) -> T:
    """Generate and parse a structured response from an LLM.

    Args:
        user_prompt: The input prompt for the LLM.
        agent: The agent configuration for the LLM.
        response_format: Type or callable for parsing/validating the response.

    Returns:
        Parsed and validated response object.

    Raises:
        ValueError: If response content is empty.
        ValidationError: If response doesn't match expected format.
    """
    swarm = Swarm()
    stream = swarm.stream(
        agent=agent,
        messages=[Message(role="user", content=user_prompt)],
    )

    async for event in stream:
        if event.type == "agent_response_chunk":
            completion = event.response_chunk.completion
            if content := completion.delta.content:
                print(f"{content}", end="", flush=True)

        if event.type == "agent_complete":
            print()

    result = await stream.get_return_value()
    response = result.agent_response
    if not response.content:
        raise ValueError("No response content")

    return parse_response(response.content, response_format)
