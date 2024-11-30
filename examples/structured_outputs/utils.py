# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.types import Agent, ContextVariables
from liteswarm.utils.typing import is_callable

from .stream import SwarmStreamHandler

T = TypeVar("T", bound=BaseModel)

ResponseFormat: TypeAlias = type[T] | Callable[[str, ContextVariables], T]


def parse_response(
    response: str,
    response_format: ResponseFormat[T],
    context: ContextVariables | None = None,
) -> T:
    if is_callable(response_format):
        return response_format(response, context or ContextVariables())

    return response_format.model_validate_json(response)


async def generate_structured_response_typed(
    user_prompt: str,
    agent: Agent,
    response_format: ResponseFormat[T],
) -> T:
    swarm = Swarm(stream_handler=SwarmStreamHandler())
    result = await swarm.execute(agent=agent, prompt=user_prompt)
    if not result.content:
        raise ValueError("No response content")

    return parse_response(result.content, response_format)
