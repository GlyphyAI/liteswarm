# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from liteswarm.core import LiteStreamHandler, Swarm
from liteswarm.types import LLM, Agent, Delta
from liteswarm.utils.pydantic import (
    remove_default_values,
    replace_default_values,
    restore_default_values,
)

T = TypeVar("T", bound=BaseModel)

AgentInstructions: TypeAlias = Callable[[type[BaseModel], BaseModel | None], str]


class StreamHandler(LiteStreamHandler):
    async def on_stream(self, delta: Delta, agent: Agent) -> None:
        print(f"{delta.content}", end="", flush=True)


async def generate_structured_response(
    user_prompt: str,
    agent_instructions: AgentInstructions,
    response_format: type[T],
    response_example: T | None = None,
    llm: LLM | None = None,
) -> T:
    response_format_patched: type[BaseModel] = remove_default_values(response_format)
    response_example_patched: BaseModel | None = None
    if response_example:
        response_example_patched = replace_default_values(
            instance=response_example,
            target_model_type=response_format_patched,
        )

    llm = llm or LLM(model="gpt-4o")
    llm.response_format = response_format_patched

    agent = Agent(
        id="structured_output_agent",
        instructions=agent_instructions(response_format_patched, response_example_patched),
        llm=llm,
    )

    swarm = Swarm(stream_handler=StreamHandler())
    result = await swarm.execute(agent=agent, prompt=user_prompt)
    if not result.content:
        raise ValueError("No response content")

    validated_result = response_format_patched.model_validate_json(result.content)
    restored_result = restore_default_values(validated_result, response_format)

    return restored_result
