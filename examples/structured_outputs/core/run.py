# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, Message


class MathResult(BaseModel):
    result: int
    thoughts: str


async def run() -> None:
    agent = Agent(
        id="math_expert",
        instructions="You are a math expert.",
        llm=LLM(
            model="gpt-4o",
            response_format=MathResult,
        ),
    )

    swarm = Swarm()

    # Method 1: Use streaming API to get structured outputs
    stream = swarm.stream(
        agent,
        messages=[Message(role="user", content="What is 2 + 2 * 2?")],
        response_format=MathResult,
    )

    # Get streaming partial parsed responses
    async for event in stream:
        if event.type == "agent_response_chunk":
            if event.response_chunk.parsed:
                print(event.response_chunk.parsed)

    # Get final parsed response
    result = await stream.get_return_value()
    if result.agent_response.parsed:
        print(result.agent_response.parsed.result)

    # Method 2: Use convenience method to get final parsed response
    result = await swarm.execute(
        agent,
        messages=[Message(role="user", content="What is 2 + 2 * 2?")],
        response_format=MathResult,
    )

    if result.agent_response.parsed:
        print(result.agent_response.parsed.result)


if __name__ == "__main__":
    asyncio.run(run())
