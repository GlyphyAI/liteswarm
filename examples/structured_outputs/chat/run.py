# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio

from pydantic import BaseModel

from liteswarm.chat.chat import LiteChat
from liteswarm.types import LLM, Agent


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

    chat = LiteChat()
    stream = chat.send_message(
        "What is 2 + 2 * 2?",
        agent=agent,
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


if __name__ == "__main__":
    asyncio.run(run())
