# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, Message
from liteswarm.utils.misc import prompt


async def main() -> None:
    agent = Agent(
        id="assistant",
        instructions="You are a helpful assistant.",
        llm=LLM(model="gpt-4o"),
    )

    swarm = Swarm()
    messages: list[Message] = []

    while True:
        # Get user input and run agent execution
        user_message = await prompt("> ")
        messages.append(Message(role="user", content=user_message))
        result = await swarm.execute(agent, messages=messages)

        # Add new messages to history
        messages.extend(result.new_messages)

        # Print the last agent response
        print(result.agent_response.content)


if __name__ == "__main__":
    asyncio.run(main())
