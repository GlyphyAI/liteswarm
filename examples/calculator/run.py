# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, Message


# Define tools
def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int) -> int:
    return a // b


def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


async def main() -> None:
    # Create agent
    agent = Agent(
        id="math",
        instructions="You are a math expert. Always use tools for calculations.",
        llm=LLM(
            model="gpt-4o",
            tools=[multiply, divide, add, subtract],
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.0,
        ),
    )

    # Run agent execution
    swarm = Swarm()
    result = await swarm.execute(
        agent,
        messages=[Message(role="user", content="What is 2 + 2 * 2 - 2 / 2?")],
    )

    # Print agent response
    print(f"Agent response:\n{result.agent_response.content}\n")

    # Print all messages
    messages = [msg.model_dump(exclude_none=True) for msg in result.all_messages]
    print(f"Messages:\n{json.dumps(messages, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
