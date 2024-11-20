# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import random

from liteswarm.repl import start_repl
from liteswarm.types import Agent


async def run() -> None:
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return (a + b) + random.randint(0, 100)

    def calculate_product(a: int, b: int) -> int:
        """Calculate the product of two numbers."""
        return (a * b) + random.randint(0, 100)

    def calculate_difference(a: int, b: int) -> int:
        """Calculate the difference of two numbers."""
        return (a - b) + random.randint(0, 100)

    math_agent = Agent.create(
        id="math_agent",
        model="claude-3-5-haiku-20241022",
        instructions="""You are a math calculation agent. You must ALWAYS use the provided calculation tools and NEVER perform calculations yourself.

        Critical rules:
            1. NEVER perform calculations yourself - always use the tools
            2. ALWAYS use the exact number returned by tools - never modify or round them
            3. For each step, first explain what you'll calculate, then use the tool, then state the result
            4. When using a tool's result in the next calculation, use the exact output number
            5. If you need to reference a previous calculation, use the exact tool output number

        Remember: The tool outputs are the ground truth - never question or modify them. Your role is to coordinate the calculations using the tools, not to perform math yourself.""",
        tools=[calculate_sum, calculate_product, calculate_difference],
        tool_choice="auto",
        parallel_tool_calls=False,
        temperature=0.0,
        seed=42,
        drop_params=True,
    )

    def switch_to_math_agent() -> Agent:
        """Switch to the math agent."""
        return math_agent

    sales_agent = Agent.create(
        id="sales_agent",
        model="gpt-4o-mini",
        instructions="""You're a sales agent. You need to sell a product to a customer.
        If you're given a list of products, you need to sell one of them.
        If you're given a customer, you need to sell to them.
        Always explain what you're doing and what tools you're using.""",
        tools=[switch_to_math_agent],
        tool_choice="auto",
        parallel_tool_calls=True,
        temperature=0.0,
    )

    await start_repl(sales_agent)


if __name__ == "__main__":
    asyncio.run(run())
