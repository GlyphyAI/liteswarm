import asyncio
import random

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, Delta, Message, ToolCallResult


class ConsoleStreamHandler:
    async def on_stream(
        self,
        chunk: Delta,
        agent: Agent | None,
    ) -> None:
        if chunk.content:
            print(f"{chunk.content}", end="", flush=True)

    async def on_error(
        self,
        error: Exception,
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Error: {str(error)}")

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None:
        print(f"[{next_agent.agent_id}] Switched to {next_agent.agent_id}")

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Completed")

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Tool call: {tool_call}")

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Tool call result: {tool_call_result}")


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
        agent_id="math_agent",
        model="claude-3-5-haiku-20241022",
        instructions="""You are a math calculation agent. You must ALWAYS use the provided calculation tools and NEVER perform calculations yourself.

        Critical rules:
            1. NEVER perform calculations yourself - always use the tools
            2. ALWAYS use the exact number returned by tools - never modify or round them
            3. For each step, first explain what you'll calculate, then use the tool, then state the result
            4. When using a tool's result in the next calculation, use the exact output number
            5. If you need to reference a previous calculation, use the exact tool output number

        Remember: The tool outputs are the ground truth - never question or modify them. Your role is to coordinate the calculations using the tools, not to perform math yourself.""",
        function_tools=[calculate_sum, calculate_product, calculate_difference],
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
        agent_id="sales_agent",
        model="gpt-4o-mini",
        instructions="""You're a sales agent. You need to sell a product to a customer.
        If you're given a list of products, you need to sell one of them.
        If you're given a customer, you need to sell to them.
        Always explain what you're doing and what tools you're using.""",
        function_tools=[switch_to_math_agent],
        tool_choice="auto",
        parallel_tool_calls=True,
        temperature=0.0,
    )

    console_handler = ConsoleStreamHandler()
    client = Swarm(stream_handler=console_handler)

    result = await client.execute(
        agent=sales_agent,
        prompt="Ok, can you calculate the sum of 100 and 200 and then multiply the result by 3 and then subtract 42 from the result?",
    )

    print(f"\n\nResult messages:\n\n{result.messages}")


if __name__ == "__main__":
    asyncio.run(run())
