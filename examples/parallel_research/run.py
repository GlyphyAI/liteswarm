import asyncio
import random
from time import sleep

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
        print(f"[{agent.id if agent else 'unknown'}] Error: {str(error)}")

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None:
        print(f"\n[{next_agent.id}] Switched to {next_agent.id}")

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None:
        print(f"\n[{agent.id if agent else 'unknown'}] Completed")

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None:
        print(f"\n[{agent.id if agent else 'unknown'}] Tool call: {tool_call.function.name}")

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None:
        print(
            f"\n[{agent.id if agent else 'unknown'}] Got result for: {tool_call_result.tool_call.function.name}"
        )


async def run() -> None:
    def query_weather_api(location: str) -> dict:
        """Query weather data for a location."""
        # Simulate API call with random delay
        sleep(random.uniform(1, 3))
        return {
            "location": location,
            "temperature": random.randint(0, 30),
            "conditions": random.choice(["sunny", "rainy", "cloudy"]),
        }

    def query_population_data(city: str) -> dict:
        """Query population statistics for a city."""
        # Simulate API call with random delay
        sleep(random.uniform(1, 3))
        return {
            "city": city,
            "population": random.randint(100000, 5000000),
            "growth_rate": round(random.uniform(-2, 5), 2),
        }

    def query_tourist_attractions(city: str) -> dict:
        """Query tourist attractions for a city."""
        # Simulate API call with random delay
        sleep(random.uniform(1, 3))
        attractions = [
            "Historic Downtown",
            "Central Park",
            "Art Museum",
            "Science Center",
            "Botanical Gardens",
        ]
        return {
            "city": city,
            "attractions": random.sample(attractions, random.randint(2, 4)),
            "annual_visitors": random.randint(100000, 1000000),
        }

    def query_economic_data(city: str) -> dict:
        """Query economic data for a city."""
        # Simulate API call with random delay
        sleep(random.uniform(1, 3))
        return {
            "city": city,
            "gdp_per_capita": random.randint(20000, 80000),
            "unemployment_rate": round(random.uniform(2, 10), 1),
            "major_industries": random.sample(
                ["Tech", "Finance", "Manufacturing", "Tourism", "Healthcare"],
                random.randint(2, 3),
            ),
        }

    research_agent = Agent.create(
        id="research_agent",
        model="claude-3-5-haiku-20241022",
        instructions="""You are a city research analyst. Your task is to gather and analyze data about cities using the available tools.

        Important guidelines:
        1. When researching a city, ALWAYS query multiple data sources in parallel
        2. Use all available tools to get a comprehensive view
        3. After gathering data, provide a concise summary highlighting key insights
        4. Compare and contrast different aspects of the data
        5. Make recommendations based on the collected data

        Remember to use parallel tool calls efficiently to gather data quickly.""",
        tools=[
            query_weather_api,
            query_population_data,
            query_tourist_attractions,
            query_economic_data,
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        temperature=0.0,
    )

    console_handler = ConsoleStreamHandler()
    client = Swarm(stream_handler=console_handler)

    result = await client.execute(
        agent=research_agent,
        prompt="Please analyze the city of Seattle and provide insights about its weather, population, tourism, and economy.",
    )

    print("\nFinal Analysis Complete!")
    print(f"\n\nResult: {result.messages}\n\n")


if __name__ == "__main__":
    asyncio.run(run())
