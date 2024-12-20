# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import os
import random
from time import sleep
from typing import Any

from liteswarm.core import ConsoleEventHandler, Swarm
from liteswarm.types import LLM, Agent
from liteswarm.utils.logging import enable_logging

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"

RESEARCH_AGENT_INSTRUCTIONS = """
You are a city research analyst. Your task is to gather and analyze data about cities using the available tools.

Important guidelines:
1. When researching a city, ALWAYS query multiple data sources in parallel
2. Use all available tools to get a comprehensive view
3. After gathering data, provide a concise summary highlighting key insights
4. Compare and contrast different aspects of the data
5. Make recommendations based on the collected data

Remember to use parallel tool calls efficiently to gather data quickly.
""".strip()


async def run() -> None:
    def query_weather_api(location: str) -> dict[str, Any]:
        """Query weather data for a location."""
        # Simulate API call with random delay
        sleep(random.uniform(1, 3))
        return {
            "location": location,
            "temperature": random.randint(0, 30),
            "conditions": random.choice(["sunny", "rainy", "cloudy"]),
        }

    def query_population_data(city: str) -> dict[str, Any]:
        """Query population statistics for a city."""
        # Simulate API call with random delay
        sleep(random.uniform(1, 3))
        return {
            "city": city,
            "population": random.randint(100000, 5000000),
            "growth_rate": round(random.uniform(-2, 5), 2),
        }

    def query_tourist_attractions(city: str) -> dict[str, Any]:
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

    def query_economic_data(city: str) -> dict[str, Any]:
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

    research_agent = Agent(
        id="research_agent",
        instructions=RESEARCH_AGENT_INSTRUCTIONS,
        llm=LLM(
            model="claude-3-5-haiku-20241022",
            tools=[
                query_weather_api,
                query_population_data,
                query_tourist_attractions,
                query_economic_data,
            ],
            tool_choice="auto",
            parallel_tool_calls=True,
            temperature=0.0,
        ),
    )

    client = Swarm(event_handler=ConsoleEventHandler())
    result = await client.execute(
        agent=research_agent,
        prompt="Please analyze the city of Seattle and provide insights about its weather, population, tourism, and economy.",
    )

    print("\nFinal Analysis Complete!")
    print(f"\n\nResult: {result.content}\n\n")


if __name__ == "__main__":
    enable_logging()
    asyncio.run(run())
