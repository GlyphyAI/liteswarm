# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json

from liteswarm.chat import LiteChat
from liteswarm.types import LLM, Agent, ContextVariables, ToolResult
from liteswarm.utils.logging import enable_logging
from liteswarm.utils.messages import dump_messages

enable_logging(default_level="DEBUG")


WEATHER_AGENT_INSTRUCTIONS = """You are a weather assistant.

The context will always include the user's location, so you can directly provide weather information without asking for location.

You can use the following tools:
- fetch_weather(location: str) -> str: Fetch the weather for a given location.
""".strip()

MAIN_AGENT_INSTRUCTIONS = """You are a helpful assistant.

When the user asks about weather, immediately switch to the weather agent without asking any questions.

You can use the following tools:
- switch_to_weather(location: str) -> str: Switch to the weather agent.
""".strip()


async def main() -> None:
    def fetch_weather(context_variables: ContextVariables) -> str:
        print(context_variables)
        return f"The weather in {context_variables.location} is sunny."

    weather_agent = Agent(
        id="weather_agent",
        instructions=WEATHER_AGENT_INSTRUCTIONS,
        llm=LLM(
            model="gpt-4o",
            tools=[fetch_weather],
            parallel_tool_calls=False,
        ),
    )

    def switch_to_weather_agent(context_variables: ContextVariables) -> ToolResult:
        return ToolResult.switch_agent(
            agent=weather_agent,
            content="Switched to weather agent",
            context_variables=context_variables,
        )

    agent = Agent(
        id="my_agent",
        instructions=MAIN_AGENT_INSTRUCTIONS,
        llm=LLM(
            model="gpt-4o",
            tools=[switch_to_weather_agent],
            parallel_tool_calls=False,
        ),
    )

    chat = LiteChat()
    session = await chat.create_session(user_id="john_doe")

    messages_queue: list[tuple[str, Agent, ContextVariables]] = [
        (
            "What is the weather today?",
            agent,
            ContextVariables(location="New York", user_id="john_doe"),
        ),
        (
            "OK, please update the weather forecast",
            agent,
            ContextVariables(location="San Francisco"),
        ),
    ]

    for message, agent, context_variables in messages_queue:
        async for event in session.send_message(
            message,
            agent=agent,
            context_variables=context_variables,
        ):
            if event.type == "agent_response_chunk":
                completion = event.response_chunk.completion
                if completion.delta.content:
                    print(completion.delta.content, end="", flush=True)
                if completion.finish_reason == "stop":
                    print()

    chat_messages = await session.get_messages()
    messages = dump_messages(chat_messages, exclude_none=True)
    print(json.dumps(messages, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
