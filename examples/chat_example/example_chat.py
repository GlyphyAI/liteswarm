# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json

from liteswarm.chat import LiteChat
from liteswarm.types import LLM, Agent, ContextVariables, ToolResult
from liteswarm.utils import dump_messages

WEATHER_AGENT_INSTRUCTIONS = """You are a weather assistant that provides weather information.

You have access to real-time weather data through the fetch_weather tool.
Use the location from the user's query to get accurate weather information.
Keep responses concise and focused on weather conditions.
""".strip()

MAIN_AGENT_INSTRUCTIONS = """You are a helpful assistant that can handle various tasks.

For weather-related queries:
- Switch to the weather agent immediately when user asks about weather
- Don't ask for clarification or location - just switch
- Use tools to make the switch

For other queries:
- Handle them directly with your general knowledge
- Provide clear and concise responses
""".strip()


async def run() -> None:
    def fetch_weather(location: str, context_variables: ContextVariables) -> str:
        if context_variables.api_key == "sk-proj-1234567890":
            return f"The weather in {location} is sunny."
        else:
            raise ValueError("Invalid API key")

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

    messages_queue: list[tuple[str, Agent, ContextVariables]] = [
        (
            "What is the weather today in New York?",
            agent,
            ContextVariables(api_key="sk-proj-1234567890"),
        ),
        (
            "OK, and what about San Francisco?",
            agent,
            ContextVariables(api_key="sk-proj-1234567890"),
        ),
    ]

    chat = LiteChat()
    for message, agent, context_variables in messages_queue:
        async for event in chat.send_message(
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

    chat_messages = await chat.get_messages()
    messages = dump_messages(chat_messages, exclude_none=True)
    print(json.dumps(messages, indent=2))
