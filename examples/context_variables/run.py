# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, AgentExecutionResult, ContextVariables, Message, ToolResult
from liteswarm.utils import dump_messages, enable_logging
from liteswarm.utils.misc import prompt

enable_logging(default_level="DEBUG")


def get_messages_json(result: AgentExecutionResult) -> str:
    messages = dump_messages(result.all_messages, exclude_none=True)
    return json.dumps(messages, indent=2, ensure_ascii=False)


async def instructions_example() -> None:
    def instructions(context_variables: ContextVariables) -> str:
        user_name: str = context_variables.get("user_name", "John")
        return f"Help the user, {user_name}, do whatever they want."

    agent = Agent(
        id="agent",
        instructions=instructions,
        llm=LLM(
            model="gpt-4o-mini",
            litellm_kwargs={"drop_params": True},
        ),
    )

    swarm = Swarm(include_usage=True)
    result = await swarm.execute(
        agent=agent,
        messages=[Message(role="user", content="Hello!")],
        context_variables=ContextVariables(user_name="John"),
    )

    print(get_messages_json(result))


async def agent_switching_example() -> None:
    def greet(language: str, context_variables: ContextVariables) -> ToolResult:
        """Greet the user in the specified language encoded as a language code like 'en' or 'es'."""
        user_name: str = context_variables.get("user_name", "John")
        greeting = "Hola" if language.lower() == "es" else "Hello"

        return ToolResult.update_context(
            content=f"{greeting}, {user_name}!",
            context_variables=ContextVariables(language=language),
        )

    def speak_instructions(context_variables: ContextVariables) -> str:
        language: str = context_variables.get("language", "en")
        return f"Speak with the user in {language} language and ask them how they are doing."

    speak_agent = Agent(
        id="speak_agent",
        instructions=speak_instructions,
        llm=LLM(model="gpt-4o-mini"),
    )

    def switch_to_speak_agent() -> ToolResult:
        """Switch to the speak agent."""
        return ToolResult.switch_agent(
            agent=speak_agent,
            content="Switched to speak agent",
        )

    welcome_agent = Agent(
        id="welcome_agent",
        instructions=(
            "You are a welcome agent that greets the user. "
            "Switch to the speak agent after greeting the user."
        ),
        llm=LLM(
            model="gpt-4o-mini",
            tools=[greet, switch_to_speak_agent],
            parallel_tool_calls=False,
        ),
    )

    swarm = Swarm(include_usage=True)
    result = await swarm.execute(
        agent=welcome_agent,
        messages=[Message(role="user", content="Hola!")],
        context_variables=ContextVariables(user_name="John"),
    )

    print(get_messages_json(result))


async def error_handling_example() -> None:
    def fetch_weather(city: str) -> str:
        """Fetch the weather for a city."""
        if city == "Orgrimmar":
            raise ValueError("Please specify a valid city.")

        return f"The weather in {city} is nice."

    agent = Agent(
        id="weather_agent",
        instructions="Fetch the weather for a city. Always use the specified city.",
        llm=LLM(
            model="gpt-4o-mini",
            tools=[fetch_weather],
            temperature=0.0,
        ),
    )

    swarm = Swarm(include_usage=True)
    result = await swarm.execute(
        agent=agent,
        messages=[Message(role="user", content="What is the weather in Orgrimmar?")],
    )

    print(get_messages_json(result))


class User(BaseModel):
    id: str
    name: str
    age: int


async def pydantic_example() -> None:
    def fetch_user_info(user_id: str) -> User:
        """Fetch user info for a user."""
        return User(id=user_id, name="John", age=27)

    agent = Agent(
        id="user_info_agent",
        instructions="Fetch user info for a user.",
        llm=LLM(
            model="gpt-4o-mini",
            tools=[fetch_user_info],
            temperature=0.0,
        ),
    )

    swarm = Swarm(include_usage=True)
    result = await swarm.execute(
        agent=agent,
        messages=[Message(role="user", content="What is the name of a user with id 1?")],
    )

    print(get_messages_json(result))


async def run_selected_example() -> None:
    """Prompt the user to select an example to run."""
    print("Select an example to run:")
    print("1. Instructions")
    print("2. Agent switching")
    print("3. Error handling")
    print("4. Pydantic output")
    choice = await prompt("Enter the number of the example to run: ")

    match choice:
        case "1":
            await instructions_example()
        case "2":
            await agent_switching_example()
        case "3":
            await error_handling_example()
        case "4":
            await pydantic_example()
        case _:
            print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    asyncio.run(run_selected_example())
