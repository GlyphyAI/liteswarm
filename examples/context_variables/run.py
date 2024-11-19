import asyncio
import os

from liteswarm.logging import enable_logging
from liteswarm.swarm import Swarm
from liteswarm.types import Agent, ContextVariables, Result

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


async def instructions_example() -> None:
    def instructions(context_variables: ContextVariables) -> str:
        user_name: str = context_variables.get("user_name", "John")
        return f"Help the user, {user_name}, do whatever they want."

    agent = Agent.create(
        id="agent",
        model="gpt-4o-mini",
        instructions=instructions,
    )

    swarm = Swarm(include_usage=True)
    result = await swarm.execute(
        agent=agent,
        prompt="Hello!",
        context_variables=ContextVariables(user_name="John"),
    )

    print(result.messages[-1].content)


async def tool_call_example() -> None:
    def greet(language: str, context_variables: ContextVariables) -> Result[str]:
        user_name: str = context_variables.get("user_name", "John")
        greeting = "Hola" if language.lower() == "spanish" else "Hello"
        print(f"{greeting}, {user_name}!")

        return Result(
            value="Done",
            context_variables=ContextVariables(language=language),
        )

    def speak_instructions(context_variables: ContextVariables) -> str:
        print(f"Speak instructions: {context_variables}")
        language: str = context_variables.get("language", "English")
        return f"Speak with the user in {language} and ask them how they are doing."

    speak_agent = Agent.create(
        id="speak_agent",
        model="gpt-4o-mini",
        instructions=speak_instructions,
    )

    def switch_to_speak_agent() -> Result[Agent]:
        """Switch to the speak agent."""
        return Result(agent=speak_agent)

    welcome_agent = Agent.create(
        id="welcome_agent",
        model="gpt-4o-mini",
        instructions="You are a welcome agent that greets the user. "
        "Switch to the speak agent after greeting the user.",
        tools=[greet, switch_to_speak_agent],
        parallel_tool_calls=False,
    )

    swarm = Swarm(include_usage=True)
    result = await swarm.execute(
        agent=welcome_agent,
        prompt="Hola!",
        context_variables=ContextVariables(user_name="John"),
    )

    print(result.messages)


if __name__ == "__main__":
    enable_logging()
    asyncio.run(instructions_example())
    asyncio.run(tool_call_example())
