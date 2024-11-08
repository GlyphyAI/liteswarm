import asyncio

from liteswarm.repl import start_repl
from liteswarm.types import Agent


async def run() -> None:
    assistant = Agent.create(
        agent_id="assistant",
        model="claude-3-5-haiku-20241022",
        instructions="""You are a helpful AI assistant.
        Your goal is to help users with their questions and tasks.
        Be concise but thorough in your responses.""",
        temperature=0.7,
    )

    await start_repl(assistant)


if __name__ == "__main__":
    asyncio.run(run())
