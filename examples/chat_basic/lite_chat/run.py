# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json

from liteswarm.chat import LiteChat
from liteswarm.types import LLM, Agent
from liteswarm.utils import dump_messages


async def run() -> None:
    # Create a simple agent
    agent = Agent(
        id="assistant",
        instructions="You are a helpful assistant that provides clear and concise responses.",
        llm=LLM(model="gpt-4o"),
    )

    # Initialize chat
    chat = LiteChat()

    # Send messages and stream responses
    user_messages = [
        "Hello! How are you?",
        "What can you help me with?",
    ]

    for message in user_messages:
        print(f"\nUser: {message}")
        print("Assistant: ", end="", flush=True)

        async for event in chat.send_message(message, agent=agent):
            if event.type == "agent_response_chunk":
                completion = event.response_chunk.completion
                if completion.delta.content:
                    print(completion.delta.content, end="", flush=True)
                if completion.finish_reason == "stop":
                    print()

    # Get conversation history
    chat_messages = await chat.get_messages()
    history = dump_messages(chat_messages, exclude_none=True)
    print("\nConversation history:")
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
