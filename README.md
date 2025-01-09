# LiteSwarm 🐝

A lightweight, LLM-agnostic framework for building AI agents with dynamic agent switching capabilities. Supports 100+ language models through [litellm](https://github.com/BerriAI/litellm).

> [!WARNING]
> LiteSwarm is currently in early preview and the API is likely to change as we gather feedback.
>
> If you find any issues or have suggestions, please open an issue in the [Issues](https://github.com/glyphyai/liteswarm/issues) section.

## Features

- **Lightweight Core**: Minimal base implementation that's easy to understand and extend
- **LLM Agnostic**: Support for OpenAI, Anthropic, Google, and many more through litellm
- **Dynamic Agent Switching**: Switch between specialized agents during execution
- **Stateful Chat Interface**: Build chat applications with built-in state management
- **Event Streaming**: Real-time streaming of agent responses and tool calls

## Installation

```bash
pip install liteswarm
```

## Requirements

- **Python**: Version 3.11 or higher
- **Async Runtime**: LiteSwarm provides only async API, so you need to use an event loop to run it
- **LLM Provider Key**: You'll need an API key from a supported LLM provider (see [supported providers](https://docs.litellm.ai/docs/providers))
  <details>
  <summary>[click to see how to set keys]</summary>

  ```python
  # Environment variable
  export OPENAI_API_KEY=sk-...
  os.environ["OPENAI_API_KEY"] = "sk-..."
  
  # .env file
  OPENAI_API_KEY=sk-...
  
  # Direct in code
  LLM(model="gpt-4o", key="sk-...")
  ```
  </details>

## Quick Start

All examples below are complete and can be run as is.

### Hello World

Here's a minimal example showing how to use LiteSwarm's core functionality:

```python
import asyncio

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, Message


async def main() -> None:
    # Create a simple agent
    agent = Agent(
        id="assistant",
        instructions="You are a helpful assistant.",
        llm=LLM(model="gpt-4o"),
    )

    # Create swarm and execute
    swarm = Swarm()
    result = await swarm.execute(
        agent=agent,
        messages=[Message(role="user", content="Hello!")],
    )
    print(result.agent_response.content)


if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming with Agent Switching

This example demonstrates real-time streaming and dynamic agent switching capabilities:

```python
import asyncio

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, Message, ToolResult


# Define a tool that can switch to another agent
def switch_to_expert(domain: str) -> ToolResult:
    return ToolResult.switch_agent(
        agent=Agent(
            id=f"{domain}-expert",
            instructions=f"You are a {domain} expert.",
            llm=LLM(
                model="gpt-4o",
                temperature=0.0,
            ),
        ),
        content=f"Switching to {domain} expert",
    )


async def main() -> None:
    # Create a router agent that can switch to experts
    router = Agent(
        id="router",
        instructions="Route questions to appropriate experts.",
        llm=LLM(
            model="gpt-4o",
            tools=[switch_to_expert],
        ),
    )

    # Stream responses in real-time
    swarm = Swarm()
    stream = swarm.stream(
        agent=router,
        messages=[Message(role="user", content="Explain quantum physics like I'm 5")],
    )

    async for event in stream:
        if event.type == "agent_response_chunk":
            completion = event.response_chunk.completion
            if completion.delta.content:
                print(completion.delta.content, end="", flush=True)
            if completion.finish_reason == "stop":
                print()

    # Optionally, get execution result from stream
    result = await stream.get_return_value()
    print(result.agent_response.content)


if __name__ == "__main__":
    asyncio.run(main())
```

### Stateful Chat

Here's how to build a stateful chat application that maintains conversation history:

```python
import asyncio

from liteswarm.chat import LiteChat
from liteswarm.types import LLM, Agent, SwarmEvent


def handle_event(event: SwarmEvent) -> None:
    if event.type == "agent_response_chunk":
        completion = event.response_chunk.completion
        if completion.delta.content:
            print(completion.delta.content, end="", flush=True)
        if completion.finish_reason == "stop":
            print()


async def main() -> None:
    # Create an agent
    agent = Agent(
        id="assistant",
        instructions="You are a helpful assistant. Provide short answers.",
        llm=LLM(model="gpt-4o"),
    )

    # Create stateful chat
    chat = LiteChat()

    # First message
    print("First message:")
    async for event in chat.send_message("Tell me about Python", agent=agent):
        handle_event(event)

    # Second message - chat remembers the context
    print("\nSecond message:")
    async for event in chat.send_message("What are its key features?", agent=agent):
        handle_event(event)

    # Access conversation history
    messages = await chat.get_messages()
    print(f"\nMessages in history: {len(messages)}")


if __name__ == "__main__":
    asyncio.run(main())
```

For more examples, check out the [examples](examples/) directory. To learn more about advanced features and API details, see our [documentation](docs/).

## Documentation

- [Advanced Features](docs/advanced.md)
- [Examples](docs/examples.md)
- [API Reference](docs/api.md)
- [Contributing](docs/contributing.md)

## Citation

If you use LiteSwarm in your research, please cite our work:

```bibtex
@software{Mozharovskii_LiteSwarm_2025,
    author = {Mozharovskii, Evgenii and {GlyphyAI}},
    license = {MIT},
    month = jan,
    title = {{LiteSwarm}},
    url = {https://github.com/glyphyai/liteswarm},
    version = {0.5.0},
    year = {2025}
}
``` 

## License

MIT License - see [LICENSE](LICENSE) file for details.
