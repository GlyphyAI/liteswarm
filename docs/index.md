# LiteSwarm

A lightweight, LLM-agnostic framework for building AI agents with dynamic agent switching capabilities. Supports 100+ language models through [litellm](https://github.com/BerriAI/litellm).

## Features

- **Lightweight Core**: Minimal base implementation that's easy to understand and extend
- **LLM Agnostic**: Support for OpenAI, Anthropic, Google, and many more through litellm
- **Dynamic Agent Switching**: Switch between specialized agents during execution
- **Stateful Chat Interface**: Build chat applications with built-in state management
- **Event Streaming**: Real-time streaming of agent responses and tool calls

## Quick Start

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

## Installation

```bash
pip install liteswarm
```

For more details, check out:

- [Advanced Usage](advanced.md) for more advanced features
- [Examples](examples.md) for more code examples 
- [API Reference](api.md) for detailed documentation
- [Contributing](contributing.md) for how to contribute to the project