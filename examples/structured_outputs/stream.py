# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.core import LiteSwarmStreamHandler
from liteswarm.types import AgentResponse


class SwarmStreamHandler(LiteSwarmStreamHandler):
    """Custom stream handler for displaying LLM responses."""

    async def on_stream(self, agent_response: AgentResponse) -> None:
        if agent_response.delta.content:
            print(f"{agent_response.delta.content}", end="", flush=True)
