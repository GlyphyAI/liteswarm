# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.core import LiteSwarmStreamHandler
from liteswarm.types import Agent, Delta


class SwarmStreamHandler(LiteSwarmStreamHandler):
    """Custom stream handler for displaying LLM responses."""

    async def on_stream(self, delta: Delta, agent: Agent) -> None:
        """Handle a stream delta update.

        Args:
            delta: The delta update containing new content.
            agent: The agent that generated the content.
        """
        if delta.content:
            print(f"{delta.content}", end="", flush=True)
