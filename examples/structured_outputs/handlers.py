# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing_extensions import override

from liteswarm.core import LiteSwarmEventHandler
from liteswarm.types import SwarmEventType


class EventHandler(LiteSwarmEventHandler):
    """Custom event handler for displaying LLM responses."""

    @override
    async def on_event(self, event: SwarmEventType) -> None:
        if event.type == "agent_response_chunk":
            completion = event.chunk.completion
            if content := completion.delta.content:
                print(f"{content}", end="", flush=True)
