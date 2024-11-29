# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.core.stream_handler import SwarmStreamHandler
from liteswarm.types.swarm import (
    Agent,
    Delta,
    Message,
    ToolCallAgentResult,
    ToolCallMessageResult,
    ToolCallResult,
)


class ReplStreamHandler(SwarmStreamHandler):
    """Stream handler for REPL interface with better formatting."""

    def __init__(self) -> None:
        """Initialize the stream handler with usage tracking."""
        self._last_agent: Agent | None = None

    async def on_stream(
        self,
        chunk: Delta,
        agent: Agent | None,
    ) -> None:
        """Handle streaming content from agents."""
        if chunk.content:
            # Show a continuation indicator if the response ended due to a length limit
            if getattr(chunk, "finish_reason", None) == "length":
                print("\n[...continuing...]", end="", flush=True)

            # Only print agent ID prefix for the first character of a new message
            if not hasattr(self, "_last_agent") or self._last_agent != agent:
                agent_id = agent.id if agent else "unknown"
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = agent

            print(f"{chunk.content}", end="", flush=True)

    async def on_error(
        self,
        error: Exception,
        agent: Agent | None,
    ) -> None:
        """Handle and display errors."""
        agent_id = agent.id if agent else "unknown"
        print(f"\n❌ [{agent_id}] Error: {str(error)}", file=sys.stderr)
        self._last_agent = None

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None:
        """Display agent switching information."""
        print(
            f"\n🔄 Switching from {previous_agent.id if previous_agent else 'none'} to {next_agent.id}..."
        )
        self._last_agent = None

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None:
        """Handle completion of agent tasks."""
        agent_id = agent.id if agent else "unknown"
        print(f"\n✅ [{agent_id}] Completed")
        self._last_agent = None

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None:
        """Display tool call information."""
        agent_id = agent.id if agent else "unknown"
        print(f"\n🔧 [{agent_id}] Using {tool_call.function.name} [{tool_call.id}]")
        self._last_agent = None

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None:
        """Display tool call results."""
        agent_id = agent.id if agent else "unknown"

        match tool_call_result:
            case ToolCallMessageResult() as tool_call_message_result:
                print(
                    f"\n📎 [{agent_id}] Got result for {tool_call_message_result.tool_call.function.name} [{tool_call_message_result.tool_call.id}]: {tool_call_message_result.message.content}"
                )
            case ToolCallAgentResult() as tool_call_agent_result:
                print(
                    f"\n🔧 [{agent_id}] Switching to: {tool_call_agent_result.agent.id} [{tool_call_agent_result.tool_call.id}]"
                )
            case _:
                print(
                    f"\n📎 [{agent_id}] Got result for: {tool_call_result.tool_call.function.name} [{tool_call_result.tool_call.id}]"
                )

        self._last_agent = None
