# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.core import StreamHandler
from liteswarm.experimental import SwarmTeamStreamHandler
from liteswarm.types import Agent, Delta, Message, Plan, Task, ToolCallResult


class SwarmStreamHandler(StreamHandler):
    """Stream handler for software team with real-time progress updates."""

    def __init__(self) -> None:
        """Initialize the stream handler."""
        self._last_agent: Agent | None = None
        self._current_content = ""

    async def on_stream(self, delta: Delta, agent: Agent | None) -> None:
        """Handle streaming content from agents.

        Args:
            delta: The content delta from the agent
            agent: The agent generating the content
        """
        if delta.content:
            # Show agent prefix for first message
            if not self._last_agent or self._last_agent != agent:
                agent_id = agent.id if agent else "unknown"
                role = getattr(agent, "role", "assistant")
                print(f"\n\n🤖 [{agent_id}] ({role})", flush=True)
                self._last_agent = agent

            # Accumulate and print content
            self._current_content += delta.content
            print(delta.content, end="", flush=True)

            # Show continuation indicator if needed
            if getattr(delta, "finish_reason", None) == "length":
                print("\n[...continuing...]", end="", flush=True)

    async def on_error(self, error: Exception, agent: Agent | None) -> None:
        """Handle and display errors.

        Args:
            error: The error that occurred
            agent: The agent that encountered the error
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n\n❌ Error from {agent_id}: {str(error)}", file=sys.stderr, flush=True)
        self._last_agent = None

    async def on_agent_switch(self, previous_agent: Agent | None, next_agent: Agent) -> None:
        """Display agent switching information.

        Args:
            previous_agent: The agent being switched from
            next_agent: The agent being switched to
        """
        prev_id = previous_agent.id if previous_agent else "none"
        print(
            f"\n\n🔄 Switching from {prev_id} to {next_agent.id}...",
            flush=True,
        )
        self._last_agent = None

    async def on_tool_call(
        self, tool_call: ChatCompletionDeltaToolCall, agent: Agent | None
    ) -> None:
        """Display tool call information.

        Args:
            tool_call: The tool being called
            agent: The agent making the call
        """
        agent_id = agent.id if agent else "unknown"
        print(
            f"\n\n🔧 [{agent_id}] Calling: {tool_call.function.name}",
            flush=True,
        )
        self._last_agent = None

    async def on_tool_call_result(self, result: ToolCallResult, agent: Agent | None) -> None:
        """Display tool call results.

        Args:
            result: The result of the tool call
            agent: The agent that made the call
        """
        agent_id = agent.id if agent else "unknown"
        print(
            f"\n\n📎 [{agent_id}] Tool result received",
            flush=True,
        )
        self._last_agent = None

    async def on_complete(self, messages: list[Message], agent: Agent | None) -> None:
        """Handle completion of agent tasks.

        Args:
            messages: The complete message history
            agent: The agent completing its task
        """
        agent_id = agent.id if agent else "unknown"
        print(f"\n\n✅ [{agent_id}] Task completed\n", flush=True)
        self._last_agent = None


class SoftwareTeamStreamHandler(SwarmTeamStreamHandler):
    async def on_plan_created(self, plan: Plan) -> None:
        """Print the created development plan."""
        print("\nDevelopment Plan Created:")
        print("-------------------------")
        for task in plan.tasks:
            print(f"\nTask: {task.title}")
            print(f"Task Type: {task.task_type}")
            print(f"Description: {task.description}")
            if task.dependencies:
                print(f"Dependencies: {', '.join(task.dependencies)}")
            if task.metadata:
                print("Metadata:")
                for key, value in task.metadata.items():
                    print(f"- {key}: {value}")
        print("-------------------------")

    async def on_task_started(self, task: Task) -> None:
        """Print when a task is started."""
        print(f"\nStarting Task: {task.title}")
        print(f"Assigned to: {task.assignee}")

    async def on_task_completed(self, task: Task) -> None:
        """Print when a task is completed."""
        print(f"\nCompleted Task: {task.title}")

    async def on_plan_completed(self, plan: Plan) -> None:
        """Print when the plan is completed."""
        print("\nPlan Completed!")
        print("All tasks have been executed successfully.")
