import sys
from typing import NoReturn

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.swarm import Swarm
from liteswarm.types import (
    Agent,
    Delta,
    Message,
    ToolCallAgentResult,
    ToolCallMessageResult,
    ToolCallResult,
)


class ReplStreamHandler:
    """Stream handler for REPL interface with better formatting."""

    async def on_stream(
        self,
        chunk: Delta,
        agent: Agent | None,
    ) -> None:
        """Handle streaming content from agents."""
        if chunk.content:
            # Only print agent ID prefix for the first character of a new message
            if not hasattr(self, "_last_agent") or self._last_agent != agent:
                agent_id = agent.agent_id if agent else "unknown"
                print(f"\n[{agent_id}] ", end="", flush=True)
                self._last_agent = agent

            print(f"{chunk.content}", end="", flush=True)

    async def on_error(
        self,
        error: Exception,
        agent: Agent | None,
    ) -> None:
        """Handle and display errors."""
        agent_id = agent.agent_id if agent else "unknown"
        print(f"\n❌ [{agent_id}] Error: {str(error)}", file=sys.stderr)
        self._last_agent = None

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None:
        """Display agent switching information."""
        print(
            f"\n🔄 Switching from {previous_agent.agent_id if previous_agent else 'none'} to {next_agent.agent_id}..."
        )
        self._last_agent = None

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None:
        """Handle completion of agent tasks."""
        agent_id = agent.agent_id if agent else "unknown"
        print(f"\n✅ [{agent_id}] Completed")
        self._last_agent = None

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None:
        """Display tool call information."""
        agent_id = agent.agent_id if agent else "unknown"
        print(f"\n🔧 [{agent_id}] Using {tool_call.function.name} [{tool_call.id}]")
        self._last_agent = None

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None:
        """Display tool call results."""
        agent_id = agent.agent_id if agent else "unknown"

        match tool_call_result:
            case ToolCallMessageResult() as tool_call_message_result:
                print(
                    f"\n📎 [{agent_id}] Got result for {tool_call_message_result.tool_call.function.name} [{tool_call_message_result.tool_call.id}]: {tool_call_message_result.message.content}"
                )
            case ToolCallAgentResult() as tool_call_agent_result:
                print(
                    f"\n🔧 [{agent_id}] Switching to: {tool_call_agent_result.agent.agent_id} [{tool_call_agent_result.tool_call.id}]"
                )
            case _:
                print(
                    f"\n📎 [{agent_id}] Got result for: {tool_call_result.tool_call.function.name} [{tool_call_result.tool_call.id}]"
                )

        self._last_agent = None


class AgentRepl:
    """Interactive REPL for agent conversations."""

    def __init__(self, agent: Agent) -> None:
        """Initialize the REPL with a starting agent.

        Args:
            agent: The initial agent to start conversations with
        """
        self.agent = agent
        self.swarm = Swarm(stream_handler=ReplStreamHandler())
        self.conversation: list[Message] = []

    def _print_welcome(self) -> None:
        """Print welcome message and usage instructions."""
        print("\n🤖 Agent REPL")
        print(f"Starting with agent: {self.agent.agent_id}")
        print("\nCommands:")
        print("  /exit    - Exit the REPL")
        print("  /help    - Show this help message")
        print("  /clear   - Clear conversation history")
        print("  /history - Show conversation history")
        print("\nEnter your queries and press Enter. Use commands above to control the REPL.")
        print("\n" + "=" * 50 + "\n")

    def _print_history(self) -> None:
        """Print the conversation history."""
        print("\n📝 Conversation History:")
        for msg in self.conversation:
            if msg.role != "system":
                content = msg.content or "[No content]"
                print(f"\n[{msg.role}]: {content}")
        print("\n" + "=" * 50 + "\n")

    def _handle_command(self, command: str) -> bool:
        """Handle REPL commands.

        Args:
            command: The command to handle

        Returns:
            True if should exit the REPL, False otherwise
        """
        match command.lower():
            case "/exit":
                print("\n👋 Goodbye!")
                return True
            case "/help":
                self._print_welcome()
            case "/clear":
                self.conversation.clear()
                print("\n🧹 Conversation history cleared")
            case "/history":
                self._print_history()
            case _:
                print("\n❌ Unknown command. Type /help for available commands.")
        return False

    async def _process_query(self, query: str) -> None:
        """Process a user query through the agent system.

        Args:
            query: The user's input query
        """
        try:
            result = await self.swarm.execute(
                agent=self.agent,
                prompt=query,
                messages=self.conversation,
            )
            self.conversation = result.messages
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}", file=sys.stderr)

    async def run(self) -> NoReturn:
        """Run the REPL loop indefinitely.

        This method runs until explicitly exited with /exit command.
        """
        self._print_welcome()

        while True:
            try:
                # Get user input
                user_input = input("\n🗣️  Enter your query: ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                    continue

                # Process regular query
                await self._process_query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 EOF received. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}", file=sys.stderr)
                continue


async def start_repl(agent: Agent) -> NoReturn:
    """Start a REPL session with the given agent.

    Args:
        agent: The agent to start the REPL with
    """
    repl = AgentRepl(agent)
    await repl.run()
