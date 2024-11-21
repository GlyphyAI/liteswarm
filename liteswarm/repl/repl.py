# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from typing import NoReturn

from liteswarm.core.summarizer import Summarizer
from liteswarm.core.swarm import Swarm
from liteswarm.repl.stream_handler import ReplStreamHandler
from liteswarm.types.swarm import Agent, Message, ResponseCost, Usage
from liteswarm.utils.logging import enable_logging
from liteswarm.utils.usage import combine_response_cost, combine_usage


class AgentRepl:
    """Interactive REPL for agent conversations."""

    def __init__(
        self,
        agent: Agent,
        summarizer: Summarizer | None = None,
        include_usage: bool = False,
        include_cost: bool = False,
        cleanup: bool = True,
    ) -> None:
        """Initialize the REPL with a starting agent.

        Args:
            agent: The initial agent to start conversations with
            summarizer: A summarizer to use for summarizing conversation history
            include_usage: Whether to include usage in the REPL stats
            include_cost: Whether to include cost statistics in responses
            cleanup: Whether to clear agent state after completion. If False,
                    maintains the last active agent for subsequent interactions
        """
        self.agent = agent
        self.cleanup = cleanup
        self.swarm = Swarm(
            stream_handler=ReplStreamHandler(),
            summarizer=summarizer,
            include_usage=include_usage,
            include_cost=include_cost,
        )
        self.conversation: list[Message] = []
        self.usage: Usage | None = None
        self.response_cost: ResponseCost | None = None
        self.active_agent: Agent | None = None
        self.agent_queue: list[Agent] = []
        self.working_history: list[Message] = []

    def _print_welcome(self) -> None:
        """Print welcome message and usage instructions."""
        print("\n🤖 Agent REPL")
        print(f"Starting with agent: {self.agent.id}")
        print("\nCommands:")
        print("  /exit    - Exit the REPL")
        print("  /help    - Show this help message")
        print("  /clear   - Clear conversation history")
        print("  /history - Show conversation history")
        print("  /stats   - Show conversation statistics")
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

    def _print_stats(self) -> None:
        """Print conversation statistics."""
        print("\n📊 Conversation Statistics:")
        print(f"Full history length: {len(self.conversation)} messages")
        print(f"Working history length: {len(self.working_history)} messages")

        if self.usage:
            print("\nToken Usage:")
            print(f"  Prompt tokens: {self.usage.prompt_tokens or 0:,}")
            print(f"  Completion tokens: {self.usage.completion_tokens or 0:,}")
            print(f"  Total tokens: {self.usage.total_tokens or 0:,}")

            if self.usage.prompt_tokens_details:
                print("\nPrompt Token Details:")
                for key, value in self.usage.prompt_tokens_details:
                    print(f"  {key}: {value:,}")

            if self.usage.completion_tokens_details:
                print("\nCompletion Token Details:")
                for key, value in self.usage.completion_tokens_details:
                    print(f"  {key}: {value:,}")

        if self.response_cost:
            total_cost = (
                self.response_cost.prompt_tokens_cost + self.response_cost.completion_tokens_cost
            )

            print("\nResponse Cost:")
            print(f"  Prompt tokens: ${self.response_cost.prompt_tokens_cost:.4f}")
            print(f"  Completion tokens: ${self.response_cost.completion_tokens_cost:.4f}")
            print(f"  Total cost: ${total_cost:.4f}")

        print("\nActive Agent:")
        if self.active_agent:
            print(f"  ID: {self.active_agent.id}")
            print(f"  Model: {self.active_agent.model}")
            print(f"  Tools: {len(self.active_agent.tools)} available")
        else:
            print("  None")

        print(f"\nPending agents in queue: {len(self.agent_queue)}")
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
            case "/stats":
                self._print_stats()
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
                cleanup=self.cleanup,
            )
            self.conversation = result.messages
            self.usage = combine_usage(self.usage, result.usage)
            self.response_cost = combine_response_cost(self.response_cost, result.response_cost)
            self.active_agent = result.agent
            self.agent_queue = result.agent_queue
            self.working_history.extend(result.messages)
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
                        sys.exit(0)

                    continue

                # Process regular query
                await self._process_query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 EOF received. Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}", file=sys.stderr)
                continue


async def start_repl(  # noqa: PLR0913
    agent: Agent,
    summarizer: Summarizer | None = None,
    include_usage: bool = False,
    include_cost: bool = False,
    cleanup: bool = True,
) -> NoReturn:
    """Start a REPL session with the given agent.

    Args:
        agent: The agent to start the REPL with
        summarizer: A summarizer to use for summarizing conversation history
        include_usage: Whether to include usage in the REPL stats
        include_cost: Whether to include cost statistics in the REPL stats
        cleanup: Whether to clear agent state after completion. If False,
                maintains the last active agent for subsequent interactions
        log_level: The log level to use for the REPL
    """
    enable_logging()
    repl = AgentRepl(agent, summarizer, include_usage, include_cost, cleanup)
    await repl.run()