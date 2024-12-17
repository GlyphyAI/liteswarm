# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import sys
from collections import deque
from typing import NoReturn

from litellm import token_counter
from pydantic import TypeAdapter

from liteswarm.core.memory import Memory
from liteswarm.core.summarizer import Summarizer
from liteswarm.core.swarm import Swarm
from liteswarm.repl.stream_handler import ReplStreamHandler
from liteswarm.types.swarm import Agent, Message, ResponseCost, Usage
from liteswarm.utils.logging import enable_logging

Messages = TypeAdapter(list[Message])
"""Type adapter for a list of messages."""


class AgentRepl:
    """Interactive REPL for agent conversations.

    Provides a command-line interface for interacting with agents in a
    Read-Eval-Print Loop (REPL) format. Features include:
    - Interactive conversation with agents
    - Command-based control (/help, /exit, etc.)
    - Conversation history management
    - Usage and cost tracking
    - Agent state monitoring
    - History summarization support

    The REPL maintains conversation state and provides real-time feedback
    on agent responses, tool usage, and state changes.

    Example:
        ```python
        agent = Agent(
            id="helper",
            instructions="You are a helpful assistant.",
            llm=LLM(model="gpt-4o"),
        )

        repl = AgentRepl(
            agent=agent,
            include_usage=True,
            include_cost=True,
        )

        await repl.run()
        ```

    Notes:
        - The REPL runs until explicitly terminated
        - Supports history summarization for long conversations
        - Maintains conversation context between queries
        - Handles interrupts and errors gracefully
    """

    def __init__(
        self,
        agent: Agent,
        memory: Memory | None = None,
        summarizer: Summarizer | None = None,
        include_usage: bool = False,
        include_cost: bool = False,
        cleanup: bool = False,
        max_iterations: int = sys.maxsize,
    ) -> None:
        """Initialize REPL with configuration.

        Args:
            agent: Initial agent for handling conversations.
            memory: Optional memory manager for history. Defaults to None.
            summarizer: Optional history summarizer. Defaults to None.
            include_usage: Whether to track token usage. Defaults to False.
            include_cost: Whether to track costs. Defaults to False.
            cleanup: Whether to reset agent state after each query. Defaults to False.
            max_iterations: Maximum conversation turns. Defaults to sys.maxsize.

        Notes:
            - Maintains conversation state between queries if cleanup=False
            - Usage and cost tracking are optional features
        """
        # Public configuration
        self.agent = agent
        self.cleanup = cleanup
        self.swarm = Swarm(
            stream_handler=ReplStreamHandler(),
            memory=memory,
            summarizer=summarizer,
            include_usage=include_usage,
            include_cost=include_cost,
            max_iterations=max_iterations,
        )

        # Internal state (private)
        self._full_history: list[Message] = []
        self._working_history: list[Message] = []
        self._usage: Usage | None = None
        self._response_cost: ResponseCost | None = None
        self._active_agent: Agent | None = None
        self._agent_queue: deque[Agent] = deque()

    def _print_welcome(self) -> None:
        """Print welcome message and usage instructions.

        Displays:
        - Initial greeting
        - Starting agent information
        - Available commands
        - Basic usage instructions

        Notes:
            Called automatically when the REPL starts and on /help command.
        """
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
        """Print the conversation history.

        Displays all non-system messages in chronological order, including:
        - Message roles (user, assistant, tool)
        - Message content
        - Visual separators for readability

        Notes:
            - System messages are filtered out for clarity
            - Empty content is shown as [No content]
        """
        print("\n📝 Conversation History:")
        for msg in self._full_history:
            if msg.role != "system":
                content = msg.content or "[No content]"
                print(f"\n[{msg.role}]: {content}")
        print("\n" + "=" * 50 + "\n")

    def _print_stats(self) -> None:
        """Print conversation statistics.

        Displays comprehensive statistics about the conversation:
        - Message counts (full and working history)
        - Token usage details (if enabled)
        - Cost information (if enabled)
        - Active agent information
        - Queue status

        Notes:
            - Token usage shown only if include_usage=True
            - Costs shown only if include_cost=True
            - Detailed breakdowns provided when available
        """
        print("\n📊 Conversation Statistics:")
        print(f"Full history length: {len(self._full_history)} messages")
        print(f"Working history length: {len(self._working_history)} messages")

        if self._usage:
            print("\nToken Usage:")
            print(f"  Prompt tokens: {self._usage.prompt_tokens or 0:,}")
            print(f"  Completion tokens: {self._usage.completion_tokens or 0:,}")
            print(f"  Total tokens: {self._usage.total_tokens or 0:,}")

            # Only print token details if they exist and are not empty
            if self._usage.prompt_tokens_details and isinstance(
                self._usage.prompt_tokens_details, dict
            ):
                print("\nPrompt Token Details:")
                for key, value in self._usage.prompt_tokens_details.items():
                    if value is not None:
                        print(f"  {key}: {value:,}")

            if self._usage.completion_tokens_details and isinstance(
                self._usage.completion_tokens_details, dict
            ):
                print("\nCompletion Token Details:")
                for key, value in self._usage.completion_tokens_details.items():
                    if value is not None:
                        print(f"  {key}: {value:,}")

        if self._response_cost:
            prompt_cost = self._response_cost.prompt_tokens_cost or 0
            completion_cost = self._response_cost.completion_tokens_cost or 0
            total_cost = prompt_cost + completion_cost

            print("\nResponse Cost:")
            print(f"  Prompt tokens: ${prompt_cost:.6f}")
            print(f"  Completion tokens: ${completion_cost:.6f}")
            print(f"  Total cost: ${total_cost:.6f}")

        print("\nActive Agent:")
        if self._active_agent:
            print(f"  ID: {self._active_agent.id}")
            print(f"  Model: {self._active_agent.llm.model}")
            print(f"  Tools: {len(self._active_agent.llm.tools or [])} available")
        else:
            print("  None")

        print(f"\nPending agents in queue: {len(self._agent_queue)}")
        print("\n" + "=" * 50 + "\n")

    def _save_history(self, filename: str = "conversation_history.json") -> None:
        """Save the conversation history to a file.

        Args:
            filename: The name of the file to save the conversation history to.

        Notes:
            - Saves both full and working history
            - Excludes system messages
            - Includes message metadata
        """
        history = {
            "full_history": [
                msg.model_dump() for msg in self._full_history if msg.role != "system"
            ],
            "working_history": [
                msg.model_dump() for msg in self._working_history if msg.role != "system"
            ],
        }

        with open(filename, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n📤 Conversation history saved to {filename}")
        print(f"Full history: {len(history['full_history'])} messages")
        print(f"Working history: {len(history['working_history'])} messages")

    def _load_history(self, filename: str = "conversation_history.json") -> None:
        """Load the conversation history from a file.

        Args:
            filename: The name of the file to load the conversation history from.

        Notes:
            - Restores both full and working history
            - Updates swarm memory state
            - Validates message format
            - Calculates token usage
        """
        try:
            with open(filename) as f:
                history: dict[str, list[Message]] = json.load(f)

            # Validate and load histories
            full_history = Messages.validate_python(history.get("full_history", []))
            working_history = Messages.validate_python(history.get("working_history", []))

            # Update internal state
            self._full_history = full_history
            self._working_history = working_history

            # Update swarm memory
            self.swarm.memory.set_history(full_history)

            print(f"\n📥 Conversation history loaded from {filename}")
            print(f"Full history: {len(full_history)} messages")
            print(f"Working history: {len(working_history)} messages")

            # Calculate token usage for working history
            messages = [msg.model_dump() for msg in working_history if msg.role != "system"]
            prompt_tokens = token_counter(model=self.agent.llm.model, messages=messages)
            print(f"Working history token count: {prompt_tokens:,}")

        except FileNotFoundError:
            print(f"\n❌ History file not found: {filename}")
        except json.JSONDecodeError:
            print(f"\n❌ Invalid JSON format in history file: {filename}")
        except Exception as e:
            print(f"\n❌ Error loading history: {str(e)}")

    def _clear_history(self) -> None:
        """Clear the conversation history.

        Resets both full and working histories, and clears the swarm memory.
        """
        self._full_history.clear()
        self._working_history.clear()
        if self.swarm.memory:
            self.swarm.memory.clear_history()

        self._usage = None
        self._response_cost = None
        self._active_agent = None
        self._agent_queue.clear()

        print("\n🧹 Conversation history cleared")

    def _handle_command(self, command: str) -> bool:
        """Handle REPL commands.

        Processes special commands that control REPL behavior:
        - /exit: Terminate the REPL
        - /help: Show usage instructions
        - /clear: Clear conversation history
        - /history: Show message history
        - /stats: Show conversation statistics
        - /save: Save conversation history to file
        - /load: Load conversation history from file

        Args:
            command: The command to handle, including the leading slash.

        Returns:
            True if the REPL should exit, False to continue running.

        Notes:
            - Commands are case-insensitive
            - Unknown commands show help message
            - Some commands have immediate effects on REPL state
        """
        match command.lower():
            case "/exit":
                print("\n👋 Goodbye!")
                return True
            case "/help":
                self._print_welcome()
            case "/clear":
                self._clear_history()
            case "/history":
                self._print_history()
            case "/stats":
                self._print_stats()
            case "/save":
                self._save_history()
            case "/load":
                self._load_history()
            case _:
                print("\n❌ Unknown command. Type /help for available commands.")

        return False

    async def _process_query(self, query: str) -> None:
        """Process a user query through the agent system.

        Handles the complete query processing lifecycle:
        - Sends query to the swarm
        - Updates conversation history
        - Tracks usage and costs
        - Maintains agent state
        - Handles errors

        Args:
            query: The user's input query to process.

        Notes:
            - Updates multiple aspects of REPL state
            - Maintains conversation continuity
            - Preserves error context for user feedback
            - Automatically updates statistics if enabled
        """
        try:
            result = await self.swarm.execute(
                agent=self.agent,
                prompt=query,
                cleanup=self.cleanup,
            )

            self._full_history = self.swarm.memory.get_full_history()
            self._working_history = self.swarm.memory.get_working_history()
            self._usage = result.usage
            self._response_cost = result.response_cost
            self._active_agent = result.agent
            self._agent_queue = self.swarm._agent_queue
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}", file=sys.stderr)

    async def run(self) -> NoReturn:
        """Run the REPL loop indefinitely.

        Provides the main interaction loop:
        - Displays welcome message
        - Processes user input
        - Handles commands
        - Manages conversation flow
        - Handles interruptions

        The loop continues until explicitly terminated by:
        - /exit command
        - Keyboard interrupt (Ctrl+C)
        - EOF signal (Ctrl+D)

        Raises:
            SystemExit: When the REPL is terminated.

        Notes:
            - Empty inputs are ignored
            - Errors don't terminate the loop
            - Graceful shutdown on interrupts
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


async def start_repl(
    agent: Agent,
    memory: Memory | None = None,
    summarizer: Summarizer | None = None,
    include_usage: bool = False,
    include_cost: bool = False,
    cleanup: bool = False,
    max_iterations: int = sys.maxsize,
) -> NoReturn:
    """Start an interactive REPL session.

    Args:
        agent: Initial agent for handling conversations.
        memory: Optional memory manager for history. Defaults to None.
        summarizer: Optional history summarizer. Defaults to None.
        include_usage: Whether to track token usage. Defaults to False.
        include_cost: Whether to track costs. Defaults to False.
        cleanup: Whether to reset agent state after each query. Defaults to False.
        max_iterations: Maximum conversation turns. Defaults to sys.maxsize.

    Example:
        ```python
        agent = Agent(
            id="helper",
            instructions="You are a helpful assistant.",
            llm=LLM(model="gpt-4o"),
        )

        await start_repl(agent=agent, include_usage=True)
        ```

    Notes:
        - Enables logging automatically
        - Runs until explicitly terminated
    """
    enable_logging()
    repl = AgentRepl(
        agent=agent,
        memory=memory,
        summarizer=summarizer,
        include_usage=include_usage,
        include_cost=include_cost,
        cleanup=cleanup,
        max_iterations=max_iterations,
    )

    await repl.run()
