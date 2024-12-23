# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import json
import shlex
import sys
from collections import deque
from typing import Any, NoReturn, get_args

from litellm import token_counter
from typing_extensions import override

from liteswarm.core.context_manager import (
    ContextManager,
    LiteContextManager,
    LiteOptimizationStrategy,
)
from liteswarm.core.message_store import MessageStore
from liteswarm.core.swarm import Swarm
from liteswarm.repl.event_handler import ReplEventHandler
from liteswarm.types.swarm import Agent, Message, ResponseCost, Usage
from liteswarm.utils.logging import enable_logging as liteswarm_enable_logging
from liteswarm.utils.messages import dump_messages, validate_messages


class ReplArgumentParser(argparse.ArgumentParser):
    """Custom argument parser that doesn't exit on error.

    Overrides the error handling to raise exceptions instead of
    calling sys.exit(), making it suitable for interactive use.
    """

    @override
    def error(self, message: str) -> NoReturn:
        """Raise an ArgumentError instead of exiting.

        Args:
            message: Error message to include in the exception.
        """
        raise argparse.ArgumentError(None, message)


class AgentRepl:
    """Interactive REPL for agent conversations.

    Provides a command-line interface for interacting with agents in a
    Read-Eval-Print Loop (REPL) format. Features include:
    - Interactive conversation with agents
    - Command-based control (/help, /exit, etc.)
    - Message store for storing messages
    - Usage and cost tracking
    - Agent state monitoring
    - Context optimization support

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
        - Supports context optimization for long conversations
        - Maintains conversation context between queries
        - Handles interrupts and errors gracefully
    """

    def __init__(
        self,
        agent: Agent,
        message_store: MessageStore[Any] | None = None,
        context_manager: ContextManager | None = None,
        include_usage: bool = False,
        include_cost: bool = False,
        max_iterations: int = sys.maxsize,
    ) -> None:
        """Initialize REPL with configuration.

        Args:
            agent: Initial agent for handling conversations.
            message_store: Optional store for messages. Defaults to None.
            context_manager: Optional context manager for optimization. Defaults to None.
            include_usage: Whether to track token usage. Defaults to False.
            include_cost: Whether to track costs. Defaults to False.
            max_iterations: Maximum conversation turns. Defaults to sys.maxsize.

        Notes:
            - Maintains conversation state between queries
            - Usage and cost tracking are optional features
            - Cleanup must be performed explicitly using /clear command
        """
        # Public configuration
        self.agent = agent
        self.swarm = Swarm(
            event_handler=ReplEventHandler(),
            message_store=message_store,
            context_manager=context_manager,
            include_usage=include_usage,
            include_cost=include_cost,
            max_iterations=max_iterations,
        )

        # Internal state (private)
        self._messages: list[Message] = []
        self._accumulated_usage: Usage | None = None
        self._accumulated_cost: ResponseCost | None = None
        self._active_agent: Agent | None = None
        self._agent_queue: deque[Agent] = deque()

    async def _print_welcome(self) -> None:
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
        print("  /clear   - Clear conversation memory")
        print("  /history - Show conversation messages")
        print("  /stats   - Show conversation statistics")
        print("  /save    - Save conversation memory to file")
        print("  /load    - Load conversation memory from file")
        print("  /optimize --strategy <strategy> [--model <model>] - Optimize context")
        print("           strategies: summarize, window, compress")
        print("  /find --query <query> [--count <n>] - Find relevant messages")
        print("\nEnter your queries and press Enter. Use commands above to control the REPL.")
        print("\n" + "=" * 50 + "\n")

    async def _print_history(self) -> None:
        """Print the conversation messages.

        Displays all non-system messages in chronological order, including:
        - Message roles (user, assistant, tool)
        - Message content
        - Visual separators for readability

        Notes:
            - System messages are filtered out for clarity
            - Empty content is shown as [No content]
        """
        print("\n📝 Conversation Messages:")
        for msg in self._messages:
            if msg.role != "system":
                content = msg.content or "[No content]"
                print(f"\n[{msg.role}]: {content}")
        print("\n" + "=" * 50 + "\n")

    async def _print_stats(self) -> None:
        """Print conversation statistics.

        Displays comprehensive statistics about the conversation:
        - Message counts
        - Accumulated token usage details (if enabled)
        - Accumulated cost information (if enabled)
        - Active agent information
        - Queue status

        Notes:
            - Token usage shown only if include_usage=True
            - Costs shown only if include_cost=True
            - Detailed breakdowns provided when available
            - Usage and costs are accumulated across all agents
        """
        print("\n📊 Conversation Statistics:")
        print(f"Message count: {len(self._messages)} messages")

        if self._accumulated_usage:
            print("\nAccumulated Token Usage:")
            print(f"  Prompt tokens: {self._accumulated_usage.prompt_tokens or 0:,}")
            print(f"  Completion tokens: {self._accumulated_usage.completion_tokens or 0:,}")
            print(f"  Total tokens: {self._accumulated_usage.total_tokens or 0:,}")

            # Only print token details if they exist and are not empty
            if self._accumulated_usage.prompt_tokens_details:
                print("\nPrompt Token Details:")
                prompt_token_details = self._accumulated_usage.prompt_tokens_details
                items = prompt_token_details.model_dump().items()
                for key, value in items:
                    if value is not None:
                        print(f"  {key}: {value:,}")

            if self._accumulated_usage.completion_tokens_details:
                print("\nCompletion Token Details:")
                completion_token_details = self._accumulated_usage.completion_tokens_details
                items = completion_token_details.model_dump().items()
                for key, value in items:
                    if value is not None:
                        print(f"  {key}: {value:,}")

        if self._accumulated_cost:
            prompt_cost = self._accumulated_cost.prompt_tokens_cost or 0
            completion_cost = self._accumulated_cost.completion_tokens_cost or 0
            total_cost = prompt_cost + completion_cost

            print("\nAccumulated Response Cost:")
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

    async def _save_history(self, filename: str = "conversation_memory.json") -> None:
        """Save the conversation memory to a file.

        Args:
            filename: The name of the file to save the conversation memory to.

        Notes:
            - Excludes system messages
            - Includes message metadata
        """
        memory = {"messages": dump_messages(self._messages)}

        with open(filename, "w") as f:
            json.dump(memory, f, indent=2)

        print(f"\n📤 Conversation memory saved to {filename}")
        print(f"Messages: {len(memory['messages'])} messages")

    async def _load_history(self, filename: str = "conversation_memory.json") -> None:
        """Load the conversation memory from a file.

        Args:
            filename: The name of the file to load the conversation memory from.

        Notes:
            - Updates memory state
            - Validates message format
            - Calculates token usage
        """
        try:
            with open(filename) as f:
                memory: dict[str, Any] = json.load(f)

            # Validate and load messages
            messages = validate_messages(memory.get("messages", []))

            # Update internal state
            self._messages = messages

            # Update swarm message store
            await self.swarm.message_store.set_messages(messages)

            print(f"\n📥 Conversation memory loaded from {filename}")
            print(f"Messages: {len(messages)} messages")

            # Calculate token usage for messages
            messages_dump = [msg.model_dump() for msg in messages if msg.role != "system"]
            prompt_tokens = token_counter(model=self.agent.llm.model, messages=messages_dump)
            print(f"Token count: {prompt_tokens:,}")

        except FileNotFoundError:
            print(f"\n❌ Memory file not found: {filename}")
        except json.JSONDecodeError:
            print(f"\n❌ Invalid JSON format in memory file: {filename}")
        except Exception as e:
            print(f"\n❌ Error loading memory: {str(e)}")

    async def _clear_history(self) -> None:
        """Clear the conversation memory.

        Resets memory and clears the swarm state.
        """
        self._messages = []
        self._accumulated_usage = None
        self._accumulated_cost = None
        self._active_agent = None
        self._agent_queue.clear()

        await self.swarm.cleanup(
            clear_agents=True,
            clear_context=True,
            clear_messages=True,
        )

        print("\n🧹 Conversation memory cleared")

    def _parse_command_args(
        self,
        parser: ReplArgumentParser,
        args_str: str,
        join_args: list[str] | None = None,
    ) -> argparse.Namespace | None:
        """Parse command arguments with error handling.

        Args:
            parser: Configured argument parser.
            args_str: Raw argument string to parse.
            join_args: List of argument names whose values should be joined.

        Returns:
            Parsed arguments or None if parsing failed.

        Notes:
            - Handles quoted strings and spaces in arguments
            - Joins multi-word values for specified arguments
            - Provides helpful error messages on failure
        """
        try:
            # Clean up args to handle quoted strings properly
            cleaned_args = []
            for arg in shlex.split(args_str):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    cleaned_args.extend([key, value])
                else:
                    cleaned_args.append(arg)

            parsed = parser.parse_args(cleaned_args)

            # Join multi-word arguments if specified
            if join_args:
                for arg_name in join_args:
                    arg_value = getattr(parsed, arg_name, None)
                    if isinstance(arg_value, list):
                        setattr(parsed, arg_name, " ".join(arg_value))

            return parsed

        except argparse.ArgumentError as e:
            print(f"\n❌ {str(e)}")
            parser.print_usage()
            return None

        except argparse.ArgumentTypeError as e:
            print(f"\n❌ {str(e)}")
            parser.print_usage()
            return None

        except (ValueError, Exception) as e:
            print(f"\n❌ Invalid command format: {str(e)}")
            parser.print_usage()
            return None

    def _create_optimize_parser(self) -> ReplArgumentParser:
        """Create argument parser for optimize command."""
        parser = ReplArgumentParser(
            prog="/optimize",
            description="Optimize conversation context using specified strategy",
            add_help=False,
        )
        parser.add_argument(
            "--strategy",
            "-s",
            required=True,
            choices=get_args(LiteOptimizationStrategy),
            help="Optimization strategy to use",
        )
        parser.add_argument(
            "--model",
            "-m",
            help="Model to optimize for (defaults to agent's model)",
        )
        parser.add_argument(
            "--query",
            "-q",
            nargs="+",  # Accept multiple words
            help="Query to use for RAG strategy",
        )
        return parser

    def _create_find_parser(self) -> ReplArgumentParser:
        """Create argument parser for find command."""
        parser = ReplArgumentParser(
            prog="/find",
            description="Find messages relevant to the given query",
            add_help=False,
        )
        parser.add_argument(
            "--query",
            "-q",
            required=True,
            nargs="+",  # Accept multiple words
            help="Search query",
        )
        parser.add_argument(
            "--count",
            "-n",
            type=int,
            help="Maximum number of messages to return",
        )
        return parser

    async def _handle_command(self, command: str) -> bool:
        """Handle REPL commands."""
        # Split command and arguments, preserving quoted strings
        parts = shlex.split(command)
        cmd = parts[0].lower()
        args = " ".join(parts[1:])

        match cmd:
            case "/exit":
                print("\n👋 Goodbye!")
                return True
            case "/help":
                await self._print_welcome()
            case "/clear":
                await self._clear_history()
            case "/history":
                await self._print_history()
            case "/stats":
                await self._print_stats()
            case "/save":
                await self._save_history()
            case "/load":
                await self._load_history()
            case "/optimize":
                await self._optimize_context(args)
            case "/find":
                await self._find_relevant(args)
            case _:
                print("\n❌ Unknown command. Type /help for available commands.")

        return False

    def _update_usage(self, new_usage: Usage | None) -> None:
        """Update accumulated usage with new usage data.

        Args:
            new_usage: New usage data to add to accumulation.
        """
        if not new_usage:
            return

        if not self._accumulated_usage:
            self._accumulated_usage = new_usage
            return

        # Prompt tokens from latest response already include all previous messages
        self._accumulated_usage.prompt_tokens = new_usage.prompt_tokens

        # Accumulate completion tokens
        self._accumulated_usage.completion_tokens += new_usage.completion_tokens

        # Update total tokens
        self._accumulated_usage.total_tokens = (
            self._accumulated_usage.prompt_tokens + self._accumulated_usage.completion_tokens
        )

        # Update token details if available
        if new_usage.prompt_tokens_details:
            self._accumulated_usage.prompt_tokens_details = new_usage.prompt_tokens_details

        if new_usage.completion_tokens_details:
            if not self._accumulated_usage.completion_tokens_details:
                self._accumulated_usage.completion_tokens_details = (
                    new_usage.completion_tokens_details
                )
            else:
                completion_token_details = self._accumulated_usage.completion_tokens_details
                items = completion_token_details.model_dump().items()
                for key, value in items:
                    if value is not None:
                        current = (
                            getattr(self._accumulated_usage.completion_tokens_details, key) or 0
                        )
                        setattr(
                            self._accumulated_usage.completion_tokens_details, key, current + value
                        )

    def _update_cost(self, new_cost: ResponseCost | None) -> None:
        """Update accumulated cost with new cost data.

        Args:
            new_cost: New cost data to add to accumulation.
        """
        if not new_cost:
            return

        if not self._accumulated_cost:
            self._accumulated_cost = new_cost
            return

        # Prompt cost from latest response already includes all previous messages
        self._accumulated_cost.prompt_tokens_cost = new_cost.prompt_tokens_cost

        # Accumulate completion cost
        self._accumulated_cost.completion_tokens_cost += new_cost.completion_tokens_cost

    async def _process_query(self, query: str) -> None:
        """Process a user query through the agent system.

        Handles the complete query processing lifecycle:
        - Sends query to the swarm
        - Updates conversation memory
        - Accumulates usage and costs
        - Maintains agent state
        - Handles errors

        Args:
            query: The user's input query to process.

        Notes:
            - Updates multiple aspects of REPL state
            - Maintains conversation continuity
            - Preserves error context for user feedback
            - Accumulates statistics if enabled
        """
        try:
            agent = self._active_agent or self.agent
            result = await self.swarm.execute(
                agent=agent,
                prompt=query,
            )

            self._messages = validate_messages(await self.swarm.message_store.get_messages())
            self._update_usage(result.usage)
            self._update_cost(result.response_cost)
            self._active_agent = result.agent
            self._agent_queue = self.swarm._agent_queue
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}", file=sys.stderr)

    async def _optimize_context(self, args: str) -> None:
        """Optimize conversation context using specified strategy.

        Command format: /optimize --strategy <strategy> [--model <model>] [--query <query>]
        - strategy: summarize, window, rag, or trim
        - model: optional model name (defaults to agent's model)
        - query: optional query for RAG strategy
        """
        try:
            parser = self._create_optimize_parser()
            parsed = self._parse_command_args(parser, args, join_args=["query"])
            if not parsed:
                print("\nUsage examples:")
                print("  /optimize -s rag -q 'search query'")
                print('  /optimize --strategy window --model "gpt-4"')
                print("  /optimize -s summarize")
                print('  /optimize -s rag -q "hello world"')
                return

            # Get current messages
            messages = await self.swarm.message_store.get_messages()
            if not messages:
                print("\n❌ No messages to optimize")
                return

            if not isinstance(self.swarm.context_manager, LiteContextManager):
                print("\n❌ Context manager does not support optimization")
                return

            # Run optimization
            optimized = await self.swarm.context_manager.optimize_context(
                model=parsed.model or self.agent.llm.model,
                strategy=parsed.strategy,
                query=parsed.query,
            )

            # Update memory
            await self.swarm.message_store.set_messages(optimized)
            self._messages = validate_messages(optimized)

            print(f"\n✨ Context optimized using {parsed.strategy} strategy")
            print(f"Messages: {len(messages)} → {len(optimized)}")

        except Exception as e:
            print(f"\n❌ Error optimizing context: {str(e)}")
            print("\nUsage examples:")
            print("  /optimize -s rag -q 'search query'")
            print('  /optimize --strategy window --model "gpt-4"')
            print("  /optimize -s summarize")
            print('  /optimize -s rag -q "hello world"')

    async def _find_relevant(self, args: str) -> None:
        """Find messages relevant to the given query.

        Command format: /find --query <query> [--count <n>]
        - query: search query
        - count: optional number of messages to return
        """
        try:
            parser = self._create_find_parser()
            parsed = self._parse_command_args(parser, args, join_args=["query"])
            if not parsed:
                print("\nUsage examples:")
                print('  /find --query "calendar view" --count 5')
                print('  /find -q "search term" -n 3')
                print("  /find --query calendar view --count 5")
                print("  /find -q calendar view -n 3")
                return

            # Get current messages
            messages = await self.swarm.message_store.get_messages()
            if not messages:
                print("\n❌ No messages to search")
                return

            if not isinstance(self.swarm.context_manager, LiteContextManager):
                print("\n❌ Context manager does not support RAG")
                return

            # Find relevant messages
            relevant = await self.swarm.context_manager.find_context(
                query=parsed.query,
                max_messages=parsed.count,
                embedding_model="text-embedding-ada-002",
            )

            # Print results
            print(f"\n🔍 Found {len(relevant)} relevant messages:")
            for msg in relevant:
                if msg.role != "system":
                    content = msg.content or "[No content]"
                    print(f"\n[{msg.role}]: {content}")

            print("\n" + "=" * 50 + "\n")

        except Exception as e:
            print(f"\n❌ Error finding relevant messages: {str(e)}")
            print("\nUsage examples:")
            print('  /find --query "calendar view" --count 5')
            print('  /find -q "search term" -n 3')
            print("  /find --query calendar view --count 5")
            print("  /find -q calendar view -n 3")

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
        await self._print_welcome()

        while True:
            try:
                # Get user input
                user_input = input("\n🗣️  Enter your query: ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if await self._handle_command(user_input):
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
    message_store: MessageStore[Any] | None = None,
    context_manager: ContextManager | None = None,
    include_usage: bool = False,
    include_cost: bool = False,
    max_iterations: int = sys.maxsize,
    enable_logging: bool = True,
) -> NoReturn:
    """Start an interactive REPL session.

    Args:
        agent: Initial agent for handling conversations.
        message_store: Optional store for messages. Defaults to None.
        context_manager: Optional context manager for optimization. Defaults to None.
        include_usage: Whether to track token usage. Defaults to False.
        include_cost: Whether to track costs. Defaults to False.
        max_iterations: Maximum conversation turns. Defaults to sys.maxsize.
        enable_logging: Whether to enable logging. Defaults to True.

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
        - State is preserved between queries
        - Use /clear command to reset state
    """
    if enable_logging:
        liteswarm_enable_logging()

    repl = AgentRepl(
        agent=agent,
        message_store=message_store,
        context_manager=context_manager,
        include_usage=include_usage,
        include_cost=include_cost,
        max_iterations=max_iterations,
    )

    await repl.run()
