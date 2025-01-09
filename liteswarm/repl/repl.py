# Copyright 2025 GlyphyAI
#
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

from liteswarm.chat import LiteChat, LiteChatMemory, LiteChatSearch
from liteswarm.chat.optimization import LiteChatOptimization, OptimizationStrategy
from liteswarm.core.swarm import Swarm
from liteswarm.repl.event_handler import ConsoleEventHandler
from liteswarm.types.chat import RAGStrategyConfig
from liteswarm.types.swarm import Agent, AgentContext, ResponseCost, Usage
from liteswarm.utils.logging import LogLevel, log_verbose
from liteswarm.utils.logging import enable_logging as liteswarm_enable_logging
from liteswarm.utils.messages import dump_messages, validate_messages
from liteswarm.utils.misc import prompt


class ReplArgumentParser(argparse.ArgumentParser):
    """Custom argument parser that raises exceptions instead of exiting.

    Designed for interactive use in REPL environments where program termination
    on error is undesirable.
    """

    @override
    def error(self, message: str) -> NoReturn:
        """Raise an ArgumentError instead of exiting.

        Args:
            message: Error message to include in the exception.
        """
        raise argparse.ArgumentError(None, message)


class AgentRepl:
    """Interactive REPL for agent-based conversations.

    Provides a command-line interface for interacting with agents through a
    Read-Eval-Print Loop (REPL). Features include conversation management,
    command-based control, message storage, usage tracking, and context
    optimization.

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
        memory: LiteChatMemory | None = None,
        search: LiteChatSearch | None = None,
        optimization: LiteChatOptimization | None = None,
        include_usage: bool = False,
        include_cost: bool = False,
        max_iterations: int = sys.maxsize,
    ) -> None:
        """Initialize REPL with configuration.

        Args:
            agent: Initial agent for handling conversations.
            memory: Memory manager for the chat.
            search: Search manager for the chat.
            optimization: Optimization manager for the chat.
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
        self.memory = memory or LiteChatMemory()
        self.search = search or LiteChatSearch(memory=self.memory)
        self.optimization = optimization or LiteChatOptimization(
            memory=self.memory,
            search=self.search,
        )
        self.chat = LiteChat(
            memory=self.memory,
            search=self.search,
            optimization=self.optimization,
            swarm=Swarm(
                include_usage=include_usage,
                include_cost=include_cost,
                max_iterations=max_iterations,
            ),
        )

        # Internal state (private)
        self._event_handler = ConsoleEventHandler()
        self._accumulated_usage: Usage | None = None
        self._accumulated_cost: ResponseCost | None = None
        self._active_agent: Agent | None = None
        self._agent_queue: deque[AgentContext] = deque()

    async def _print_welcome(self) -> None:
        """Display welcome message and available commands."""
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
        """Display all non-system messages in chronological order."""
        print("\n📝 Conversation Messages:")
        messages = await self.chat.get_messages()
        for msg in messages:
            if msg.role != "system":
                content = msg.content or "[No content]"
                print(f"\n[{msg.role}]: {content}")
        print("\n" + "=" * 50 + "\n")

    async def _print_stats(self) -> None:
        """Display conversation statistics including message counts, token usage, and costs."""
        messages = await self.chat.get_messages()
        print("\n📊 Conversation Statistics:")
        print(f"Message count: {len(messages)} messages")

        if self._accumulated_usage:
            print("\nAccumulated Token Usage:")
            print(f"  Prompt tokens: {self._accumulated_usage.prompt_tokens or 0:,}")
            print(f"  Completion tokens: {self._accumulated_usage.completion_tokens or 0:,}")
            print(f"  Total tokens: {self._accumulated_usage.total_tokens or 0:,}")

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
        """Save conversation memory to a JSON file.

        Args:
            filename: The target file path.
        """
        messages = await self.chat.get_messages()
        memory = {"messages": dump_messages(messages)}

        with open(filename, "w") as f:
            json.dump(memory, f, indent=2)

        print(f"\n📤 Conversation memory saved to {filename}")
        print(f"Messages: {len(memory['messages'])} messages")

    async def _load_history(self, filename: str = "conversation_memory.json") -> None:
        """Load conversation memory from a JSON file.

        Args:
            filename: The source file path.
        """
        try:
            with open(filename) as f:
                memory: dict[str, Any] = json.load(f)

            messages = validate_messages(memory.get("messages", []))
            await self.memory.add_messages(messages)

            print(f"\n📥 Conversation memory loaded from {filename}")
            print(f"Messages: {len(messages)} messages")

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
        """Clear conversation memory and reset REPL state."""
        self._accumulated_usage = None
        self._accumulated_cost = None
        self._active_agent = None
        self._agent_queue.clear()

        await self.memory.clear()
        await self.search.clear()

        print("\n🧹 Conversation memory cleared")

    def _parse_command_args(
        self,
        parser: ReplArgumentParser,
        args_str: str,
        join_args: list[str] | None = None,
    ) -> argparse.Namespace | None:
        """Parse command arguments with error handling.

        Args:
            parser: The argument parser to use.
            args_str: Raw argument string to parse.
            join_args: List of argument names whose values should be joined.

        Returns:
            Parsed arguments or None if parsing failed.
        """
        try:
            cleaned_args = []
            for arg in shlex.split(args_str):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    cleaned_args.extend([key, value])
                else:
                    cleaned_args.append(arg)

            parsed = parser.parse_args(cleaned_args)

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
        """Create argument parser for the optimize command."""
        parser = ReplArgumentParser(
            prog="/optimize",
            description="Optimize conversation context using specified strategy",
            add_help=False,
        )
        parser.add_argument(
            "--strategy",
            "-s",
            required=True,
            choices=get_args(OptimizationStrategy),
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
            nargs="+",
            help="Query to use for RAG strategy",
        )
        return parser

    def _create_find_parser(self) -> ReplArgumentParser:
        """Create argument parser for the find command."""
        parser = ReplArgumentParser(
            prog="/find",
            description="Find messages relevant to the given query",
            add_help=False,
        )
        parser.add_argument(
            "--query",
            "-q",
            required=True,
            nargs="+",
            help="Search query",
        )
        parser.add_argument(
            "--count",
            "-n",
            type=int,
            help="Maximum number of messages to return",
        )
        parser.add_argument(
            "--threshold",
            "-t",
            type=float,
            default=0.5,
            help="Minimum similarity score (0.0 to 1.0)",
        )
        return parser

    async def _handle_command(self, command: str) -> bool:
        """Handle REPL commands and return whether to exit.

        Args:
            command: The command to handle.

        Returns:
            True if the REPL should exit, False otherwise.
        """
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
        """Update accumulated usage statistics.

        Args:
            new_usage: New usage data to add.
        """
        if not new_usage:
            return

        if not self._accumulated_usage:
            self._accumulated_usage = new_usage
            return

        self._accumulated_usage.prompt_tokens = new_usage.prompt_tokens
        self._accumulated_usage.completion_tokens += new_usage.completion_tokens
        self._accumulated_usage.total_tokens = (
            self._accumulated_usage.prompt_tokens + self._accumulated_usage.completion_tokens
        )

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
        """Update accumulated cost statistics.

        Args:
            new_cost: New cost data to add.
        """
        if not new_cost:
            return

        if not self._accumulated_cost:
            self._accumulated_cost = new_cost
            return

        self._accumulated_cost.prompt_tokens_cost = new_cost.prompt_tokens_cost
        self._accumulated_cost.completion_tokens_cost += new_cost.completion_tokens_cost

    async def _process_query(self, query: str) -> None:
        """Process a user query through the agent system.

        Args:
            query: The user's input query.
        """
        try:
            agent = self._active_agent or self.agent
            stream = self.chat.send_message(query, agent=agent)

            async for event in stream:
                self._event_handler.on_event(event)
                if event.type == "agent_switch":
                    self._active_agent = event.next_agent

            chat_response = await stream.get_return_value()
            result = chat_response.agent_execution
            for response in result.agent_responses:
                if response.usage:
                    self._update_usage(response.usage)
                if response.response_cost:
                    self._update_cost(response.response_cost)

            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}", file=sys.stderr)

    async def _optimize_context(self, args: str) -> None:
        """Optimize conversation context using the specified strategy.

        Args:
            args: Raw command arguments string.
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

            log_verbose(f"Optimizing context with {parsed}")

            messages = await self.chat.get_messages()
            if not messages:
                print("\n❌ No messages to optimize")
                return

            optimized = await self.chat.optimize_messages(
                model=parsed.model or self.agent.llm.model,
                strategy=parsed.strategy,
                rag_config=RAGStrategyConfig(
                    query=parsed.query,
                    max_messages=10,
                    score_threshold=0.5,
                ),
            )

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

        Args:
            args: Raw command arguments string.
        """
        try:
            parser = self._create_find_parser()
            parsed = self._parse_command_args(parser, args, join_args=["query"])
            if not parsed:
                print("\nUsage examples:")
                print('  /find --query "calendar view" --count 5')
                print('  /find -q "search term" -n 3 -t 0.7')
                print("  /find --query calendar view --threshold 0.8")
                print("  /find -q calendar view -n 3 --threshold 0.6")
                return

            messages = await self.chat.search_messages(
                query=parsed.query,
                max_results=parsed.count,
                score_threshold=parsed.threshold,
                index_messages=True,
            )

            if not messages:
                print("\n❌ No relevant messages found")
                return

            print(f"\n🔍 Found {len(messages)} relevant messages:")
            for msg in messages:
                if msg.role != "system":
                    content = msg.content or "[No content]"
                    print(f"\n[{msg.role}]: {content}")

            print("\n" + "=" * 50 + "\n")

        except Exception as e:
            print(f"\n❌ Error finding relevant messages: {str(e)}")
            print("\nUsage examples:")
            print('  /find --query "calendar view" --count 5')
            print('  /find -q "search term" -n 3 -t 0.7')
            print("  /find --query calendar view --threshold 0.8")
            print("  /find -q calendar view -n 3 --threshold 0.6")

    async def run(self) -> NoReturn:
        """Run the REPL loop until explicitly terminated."""
        await self._print_welcome()

        while True:
            try:
                user_input = await prompt("\n🗣️  Enter your query: ")

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    log_verbose(f"Handling command: {user_input}")
                    if await self._handle_command(user_input):
                        sys.exit(0)

                    continue

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
    memory: LiteChatMemory | None = None,
    search: LiteChatSearch | None = None,
    optimization: LiteChatOptimization | None = None,
    include_usage: bool = False,
    include_cost: bool = False,
    max_iterations: int = sys.maxsize,
    enable_logging: bool = True,
    log_level: LogLevel = "INFO",
) -> NoReturn:
    """Start an interactive REPL session with the specified agent.

    Args:
        agent: Initial agent for handling conversations.
        memory: Memory manager for the chat.
        search: Search manager for the chat.
        optimization: Optimization manager for the chat.
        include_usage: Whether to track token usage.
        include_cost: Whether to track costs.
        max_iterations: Maximum conversation turns.
        enable_logging: Whether to enable logging.
        log_level: Log level to use for logging.

    Example:
        ```python
        agent = Agent(
            id="helper",
            instructions="You are a helpful assistant.",
            llm=LLM(model="gpt-4o"),
        )

        await start_repl(agent=agent, include_usage=True)
        ```
    """
    if enable_logging:
        liteswarm_enable_logging(default_level=log_level)

    repl = AgentRepl(
        agent=agent,
        memory=memory,
        search=search,
        optimization=optimization,
        include_usage=include_usage,
        include_cost=include_cost,
        max_iterations=max_iterations,
    )

    await repl.run()
