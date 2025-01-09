# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
from typing import Any, Literal, Protocol

from litellm import acompletion
from typing_extensions import override

from liteswarm.chat.memory import ChatMemory
from liteswarm.chat.search import ChatSearch
from liteswarm.types.chat import ChatMessage, RAGStrategyConfig
from liteswarm.types.llm import LLM
from liteswarm.types.swarm import Message
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import dump_messages, filter_tool_call_pairs, trim_messages

OptimizationStrategy = Literal["trim", "window", "summarize", "rag"]
"""Available context optimization strategies."""

SUMMARIZER_SYSTEM_PROMPT = """\
You are a precise conversation summarizer that distills complex interactions into essential points.

Your summaries must capture:
- Key decisions and outcomes
- Essential context needed for future interactions
- Tool calls and their results
- Important user requirements or constraints

Focus on factual information and exclude:
- Greetings and acknowledgments
- Routine interactions
- Redundant information
- Conversational fillers

Be extremely concise while preserving all critical details.\
"""

SUMMARIZER_USER_PROMPT = """\
Create a 2-3 sentence summary of this conversation segment that captures only:
1. Key decisions and actions taken
2. Essential context for future reference
3. Important tool interactions and their outcomes

Be direct and factual. Exclude any unnecessary details or pleasantries.\
"""


class ChatOptimization(Protocol):
    """Protocol for managing conversation context optimization.

    Provides a standard interface for optimizing chat context to maintain
    efficient memory usage and improve response quality. Supports various
    optimization strategies to handle long conversations effectively.

    The protocol is designed to:
        - Reduce context size while preserving critical information
        - Support multiple optimization strategies
        - Maintain message ordering and relationships
        - Handle tool calls and system messages appropriately

    Examples:
        ```python
        class MyOptimizer(ChatOptimization):
            def __init__(self) -> None:
                self._memory = {}

            async def optimize_context(
                self,
                model: str,
                strategy: str | None = None,
            ) -> list[ChatMessage]:
                # Implement optimization logic
                messages = await self.memory.get_messages()
                return self._apply_strategy(messages, strategy)


        # Use custom optimizer
        optimizer = MyOptimizer()
        optimized = await optimizer.optimize_context(
            model="gpt-4o",
            strategy="summarize",
        )
        ```

    Notes:
        - Implementations should preserve message relationships
        - System messages require special handling
        - Tool call pairs must be kept together
        - All operations are asynchronous by framework design
    """

    async def optimize_context(
        self,
        model: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Optimize conversation context.

        Args:
            model: Target language model identifier.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            Optimized list of messages.
        """
        ...


class LiteChatOptimization(ChatOptimization):
    """In-memory implementation of chat context optimization.

    Provides multiple strategies for efficient context management with
    support for token-based trimming, windowing, summarization, and
    relevance-based filtering through RAG.

    The implementation offers:
        - Token-based trimming that preserves message order
        - Sliding window that keeps N most recent messages
        - Summarization of older messages with recent preservation
        - Semantic search with query-based optimization

    Examples:
        ```python
        optimizer = LiteChatOptimization(
            memory=memory,
            search=search,
            window_size=50,
            preserve_recent=25,
        )

        # Use token-based trimming
        trimmed = await optimizer.optimize_context(
            model="gpt-4o",
            strategy="trim",
        )

        # Use semantic search
        relevant = await optimizer.optimize_context(
            model="gpt-4o",
            strategy="rag",
            rag_config=RAGStrategyConfig(
                query="project requirements",
                max_messages=20,
            ),
        )
        ```

    Notes:
        - System messages are preserved across all strategies
        - Tool call pairs are kept together during optimization
        - Summarization uses a specialized LLM prompt
        - RAG requires proper vector search setup
    """

    def __init__(
        self,
        memory: ChatMemory,
        search: ChatSearch,
        optimization_llm: LLM | None = None,
        window_size: int = 50,
        preserve_recent: int = 25,
        chunk_size: int = 10,
        default_strategy: OptimizationStrategy = "trim",
    ) -> None:
        """Initialize a new context optimizer instance.

        Creates an optimizer with configurable parameters for different
        optimization strategies. Each strategy can be fine-tuned through
        its specific parameters while maintaining consistent behavior.

        Args:
            memory: Storage for message persistence and retrieval.
            search: Search functionality for semantic retrieval.
            optimization_llm: LLM for summarization, defaults to gpt-4o.
            window_size: Maximum messages in sliding window.
            preserve_recent: Messages to keep during summarization.
            chunk_size: Messages per summary chunk.
            default_strategy: Default strategy ("trim", "window", "summarize", "rag").

        Notes:
            - Memory and search are required
            - Window size must account for tool call pairs
            - Preserve recent should be less than window size
        """
        self._memory = memory
        self._search = search
        self._optimization_llm = optimization_llm or LLM(model="gpt-4o")
        self._window_size = window_size
        self._preserve_recent = preserve_recent
        self._chunk_size = chunk_size
        self._default_strategy = default_strategy

    def _split_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Split messages into system and non-system groups.

        Separates system messages from regular conversation messages to
        ensure proper handling during optimization.

        Args:
            messages: List of messages to split.

        Returns:
            Tuple of (system_messages, non_system_messages).
        """
        system_messages: list[ChatMessage] = []
        non_system_messages: list[ChatMessage] = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        return system_messages, non_system_messages

    def _create_message_chunks(
        self,
        messages: list[ChatMessage],
    ) -> list[list[ChatMessage]]:
        """Create chunks of messages for summarization.

        Groups messages into chunks while preserving tool call pairs and
        maintaining proper message relationships. Ensures that tool calls
        and their results stay together.

        Args:
            messages: List of messages to chunk.

        Returns:
            List of message chunks ready for summarization.
        """
        if not messages:
            return []

        chunks: list[list[ChatMessage]] = []
        current_chunk: list[ChatMessage] = []
        pending_tool_calls: dict[str, ChatMessage] = {}

        def add_chunk() -> None:
            if current_chunk:
                filtered_chunk = filter_tool_call_pairs(current_chunk)
                if filtered_chunk:
                    chunks.append(filtered_chunk)
                current_chunk.clear()
                pending_tool_calls.clear()

        def add_chunk_if_needed() -> None:
            if len(current_chunk) >= self._chunk_size and not pending_tool_calls:
                add_chunk()

        for message in messages:
            add_chunk_if_needed()

            if message.role == "assistant" and message.tool_calls:
                current_chunk.append(message)
                for tool_call in message.tool_calls:
                    if tool_call.id:
                        pending_tool_calls[tool_call.id] = message

            elif message.role == "tool" and message.tool_call_id:
                current_chunk.append(message)
                pending_tool_calls.pop(message.tool_call_id, None)
                add_chunk_if_needed()

            else:
                current_chunk.append(message)
                add_chunk_if_needed()

        if current_chunk:
            add_chunk()

        return chunks

    async def _summarize_chunk(
        self,
        messages: list[ChatMessage],
    ) -> str:
        """Create a concise summary of a message chunk.

        Uses the optimization LLM to generate a focused summary that
        captures key information while removing unnecessary details.

        Args:
            messages: List of messages to summarize.

        Returns:
            Concise summary of the message chunk.
        """
        log_verbose(
            f"Summarizing chunk of {len(messages)} messages",
            level="DEBUG",
        )

        system_message = Message(role="system", content=SUMMARIZER_SYSTEM_PROMPT)
        user_message = Message(role="user", content=SUMMARIZER_USER_PROMPT)

        input_messages: list[ChatMessage] = [
            ChatMessage.from_message(system_message),
            *messages,
            ChatMessage.from_message(user_message),
        ]

        response = await acompletion(
            model=self._optimization_llm.model,
            messages=dump_messages(input_messages),
        )

        summary = response.choices[0].message.content or "No summary available."

        log_verbose(
            f"Generated summary of length {len(summary)}",
            level="DEBUG",
        )

        return summary

    async def _trim_strategy(
        self,
        messages: list[ChatMessage],
        model: str,
        trim_ratio: float = 0.75,
    ) -> list[ChatMessage]:
        """Optimize context using token-based trimming.

        Reduces context size by removing messages while preserving order
        and maintaining token count within model limits.

        Args:
            messages: List of messages to trim.
            model: Target language model identifier.
            trim_ratio: Target ratio of max token limit to preserve.

        Returns:
            Trimmed list of messages.
        """
        log_verbose(
            f"Trimming messages to {trim_ratio:.0%} of {model} context limit",
            level="DEBUG",
        )

        trimmed = trim_messages(
            messages=list(messages),
            model=model,
            trim_ratio=trim_ratio,
        )

        log_verbose(
            f"Trimmed messages from {len(messages)} to {len(trimmed.messages)}",
            level="DEBUG",
        )

        return trimmed.messages

    async def _window_strategy(
        self,
        messages: list[ChatMessage],
        model: str,
    ) -> list[ChatMessage]:
        """Keep only the most recent messages.

        Maintains a sliding window of recent messages while ensuring
        tool call pairs stay together and context fits model limits.

        Args:
            messages: List of messages to window.
            model: Target language model identifier.

        Returns:
            Windowed list of messages.
        """
        if len(messages) <= self._window_size:
            return list(messages)

        log_verbose(
            f"Applying window strategy with size {self._window_size}",
            level="DEBUG",
        )

        recent = list(messages[-self._window_size :])
        filtered = filter_tool_call_pairs(recent)
        trimmed = trim_messages(filtered, model)

        log_verbose(
            f"Window strategy reduced messages from {len(messages)} to {len(trimmed.messages)}",
            level="DEBUG",
        )

        return trimmed.messages

    async def _summarize_strategy(
        self,
        messages: list[ChatMessage],
        model: str,
    ) -> list[ChatMessage]:
        """Summarize older messages while preserving recent ones.

        Creates concise summaries of older messages while keeping recent
        messages intact. Ensures proper handling of tool calls and system
        messages.

        Args:
            messages: List of messages to process.
            model: Target language model identifier.

        Returns:
            List of messages with older ones summarized.
        """
        if len(messages) <= self._preserve_recent:
            return list(messages)

        to_preserve = filter_tool_call_pairs(list(messages[-self._preserve_recent :]))
        to_summarize = filter_tool_call_pairs(list(messages[: -self._preserve_recent]))

        if not to_summarize:
            return to_preserve

        chunks = self._create_message_chunks(to_summarize)
        summaries = await asyncio.gather(*[self._summarize_chunk(chunk) for chunk in chunks])

        summary_message = Message(
            role="assistant",
            content=f"Previous conversation summary:\n{' '.join(summaries)}",
        )

        combined_messages = [ChatMessage.from_message(summary_message), *to_preserve]
        trimmed = trim_messages(combined_messages, model)
        return trimmed.messages

    async def _rag_strategy(
        self,
        messages: list[ChatMessage],
        model: str,
        config: RAGStrategyConfig | None = None,
    ) -> list[ChatMessage]:
        """Optimize context using semantic search.

        Uses query-based retrieval to find relevant messages from the
        conversation history. Falls back to trimming if search fails
        or no configuration is provided.

        Args:
            messages: List of messages to process.
            model: Target language model identifier.
            config: RAG strategy configuration.

        Returns:
            List of relevant messages fitting context limits.
        """
        if not config or not config.query:
            log_verbose(
                "No query provided, falling back to trim strategy",
                level="DEBUG",
            )
            return await self._trim_strategy(messages, model)

        log_verbose(
            f"Searching for relevant messages with query: {config.query}",
            level="DEBUG",
        )

        search_results = await self._search.search(
            query=config.query or "",
            max_results=config.max_messages,
            score_threshold=config.score_threshold,
        )

        if not search_results:
            log_verbose(
                "No relevant messages found, falling to trim strategy",
                level="DEBUG",
            )
            return await self._trim_strategy(messages, model)

        log_verbose(
            "Trimming relevant messages to fit context",
            level="DEBUG",
        )

        relevant = [msg for msg, _ in search_results]
        return await self._trim_strategy(relevant, model)

    @override
    async def optimize_context(
        self,
        model: str,
        strategy: OptimizationStrategy | None = None,
        rag_config: RAGStrategyConfig | None = None,
    ) -> list[ChatMessage]:
        """Optimize conversation context using specified strategy.

        Applies the selected optimization strategy to reduce context size
        while preserving important information and relationships. Handles
        system messages separately and ensures proper message ordering.

        Args:
            model: Target language model identifier.
            strategy: Optimization strategy to use.
            rag_config: Configuration for RAG strategy.

        Returns:
            Optimized list of messages.

        Raises:
            ValueError: If unknown strategy is specified.
        """
        messages = await self._memory.get_messages()
        system_messages, non_system_messages = self._split_messages(messages)
        strategy = strategy or self._default_strategy

        log_verbose(
            f"Optimizing context with strategy '{strategy}' for model {model}",
            level="DEBUG",
        )

        match strategy:
            case "trim":
                optimized = await self._trim_strategy(non_system_messages, model)
            case "window":
                optimized = await self._window_strategy(non_system_messages, model)
            case "summarize":
                optimized = await self._summarize_strategy(non_system_messages, model)
            case "rag":
                optimized = await self._rag_strategy(
                    messages=non_system_messages,
                    model=model,
                    config=rag_config,
                )
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

        log_verbose(
            f"Context optimized from {len(non_system_messages)} to {len(optimized)} messages",
            level="DEBUG",
        )

        return [*system_messages, *optimized]
