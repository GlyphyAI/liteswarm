# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias

from litellm import acompletion
from typing_extensions import override

from liteswarm.core.message_index import LiteMessageIndex, MessageIndex
from liteswarm.types.llm import LLM
from liteswarm.types.messages import MessageRecord
from liteswarm.types.swarm import Message
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import dump_messages, filter_tool_call_pairs, trim_messages


class ContextManager(Protocol):
    """Protocol for managing conversation context size and relevance.

    Provides context optimization and relevance filtering capabilities while
    delegating storage to MessageStore. Supports various optimization strategies
    like summarization, windowing, and semantic search.

    Examples:
        Basic implementation:
            ```python
            class SimpleManager(ContextManager):
                async def optimize(
                    self,
                    messages: Sequence[MessageRecord],
                    model: str,
                ) -> list[MessageRecord]:
                    # Trim to fit model context
                    return await self._trim_messages(messages, model)

                async def get_relevant_context(
                    self,
                    messages: Sequence[MessageRecord],
                    query: str,
                ) -> list[MessageRecord]:
                    # Find relevant messages
                    return await self._search_context(messages, query)
            ```
    """

    async def optimize(
        self,
        messages: Sequence[MessageRecord],
        model: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[MessageRecord]:
        """Optimize context to fit within model limits.

        Reduces context size while preserving important information. System
        messages are always preserved at the start of the context.

        Args:
            messages: Messages to optimize.
            model: Model identifier for context limits.
            *args: Implementation-specific positional arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            Optimized list of messages that fits model context.
        """
        ...

    async def get_relevant_context(
        self,
        messages: Sequence[MessageRecord],
        query: str,
        max_messages: int | None = None,
        embedding_model: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[MessageRecord]:
        """Find messages most relevant to the current query.

        Uses semantic search when embedding model is provided, otherwise
        falls back to recency-based selection.

        Args:
            messages: Messages to search through.
            query: Current conversation query.
            max_messages: Maximum number of messages to return.
            embedding_model: Model for computing embeddings.
            *args: Implementation-specific positional arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of messages most relevant to the query.
        """
        ...


LiteOptimizationStrategy: TypeAlias = Literal["trim", "window", "summarize", "rag"]
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


class LiteContextManager(ContextManager):
    """Lightweight implementation of the ContextManager protocol.

    Provides multiple optimization strategies and semantic search capabilities.
    Suitable for managing context in agentic systems like liteswarm.

    Examples:
        Basic usage:
            ```python
            manager = LiteContextManager()

            # Optimize context for model
            optimized = await manager.optimize(
                messages=messages,
                model="gpt-4o",
                strategy="summarize",
            )

            # Find relevant context
            relevant = await manager.get_relevant_context(
                messages=messages,
                query="How do I deploy?",
                max_messages=10,
            )
            ```
    """

    def __init__(
        self,
        llm: LLM | None = None,
        window_size: int = 50,
        preserve_recent: int = 25,
        relevant_window_size: int = 10,
        chunk_size: int = 10,
        message_index: MessageIndex | None = None,
        default_strategy: LiteOptimizationStrategy = "trim",
        default_embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the context manager.

        Args:
            llm: Language model for summarization.
            window_size: Maximum messages in sliding window.
            preserve_recent: Messages to keep when summarizing.
            relevant_window_size: Maximum relevant messages to return.
            chunk_size: Messages per summary chunk.
            message_index: Index for semantic search.
            default_strategy: Default optimization strategy.
            default_embedding_model: Default model for embeddings.
        """
        self.llm = llm or LLM(model="gpt-4o")
        self.window_size = window_size
        self.preserve_recent = preserve_recent
        self.relevant_window_size = relevant_window_size
        self.chunk_size = chunk_size
        self.message_index = message_index or LiteMessageIndex()
        self.default_strategy = default_strategy
        self.default_embedding_model = default_embedding_model

    # ================================================
    # MARK: Message Processing
    # ================================================

    def _split_messages(
        self,
        messages: Sequence[MessageRecord],
    ) -> tuple[list[MessageRecord], list[MessageRecord]]:
        """Split messages into system and non-system groups.

        Args:
            messages: Messages to split.

        Returns:
            Tuple of (system_messages, non_system_messages).
        """
        system_messages: list[MessageRecord] = []
        non_system_messages: list[MessageRecord] = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        return system_messages, non_system_messages

    def _create_message_chunks(
        self,
        messages: Sequence[MessageRecord],
    ) -> list[list[MessageRecord]]:
        """Create chunks of messages for summarization.

        Preserves tool call/result pairs within chunks and handles
        pending tool calls across chunk boundaries.

        Args:
            messages: Messages to chunk.

        Returns:
            List of message chunks ready for summarization.
        """
        if not messages:
            return []

        chunks: list[list[MessageRecord]] = []
        current_chunk: list[MessageRecord] = []
        pending_tool_calls: dict[str, MessageRecord] = {}

        def add_chunk() -> None:
            if current_chunk:
                filtered_chunk = filter_tool_call_pairs(current_chunk)
                if filtered_chunk:
                    chunks.append(filtered_chunk)
                current_chunk.clear()
                pending_tool_calls.clear()

        def add_chunk_if_needed() -> None:
            if len(current_chunk) >= self.chunk_size and not pending_tool_calls:
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

    # ================================================
    # MARK: Summarization Helpers
    # ================================================

    async def _summarize_chunk(self, messages: Sequence[MessageRecord]) -> str:
        """Create a concise summary of messages.

        Args:
            messages: Messages to summarize.

        Returns:
            Concise summary focusing on key points.
        """
        log_verbose(
            f"Summarizing chunk of {len(messages)} messages",
            level="DEBUG",
        )

        input_messages = [
            Message(role="system", content=SUMMARIZER_SYSTEM_PROMPT),
            *messages,
            Message(role="user", content=SUMMARIZER_USER_PROMPT),
        ]

        response = await acompletion(
            model=self.llm.model,
            messages=dump_messages(input_messages),
        )

        summary = response.choices[0].message.content or "No summary available."

        log_verbose(
            f"Generated summary of length {len(summary)}",
            level="DEBUG",
        )

        return summary

    # ================================================
    # MARK: Optimization Strategies
    # ================================================

    async def _trim_strategy(
        self,
        messages: Sequence[MessageRecord],
        model: str,
        trim_ratio: float = 0.75,
    ) -> list[MessageRecord]:
        """Optimize context using token-based trimming.

        Preserves message order and tool call pairs while fitting
        within model's token limit.

        Args:
            messages: Messages to optimize.
            model: Model to determine token limits.
            trim_ratio: Proportion of model's context to use.

        Returns:
            Trimmed message list that fits model limits.
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

    async def _window_strategy(self, messages: Sequence[MessageRecord]) -> list[MessageRecord]:
        """Keep only the most recent messages.

        Maintains chronological order and tool call pairs while
        limiting context to window size.

        Args:
            messages: Messages to optimize.

        Returns:
            Most recent messages that fit within window.
        """
        if len(messages) <= self.window_size:
            return list(messages)

        log_verbose(
            f"Applying window strategy with size {self.window_size}",
            level="DEBUG",
        )

        recent = list(messages[-self.window_size :])
        filtered = filter_tool_call_pairs(recent)
        trimmed = trim_messages(filtered, self.llm.model)

        log_verbose(
            f"Window strategy reduced messages from {len(messages)} to {len(trimmed.messages)}",
            level="DEBUG",
        )

        return trimmed.messages

    async def _summarize_strategy(self, messages: Sequence[MessageRecord]) -> list[MessageRecord]:
        """Summarize older messages while preserving recent ones.

        Creates concise summaries of older messages while keeping
        recent messages intact.

        Args:
            messages: Messages to optimize.

        Returns:
            Combined summary and recent messages.
        """
        if len(messages) <= self.preserve_recent:
            return list(messages)

        to_preserve = filter_tool_call_pairs(list(messages[-self.preserve_recent :]))
        to_summarize = filter_tool_call_pairs(list(messages[: -self.preserve_recent]))

        if not to_summarize:
            return to_preserve

        chunks = self._create_message_chunks(to_summarize)
        summaries = await asyncio.gather(*[self._summarize_chunk(chunk) for chunk in chunks])

        summary_message = Message(
            role="assistant",
            content=f"Previous conversation summary:\n{' '.join(summaries)}",
        )

        combined_messages = [MessageRecord.from_message(summary_message), *to_preserve]
        trimmed = trim_messages(combined_messages, self.llm.model)
        return trimmed.messages

    async def _rag_strategy(
        self,
        messages: Sequence[MessageRecord],
        model: str,
        query: str | None = None,
    ) -> list[MessageRecord]:
        """Optimize context using semantic search.

        Uses query-based relevance when available, falls back to
        trimming when no query is provided.

        Args:
            messages: Messages to optimize.
            model: Target model for context limits.
            query: Optional query for semantic search.

        Returns:
            Optimized messages based on relevance.
        """
        if not query:
            log_verbose(
                "No query provided, falling to trim strategy",
                level="DEBUG",
            )
            return await self._trim_strategy(messages, model)

        log_verbose(
            f"Searching for relevant messages with query: {query}",
            level="DEBUG",
        )

        relevant = await self.get_relevant_context(
            messages=messages,
            query=query,
            max_messages=self.relevant_window_size,
            embedding_model=self.default_embedding_model,
        )

        if not relevant:
            log_verbose(
                "No relevant messages found, falling to trim strategy",
                level="DEBUG",
            )
            return await self._trim_strategy(messages, model)

        log_verbose(
            "Trimming relevant messages to fit context",
            level="DEBUG",
        )

        return await self._trim_strategy(relevant, model)

    # ================================================
    # MARK: Public API
    # ================================================

    @override
    async def optimize(
        self,
        messages: Sequence[MessageRecord],
        model: str,
        strategy: LiteOptimizationStrategy | None = None,
        query: str | None = None,
    ) -> list[MessageRecord]:
        """Optimize conversation context using selected strategy.

        Available strategies:
        - "trim": Token-based trimming without summarization
        - "window": Keep N most recent messages
        - "summarize": Summarize older messages, keep recent ones
        - "rag": Semantic search with query-based optimization

        Args:
            messages: Messages to optimize.
            model: Model identifier for context limits.
            strategy: Optimization strategy to use.
            query: Optional query for RAG strategy.

        Returns:
            Optimized messages that fit model context.
        """
        system_messages, non_system_messages = self._split_messages(messages)
        strategy = strategy or self.default_strategy

        log_verbose(
            f"Optimizing context with strategy '{strategy}' for model {model}",
            level="DEBUG",
        )

        match strategy:
            case "trim":
                optimized = await self._trim_strategy(non_system_messages, model)
            case "window":
                optimized = await self._window_strategy(non_system_messages)
            case "summarize":
                optimized = await self._summarize_strategy(non_system_messages)
            case "rag":
                optimized = await self._rag_strategy(non_system_messages, model, query)
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

        log_verbose(
            f"Context optimized from {len(non_system_messages)} to {len(optimized)} messages",
            level="DEBUG",
        )

        return [*system_messages, *optimized]

    @override
    async def get_relevant_context(
        self,
        messages: Sequence[MessageRecord],
        query: str,
        max_messages: int | None = None,
        embedding_model: str | None = None,
    ) -> list[MessageRecord]:
        """Find messages most relevant to the current query.

        Uses semantic search with embeddings when model is provided,
        falls back to recency-based selection otherwise.

        Args:
            messages: Messages to search through.
            query: Current query to find context for.
            max_messages: Maximum messages to return.
            embedding_model: Model for computing embeddings.

        Returns:
            Messages most relevant to the query.
        """
        system_messages, non_system_messages = self._split_messages(messages)
        embedding_model = embedding_model or self.default_embedding_model

        if not non_system_messages:
            max_messages = max_messages or self.window_size
            recent = list(non_system_messages[-max_messages:])
            return [*system_messages, *recent]

        log_verbose(
            f"Searching for relevant messages with query: '{query}' using {embedding_model}",
            level="DEBUG",
        )

        await self.message_index.index(non_system_messages)
        result = await self.message_index.search(
            query=query,
            max_results=max_messages or self.window_size,
        )

        relevant_messages = [msg for msg, _ in result]

        log_verbose(
            f"Found {len(relevant_messages)} relevant messages out of {len(non_system_messages)}",
            level="DEBUG",
        )

        return [*system_messages, *relevant_messages]
