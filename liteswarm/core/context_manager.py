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
from liteswarm.core.message_store import MessageStore
from liteswarm.types.context import ContextVariables
from liteswarm.types.context_manager import RAGStrategyConfig
from liteswarm.types.llm import LLM
from liteswarm.types.messages import MessageRecord, MessageT
from liteswarm.types.swarm import Agent, Message
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import dump_messages, filter_tool_call_pairs, trim_messages
from liteswarm.utils.unwrap import unwrap_instructions

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


class ContextManager(Protocol):
    """Protocol for managing agent execution contexts.

    Manages the context provided to language models during agent execution. Creates
    appropriate context from available messages. Optimizes context size to fit model
    limits. Provides semantic search capabilities for finding relevant messages.

    The manager is responsible for fetching and managing context on its own.
    Typically this would be done using a MessageStore, but implementations are free
    to source context in any way that suits their needs.

    Core responsibilities include context creation, optimization, and relevance
    filtering. The manager ensures proper message ordering and handles dynamic
    resolution of agent instructions.

    Examples:
        Basic usage with message store:
            ```python
            class BasicContextManager(ContextManager):
                def __init__(self, message_store: MessageStore[Any]) -> None:
                    self.message_store = message_store
                    self.window_size = 50

                async def create_context(
                    self,
                    agent: Agent,
                    prompt: str | None = None,
                    context_variables: ContextVariables | None = None,
                ) -> list[Message]:
                    # Get history from store
                    history = await self.message_store.get_messages()

                    # Create context with system message and history
                    messages = [Message(role="system", content=agent.instructions), *history]

                    # Add prompt if provided
                    if prompt:
                        messages.append(Message(role="user", content=prompt))
                    return messages

                async def optimize_context(
                    self,
                    model: str,
                    strategy: str | None = None,
                ) -> list[Message]:
                    messages = await self.message_store.get_messages()
                    # Keep only recent messages in window
                    return messages[-self.window_size :]

                async def find_context(
                    self,
                    query: str,
                    max_messages: int | None = None,
                ) -> list[Message]:
                    messages = await self.message_store.get_messages()
                    # In practice, you'd use semantic search here
                    return messages[-10:]  # Return 10 most recent for simplicity
            ```
    """

    async def create_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        """Create an execution context for an agent.

        Prepares a list of messages suitable for the agent's next execution.
        Resolves agent instructions with provided context variables. Combines
        system message, relevant history, and optional user prompt.

        Args:
            agent: Agent requiring context creation.
            prompt: Optional user prompt to include.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            List of messages ready for execution.

        Examples:
            Basic usage:
                ```python
                context = await manager.create_context(
                    agent=agent,
                    prompt="Hello!",
                )
                ```

            With context variables:
                ```python
                context = await manager.create_context(
                    agent=agent,
                    prompt="Help with task",
                    context_variables=ContextVariables(
                        user_name="Alice",
                        task_type="analysis",
                    ),
                )
                ```
        """
        ...

    async def optimize_context(
        self,
        model: str,
        strategy: Any | None = None,
    ) -> list[Message]:
        """Optimize context to fit within model limits.

        Reduces context size to fit within model token limits. Preserves important
        information based on the chosen strategy. Ensures system messages remain
        at the start of context.

        Args:
            model: Model identifier for context limits.
            strategy: Optimization strategy to use.

        Returns:
            Optimized list of messages that fits model context.

        Examples:
            Basic optimization:
                ```python
                optimized = await manager.optimize_context(
                    model="gpt-4o",
                    strategy="window",
                )
                ```
        """
        ...

    async def find_context(
        self,
        query: str,
        context: Sequence[Message] | None = None,
        max_messages: int | None = None,
        score_threshold: float | None = None,
        embedding_model: str | None = None,
    ) -> list[Message]:
        """Find messages matching the search query.

        Searches through available messages to find those matching the query.
        Uses semantic search or other relevance metrics based on the
        implementation. Preserves message relationships in the results.

        Args:
            query: Search query text.
            context: Optional context to search within.
            max_messages: Optional maximum messages to return.
            score_threshold: Optional minimum score to return.
            embedding_model: Optional model for computing embeddings.

        Returns:
            List of matching messages.

        Examples:
            Basic search:
                ```python
                matches = await manager.find_context(
                    query="How do I deploy?",
                    max_messages=10,
                )
                ```
        """
        ...


class LiteContextManager(ContextManager):
    """Default implementation of context management.

    Uses MessageStore as the source of context for all operations. Provides
    multiple strategies for context optimization and semantic search capabilities.

    Context optimization strategies:
    - "trim": Token-based trimming that preserves message order and relationships
    - "window": Sliding window that keeps N most recent messages
    - "summarize": Creates concise summaries of older messages while preserving recent ones
    - "rag": Semantic search with query-based optimization, falls back to trim if no query

    Context search is implemented using a MessageIndex for semantic operations.
    The index computes embeddings for messages and performs similarity search
    to find the most relevant context. System messages are always preserved
    in search results.

    All context operations fetch messages from the underlying MessageStore:
    - create_context gets history and adds system message
    - optimize_context fetches and processes messages using chosen strategy
    - find_context searches through stored messages using semantic index
    """

    def __init__(
        self,
        message_store: MessageStore[Any],
        message_index: MessageIndex | None = None,
        optimization_llm: LLM | None = None,
        window_size: int = 50,
        preserve_recent: int = 25,
        relevant_window_size: int = 10,
        relevant_score_threshold: float = 0.5,
        chunk_size: int = 10,
        default_optimization_strategy: LiteOptimizationStrategy = "trim",
        default_embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the context manager.

        Args:
            message_store: Store for message persistence.
            message_index: Index for semantic search operations.
            optimization_llm: Language model used for context summarization.
            window_size: Maximum messages in sliding window strategy.
            preserve_recent: Messages to preserve when summarizing.
            relevant_window_size: Maximum messages to return in search results.
            relevant_score_threshold: Minimum score to return in search results.
            chunk_size: Number of messages per summary chunk.
            default_optimization_strategy: Default strategy for context optimization.
            default_embedding_model: Default model for computing embeddings.
        """
        self.message_store = message_store
        self.message_index = message_index or LiteMessageIndex()
        self.optimization_llm = optimization_llm or LLM(model="gpt-4o")
        self.window_size = window_size
        self.preserve_recent = preserve_recent
        self.relevant_window_size = relevant_window_size
        self.relevant_score_threshold = relevant_score_threshold
        self.chunk_size = chunk_size
        self.default_optimization_strategy = default_optimization_strategy
        self.default_embedding_model = default_embedding_model

    # ================================================
    # MARK: Message Processing
    # ================================================

    def _split_messages(
        self,
        messages: Sequence[MessageT],
    ) -> tuple[list[MessageT], list[MessageT]]:
        """Split messages into system and non-system groups.

        Args:
            messages: Messages to split.

        Returns:
            Tuple of (system_messages, non_system_messages).
        """
        system_messages: list[MessageT] = []
        non_system_messages: list[MessageT] = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        return system_messages, non_system_messages

    def _create_message_chunks(
        self,
        messages: Sequence[MessageT],
    ) -> list[list[MessageT]]:
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

        chunks: list[list[MessageT]] = []
        current_chunk: list[MessageT] = []
        pending_tool_calls: dict[str, MessageT] = {}

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

    async def _summarize_chunk(self, messages: Sequence[Message]) -> str:
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
            model=self.optimization_llm.model,
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
        messages: Sequence[Message],
        model: str,
        trim_ratio: float = 0.75,
    ) -> list[Message]:
        """Optimize context using token-based trimming.

        Preserves message order and tool call pairs while fitting
        within model's token limit.

        Args:
            messages: Messages to optimize.
            model: Model identifier to determine context limits.
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

    async def _window_strategy(
        self,
        messages: Sequence[Message],
        model: str,
    ) -> list[Message]:
        """Keep only the most recent messages.

        Maintains chronological order and tool call pairs while
        limiting context to window size.

        Args:
            messages: Messages to optimize.
            model: Model identifier to determine context limits.

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
        trimmed = trim_messages(filtered, model)

        log_verbose(
            f"Window strategy reduced messages from {len(messages)} to {len(trimmed.messages)}",
            level="DEBUG",
        )

        return trimmed.messages

    async def _summarize_strategy(
        self,
        messages: Sequence[Message],
        model: str,
    ) -> list[Message]:
        """Summarize older messages while preserving recent ones.

        Creates concise summaries of older messages while keeping
        recent messages intact.

        Args:
            messages: Messages to optimize.
            model: Model identifier to determine context limits.

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
        trimmed = trim_messages(combined_messages, model)
        return trimmed.messages

    async def _rag_strategy(
        self,
        messages: Sequence[Message],
        model: str,
        config: RAGStrategyConfig | None = None,
    ) -> list[Message]:
        """Optimize context using semantic search.

        Uses query-based relevance when available, falls back to
        trimming when no query is provided.

        Args:
            messages: Messages to optimize.
            model: Model identifier to determine context limits.
            config: Configuration for RAG strategy.

        Returns:
            Optimized messages based on relevance.
        """
        if config is None:
            log_verbose(
                "No query provided, falling to trim strategy",
                level="DEBUG",
            )
            return await self._trim_strategy(messages, model)

        log_verbose(
            f"Searching for relevant messages with query: {config.query}",
            level="DEBUG",
        )

        relevant = await self.find_context(
            context=messages,
            query=config.query,
            max_messages=config.max_messages,
            score_threshold=config.score_threshold,
            embedding_model=config.embedding_model,
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
    async def create_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        """Create an execution context for an agent.

        Prepares a list of messages suitable for the agent's next execution.
        Resolves agent instructions with provided context variables. Combines
        system message, relevant history, and optional user prompt.

        Args:
            agent: Agent requiring context creation.
            prompt: Optional user prompt to include.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            List of messages ready for execution.

        Examples:
            Basic usage:
                ```python
                context = await manager.create_context(
                    agent=agent,
                    prompt="Hello!",
                )
                ```

            With context variables:
                ```python
                context = await manager.create_context(
                    agent=agent,
                    prompt="Help with task",
                    context_variables=ContextVariables(
                        user_name="Alice",
                        task_type="analysis",
                    ),
                )
                ```
        """
        instructions = unwrap_instructions(agent.instructions, context_variables)
        history = await self.message_store.get_messages()
        history = [msg for msg in history if msg.role != "system"]

        messages = [Message(role="system", content=instructions), *history]
        if prompt:
            messages.append(Message(role="user", content=prompt))

        return messages

    @override
    async def optimize_context(
        self,
        model: str,
        strategy: LiteOptimizationStrategy | None = None,
        rag_config: RAGStrategyConfig | None = None,
    ) -> list[Message]:
        """Optimize context to fit model limits.

        Reduces context size to fit within model token limits. Applies the chosen
        strategy to preserve the most important information. Maintains message
        relationships and ensures proper ordering in the result.

        Available strategies:
        - "trim": Token-based trimming without summarization
        - "window": Keep N most recent messages
        - "summarize": Summarize older messages, keep recent ones
        - "rag": Semantic search with query-based optimization

        Args:
            model: Model identifier to determine context limits.
            strategy: Optimization strategy to use.
            rag_config: Configuration for RAG strategy.

        Returns:
            Optimized list of messages.

        Examples:
            Basic optimization:
                ```python
                optimized = await manager.optimize_context(
                    model="gpt-4o",
                    strategy="window",
                )
                ```
        """
        messages = await self.message_store.get_messages()
        system_messages, non_system_messages = self._split_messages(messages)
        strategy = strategy or self.default_optimization_strategy

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

        optimized_messages = [*system_messages, *optimized]
        await self.message_store.set_messages(optimized_messages)

        return optimized_messages

    @override
    async def find_context(
        self,
        query: str,
        context: Sequence[Message] | None = None,
        max_messages: int | None = None,
        score_threshold: float | None = None,
        embedding_model: str | None = None,
    ) -> list[Message]:
        """Find messages matching the search query.

        Searches through available messages to find those matching the query.
        Uses semantic search or other relevance metrics based on the
        implementation. Preserves message relationships in the results.

        Args:
            query: Search query text.
            context: Optional context to search within.
            max_messages: Optional maximum messages to return.
            score_threshold: Optional minimum score to return.
            embedding_model: Optional model for computing embeddings.

        Returns:
            List of matching messages.

        Examples:
            Basic search:
                ```python
                matches = await manager.find_context(
                    query="How do I deploy?",
                    max_messages=10,
                )
                ```
        """
        if context is not None:
            messages = [MessageRecord.from_message(msg) for msg in context]
        else:
            messages = await self.message_store.get_messages()

        system_messages, non_system_messages = self._split_messages(messages)
        embedding_model = embedding_model or self.default_embedding_model

        if not non_system_messages:
            return [*system_messages]

        log_verbose(
            f"Searching for relevant messages with query: '{query}' using {embedding_model}",
            level="DEBUG",
        )

        await self.message_index.index(non_system_messages)
        result = await self.message_index.search(
            query=query,
            max_results=max_messages or self.relevant_window_size,
            score_threshold=score_threshold or self.relevant_score_threshold,
        )

        relevant_messages = [msg for msg, _ in result]

        log_verbose(
            f"Found {len(relevant_messages)} relevant messages",
            level="DEBUG",
        )

        return [*system_messages, *relevant_messages]
