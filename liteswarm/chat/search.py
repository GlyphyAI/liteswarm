# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Protocol

from typing_extensions import override

from liteswarm.chat.index import LiteMessageIndex
from liteswarm.chat.memory import ChatMemory
from liteswarm.types.chat import ChatMessage


class ChatSearch(Protocol):
    """Protocol for managing semantic search capabilities in chat conversations.

    Defines a standard interface for semantic search operations that can be
    implemented by different search backends. Supports indexing and searching
    messages within isolated conversation contexts.

    Examples:
        ```python
        class MySearch(ChatSearch):
            def __init__(self) -> None:
                self._index = {}

            async def index(self) -> None:
                # Index new messages
                messages = await self.memory.get_messages()
                await self._update_index(messages)

            async def search(
                self,
                query: str,
                max_results: int | None = None,
            ) -> list[tuple[ChatMessage, float]]:
                # Find semantically similar messages
                return await self._compute_similarity(
                    query=query,
                    limit=max_results,
                )


        # Use custom search
        search = MySearch()
        await search.index()
        results = await search.search(
            query="project requirements",
            max_results=5,
        )
        ```

    Notes:
        - Implementations must handle concurrent access
        - Index updates should be atomic
        - All operations are asynchronous by framework design
        - Search results should be ordered by relevance
    """

    async def index(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Index messages for semantic search.

        Updates the search index with current messages. Should be called
        after adding new messages to ensure they are searchable.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[ChatMessage, float]]:
        """Search for semantically similar messages.

        Finds messages that are semantically similar to the query.
        Results are sorted by similarity score and can be filtered
        by score threshold.

        Args:
            query: Text to search for similar messages.
            max_results: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of (message, score) tuples sorted by descending score.
        """
        ...

    async def clear(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Clear search indices.

        Removes search indices and frees associated memory.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...


class LiteChatSearch(ChatSearch):
    """In-memory implementation of semantic search using vector embeddings.

    Provides efficient semantic search capabilities using vector embeddings
    to represent messages and compute similarities. Each instance maintains
    its own index for conversation isolation.

    The implementation offers:
        - Fast in-memory vector search
        - Automatic embedding computation
        - Configurable similarity thresholds
        - Conversation isolation
        - Efficient batch processing

    Examples:
        ```python
        # Create search instance
        memory = LiteChatMemory()
        search = LiteChatSearch(
            memory=memory,
            embedding_model="text-embedding-3-small",
            embedding_batch_size=16,
        )

        # Index and search messages
        await search.index()
        results = await search.search(
            query="project requirements",
            max_results=5,
            score_threshold=0.7,
        )

        # Process results
        for message, score in results:
            print(f"Score {score:.2f}: {message.content}")
        ```

    Notes:
        - Indices are volatile and reset on restart
        - Embedding computation may impact latency
        - Higher batch sizes improve indexing speed
        - Index updates require full recomputation
    """

    def __init__(
        self,
        memory: ChatMemory,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 16,
    ) -> None:
        """Initialize a new search instance.

        Creates a search manager with configurable embedding settings.
        Initializes empty indices that will be populated on demand.

        Args:
            memory: Storage backend for message access.
            embedding_model: Model for computing text embeddings.
            embedding_batch_size: Messages to embed in parallel.
        """
        self._memory = memory
        self._embedding_model = embedding_model
        self._embedding_batch_size = embedding_batch_size
        self._index = LiteMessageIndex(
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
        )

    @override
    async def index(self) -> None:
        """Index all messages in this conversation.

        Retrieves messages from storage and updates the vector index.
        Creates a new index if none exists.
        """
        messages = await self._memory.get_messages()
        if messages:
            await self._index.index(messages)

    @override
    async def search(
        self,
        query: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ChatMessage, float]]:
        """Search for similar messages in this conversation.

        Finds messages that are semantically similar to the query text.
        Returns empty list if no index exists or no matches found.

        Args:
            query: Text to find similar messages for.
            max_results: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of (message, score) tuples sorted by score.

        Notes:
            Index should be updated before search if messages changed.
        """
        return await self._index.search(
            query=query,
            max_results=max_results,
            score_threshold=score_threshold,
        )

    @override
    async def clear(self) -> None:
        """Clear search index.

        Removes vector index and frees associated memory.
        """
        await self._index.clear()
