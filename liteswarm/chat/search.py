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
    """Protocol for managing semantic search capabilities in chat sessions.

    Defines a standard interface for semantic search operations that can be
    implemented by different search backends. Supports indexing and searching
    messages within isolated session contexts.

    Examples:
        ```python
        class MySearch(ChatSearch):
            async def index(self, session_id: str) -> None:
                # Index new messages
                messages = await self.memory.get_messages(session_id)
                await self._update_index(messages)

            async def search(
                self,
                query: str,
                session_id: str,
                max_results: int | None = None,
            ) -> list[tuple[ChatMessage, float]]:
                # Find semantically similar messages
                return await self._compute_similarity(
                    query=query,
                    session_id=session_id,
                    limit=max_results,
                )


        # Use custom search
        search = MySearch()
        await search.index(session_id="session_123")
        results = await search.search(
            query="project requirements",
            session_id="session_123",
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
        session_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Index session messages for semantic search.

        Updates the search index with current messages from the specified
        session. Should be called after adding new messages to ensure
        they are searchable.

        Args:
            session_id: Session to index messages for.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def search(
        self,
        query: str,
        session_id: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[ChatMessage, float]]:
        """Search for semantically similar messages.

        Finds messages in the specified session that are semantically
        similar to the query. Results are sorted by similarity score
        and can be filtered by score threshold.

        Args:
            query: Text to search for similar messages.
            session_id: Session to search within.
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
        session_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Clear search indices.

        Removes search indices for the specified session or all sessions
        if no session ID is provided. Useful for cleanup and maintenance.

        Args:
            session_id: Session to clear, or None for all sessions.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...


class LiteChatSearch(ChatSearch):
    """In-memory implementation of semantic search using vector embeddings.

    Provides efficient semantic search capabilities using vector embeddings
    to represent messages and compute similarities. Maintains separate
    indices per session for isolation and concurrent access.

    The implementation offers:
        - Fast in-memory vector search
        - Automatic embedding computation
        - Configurable similarity thresholds
        - Per-session index isolation
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
        await search.index(session_id="session_123")
        results = await search.search(
            query="project requirements",
            session_id="session_123",
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
        self._indices: dict[str, LiteMessageIndex] = {}  # session_id -> index

    def _get_or_create_index(self, session_id: str) -> LiteMessageIndex:
        """Get or create a vector index for a session.

        Retrieves an existing index or creates a new one if none exists.
        Ensures consistent embedding settings across index creation.

        Args:
            session_id: Session to get/create index for.

        Returns:
            Vector index instance for the session.
        """
        if session_id not in self._indices:
            self._indices[session_id] = LiteMessageIndex(
                embedding_model=self._embedding_model,
                embedding_batch_size=self._embedding_batch_size,
            )
        return self._indices[session_id]

    @override
    async def index(self, session_id: str) -> None:
        """Index all messages in a session.

        Retrieves messages from storage and updates the vector index.
        Creates a new index if none exists for the session.

        Args:
            session_id: Session to index messages for.
        """
        messages = await self._memory.get_messages(session_id)
        if messages:
            index = self._get_or_create_index(session_id)
            await index.index(messages)

    @override
    async def search(
        self,
        query: str,
        session_id: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ChatMessage, float]]:
        """Search for similar messages in a session.

        Finds messages that are semantically similar to the query text.
        Returns empty list if session has no index or no matches found.

        Args:
            query: Text to find similar messages for.
            session_id: Session to search within.
            max_results: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of (message, score) tuples sorted by score.

        Notes:
            Index should be updated before search if messages changed.
        """
        if session_id not in self._indices:
            return []

        return await self._indices[session_id].search(
            query=query,
            max_results=max_results,
            score_threshold=score_threshold,
        )

    @override
    async def clear(self, session_id: str | None = None) -> None:
        """Clear search indices.

        Removes vector indices and frees associated memory. Cleans up
        either a specific session or all sessions.

        Args:
            session_id: Session to clear, or None for all sessions.
        """
        if session_id is None:
            for index in self._indices.values():
                await index.clear()
            self._indices.clear()
        elif session_id in self._indices:
            await self._indices[session_id].clear()
            del self._indices[session_id]
