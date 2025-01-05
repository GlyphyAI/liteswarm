# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from litellm import aembedding
from numpy.typing import NDArray
from typing_extensions import override

from liteswarm.types.chat import ChatMessage


class MessageIndex(Protocol):
    """Protocol for indexing and searching messages by semantic similarity.

    Provides semantic search capabilities over messages while delegating storage
    to MessageStore implementations. Supports various indexing strategies like
    in-memory arrays, vector indices (FAISS, Annoy), and vector databases.

    Examples:
        Basic usage:
            ```python
            class SimpleIndex(MessageIndex):
                async def index(self, messages: Sequence[MessageRecord]) -> None:
                    # Index messages using embeddings
                    await self._update_index(messages)

                async def search(
                    self,
                    query: str,
                    max_results: int | None = None,
                ) -> list[tuple[MessageRecord, float]]:
                    # Find similar messages
                    return await self._search_index(query, max_results)
            ```
    """

    async def index(
        self,
        messages: Sequence[ChatMessage],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Index new messages or update existing ones.

        Processes messages in batches and updates the index with their embeddings.
        Implementations should handle deduplication and error cases gracefully.

        Args:
            messages: Messages to index. Messages without content are skipped.
            *args: Implementation-specific positional arguments.
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
        """Find messages most semantically similar to the query.

        Computes query embedding and returns messages sorted by similarity score.
        Empty query or index returns empty list.

        Args:
            query: Search query text.
            max_results: Maximum number of results to return.
            score_threshold: Minimum similarity score (0 to 1) to include.
            *args: Implementation-specific positional arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of (message, score) pairs sorted by decreasing similarity.
        """
        ...

    async def clear(self) -> None:
        """Clear the index while preserving original messages.

        Removes all indexed embeddings and resets internal data structures
        without affecting the original messages in the store.
        """
        ...


class LiteMessageIndex(MessageIndex):
    """Simple in-memory implementation of the MessageIndex protocol.

    Uses numpy arrays for storing embeddings and computing similarities.
    Suitable for small to medium message sets in development and testing.
    For production with large sets, prefer FAISS or vector databases.

    Examples:
        Basic usage:
            ```python
            index = LiteMessageIndex()

            # Index some messages
            await index.index(
                [
                    MessageRecord(content="Hello world"),
                    MessageRecord(content="How are you?"),
                ]
            )

            # Search for similar messages
            results = await index.search(
                query="Hi there",
                max_results=5,
                score_threshold=0.7,
            )

            # Process results
            for message, score in results:
                print(f"{message.content}: {score:.2f}")
            ```
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 16,
    ) -> None:
        """Initialize an empty in-memory message index.

        Args:
            embedding_model: OpenAI model for computing embeddings.
            embedding_batch_size: Number of texts to embed in parallel.
        """
        self._messages: dict[str, ChatMessage] = {}
        self._message_embeddings: dict[str, NDArray[np.float32]] = {}
        self._embedding_model = embedding_model
        self._batch_size = embedding_batch_size

    async def _get_embeddings(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[NDArray[np.float32]]:
        """Compute embeddings for a list of texts.

        Processes texts in batches for efficiency. Empty or whitespace-only
        texts are filtered out before processing.

        Args:
            texts: List of texts to embed.
            batch_size: Optional override for batch size.

        Returns:
            List of embedding arrays, one per valid input text.
        """
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return []

        batch_size = batch_size or self._batch_size
        batches = [valid_texts[i : i + batch_size] for i in range(0, len(valid_texts), batch_size)]

        responses = await asyncio.gather(
            *[aembedding(model=self._embedding_model, input=batch) for batch in batches]
        )

        embeddings: list[NDArray[np.float32]] = []
        for response in responses:
            batch_embeddings = [
                np.array(item["embedding"], dtype=np.float32).reshape(-1) for item in response.data
            ]
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> list[tuple[ChatMessage, NDArray[np.float32]]]:
        """Generate embeddings for a sequence of messages.

        Filters out messages without content and computes embeddings in batches.
        Returns only messages with valid embeddings.

        Args:
            messages: Messages to embed.

        Returns:
            List of (message, embedding) pairs for successfully embedded messages.
        """
        valid_messages = [msg for msg in messages if msg.content and msg.content.strip()]

        if not valid_messages:
            return []

        contents = [msg.content for msg in valid_messages if msg.content]
        embeddings = await self._get_embeddings(contents)
        embedding_pairs = zip(valid_messages, embeddings, strict=True)

        return [(msg, embedding) for msg, embedding in embedding_pairs if embedding.size > 0]

    @override
    async def index(self, messages: Sequence[ChatMessage]) -> None:
        """Add new messages to the index.

        Computes embeddings for messages not already in the index. Messages
        without content or with existing embeddings are skipped.

        Args:
            messages: Messages to index.
        """
        to_index = [msg for msg in messages if msg.id not in self._message_embeddings]
        if not to_index:
            return

        indexed = await self._embed_messages(to_index)

        for msg, embedding in indexed:
            self._messages[msg.id] = msg
            self._message_embeddings[msg.id] = embedding

    @override
    async def search(
        self,
        query: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ChatMessage, float]]:
        """Search for messages similar to the query.

        Computes cosine similarity between the query embedding and all indexed
        messages. Returns messages sorted by similarity score in descending order.

        Args:
            query: Text to search for.
            max_results: Maximum number of results to return.
            score_threshold: Minimum similarity score (0 to 1) to include.

        Returns:
            List of (message, score) pairs sorted by decreasing similarity.
        """
        if not self._message_embeddings:
            return []

        query_embeddings = await self._get_embeddings([query])
        if not query_embeddings:
            return []

        query_embedding = query_embeddings[0]

        similarity_scores: list[tuple[ChatMessage, float]] = []
        for msg_id, emb in self._message_embeddings.items():
            score = np.dot(emb, query_embedding)
            score /= np.linalg.norm(emb) * np.linalg.norm(query_embedding)
            if score >= (score_threshold or -float("inf")):
                similarity_scores.append((self._messages[msg_id], score))

        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        if max_results:
            similarity_scores = similarity_scores[:max_results]

        return similarity_scores

    @override
    async def clear(self) -> None:
        """Clear all indexed embeddings while preserving messages.

        Removes all indexed embeddings and resets internal data structures
        without affecting the original messages in the store.
        """
        self._message_embeddings.clear()
