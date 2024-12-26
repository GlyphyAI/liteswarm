# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pydantic import BaseModel


class RAGStrategyConfig(BaseModel):
    """Configuration for the RAG (Retrieval-Augmented Generation) optimization strategy.

    This class defines parameters for controlling how relevant messages are retrieved
    and selected during context optimization. It allows customization of the search
    query, result limits, relevance thresholds, and embedding model selection.

    Example:
        ```python
        config = RAGStrategyConfig(
            query="weather in London",
            max_messages=10,
            score_threshold=0.6,
            embedding_model="text-embedding-3-small",
        )
        ```
    """

    query: str
    """The search query used to find relevant messages."""

    max_messages: int | None = None
    """Maximum number of messages to retrieve."""

    score_threshold: float | None = None
    """Minimum similarity score (0-1) for including messages."""

    embedding_model: str | None = None
    """Name of the embedding model to use for semantic search."""
