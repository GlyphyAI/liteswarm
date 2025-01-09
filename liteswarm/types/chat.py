# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import uuid
from datetime import datetime
from typing import Any, Generic

from pydantic import BaseModel, ConfigDict

from liteswarm.types.llm import ResponseFormatPydantic
from liteswarm.types.swarm import AgentExecutionResult, Message


class ChatMessage(Message):
    """Message for multi-user chat applications.

    Extends the base Message with unique identification and
    application-specific metadata. Used to build chat applications.

    Examples:
        Basic usage:
            ```python
            message = ChatMessage(
                id="msg_123",
                role="user",
                content="Hello",
            )
            ```

        With metadata:
            ```python
            chat_msg = ChatMessage.from_message(
                Message(role="user", content="Hello"),
                metadata={"user_id": "user_123"},
            )
            ```
    """

    id: str
    """Unique message identifier."""

    created_at: datetime = datetime.now()
    """Message creation timestamp."""

    metadata: dict[str, Any] | None = None
    """Application-specific message data."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @classmethod
    def from_message(
        cls,
        message: Message,
        /,
        *,
        id: str | None = None,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatMessage":
        """Create a ChatMessage from a base Message.

        Converts a base Message to a ChatMessage by copying all fields
        and adding identification information. If the input is already a
        ChatMessage, returns a copy.

        Args:
            message: Base Message to convert.
            id: Optional unique message identifier.
            created_at: Optional message creation timestamp.
            metadata: Optional message metadata.

        Returns:
            New ChatMessage with identification fields.
        """
        if isinstance(message, ChatMessage):
            return message.model_copy()

        return cls(
            # Copy Message fields
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            audio=message.audio,
            # Add identification fields
            id=id or str(uuid.uuid4()),
            created_at=created_at or datetime.now(),
            metadata=metadata,
        )


class ChatResponse(BaseModel, Generic[ResponseFormatPydantic]):
    """Response object containing execution results and conversation state.

    Encapsulates the complete state of a chat interaction, including the final
    agent that responded, new messages generated during execution, the complete
    conversation history, and all intermediate agent responses.
    """

    agent_execution: AgentExecutionResult[ResponseFormatPydantic]
    """Complete agent execution result."""


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

    query: str | None = None
    """The search query used to find relevant messages."""

    max_messages: int | None = None
    """Maximum number of messages to retrieve."""

    score_threshold: float | None = None
    """Minimum similarity score (0-1) for including messages."""

    embedding_model: str | None = None
    """Name of the embedding model to use for semantic search."""
