# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import uuid
from datetime import datetime
from typing import Any, Generic, Self

from pydantic import BaseModel, ConfigDict, Field

from liteswarm.types.llm import ResponseFormatPydantic
from liteswarm.types.swarm import AgentExecutionResult, Message


class ChatMessage(Message):
    """A wrapper around base Message type for conversational applications.

    ChatMessage extends Message to provide additional fields needed for chat
    applications. It preserves all base message attributes while adding
    identification and metadata support.

    Notes:
        - Inherits all Message capabilities
        - Adds unique identification
        - Supports application metadata
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


class ChatResponse(AgentExecutionResult[ResponseFormatPydantic], Generic[ResponseFormatPydantic]):
    """Chat-optimized execution result for conversational applications.

    ChatResponse extends AgentExecutionResult to provide a chat-friendly interface
    for conversational applications. It preserves all execution details while allowing
    for future chat-specific extensions like conversation state and metadata.

    Type Parameters:
        ResponseFormatPydantic: The type of the parsed response format.

    Notes:
        - Inherits all AgentExecutionResult capabilities
        - Optimized for chat applications
        - Supports future chat-specific extensions
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique response identifier."""

    @classmethod
    def from_agent_execution(
        cls,
        agent_execution: AgentExecutionResult[ResponseFormatPydantic],
        /,
        *,
        id: str | None = None,
    ) -> Self:
        """Create a ChatResponse from an AgentExecutionResult.

        Converts a standard AgentExecutionResult into a chat-optimized response
        while preserving all execution details. This allows reuse of Swarm's
        execution capabilities in chat applications.

        Args:
            agent_execution: Agent execution result to convert.
            id: Optional unique response identifier.

        Returns:
            New ChatResponse with all execution details.
        """
        return cls(
            id=id or str(uuid.uuid4()),
            agent=agent_execution.agent,
            agent_response=agent_execution.agent_response,
            agent_responses=agent_execution.agent_responses,
            new_messages=agent_execution.new_messages,
            all_messages=agent_execution.all_messages,
            context_variables=agent_execution.context_variables,
        )


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
