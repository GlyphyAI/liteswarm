# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import uuid
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from liteswarm.types.swarm import Message

MessageT = TypeVar("MessageT", bound=Message)
"""Generic type variable for Message-based types.

Enables type-safe operations on Message subclasses while
maintaining their specific type information.
"""

MessageAdapter = TypeAdapter(list[Message])
"""Type adapter for converting between Message objects and dictionaries.

Used for serialization and deserialization of message lists in:
- API requests/responses
- Token counting
- Message validation
"""


class MessageRecord(Message):
    """Message with unique identification and metadata support.

    Extends the base Message with additional fields for tracking,
    identification, and enhanced functionality. Used throughout the
    system for message management and retrieval.

    Examples:
        Create message records:
            ```python
            # Basic message record
            record = MessageRecord(
                id="msg_123",
                role="user",
                content="Hello",
                timestamp=datetime.now(),
            )

            # With metadata for search
            search_record = MessageRecord(
                id="msg_456",
                role="assistant",
                content="Python info",
                timestamp=datetime.now(),
                metadata={
                    "source": "python_docs",
                    "relevance": 0.95,
                    "tags": ["python", "docs"],
                },
            )

            # Tool call record
            tool_record = MessageRecord(
                id="msg_789",
                role="assistant",
                content="Let me calculate that.",
                tool_calls=[
                    ToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2, "b": 2}'},
                    )
                ],
                metadata={"tool_type": "calculator"},
            )
            ```
    """

    id: str
    """Unique identifier for message referencing and tracking."""

    timestamp: datetime
    """When the message was created or processed."""

    metadata: dict[str, Any] | None
    """Optional metadata for enhanced functionality (e.g., search, analytics)."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @classmethod
    def from_message(
        cls,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> "MessageRecord":
        """Create a MessageRecord from a base Message.

        Creates a new MessageRecord by copying fields from the base Message
        and adding tracking fields (ID, timestamp, metadata).

        Args:
            message: Base Message instance to convert.
            metadata: Optional metadata to associate with the message.

        Returns:
            New MessageRecord with tracking fields.

        Examples:
            ```python
            # Basic conversion
            record = MessageRecord.from_message(Message(role="user", content="Hello"))

            # With metadata
            record = MessageRecord.from_message(
                Message(role="assistant", content="Python info"),
                metadata={"source": "docs", "relevance": 0.95},
            )

            # With tool calls
            record = MessageRecord.from_message(
                Message(
                    role="assistant",
                    content="Let me help",
                    tool_calls=[
                        ToolCall(
                            id="calc_1",
                            function={"name": "add", "arguments": '{"a": 2, "b": 2}'},
                        )
                    ],
                ),
                metadata={"tool_type": "calculator"},
            )
            ```
        """
        if isinstance(message, MessageRecord):
            return message.model_copy()

        return cls(
            # Copy all fields from Message
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            audio=message.audio,
            # Add MessageRecord specific fields
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            metadata=metadata,
        )


class TrimmedMessages(BaseModel, Generic[MessageT]):
    """Result of trimming messages in a conversation.

    Manages conversation history within model context limits by providing
    both the trimmed message list and available response tokens. Used
    throughout the system for context window management and token
    optimization.

    Examples:
        Basic trimming:
            ```python
            # Trim with default settings
            result = TrimmedMessages(
                messages=[
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi!"),
                ],
                response_tokens=1000,  # Tokens left for response
            )

            # Access results
            messages = result.messages
            available_tokens = result.response_tokens
            ```

        Model-specific trimming:
            ```python
            # Trim for specific model
            result = TrimmedMessages(
                messages=[
                    Message(role="system", content="You are helpful."),
                    Message(role="user", content="Complex question..."),
                    Message(role="assistant", content="Detailed answer..."),
                ],
                response_tokens=2048,  # GPT-4's typical response space
            )

            # Check if enough space for response
            if result.response_tokens >= 500:
                print("Sufficient space for detailed response")
            ```

        Tool interaction preservation:
            ```python
            # Trim while keeping tool pairs
            result = TrimmedMessages(
                messages=[
                    Message(
                        role="assistant",
                        content="Let me calculate.",
                        tool_calls=[
                            ToolCall(
                                id="calc_1",
                                function={"name": "add", "arguments": '{"a": 2, "b": 2}'},
                            )
                        ],
                    ),
                    Message(
                        role="tool",
                        content="4",
                        tool_call_id="calc_1",
                    ),
                ],
                response_tokens=800,
            )
            ```
    """

    messages: list[MessageT]
    """Messages remaining after trimming operation."""

    response_tokens: int
    """Number of tokens available for model response after trimming."""
