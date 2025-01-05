# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import uuid
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from liteswarm.types.llm import AudioResponse, MessageRole, ToolCall

MessageT = TypeVar("MessageT", bound="Message")
"""Type variable for Message-based types.

Used for generic operations on Message subclasses while preserving
their specific type information. Enables type-safe message handling
throughout the system.

Examples:
    Type-safe message container:
        ```python
        class MessageContainer(Generic[MessageT]):
            def __init__(self, messages: list[MessageT]) -> None:
                self.messages = messages

            def get_first(self) -> MessageT:
                return self.messages[0]
        ```
"""

MessageAdapter = TypeAdapter(list["Message"])
"""Type adapter for Message list serialization.

Handles conversion between Message objects and dictionaries for:
- API request/response serialization
- Token counting and validation
- Message persistence

Examples:
    Serialization:
        ```python
        messages = [Message(role="user", content="Hello")]
        data = MessageAdapter.dump_python(messages)
        restored = MessageAdapter.validate_python(data)
        ```
"""


class Message(BaseModel):
    """Message in a conversation between user, assistant, and tools.

    Represents a message in a conversation between participants
    with content and optional tool interactions. Each message has
    a specific role and may include tool calls or responses.

    Examples:
        System message:
            ```python
            system_msg = Message(
                role="system",
                content="You are a helpful assistant.",
            )
            ```

        Assistant with tool:
            ```python
            assistant_msg = Message(
                role="assistant",
                content="Let me calculate that.",
                tool_calls=[
                    ToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2, "b": 2}'},
                        type="function",
                        index=0,
                    )
                ],
            )
            ```

        Tool response:
            ```python
            tool_msg = Message(
                role="tool",
                content="4",
                tool_call_id="calc_1",
            )
            ```
    """

    role: MessageRole
    """Role of the message author."""

    content: str | None = None
    """Text content of the message."""

    tool_calls: list[ToolCall] | None = None
    """Tool calls made in this message."""

    tool_call_id: str | None = None
    """ID of the tool call this message responds to."""

    audio: AudioResponse | None = None
    """Audio response data if available."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ChatMessage(Message):
    """Message for multi-user chat applications.

    Extends the base Message with unique identification, session support,
    and application-specific metadata. Used to build chat applications.

    Examples:
        Basic usage:
            ```python
            message = ChatMessage(
                id="msg_123",
                role="user",
                content="Hello",
                session_id="session_1",
            )
            ```

        With metadata:
            ```python
            chat_msg = ChatMessage.from_message(
                Message(role="user", content="Hello"),
                session_id="session_1",
                metadata={"user_id": "user_123"},
            )
            ```
    """

    id: str
    """Unique message identifier."""

    session_id: str | None = None
    """Identifier of session the message belongs to."""

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
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatMessage":
        """Create a ChatMessage from a base Message.

        Converts a base Message to a ChatMessage by copying all fields
        and adding identification information. If the input is already a
        ChatMessage, returns a copy.

        Args:
            message: Base Message to convert.
            session_id: Optional session identifier.
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
            id=str(uuid.uuid4()),
            session_id=session_id,
            created_at=datetime.now(),
            metadata=metadata,
        )


class TrimMessagesResult(BaseModel, Generic[MessageT]):
    """Result of message trimming operation.

    Contains messages that fit within model context limits and
    the remaining tokens available for response. Used for context
    window management and token optimization.

    Example:
        ```python
        result = TrimmedMessages(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            response_tokens=1000,
        )
        ```
    """

    messages: list[MessageT]
    """Messages that fit within context limits."""

    response_tokens: int
    """Tokens available for model response."""
