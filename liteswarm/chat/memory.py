# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Protocol

from typing_extensions import override

from liteswarm.chat.index import LiteMessageIndex
from liteswarm.types.chat import ChatMessage
from liteswarm.types.swarm import Message


class ChatMemory(Protocol):
    """Protocol for managing chat message persistence and retrieval.

    Provides a standard interface for message storage operations that can be
    implemented by different storage backends. Supports individual and batch
    operations for adding, retrieving, and removing messages.

    Examples:
        ```python
        class MyStorage(ChatMemory):
            def __init__(self) -> None:
                self._messages = {}

            async def add_message(self, message: Message) -> None:
                # Store message in custom backend
                ...

            async def get_messages(self) -> list[ChatMessage]:
                # Retrieve messages from custom backend
                ...


        # Use custom storage
        storage = MyStorage()
        await storage.add_message(
            message=Message(role="user", content="Hello!"),
        )
        ```

    Notes:
        - All operations are asynchronous by framework design
        - Implementations should preserve message order
        - Methods accept custom arguments for extensibility
    """

    async def add_message(
        self,
        message: Message,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a single message to storage.

        Stores a message and makes it available for future retrieval.

        Args:
            message: Message to store.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def add_messages(
        self,
        messages: list[Message],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add multiple messages to storage.

        Stores a batch of messages. More efficient than adding messages
        individually.

        Args:
            messages: List of messages to store.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def get_messages(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Get all messages.

        Retrieves messages in chronological order.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of messages in chronological order.
        """
        ...

    async def remove_message(
        self,
        message_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Remove a single message from storage.

        Deletes a message and its associated data. Does nothing if the
        message doesn't exist.

        Args:
            message_id: Identifier of the message to remove.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def remove_messages(
        self,
        message_ids: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Remove multiple messages from storage.

        Deletes a batch of messages and their associated data. More efficient
        than removing messages individually.

        Args:
            message_ids: List of message identifiers to remove.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def clear(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Remove all messages from storage.

        Clears all messages and their associated data, effectively resetting
        the storage to its initial state.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...


class LiteChatMemory(ChatMemory):
    """In-memory implementation of chat message storage with vector search.

    Stores messages in memory with vector embeddings for semantic search.
    Each instance maintains its own message store for a single conversation.

    Examples:
        ```python
        # Create memory instance
        memory = LiteChatMemory(
            embedding_model="text-embedding-3-small",
            embedding_batch_size=16,
        )

        # Add messages
        await memory.add_message(
            message=Message(role="user", content="Hello!"),
        )

        # Get messages
        messages = await memory.get_messages()
        ```

    Notes:
        - Messages are stored in memory and lost on restart
        - Vector search requires embedding computation
        - Batch operations are more efficient
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 16,
    ) -> None:
        """Initialize a new memory instance.

        Creates an in-memory message store with vector indexing capabilities.
        Messages are embedded using the specified model for semantic search.

        Args:
            embedding_model: Model to use for creating message embeddings.
            embedding_batch_size: Number of messages to embed in parallel.
        """
        self._messages: dict[str, ChatMessage] = {}  # message_id -> ChatMessage
        self._message_ids: set[str] = set()  # Ordered set of message IDs
        self._index = LiteMessageIndex(
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
        )

    @override
    async def add_message(self, message: Message) -> None:
        """Add a single message to memory.

        Converts the message to a chat message, stores it in memory, and
        updates the vector index.

        Args:
            message: Message to store.
        """
        chat_message = ChatMessage.from_message(message)
        self._messages[chat_message.id] = chat_message
        self._message_ids.add(chat_message.id)
        await self._index.index([chat_message])

    @override
    async def add_messages(self, messages: list[Message]) -> None:
        """Add multiple messages to memory.

        Converts messages to chat messages, stores them in memory, and
        updates the vector index in a single batch.

        Args:
            messages: List of messages to store.
        """
        chat_messages: list[ChatMessage] = []

        for message in messages:
            chat_message = ChatMessage.from_message(message)
            self._messages[chat_message.id] = chat_message
            self._message_ids.add(chat_message.id)
            chat_messages.append(chat_message)

        await self._index.index(chat_messages)

    @override
    async def get_messages(self) -> list[ChatMessage]:
        """Get all messages in chronological order.

        Retrieves messages and sorts them by timestamp.

        Returns:
            List of messages in chronological order.
        """
        messages = [self._messages[mid] for mid in self._message_ids if mid in self._messages]
        messages.sort(key=lambda m: m.created_at)
        return messages

    @override
    async def remove_message(self, message_id: str) -> None:
        """Remove a single message from memory.

        Removes the message from both memory storage and message list.
        Does nothing if the message doesn't exist.

        Args:
            message_id: Identifier of the message to remove.
        """
        if message_id in self._messages:
            self._messages.pop(message_id)
            self._message_ids.discard(message_id)

    @override
    async def remove_messages(self, message_ids: list[str]) -> None:
        """Remove multiple messages from memory.

        Removes messages one by one from both memory storage and message list.

        Args:
            message_ids: List of message identifiers to remove.
        """
        for message_id in message_ids:
            await self.remove_message(message_id)

    @override
    async def clear(self) -> None:
        """Remove all messages from memory.

        Clears all messages and the vector index.
        """
        self._messages.clear()
        self._message_ids.clear()
        await self._index.clear()
