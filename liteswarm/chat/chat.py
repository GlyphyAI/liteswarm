# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Protocol, TypeVar
from uuid import uuid4

from typing_extensions import override

from liteswarm.chat.memory import LiteChatMemory
from liteswarm.chat.optimization import LiteChatOptimization
from liteswarm.chat.search import LiteChatSearch
from liteswarm.chat.session import ChatSession, LiteChatSession
from liteswarm.core.swarm import Swarm

ChatSessionType = TypeVar("ChatSessionType", bound=ChatSession[Any])
"""Type variable for chat session type with arbitrary return type."""


class Chat(Protocol[ChatSessionType]):
    """Protocol for managing chat sessions and user interactions.

    Provides a standard interface for chat session lifecycle management,
    including creation, retrieval, and cleanup. Implementations can use
    different storage backends while maintaining consistent session access.

    Each session represents an isolated conversation context that can be
    accessed and modified independently of other sessions.

    Type Parameters:
        ChatSessionType: Type of chat session managed by this implementation.

    Examples:
        ```python
        class MyChat(Chat[MyChatSession]):
            async def create_session(
                self,
                user_id: str,
                session_id: str | None = None,
            ) -> MyChatSession:
                # Create and store new session
                ...

            async def get_session(
                self,
                session_id: str,
            ) -> MyChatSession | None:
                # Retrieve existing session
                ...


        # Use custom chat manager
        chat = MyChat()
        session = await chat.create_session(user_id="user_123")
        async for event in session.send_message("Hello!", agent=my_agent):
            print(event)
        ```

    Notes:
        - Sessions should be isolated from each other
        - User-session mappings must be maintained
        - All operations are asynchronous by framework design
        - Implementations should handle concurrent access
    """

    async def create_session(
        self,
        user_id: str,
        session_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> ChatSessionType:
        """Create a new chat session for a user.

        Creates and initializes a new session with optional custom ID.
        If a session with the given ID exists, returns the existing session.

        Args:
            user_id: Identifier of the user owning the session.
            session_id: Optional custom session identifier.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            Newly created or existing chat session.
        """
        ...

    async def get_session(
        self,
        session_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> ChatSessionType | None:
        """Get a specific chat session.

        Retrieves a session by its ID if it exists.

        Args:
            session_id: Identifier of the session to retrieve.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            Chat session if found, None otherwise.
        """
        ...

    async def get_user_sessions(
        self,
        user_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[ChatSessionType]:
        """Get all sessions belonging to a user.

        Retrieves all active sessions associated with the given user ID.

        Args:
            user_id: Identifier of the user.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of user's chat sessions.
        """
        ...

    async def delete_session(
        self,
        session_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Delete a chat session.

        Removes a session and its associated data. Does nothing if the
        session doesn't exist.

        Args:
            session_id: Identifier of the session to delete.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...

    async def delete_user_sessions(
        self,
        user_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Delete all sessions belonging to a user.

        Removes all sessions and their data for the given user.

        Args:
            user_id: Identifier of the user.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        ...


class LiteChat(Chat[LiteChatSession]):
    """In-memory implementation of chat session management.

    Manages chat sessions using in-memory storage with support for message
    history, search, and context optimization. Each session maintains its
    own conversation state and can be accessed concurrently.

    Examples:
        ```python
        # Create chat manager
        chat = LiteChat(
            memory=LiteChatMemory(),
            search=LiteChatSearch(),
            optimization=LiteChatOptimization(),
        )

        # Create and use session
        session = await chat.create_session(user_id="user_123")
        async for event in session.send_message(
            message="Hello!",
            agent=my_agent,
        ):
            if event.type == "agent_response_chunk":
                print(event.chunk.content)
        ```

    Notes:
        - Sessions are stored in memory and lost on restart
        - Each session can be used concurrently
        - Memory, search, and optimization are optional
    """

    def __init__(
        self,
        memory: LiteChatMemory | None = None,
        search: LiteChatSearch | None = None,
        optimization: LiteChatOptimization | None = None,
        swarm: Swarm | None = None,
    ) -> None:
        """Initialize a new LiteChat instance.

        Creates a chat manager with optional components for message storage,
        search, and context optimization. Each component is initialized with
        default implementations if not provided.

        Args:
            memory: Storage for chat history and metadata.
            search: Search functionality for chat history.
            optimization: Context optimization for long conversations.
            swarm: Swarm instance for agent execution.
        """
        self._memory = memory or LiteChatMemory()
        self._search = search or LiteChatSearch(memory=self._memory)
        self._optimization = optimization or LiteChatOptimization(
            memory=self._memory,
            search=self._search,
        )
        self._swarm = swarm or Swarm()
        self._sessions: dict[str, LiteChatSession] = {}
        self._user_sessions: dict[str, set[str]] = {}

    @override
    async def create_session(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> LiteChatSession:
        """Create a new chat session for a user.

        Creates a session with the given ID or generates a UUID if not provided.
        Returns existing session if ID is already in use.

        Args:
            user_id: Identifier of the user owning the session.
            session_id: Optional custom session identifier.

        Returns:
            New or existing chat session.
        """
        session_id = session_id or str(uuid4())
        if session_id in self._sessions:
            return self._sessions[session_id]

        session = LiteChatSession(
            session_id=session_id,
            memory=self._memory,
            search=self._search,
            optimization=self._optimization,
            swarm=self._swarm,
        )

        self._sessions[session_id] = session
        self._user_sessions.setdefault(user_id, set()).add(session_id)

        return session

    @override
    async def get_session(self, session_id: str) -> LiteChatSession | None:
        """Get a specific chat session.

        Retrieves a session by its ID from in-memory storage.

        Args:
            session_id: Identifier of the session to retrieve.

        Returns:
            Chat session if found, None otherwise.
        """
        if session_id not in self._sessions:
            return None

        return self._sessions[session_id]

    @override
    async def get_user_sessions(self, user_id: str) -> list[LiteChatSession]:
        """Get all sessions belonging to a user.

        Retrieves all active sessions for the user from in-memory storage.

        Args:
            user_id: Identifier of the user.

        Returns:
            List of user's chat sessions.
        """
        session_ids = self._user_sessions.get(user_id, set())
        return [self._sessions[sid] for sid in session_ids]

    @override
    async def delete_session(self, session_id: str) -> None:
        """Delete a chat session.

        Removes a session from in-memory storage and user mappings.
        Does nothing if session doesn't exist.

        Args:
            session_id: Identifier of the session to delete.
        """
        if session_id not in self._sessions:
            return

        self._sessions.pop(session_id)
        for sessions in self._user_sessions.values():
            sessions.discard(session_id)

    @override
    async def delete_user_sessions(self, user_id: str) -> None:
        """Delete all sessions belonging to a user.

        Removes all user's sessions from in-memory storage.

        Args:
            user_id: Identifier of the user.
        """
        for session_id in self._user_sessions.get(user_id, set()):
            await self.delete_session(session_id)
