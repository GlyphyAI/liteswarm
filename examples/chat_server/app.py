# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from liteswarm.chat import LiteChat, LiteChatMemory, LiteChatOptimization, LiteChatSearch
from liteswarm.core import Swarm
from liteswarm.types import ChatMessage, ResponseCost, Usage
from liteswarm.types.context import ContextVariables
from liteswarm.types.events import ErrorEvent, SwarmEvent
from liteswarm.utils import enable_logging

from .agents import create_agent_team

enable_logging(default_level="DEBUG")


AGENT_TEAM = create_agent_team()
DEFAULT_AGENT = AGENT_TEAM.router


@dataclass
class ChatState:
    """State container for chat components."""

    chat: LiteChat
    user_store: "UserStore"


@dataclass
class UserStore:
    """In-memory user management for chat server."""

    _users: dict[str, "User"] = field(default_factory=dict)

    async def add_user(
        self,
        user_id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add new user."""
        if user_id in self._users:
            raise ValueError("User already exists")

        self._users[user_id] = User(
            user_id=user_id,
            name=name,
            metadata=metadata,
        )

    async def get_user(self, user_id: str) -> "User | None":
        """Get user by ID."""
        return self._users.get(user_id)

    async def list_users(self) -> dict[str, "User"]:
        """Get all users."""
        return self._users.copy()

    async def delete_user(self, user_id: str) -> None:
        """Delete user and their sessions."""
        if user_id not in self._users:
            return
        self._users.pop(user_id)


@dataclass
class StateManager:
    """Singleton manager for application state."""

    _chat_state: ChatState | None = None

    @classmethod
    def get_chat_state(cls) -> ChatState:
        """Get or create chat state."""
        if cls._chat_state is None:
            memory = LiteChatMemory()
            search = LiteChatSearch(memory=memory)
            optimization = LiteChatOptimization(memory=memory, search=search)
            swarm = Swarm(include_usage=True, include_cost=True)
            chat = LiteChat(
                memory=memory,
                search=search,
                optimization=optimization,
                swarm=swarm,
            )

            cls._chat_state = ChatState(
                chat=chat,
                user_store=UserStore(),
            )

        return cls._chat_state


app = FastAPI(
    title="Chat Session API",
    description="API for managing chat sessions and messages",
    version="1.0.0",
)


class User(BaseModel):
    """User model for chat server."""

    user_id: str
    name: str | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class CreateSessionRequest(BaseModel):
    """Request model for creating a session."""

    user_id: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class SessionResponse(BaseModel):
    """Response model for session operations."""

    session_id: str
    user_id: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class SendMessageRequest(BaseModel):
    """Request model for sending messages."""

    message: str
    agent_id: str | None = None
    context_variables: dict[str, Any] | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class MessageResponse(BaseModel):
    """Response model for messages."""

    content: str
    agent_id: str | None = None
    usage: Usage | None = None
    response_cost: ResponseCost | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    session_id: str
    user_id: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class MessageListResponse(BaseModel):
    """Response model for listing messages."""

    messages: list[ChatMessage]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class CreateUserRequest(BaseModel):
    """Request model for creating users."""

    user_id: str
    name: str | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class UserResponse(BaseModel):
    """Response model for user operations."""

    user_id: str
    name: str | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


async def format_chunk(event: SwarmEvent) -> bytes:
    """Format a swarm event as a JSON chunk.

    Args:
        event: The swarm event to format.

    Returns:
        JSON-encoded bytes for the event.
    """
    return (event.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


async def generate_json_chunks(
    session_id: str,
    request: SendMessageRequest,
    state: ChatState,
) -> AsyncGenerator[bytes, None]:
    """Generate JSON chunks from chat responses."""
    try:
        session = await state.chat.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        agent = DEFAULT_AGENT
        if request.agent_id:
            requested_agent = AGENT_TEAM.agents.get(request.agent_id)
            if requested_agent is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            agent = requested_agent

        context_variables = ContextVariables(request.context_variables or {})
        stream = session.send_message(
            request.message,
            agent=agent,
            context_variables=context_variables,
        )

        async for event in stream:
            yield await format_chunk(event)

        await stream.get_return_value()

    except HTTPException:
        raise

    except Exception as e:
        error_event = ErrorEvent(error=str(e), agent=None)
        yield await format_chunk(error_event)
        print(f"Error during streaming: {e}")


@app.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> SessionResponse:
    """Create a new chat session.

    Args:
        request: CreateSessionRequest containing user ID.
        state: ChatState containing chat components.

    Returns:
        SessionResponse with session details.
    """
    session = await state.chat.create_session(user_id=request.user_id)
    return SessionResponse(
        session_id=session.session_id,
        user_id=request.user_id,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> None:
    """Delete a chat session.

    Args:
        session_id: ID of the session to delete.
        state: ChatState containing chat components.
    """
    try:
        session = await state.chat.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        await state.chat.delete_session(session_id)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/sessions/{session_id}/messages/stream")
async def stream_message(
    session_id: str,
    request: SendMessageRequest,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> StreamingResponse:
    """Stream chat messages using chunked transfer encoding.

    Args:
        session_id: ID of the chat session.
        request: SendMessageRequest containing message and agent details.
        state: ChatState containing chat components.

    Returns:
        StreamingResponse with chunked JSON data.
    """
    return StreamingResponse(
        generate_json_chunks(session_id, request, state),
        media_type="application/json",
    )


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> MessageResponse:
    """Send a message to a chat session.

    Args:
        session_id: ID of the chat session.
        request: SendMessageRequest containing message and agent details.
        state: ChatState containing chat components.

    Returns:
        MessageResponse with response content and metadata.
    """
    try:
        session = await state.chat.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        agent = DEFAULT_AGENT if request.agent_id == "assistant" else None
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        context_variables = ContextVariables(request.context_variables or {})
        stream = session.send_message(
            request.message,
            agent=agent,
            context_variables=context_variables,
        )

        result = await stream.get_return_value()

        content: str = ""
        total_usage: Usage | None = None
        total_cost: ResponseCost | None = None

        for response in result.agent_responses:
            content += response.content or ""
            if response.usage:
                total_usage = response.usage
            if response.response_cost:
                total_cost = response.response_cost

        return MessageResponse(
            content=content,
            agent_id=agent.id,
            usage=total_usage,
            response_cost=total_cost,
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(
    session_id: str,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> MessageListResponse:
    """Get all messages from a chat session.

    Args:
        session_id: ID of the chat session.
        state: ChatState containing chat components.

    Returns:
        List of messages in the session.
    """
    try:
        session = await state.chat.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = await session.get_messages()
        return MessageListResponse(messages=messages)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/sessions", response_model=list[SessionListResponse])
async def list_sessions(
    user_id: str,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> list[SessionListResponse]:
    """List all chat sessions for a user.

    Args:
        user_id: ID of the user to list sessions for.
        state: ChatState containing chat components.

    Returns:
        List of sessions with their details.
    """
    try:
        sessions = await state.chat.get_user_sessions(user_id)
        return [
            SessionListResponse(session_id=session.session_id, user_id=user_id)
            for session in sessions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/users", response_model=UserResponse)
async def create_user(
    request: CreateUserRequest,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> UserResponse:
    """Create a new user.

    Args:
        request: CreateUserRequest containing user details.
        state: ChatState containing chat components.

    Returns:
        UserResponse with user details.
    """
    try:
        await state.user_store.add_user(
            user_id=request.user_id,
            name=request.name,
            metadata=request.metadata,
        )

        return UserResponse(
            user_id=request.user_id,
            name=request.name,
            metadata=request.metadata,
        )

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> None:
    """Delete a user and all associated sessions.

    Args:
        user_id: ID of user to delete.
        state: ChatState containing chat components.
    """
    try:
        user = await state.user_store.get_user(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        await state.user_store.delete_user(user_id)
        await state.chat.delete_user_sessions(user_id)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/users", response_model=list[UserResponse])
async def list_users(
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> list[UserResponse]:
    """List all users.

    Args:
        state: ChatState containing chat components.

    Returns:
        List of users with their details.
    """
    try:
        users = await state.user_store.list_users()
        return [
            UserResponse(
                user_id=user_id,
                name=user.name,
                metadata=user.metadata,
            )
            for user_id, user in users.items()
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/users/{user_id}/switch", response_model=list[SessionResponse])
async def switch_user(
    user_id: str,
    state: Annotated[ChatState, Depends(StateManager.get_chat_state)],
) -> list[SessionResponse]:
    """Switch to a different user and get their sessions.

    Args:
        user_id: ID of user to switch to.
        state: ChatState containing chat components.

    Returns:
        List of user's active sessions.
    """
    try:
        user = await state.user_store.get_user(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        sessions = await state.chat.get_user_sessions(user_id)
        return [
            SessionResponse(session_id=session.session_id, user_id=user_id) for session in sessions
        ]

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
