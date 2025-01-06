# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json
import random
import sys
from typing import Any, NoReturn

import aiohttp
from aiohttp import ClientTimeout
from pydantic import BaseModel, ConfigDict

from liteswarm.types.context import ContextVariables
from liteswarm.utils.misc import prompt
from liteswarm.utils.retry import retry

BASE_URL = "http://localhost:8000"
TIMEOUT = ClientTimeout(total=None, connect=30, sock_read=30, sock_connect=30)
DEBUG = False

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 16.0
RETRY_MULTIPLIER = 2.0
EXCEPTION = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    ConnectionError,
    aiohttp.ServerDisconnectedError,
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


class SessionResponse(BaseModel):
    """Response model for session operations."""

    session_id: str
    user_id: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class CreateSessionRequest(BaseModel):
    """Request model for creating sessions."""

    user_id: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


def debug_print(message: str) -> None:
    """Print debug message if debug mode is enabled."""
    if DEBUG:
        print(f"\n[DEBUG] {message}", file=sys.stderr)


def get_retry_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(INITIAL_RETRY_DELAY * (RETRY_MULTIPLIER ** (attempt - 1)), MAX_RETRY_DELAY)
    return delay * (0.5 + random.random())  # Add jitter


def get_agent_title(agent_id: str) -> str:
    """Get a human-readable title for an agent."""
    titles = {
        "router": "Router",
        "product_manager": "Product Manager",
        "designer": "Designer",
        "engineer": "Engineer",
        "qa": "QA Engineer",
    }

    return titles.get(agent_id, agent_id.title())


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def create_session(session: aiohttp.ClientSession, user_id: str) -> SessionResponse:
    """Create a new chat session."""
    request = CreateSessionRequest(user_id=user_id)
    async with session.post(
        f"{BASE_URL}/sessions",
        json=request.model_dump(),
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return SessionResponse(**data)


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def send_message_and_stream(  # noqa: PLR0915
    session: aiohttp.ClientSession,
    session_id: str,
    message: str,
    agent_id: str | None = None,
    context_variables: ContextVariables | None = None,
    user_id: str | None = None,
) -> str | None:
    """Send a message and stream the response.

    Returns:
        The ID of the last active agent, if any.
    """
    request = SendMessageRequest(
        message=message,
        agent_id=agent_id,
        context_variables=context_variables,
    )

    try:
        async with session.post(
            f"{BASE_URL}/sessions/{session_id}/messages/stream",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=request.model_dump(),
            timeout=TIMEOUT,
        ) as response:
            if response.status == 404 and user_id:
                # Session not found, try to create a new one
                print("\n⚠️ Session expired, creating a new one...")
                new_session = await create_session(session, user_id=user_id)
                print(f"✨ Created new session: {new_session.session_id}")

                # Retry with new session
                async with session.post(
                    f"{BASE_URL}/sessions/{new_session.session_id}/messages/stream",
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    json=request.model_dump(),
                    timeout=TIMEOUT,
                ) as retry_response:
                    retry_response.raise_for_status()
                    return await process_stream_response(retry_response, agent_id)

            response.raise_for_status()
            return await process_stream_response(response, agent_id)

    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            print("\n❌ Session not found and couldn't recreate it")
        else:
            print(f"\n❌ Error: {str(e)}")

        return None

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise e


async def process_stream_response(
    response: aiohttp.ClientResponse, agent_id: str | None
) -> str | None:
    """Process streaming response and return the last active agent ID.

    Args:
        response: The streaming response to process.
        agent_id: Current agent ID.

    Returns:
        The ID of the last active agent, if any.
    """
    if agent_id:
        print(f"\n🤖 {get_agent_title(agent_id)} is thinking...")
    else:
        print("\n🤖 Agent is thinking...")

    current_agent = agent_id
    first_chunk = True
    last_active_agent = None

    async for line in response.content:
        if not line:
            continue

        try:
            event_data = json.loads(line)
            debug_print(f"Received event: {event_data!r}")

            match event_data["type"]:
                case "agent_switch":
                    prev_agent = event_data.get("prev_agent", {}).get("id", current_agent)
                    next_agent = event_data["next_agent"]["id"]

                    if prev_agent:
                        prev_agent_title = get_agent_title(prev_agent)
                        next_agent_title = get_agent_title(next_agent)
                        print(f"\n🔄 Switching from {prev_agent_title} to {next_agent_title}...")
                    else:
                        next_agent_title = get_agent_title(next_agent)
                        print(f"\n🔄 Starting with {next_agent_title}...")

                    current_agent = next_agent
                    first_chunk = True

                case "agent_response_chunk":
                    if completion := event_data.get("response_chunk", {}).get("completion"):
                        # Get agent ID from the response chunk
                        chunk_agent_id = event_data.get("agent", {}).get("id")
                        if chunk_agent_id:
                            current_agent = chunk_agent_id

                        if content := completion.get("delta", {}).get("content"):
                            if first_chunk:
                                if current_agent:
                                    prefix = f"\n🤖 [{get_agent_title(current_agent)}]: "
                                else:
                                    prefix = "\n🤖 [Agent]: "
                                print(prefix, end="", flush=True)
                                first_chunk = False
                            print(content, end="", flush=True)

                case "agent_complete":
                    if agent := event_data.get("agent"):
                        last_active_agent = agent.get("id")

                case "complete":
                    if DEBUG:
                        print("\n✅ Stream complete.")
                    break

                case "error":
                    print(f"\n❌ Error: {event_data.get('error')}", file=sys.stderr)
                    return None

        except json.JSONDecodeError as e:
            print(f"\n❌ Error parsing response: {e}", file=sys.stderr)
            debug_print(f"Error details: {e}")
            continue

    print()
    return last_active_agent


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def delete_session(session: aiohttp.ClientSession, session_id: str) -> None:
    """Delete a chat session."""
    async with session.delete(
        f"{BASE_URL}/sessions/{session_id}",
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def list_sessions(session: aiohttp.ClientSession, user_id: str) -> list[SessionResponse]:
    """List all sessions for a user."""
    async with session.get(
        f"{BASE_URL}/sessions",
        params={"user_id": user_id},
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return [SessionResponse(**item) for item in data]


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def get_messages(session: aiohttp.ClientSession, session_id: str) -> list[dict[str, Any]]:
    """Get all messages from a chat session."""
    async with session.get(
        f"{BASE_URL}/sessions/{session_id}/messages",
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        response_data = await response.json()
        return response_data.get("messages", [])


async def display_messages(session: aiohttp.ClientSession, session_id: str) -> None:
    """Display message history for a session."""
    async with session.get(
        f"{BASE_URL}/sessions/{session_id}/messages",
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        data = await response.json()
        messages = data["messages"]

        if not messages:
            print("\n📭 No messages in this session")
            return

        print("\n📝 Message History:")
        print("=" * 50)
        for msg in messages:
            role = msg["role"].title()
            content = msg["content"]
            print(f"\n{role}: {content}")
        print("\n" + "=" * 50)


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def create_user(
    session: aiohttp.ClientSession,
    user_id: str,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> UserResponse:
    """Create a new user."""
    request = CreateUserRequest(user_id=user_id, name=name, metadata=metadata)
    async with session.post(
        f"{BASE_URL}/users",
        json=request.model_dump(),
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return UserResponse(**data)


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def delete_user(session: aiohttp.ClientSession, user_id: str) -> None:
    """Delete a user and their sessions."""
    async with session.delete(
        f"{BASE_URL}/users/{user_id}",
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def list_users(session: aiohttp.ClientSession) -> list[UserResponse]:
    """List all users."""
    async with session.get(
        f"{BASE_URL}/users",
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return [UserResponse(**user) for user in data]


@retry(
    max_retries=MAX_RETRIES,
    initial_delay=get_retry_delay(1),
    exception=EXCEPTION,
)
async def switch_user(session: aiohttp.ClientSession, user_id: str) -> list[dict[str, str]]:
    """Switch to a different user and get their sessions."""
    async with session.post(
        f"{BASE_URL}/users/{user_id}/switch",
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        return await response.json()


async def display_user_info(session: aiohttp.ClientSession, user_id: str) -> None:
    """Display user information."""
    users = await list_users(session)
    user = next((u for u in users if u.user_id == user_id), None)
    if not user:
        print("\n❌ User not found")
        return

    print("\n👤 User Information:")
    print("=" * 50)
    print(f"User ID: {user.user_id}")
    if user.name:
        print(f"Name: {user.name}")
    if user.metadata:
        print("Metadata:")
        for key, value in user.metadata.items():
            print(f"  {key}: {value}")
    print("=" * 50)


async def display_session_info(
    session: aiohttp.ClientSession,
    session_id: str,
    user_id: str,
    last_agent_id: str | None,
) -> None:
    """Display session information."""
    sessions = await list_sessions(session, user_id)
    current_session = next((s for s in sessions if s.session_id == session_id), None)
    if not current_session:
        print("\n❌ Session not found")
        return

    print("\n💬 Session Information:")
    print("=" * 50)
    print(f"Session ID: {session_id}")
    print(f"User ID: {current_session.user_id}")
    if last_agent_id:
        print(f"Last Agent: {get_agent_title(last_agent_id)}")
    print("=" * 50)


async def handle_session_command(
    session: aiohttp.ClientSession,
    command: str,
    args: list[str],
    user_id: str,
    current_session_id: str,
    last_agent_id: str | None,
) -> tuple[str, str | None]:
    """Handle session-related commands.

    Args:
        session: HTTP client session.
        command: The session command to execute.
        args: Command arguments.
        user_id: Current user ID.
        current_session_id: Current session ID.
        last_agent_id: Last active agent ID.

    Returns:
        Tuple of (new_session_id, new_last_agent_id).
    """
    match command:
        case "create":
            try:
                response = await create_session(session, user_id=user_id)
                print(f"\n✨ Created session: {response.session_id}")
                return response.session_id, None
            except Exception as e:
                print(f"\n❌ Error creating session: {str(e)}")
                return current_session_id, last_agent_id

        case "delete":
            if len(args) != 1:
                print("\n❌ Usage: /session delete <session_id>")
                return current_session_id, last_agent_id

            target_session_id = args[0]
            try:
                # Verify session exists and belongs to user
                sessions = await list_sessions(session, user_id)
                if not any(s.session_id == target_session_id for s in sessions):
                    print("\n❌ Session not found or belongs to another user")
                    return current_session_id, last_agent_id

                await delete_session(session, target_session_id)
                print(f"\n🗑️  Removed session: {target_session_id}")

                # If we removed the current session, create a new one
                if target_session_id == current_session_id:
                    response = await create_session(session, user_id=user_id)
                    print(f"✨ Created new session: {response.session_id}")
                    return response.session_id, None
                return current_session_id, last_agent_id

            except Exception as e:
                print(f"\n❌ Error removing session: {str(e)}")
                return current_session_id, last_agent_id

        case "list":
            try:
                sessions = await list_sessions(session, user_id)
                print("\nAvailable sessions:")
                for s in sessions:
                    current = "📍 " if s.session_id == current_session_id else "   "
                    print(f"{current}{s.session_id} (User: {s.user_id})")
                return current_session_id, last_agent_id
            except Exception as e:
                print(f"\n❌ Error listing sessions: {str(e)}")
                return current_session_id, last_agent_id

        case "switch":
            if len(args) != 1:
                print("\n❌ Usage: /session switch <session_id>")
                return current_session_id, last_agent_id

            new_session_id = args[0]
            try:
                # Verify session exists and belongs to user
                sessions = await list_sessions(session, user_id)
                if not any(s.session_id == new_session_id for s in sessions):
                    print("\n❌ Session not found or belongs to another user")
                    return current_session_id, last_agent_id

                print(f"\n🔄 Switched to session: {new_session_id}")
                return new_session_id, None

            except Exception as e:
                print(f"\n❌ Error switching session: {str(e)}")
                return current_session_id, last_agent_id

        case "info":
            await display_session_info(
                session=session,
                session_id=current_session_id,
                user_id=user_id,
                last_agent_id=last_agent_id,
            )
            return current_session_id, last_agent_id

        case _:
            print("\n❌ Unknown session command. Available commands:")
            print("  /session create - Create a new session")
            print("  /session delete <session_id> - Delete a session")
            print("  /session list - List all sessions")
            print("  /session switch <session_id> - Switch to another session")
            print("  /session info - Show current session info")
            return current_session_id, last_agent_id


async def run_repl() -> NoReturn:
    """Run the chat REPL."""
    print("\n🤖 Chat Client")
    print("\nCommands:")
    print("  /exit - Exit the client")
    print("  /help - Show this help message")
    print("  /debug - Toggle debug mode")
    print("  /history - Show message history")
    print("\nSession Commands:")
    print("  /session create - Create a new session")
    print("  /session delete <session_id> - Delete a session")
    print("  /session list - List all sessions")
    print("  /session switch <session_id> - Switch to another session")
    print("  /session info - Show current session info")
    print("\nUser Commands:")
    print("  /user create <user_id> [name] - Create a new user")
    print("  /user delete <user_id> - Delete a user")
    print("  /user list - List all users")
    print("  /user info - Show current user info")
    print("  /user switch <user_id> - Switch to another user")
    print("\nEnter your messages and press Enter.")
    print("\n" + "=" * 50 + "\n")

    # Create HTTP session with default settings
    connector = aiohttp.TCPConnector(
        limit=100,
        keepalive_timeout=120,
        force_close=False,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=TIMEOUT,
        headers={"Connection": "keep-alive"},
    ) as session:
        try:
            user_id = "repl_user"
            try:
                await create_user(session, user_id=user_id)
                print(f"\n✨ Created user: {user_id}")
            except Exception as e:
                if "User already exists" not in str(e):
                    raise
                print(f"\n📍 Using existing user: {user_id}")

            await display_user_info(session, user_id)

            # Create chat session
            response = await create_session(session, user_id=user_id)
            session_id = response.session_id
            print(f"\n✨ Created session: {session_id}")

            # Track the last active agent
            last_agent_id: str | None = None

            while True:
                try:
                    # Get user input with prompt_toolkit
                    user_input = await prompt("\n🗣️  Enter your message: ")

                    # Skip empty input
                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        parts = user_input.split()
                        command = parts[0]

                        match command:
                            case "/exit":
                                print("\n👋 Goodbye!")
                                sys.exit(0)
                            case "/help":
                                print("\nCommands:")
                                print("  /exit - Exit the client")
                                print("  /help - Show this help message")
                                print("  /debug - Toggle debug mode")
                                print("  /history - Show message history")
                                print("\nSession Commands:")
                                print("  /session create - Create a new session")
                                print("  /session delete <session_id> - Delete a session")
                                print("  /session list - List all sessions")
                                print("  /session switch <session_id> - Switch to another session")
                                print("  /session info - Show current session info")
                                print("\nUser Commands:")
                                print("  /user create <user_id> [name] - Create a new user")
                                print("  /user delete <user_id> - Delete a user")
                                print("  /user list - List all users")
                                print("  /user info - Show current user info")
                                print("  /user switch <user_id> - Switch to another user")
                                continue
                            case "/debug":
                                global DEBUG  # noqa: PLW0603
                                DEBUG = not DEBUG
                                print(f"\n🔧 Debug mode: {'enabled' if DEBUG else 'disabled'}")
                                continue
                            case "/history":
                                try:
                                    await display_messages(session, session_id)
                                except Exception as e:
                                    print(f"\n❌ Error displaying message history: {str(e)}")
                                continue
                            case "/session":
                                if len(parts) < 2:
                                    print("\n❌ Usage: /session <command> [args...]")
                                    continue

                                session_command = parts[1]
                                session_args = parts[2:]
                                session_id, last_agent_id = await handle_session_command(
                                    session=session,
                                    command=session_command,
                                    args=session_args,
                                    user_id=user_id,
                                    current_session_id=session_id,
                                    last_agent_id=last_agent_id,
                                )
                                continue
                            case "/user":
                                if len(parts) < 2:
                                    print("\n❌ Usage: /user <command> [args...]")
                                    continue

                                user_command = parts[1]
                                match user_command:
                                    case "create":
                                        if len(parts) < 3:
                                            print("\n❌ Usage: /user create <user_id> [name]")
                                            continue
                                        new_user_id = parts[2]
                                        new_user_name = (
                                            " ".join(parts[3:]) if len(parts) > 3 else None
                                        )
                                        try:
                                            await create_user(
                                                session,
                                                user_id=new_user_id,
                                                name=new_user_name,
                                            )
                                            print(f"\n✨ Created user: {new_user_id}")
                                        except Exception as e:
                                            print(f"\n❌ Error creating user: {str(e)}")
                                        continue

                                    case "delete":
                                        if len(parts) != 3:
                                            print("\n❌ Usage: /user delete <user_id>")
                                            continue
                                        target_user_id = parts[2]
                                        try:
                                            await delete_user(session, target_user_id)
                                            print(f"\n🗑️  Deleted user: {target_user_id}")
                                            if target_user_id == user_id:
                                                # Create a new user if we deleted the current one
                                                user_id = "repl_user"
                                                await create_user(session, user_id=user_id)
                                                response = await create_session(
                                                    session, user_id=user_id
                                                )
                                                session_id = response.session_id
                                                print(f"✨ Created new user: {user_id}")
                                                print(f"✨ Created new session: {session_id}")
                                                last_agent_id = None
                                        except Exception as e:
                                            print(f"\n❌ Error deleting user: {str(e)}")
                                        continue

                                    case "list":
                                        try:
                                            users = await list_users(session)
                                            print("\n👥 Available users:")
                                            for u in users:
                                                current = "📍 " if u.user_id == user_id else "   "
                                                name_str = f" ({u.name})" if u.name else ""
                                                print(f"{current}{u.user_id}{name_str}")
                                        except Exception as e:
                                            print(f"\n❌ Error listing users: {str(e)}")
                                        continue

                                    case "info":
                                        try:
                                            await display_user_info(session, user_id)
                                        except Exception as e:
                                            print(f"\n❌ Error displaying user info: {str(e)}")
                                        continue

                                    case "switch":
                                        if len(parts) != 3:
                                            print("\n❌ Usage: /user switch <user_id>")
                                            continue
                                        new_user_id = parts[2]
                                        try:
                                            # Get user's sessions
                                            sessions = await switch_user(session, new_user_id)
                                            if not sessions:
                                                # Create a new session if user has none
                                                response = await create_session(
                                                    session, user_id=new_user_id
                                                )
                                                session_id = response.session_id
                                                print(f"✨ Created new session: {session_id}")
                                            else:
                                                # Use the first available session
                                                first_session = SessionResponse(**sessions[0])
                                                session_id = first_session.session_id
                                                print(f"🔄 Using existing session: {session_id}")

                                            user_id = new_user_id
                                            last_agent_id = None
                                            print(f"\n🔄 Switched to user: {user_id}")

                                        except Exception as e:
                                            print(f"\n❌ Error switching user: {str(e)}")
                                        continue

                                    case _:
                                        print(
                                            "\n❌ Unknown user command. Type /help for available commands."
                                        )
                                        continue

                            case _:
                                print("\n❌ Unknown command. Type /help for available commands.")
                                continue

                    completed_agent_id = await send_message_and_stream(
                        session=session,
                        session_id=session_id,
                        message=user_input,
                        agent_id=last_agent_id,
                        user_id=user_id,  # Pass user_id for session recreation
                    )

                    if completed_agent_id:
                        last_agent_id = completed_agent_id

                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted by user. Goodbye!")
                    sys.exit(0)
                except EOFError:
                    print("\n\n👋 EOF received. Goodbye!")
                    sys.exit(0)
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}", file=sys.stderr)
                    continue

        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_repl())
