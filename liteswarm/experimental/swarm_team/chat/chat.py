# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from uuid import uuid4

from typing_extensions import override

from liteswarm.chat.chat import Chat
from liteswarm.chat.memory import LiteChatMemory
from liteswarm.chat.optimization import LiteChatOptimization
from liteswarm.chat.search import LiteChatSearch
from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.chat.session import LiteTeamChatSession
from liteswarm.experimental.swarm_team.planning import PlanningAgent
from liteswarm.experimental.swarm_team.response_repair import ResponseRepairAgent
from liteswarm.types.swarm_team import TaskDefinition, TeamMember


class LiteTeamChat(Chat[LiteTeamChatSession]):
    """In-memory implementation of team chat session management.

    Manages multiple team sessions with support for task execution,
    message persistence, and context optimization. Each session maintains
    its own team composition and task definitions.

    The implementation offers:
        - Team-based session management
        - Task definition handling
        - Member role management
        - Planning agent integration
        - Response repair capabilities

    Examples:
        ```python
        # Create team chat manager
        memory = LiteChatMemory()
        search = LiteChatSearch(memory=memory)
        optimization = LiteChatOptimization(memory=memory, search=search)
        chat = LiteTeamChat(
            swarm=Swarm(),
            memory=memory,
            search=search,
            optimization=optimization,
        )

        # Create session with team configuration
        members = create_members()
        task_definitions = create_task_definitions()
        session = await chat.create_session(
            user_id="user_123",
            members=members,
            task_definitions=task_definitions,
        )

        # Send message to team
        async for event in session.send_message(
            "Create a Flutter app",
            context_variables=ContextVariables(project="my_app"),
        ):
            if event.type == "agent_response_chunk":
                print(event.chunk.content)
        ```

    Notes:
        - Sessions are stored in memory and lost on restart
        - Each session maintains isolated team state
        - Optional components use default implementations if not provided
        - Planning and repair agents can be customized per session
    """

    def __init__(
        self,
        swarm: Swarm,
        memory: LiteChatMemory | None = None,
        search: LiteChatSearch | None = None,
        optimization: LiteChatOptimization | None = None,
    ) -> None:
        """Initialize a new team chat manager instance.

        Creates a manager with configurable components for message storage,
        search, and optimization. Initializes empty session mappings.

        Args:
            swarm: Agent execution and event streaming.
            memory: Storage for message persistence and retrieval.
            search: Semantic search over conversation history.
            optimization: Context optimization strategies.

        Notes:
            - Swarm instance is required and cannot be None
            - Optional components use default implementations
            - Components should share compatible configurations
        """
        self._swarm = swarm
        self._memory = memory or LiteChatMemory()
        self._search = search or LiteChatSearch(memory=self._memory)
        self._optimization = optimization or LiteChatOptimization(
            memory=self._memory,
            search=self._search,
        )
        self._sessions: dict[str, LiteTeamChatSession] = {}
        self._user_sessions: dict[str, set[str]] = {}

    @override
    async def create_session(
        self,
        user_id: str,
        session_id: str | None = None,
        *,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        planning_agent: PlanningAgent | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
    ) -> LiteTeamChatSession:
        """Create a new team session for a user.

        Creates a session with specified team composition and task
        definitions. Returns existing session if ID is already in use.

        Args:
            user_id: Identifier of the user owning the session.
            session_id: Optional custom session identifier.
            members: List of team members with roles.
            task_definitions: Available tasks for the team.
            planning_agent: Custom agent for task planning.
            response_repair_agent: Custom agent for response repair.

        Returns:
            New or existing team session.

        Notes:
            Uses UUID4 if no session ID is provided.
        """
        session_id = session_id or str(uuid4())
        if session_id in self._sessions:
            return self._sessions[session_id]

        session = LiteTeamChatSession(
            session_id=session_id,
            members=members,
            task_definitions=task_definitions,
            swarm=self._swarm,
            memory=self._memory,
            search=self._search,
            optimization=self._optimization,
            planning_agent=planning_agent,
            response_repair_agent=response_repair_agent,
        )

        self._sessions[session_id] = session
        self._user_sessions.setdefault(user_id, set()).add(session_id)

        return session

    @override
    async def get_session(self, session_id: str) -> LiteTeamChatSession | None:
        """Get a specific team session.

        Retrieves a session by its ID from in-memory storage.

        Args:
            session_id: Identifier of the session to retrieve.

        Returns:
            Team session if found, None otherwise.
        """
        if session_id not in self._sessions:
            return None

        return self._sessions[session_id]

    @override
    async def get_user_sessions(self, user_id: str) -> list[LiteTeamChatSession]:
        """Get all sessions belonging to a user.

        Retrieves all active team sessions for the user from storage.

        Args:
            user_id: Identifier of the user.

        Returns:
            List of user's team sessions.
        """
        session_ids = self._user_sessions.get(user_id, set())
        return [self._sessions[sid] for sid in session_ids]

    @override
    async def delete_session(self, session_id: str) -> None:
        """Delete a team session.

        Removes a session from storage and user mappings. Does nothing
        if the session doesn't exist.

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

        Removes all user's team sessions from storage.

        Args:
            user_id: Identifier of the user.
        """
        for session_id in self._user_sessions.get(user_id, set()):
            await self.delete_session(session_id)
