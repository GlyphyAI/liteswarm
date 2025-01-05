# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing_extensions import override

from liteswarm.chat.memory import LiteChatMemory
from liteswarm.chat.optimization import LiteChatOptimization, OptimizationStrategy
from liteswarm.chat.search import LiteChatSearch
from liteswarm.chat.session import ChatSession
from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planning import PlanningAgent
from liteswarm.experimental.swarm_team.response_repair import ResponseRepairAgent
from liteswarm.experimental.swarm_team.swarm_team import SwarmTeam
from liteswarm.types.chat import ChatMessage, RAGStrategyConfig
from liteswarm.types.collections import AsyncStream, ReturnItem, YieldItem, returnable
from liteswarm.types.context import ContextVariables
from liteswarm.types.events import SwarmEvent
from liteswarm.types.swarm import Agent, Message
from liteswarm.types.swarm_team import Artifact, PlanFeedbackCallback, TaskDefinition, TeamMember
from liteswarm.utils.messages import validate_messages
from liteswarm.utils.unwrap import unwrap_instructions


class LiteTeamChatSession(ChatSession[Artifact]):
    """In-memory implementation of team chat session execution.

    Manages a single team conversation with support for task planning,
    execution, and artifact generation. Uses SwarmTeam for coordinated
    task handling while maintaining conversation state.

    The implementation offers:
        - Team-based task execution
        - Plan creation and feedback
        - Artifact generation
        - Message persistence
        - Context optimization
        - Semantic search

    Examples:
        ```python
        # Create session with team configuration
        members = create_members()
        task_definitions = create_task_definitions()

        memory = LiteChatMemory()
        search = LiteChatSearch(memory=memory)
        optimization = LiteChatOptimization(memory=memory, search=search)

        session = LiteTeamChatSession(
            session_id="session_123",
            members=members,
            task_definitions=task_definitions,
            swarm=Swarm(),
            memory=memory,
            search=search,
            optimization=optimization,
        )


        # Optional feedback callback to approve or reject a plan
        def feedback_callback(plan: Plan) -> PlanFeedback:
            return ApprovePlan(type="approve")


        # Send message with feedback
        async for event in session.send_message(
            "Create a simple TODO list app",
            context_variables=ContextVariables(project="my_app"),
            feedback_callback=feedback_callback,
        ):
            if event.type == "agent_response_chunk":
                print(event.chunk.content)
        ```

    Notes:
        - Messages are stored in memory and lost on restart
        - Plan feedback can pause execution for user input
        - Team composition is fixed after initialization
        - Task definitions cannot be modified during execution
        - System messages are preserved between operations
    """

    def __init__(
        self,
        session_id: str,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        swarm: Swarm,
        memory: LiteChatMemory,
        search: LiteChatSearch,
        optimization: LiteChatOptimization,
        planning_agent: PlanningAgent | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
        max_feedback_attempts: int = 3,
    ) -> None:
        """Initialize a new team session instance.

        Creates a session with specified team composition and task
        definitions. Initializes SwarmTeam for execution coordination.

        Args:
            session_id: Unique identifier for this session.
            members: List of specialized agents with task capabilities.
            task_definitions: Task execution blueprints with instructions.
            swarm: Agent execution and event streaming.
            memory: Storage for message persistence.
            search: Semantic search over conversation history.
            optimization: Context optimization strategies.
            planning_agent: Custom agent for task planning.
            response_repair_agent: Custom agent for response repair.
            max_feedback_attempts: Maximum number of plan feedback attempts (default: 3).

        Notes:
            - All components are required and cannot be None
            - Team composition is fixed after initialization
            - Components should share compatible configurations
            - System messages are preserved between operations
        """
        self.session_id = session_id
        self._members = members
        self._task_definitions = task_definitions
        self._team = SwarmTeam(
            swarm=swarm,
            members=members,
            task_definitions=task_definitions,
            planning_agent=planning_agent,
            response_repair_agent=response_repair_agent,
        )
        self._memory = memory
        self._search = search
        self._optimization = optimization
        self._last_agent: Agent | None = None
        self._last_instructions: str | None = None
        self._max_feedback_attempts = max_feedback_attempts

    def _get_instructions(
        self,
        agent: Agent,
        context_variables: ContextVariables | None = None,
    ) -> str:
        """Get agent instructions with context variables applied.

        Args:
            agent: Agent to get instructions for.
            context_variables: Variables for instruction resolution.

        Returns:
            Processed instruction string.
        """
        return unwrap_instructions(agent.instructions, context_variables)

    @override
    async def search_messages(
        self,
        query: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ChatMessage]:
        """Search for messages in this session.

        Finds messages that are semantically similar to the query text.
        Results can be limited and filtered by similarity score.

        Args:
            query: Text to search for similar messages.
            max_results: Maximum number of messages to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of matching messages sorted by relevance.
        """
        search_results = await self._search.search(
            query=query,
            session_id=self.session_id,
            max_results=max_results,
            score_threshold=score_threshold,
        )

        return [message for message, _ in search_results]

    @override
    async def optimize_messages(
        self,
        model: str,
        strategy: OptimizationStrategy | None = None,
        rag_config: RAGStrategyConfig | None = None,
    ) -> list[ChatMessage]:
        """Optimize conversation messages using specified strategy.

        Applies optimization to reduce context size while preserving
        important information. Strategy determines the optimization
        approach.

        Args:
            model: Target language model identifier.
            strategy: Optimization strategy to use.
            rag_config: Configuration for RAG strategy.

        Returns:
            Optimized list of messages.
        """
        return await self._optimization.optimize_context(
            session_id=self.session_id,
            model=model,
            strategy=strategy,
            rag_config=rag_config,
        )

    @override
    async def get_messages(self) -> list[ChatMessage]:
        """Get all messages in this session.

        Retrieves the complete conversation history in chronological
        order from storage.

        Returns:
            List of messages in chronological order.
        """
        return await self._memory.get_messages(self.session_id)

    @override
    @returnable
    async def send_message(
        self,
        message: str,
        /,
        context_variables: ContextVariables | None = None,
        feedback_callback: PlanFeedbackCallback | None = None,
    ) -> AsyncStream[SwarmEvent, Artifact]:
        """Send a message to the team and stream execution events.

        Processes the message through plan creation and execution phases.
        Supports optional user feedback on the generated plan before
        proceeding with execution. Preserves system messages between
        operations for better context management.

        Args:
            message: Message content to process.
            context_variables: Variables for instruction resolution.
            feedback_callback: Optional callback for plan feedback.
                If provided, execution pauses after plan creation
                for user approval or rejection.

        Returns:
            ReturnableAsyncGenerator yielding events and returning final Artifact.

        Notes:
            - Plan rejection triggers replanning with feedback (max 3 attempts)
            - Messages are persisted after successful execution
            - Context variables affect all execution phases
            - System messages are preserved between operations
        """
        chat_messages = await self.get_messages()
        context_messages = validate_messages(chat_messages)
        context_messages.append(Message(role="user", content=message))

        feedback_attempts = 0
        while feedback_attempts < self._max_feedback_attempts:
            plan_stream = self._team.create_plan(
                messages=context_messages,
                context_variables=context_variables,
            )

            async for event in plan_stream:
                if event.type == "agent_begin":
                    instructions = self._get_instructions(event.agent, context_variables)
                    context_messages.append(Message(role="system", content=instructions))
                    self._last_agent = event.agent
                    self._last_instructions = instructions

                yield YieldItem(event)

            plan_result = await plan_stream.get_return_value()
            if plan_result.new_messages:
                context_messages.extend(plan_result.new_messages)

            if feedback_callback:
                feedback = await feedback_callback(plan_result.plan)
                if feedback.type == "reject":
                    feedback_attempts += 1
                    if feedback_attempts >= self._max_feedback_attempts:
                        raise ValueError(
                            f"Maximum feedback attempts ({self._max_feedback_attempts}) reached"
                        )
                    context_messages.append(Message(role="user", content=feedback.feedback))
                    continue

            execution_stream = self._team.execute_plan(
                plan_result.plan,
                messages=context_messages,
                context_variables=context_variables,
            )

            async for event in execution_stream:
                if event.type == "agent_begin":
                    instructions = self._get_instructions(event.agent, context_variables)
                    context_messages.append(Message(role="system", content=instructions))
                    self._last_agent = event.agent
                    self._last_instructions = instructions

                if event.type == "agent_complete":
                    context_messages.extend(event.messages)

                if event.type == "task_started":
                    for msg in reversed(event.messages):
                        if msg.role == "user":
                            context_messages.append(msg)
                            break

                yield YieldItem(event)

            artifact = await execution_stream.get_return_value()

            await self._memory.add_messages(context_messages, self.session_id)

            yield ReturnItem(artifact)
            return
