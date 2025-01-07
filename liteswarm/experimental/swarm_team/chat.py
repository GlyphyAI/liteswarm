# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from typing_extensions import override

from liteswarm.chat.chat import Chat
from liteswarm.chat.memory import LiteChatMemory
from liteswarm.chat.optimization import LiteChatOptimization, OptimizationStrategy
from liteswarm.chat.search import LiteChatSearch
from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planning import LitePlanningAgent, PlanningAgent
from liteswarm.experimental.swarm_team.response_repair import ResponseRepairAgent
from liteswarm.experimental.swarm_team.swarm_team import SwarmTeam
from liteswarm.types.chat import ChatMessage, RAGStrategyConfig
from liteswarm.types.collections import AsyncStream, ReturnItem, YieldItem, returnable
from liteswarm.types.context import ContextVariables
from liteswarm.types.events import SwarmEvent
from liteswarm.types.swarm import Agent, Message
from liteswarm.types.swarm_team import (
    Artifact,
    PlanFeedbackCallback,
    TaskDefinition,
    TeamMember,
)
from liteswarm.utils.messages import validate_messages
from liteswarm.utils.misc import resolve_agent_instructions


class LiteTeamChat(Chat[Artifact]):
    """In-memory implementation of team chat conversation execution.

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
        # Create chat with team configuration
        members = create_members()
        task_definitions = create_task_definitions()

        memory = LiteChatMemory()
        search = LiteChatSearch(memory=memory)
        optimization = LiteChatOptimization(memory=memory, search=search)

        chat = LiteTeamChat(
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
        async for event in chat.send_message(
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
        swarm: Swarm | None = None,
        members: list[TeamMember] | None = None,
        task_definitions: list[TaskDefinition[Any, Any]] | None = None,
        memory: LiteChatMemory | None = None,
        search: LiteChatSearch | None = None,
        optimization: LiteChatOptimization | None = None,
        planning_agent: PlanningAgent | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
        max_feedback_attempts: int = 3,
    ) -> None:
        """Initialize a new team chat instance.

        Creates a chat with specified team composition and task
        definitions. Initializes SwarmTeam for execution coordination.

        Args:
            swarm: Agent execution and event streaming.
            members: List of specialized agents with task capabilities.
            task_definitions: Task execution blueprints with instructions.
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
        self._swarm = swarm or Swarm()
        self._members = members or []
        self._task_definitions = task_definitions or []
        self._planning_agent = planning_agent or LitePlanningAgent(
            swarm=self._swarm,
            task_definitions=self._task_definitions,
            response_repair_agent=response_repair_agent,
        )
        self._team = SwarmTeam(
            swarm=self._swarm,
            members=self._members,
            task_definitions=self._task_definitions,
            planning_agent=self._planning_agent,
            response_repair_agent=response_repair_agent,
        )
        self._memory = memory or LiteChatMemory()
        self._search = search or LiteChatSearch(memory=self._memory)
        self._optimization = optimization or LiteChatOptimization(
            memory=self._memory,
            search=self._search,
        )
        self._last_agent: Agent | None = None
        self._last_instructions: str | None = None
        self._max_feedback_attempts = max_feedback_attempts

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

        def add_system_message(agent: Agent) -> None:
            instructions = resolve_agent_instructions(agent, context_variables)
            context_messages.append(Message(role="system", content=instructions))
            self._last_agent = agent
            self._last_instructions = instructions

        feedback_attempts = 0
        while feedback_attempts < self._max_feedback_attempts:
            plan_stream = self._team.create_plan(
                messages=[*context_messages, Message(role="user", content=message)],
                context_variables=context_variables,
            )

            async for event in plan_stream:
                if event.type == "execution_start":
                    if self._last_agent != event.agent:
                        add_system_message(event.agent)

                    context_messages.append(Message(role="user", content=message))

                if event.type == "agent_start":
                    if self._last_agent != event.agent:
                        add_system_message(event.agent)

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

            task_instructions: str | None = None
            async for event in execution_stream:
                if event.type == "task_start":
                    task_instructions = event.task_instructions

                if event.type == "execution_start":
                    if self._last_agent != event.agent:
                        add_system_message(event.agent)

                    if task_instructions:
                        context_messages.append(Message(role="user", content=task_instructions))

                if event.type == "agent_complete":
                    context_messages.extend(event.messages)

                yield YieldItem(event)

            artifact = await execution_stream.get_return_value()

            await self._memory.add_messages(context_messages)

            yield ReturnItem(artifact)
            return

    @override
    async def get_messages(self) -> list[ChatMessage]:
        """Get all messages in this conversation.

        Retrieves the complete conversation history in chronological
        order from storage.

        Returns:
            List of messages in chronological order.
        """
        return await self._memory.get_messages()

    @override
    async def search_messages(
        self,
        query: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
        index_messages: bool = True,
    ) -> list[ChatMessage]:
        """Search for messages in this conversation.

        Finds messages that are semantically similar to the query text.
        Results can be limited and filtered by similarity score.

        Args:
            query: Text to search for similar messages.
            max_results: Maximum number of messages to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).
            index_messages: Whether to update index before search.

        Returns:
            List of matching messages sorted by relevance.
        """
        if index_messages:
            await self._search.index()

        search_results = await self._search.search(
            query=query,
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
            model=model,
            strategy=strategy,
            rag_config=rag_config,
        )
