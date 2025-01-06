# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Protocol, TypeVar

from typing_extensions import override

from liteswarm.chat.memory import LiteChatMemory
from liteswarm.chat.optimization import LiteChatOptimization, OptimizationStrategy
from liteswarm.chat.search import LiteChatSearch
from liteswarm.core.swarm import Swarm
from liteswarm.types.chat import ChatMessage, ChatResponse, RAGStrategyConfig
from liteswarm.types.collections import (
    AsyncStream,
    ReturnableAsyncGenerator,
    ReturnItem,
    YieldItem,
    returnable,
)
from liteswarm.types.context import ContextVariables
from liteswarm.types.events import SwarmEvent
from liteswarm.types.swarm import Agent, Message
from liteswarm.utils.messages import validate_messages
from liteswarm.utils.unwrap import unwrap_instructions

ReturnType = TypeVar("ReturnType")
"""Type variable for chat return type."""


class Chat(Protocol[ReturnType]):
    """Protocol for stateful conversations using Swarm runtime.

    Provides a standard interface for maintaining conversation state
    while using Swarm for agent execution. Implementations can use
    different storage backends while maintaining consistent state access.

    Type Parameters:
        ReturnType: Type returned by message sending operations.

    Examples:
        ```python
        class MyChat(Chat[ChatResponse]):
            async def send_message(
                self,
                message: str,
                /,
                agent: Agent,
                context_variables: ContextVariables | None = None,
            ) -> ReturnableAsyncGenerator[SwarmEvent, ChatResponse]:
                # Process message and generate response
                async for event in self._process_message(message, agent):
                    yield YieldItem(event)
                yield ReturnItem(ChatResponse(...))


        # Use custom chat implementation
        chat = MyChat()
        async for event in chat.send_message(
            "Hello!",
            agent=my_agent,
            context_variables={"user": "Alice"},
        ):
            print(event)
        ```

    Notes:
        - Each chat maintains isolated conversation state
        - All operations are asynchronous by framework design
        - Message order must be preserved
        - Search and optimization are optional capabilities
    """

    def send_message(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> ReturnableAsyncGenerator[SwarmEvent, ReturnType]:
        """Send message and get response with conversation history.

        Processes the message using the specified agent, applying context
        and streaming events for real-time updates.

        Args:
            message: Message content to send.
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            ReturnableAsyncGenerator yielding events and returning response.
        """
        ...

    async def get_messages(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Get conversation history.

        Retrieves the complete conversation history in chronological
        order from storage.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of messages in chronological order.
        """
        ...

    async def search_messages(
        self,
        query: str,
        max_results: int | None = None,
        score_threshold: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Search conversation history.

        Finds messages that are semantically similar to the query text.
        Results can be limited and filtered by similarity score.

        Args:
            query: Text to search for similar messages.
            max_results: Maximum number of messages to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            List of matching messages sorted by relevance.
        """
        ...

    async def optimize_messages(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Optimize conversation history to reduce context size.

        Applies optimization strategies to reduce context size while
        preserving important information and relationships.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.

        Returns:
            Optimized list of messages.
        """
        ...


class LiteChat(Chat[ChatResponse]):
    """In-memory implementation of stateful chat conversations.

    Manages conversation state using in-memory storage while leveraging
    Swarm for message processing. Supports search, optimization, and
    context management through optional components.

    The implementation offers:
        - Message persistence with ChatMemory
        - Semantic search capabilities
        - Context optimization strategies
        - Agent execution through Swarm
        - Real-time event streaming

    Examples:
        ```python
        # Create chat with components
        chat = LiteChat(
            swarm=Swarm(),
            memory=LiteChatMemory(),
            search=LiteChatSearch(),
            optimization=LiteChatOptimization(),
        )

        # Send message with context
        async for event in chat.send_message(
            "Hello!",
            agent=my_agent,
            context_variables=ContextVariables(user_name="Alice"),
        ):
            if event.type == "agent_response_chunk":
                print(event.chunk.content)
        ```

    Notes:
        - Messages are stored in memory and lost on restart
        - Agent state persists within conversation scope
        - Search requires proper index maintenance
        - Optimization affects response latency
    """

    def __init__(
        self,
        swarm: Swarm | None = None,
        memory: LiteChatMemory | None = None,
        search: LiteChatSearch | None = None,
        optimization: LiteChatOptimization | None = None,
    ) -> None:
        """Initialize a new chat instance.

        Creates a chat with message storage, search, and optimization
        capabilities. Maintains conversation state and agent execution
        through the provided components.

        Args:
            swarm: Agent execution and event streaming.
            memory: Storage for message persistence and retrieval.
            search: Semantic search over conversation history.
            optimization: Context optimization strategies.

        Notes:
            - Components are initialized with defaults if not provided
            - Components should share compatible configurations
            - State is isolated from other chat instances
        """
        self._swarm = swarm or Swarm()
        self._memory = memory or LiteChatMemory()
        self._search = search or LiteChatSearch(memory=self._memory)
        self._optimization = optimization or LiteChatOptimization(
            memory=self._memory,
            search=self._search,
        )
        self._last_agent: Agent | None = None
        self._last_instructions: str | None = None

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
    @returnable
    async def send_message(
        self,
        message: str,
        /,
        agent: Agent,
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEvent, ChatResponse]:
        """Send message and stream response events.

        Processes the message using the specified agent, applying context
        and streaming events for real-time updates. Maintains agent state
        and instruction history within the conversation.

        Args:
            message: Message content to send.
            agent: Agent to process the message.
            context_variables: Variables for instruction resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning ChatResponse.

        Notes:
            System instructions are added when agent or variables change.
        """
        context_messages: list[Message] = []
        instructions = self._get_instructions(agent, context_variables)

        if self._last_agent != agent or self._last_instructions != instructions:
            context_messages.append(Message(role="system", content=instructions))
            self._last_agent = agent
            self._last_instructions = instructions

        context_messages.append(Message(role="user", content=message))

        chat_messages = await self.get_messages()
        stream = self._swarm.stream(
            agent=agent,
            messages=[*validate_messages(chat_messages), *context_messages],
            context_variables=context_variables,
        )

        async for event in stream:
            if event.type == "agent_switch":
                instructions = self._get_instructions(event.next_agent, context_variables)
                context_messages.append(Message(role="system", content=instructions))
                self._last_agent = event.next_agent
                self._last_instructions = instructions

            if event.type == "agent_complete":
                context_messages.extend(event.messages)

            yield YieldItem(event)

        await self._memory.add_messages(context_messages)

        result = await stream.get_return_value()
        yield ReturnItem(
            ChatResponse(
                last_agent=result.last_agent,
                new_messages=result.new_messages,
                all_messages=result.all_messages,
                agent_responses=result.agent_responses,
            )
        )

    @override
    async def get_messages(self) -> list[ChatMessage]:
        """Get all messages in conversation history.

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
        """Search for messages in conversation history.

        Finds messages that are semantically similar to the query text.
        Updates the search index before searching.

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
        """Optimize conversation history using specified strategy.

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
