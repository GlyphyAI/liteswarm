import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from copy import deepcopy

import litellm
import orjson
from litellm import CustomStreamWrapper, acompletion
from litellm.types.utils import ChatCompletionDeltaToolCall, StreamingChoices

from liteswarm.types import (
    Agent,
    AgentResponse,
    CompletionResponse,
    ConversationState,
    Delta,
    Message,
    StreamHandler,
    ToolCallAgentResult,
    ToolCallMessageResult,
    ToolCallResult,
)
from liteswarm.utils import function_to_json

litellm.modify_params = True


class Swarm:
    def __init__(self, stream_handler: StreamHandler | None = None) -> None:
        self.active_agent: Agent | None = None
        self.agent_messages: list[Message] = []
        self.agent_queue: deque[Agent] = deque()
        self.stream_handler = stream_handler
        self.messages: list[Message] = []

    def _get_last_messages(
        self,
        limit: int | None = None,
        roles: list[str] | None = None,
    ) -> list[Message]:
        """Retrieve messages from the conversation history with optional filtering.

        Allows filtering the conversation history by message roles (e.g., 'user', 'assistant', 'tool')
        and limiting the number of messages returned. Messages are returned in chronological order.

        Args:
            limit: Maximum number of messages to return. If None, returns all matching messages.
            roles: List of roles to filter by (e.g., ['user', 'assistant']). If None, returns all roles.

        Returns:
            A list of Message objects matching the specified criteria, ordered chronologically.
        """
        if not roles:
            return self.messages[-limit:] if limit else self.messages

        filtered_messages = [msg for msg in self.messages if msg.role in roles and msg.content]

        return filtered_messages[-limit:] if limit else filtered_messages

    def _get_initial_conversation(
        self,
        agent: Agent,
        prompt: str | None = None,
    ) -> list[Message]:
        """Initialize a conversation with an agent's system instructions and optional prompt.

        Creates the initial conversation state for an agent by setting up the system message
        with the agent's instructions and optionally adding a user prompt message.

        Args:
            agent: The agent whose instructions should be used for the system message
            prompt: Optional initial user message to add to the conversation

        Returns:
            A list of Message objects containing the system instructions and optional prompt
        """
        conversation = [Message(role="system", content=agent.instructions)]

        if prompt:
            conversation.extend([Message(role="user", content=prompt)])

        return conversation

    def _summarize_messages(
        self,
        messages: list[Message],
        max_length: int,
        context_size: int,
    ) -> list[Message]:
        """Summarize a conversation history to maintain a manageable context window.

        Creates a condensed version of the conversation history by keeping the initial task,
        adding a summary placeholder, and including the most recent messages. This helps
        maintain context while preventing the conversation from growing too large.

        Args:
            messages: The complete list of messages to summarize
            max_length: Maximum number of messages to keep before summarizing
            context_size: Number of recent messages to preserve when summarizing

        Returns:
            A list of Message objects containing the summarized conversation
        """
        if len(messages) <= max_length:
            return messages

        first_message = messages[0]
        recent_messages = messages[-context_size:]
        if first_message in recent_messages:
            recent_messages.remove(first_message)

        return [
            first_message,
            Message(
                role="assistant",
                content="... Previous conversation summarized ...",
            ),
            *recent_messages,
        ]

    def _process_tool_call_messages(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Process messages to maintain proper tool call context and relationships.

        Ensures that tool response messages are properly paired with their corresponding
        tool call messages from the assistant. This maintains the logical flow of tool-based
        interactions while removing orphaned tool responses.

        Args:
            messages: List of messages to process

        Returns:
            A list of Message objects with properly maintained tool call relationships,
            where each tool response is preceded by its corresponding tool call
        """
        processed_messages: list[Message] = []
        tool_call_map: dict[str, Message] = {}

        for message in messages:
            if message.role == "assistant":
                tool_calls = message.tool_calls or []
                for tool_call in tool_calls:
                    if tool_call_id := tool_call.id:
                        tool_call_map[tool_call_id] = message
                processed_messages.append(message)
            elif message.role == "tool":
                tool_call_id = message.tool_call_id
                if tool_call_id and tool_call_id in tool_call_map:
                    processed_messages.append(message)
            else:
                processed_messages.append(message)

        return processed_messages

    def _prepare_agent_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        max_length: int = 6,
        context_size: int = 5,
    ) -> list[Message]:
        """Prepare context for a new agent, maintaining conversation history.

        Args:
            agent: The agent to prepare context for
            prompt: The initial user message to add to the conversation
            max_length: Maximum total messages to include
            context_size: Number of recent messages to keep when truncating

        Returns:
            A list of prepared messages for the agent
        """
        messages = self._get_initial_conversation(agent, prompt)

        # Get full relevant history
        relevant_history = self._get_last_messages(roles=["user", "assistant", "tool"])

        # Summarize relevant history if necessary
        relevant_history = self._summarize_messages(relevant_history, max_length, context_size)

        # Process messages to maintain tool call context
        relevant_history = self._process_tool_call_messages(relevant_history)

        # Make sure the last message is a user message
        last_message = relevant_history[-1]
        if last_message.role != "user":
            relevant_history.append(
                Message(
                    role="user",
                    content="Please continue with the task based on the context above.",
                )
            )

        messages.extend(relevant_history)

        return messages

    async def _process_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent,
    ) -> ToolCallResult | None:
        """Process a single tool call and return its result.

        This method handles the execution of a single function call, including error handling
        and result transformation. It supports both regular function results and agent switching.

        Args:
            tool_call: The tool call to process, containing function name and arguments
            agent: The agent that initiated the tool call

        Returns:
            A ToolCallResult object if the call was successful, None if the tool call failed
            or the function wasn't found

        Raises:
            ValueError: If the function arguments are invalid
            Exception: Any exception raised by the function tool itself
        """
        function_name = tool_call.function.name
        function_tools_map = {tool.__name__: tool for tool in agent.tools}

        if function_name not in function_tools_map:
            return None

        if self.stream_handler:
            await self.stream_handler.on_tool_call(tool_call, agent)

        try:
            args = orjson.loads(tool_call.function.arguments)
            function_tool = function_tools_map[function_name]
            function_result = function_tool(**args)

            tool_call_result: ToolCallResult | None = None
            if isinstance(function_result, Agent):
                tool_call_result = ToolCallAgentResult(
                    tool_call=tool_call,
                    agent=function_result,
                )
            else:
                tool_call_result = ToolCallMessageResult(
                    tool_call=tool_call,
                    message=Message(
                        role="tool",
                        content=orjson.dumps(function_result).decode(),
                        tool_call_id=tool_call.id,
                    ),
                )

            if self.stream_handler:
                await self.stream_handler.on_tool_call_result(tool_call_result, agent)

            return tool_call_result

        except Exception as e:
            if self.stream_handler:
                await self.stream_handler.on_error(e, agent)

            return None

    async def _process_tool_calls(
        self,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> list[ToolCallResult]:
        """Process multiple tool calls efficiently.

        For a single tool call, processes it directly to avoid concurrency overhead.
        For multiple tool calls, processes them concurrently using asyncio.gather().

        Args:
            tool_calls: List of tool calls to process

        Returns:
            List of successful tool call results, filtering out any failed calls

        Raises:
            ValueError: If there is no active agent to process the tool calls
        """
        agent = self.active_agent
        if not agent:
            raise ValueError("No active agent to process tool calls.")

        # Process single tool calls directly
        if len(tool_calls) == 1:
            result = await self._process_tool_call(tool_calls[0], agent)
            return [result] if result is not None else []

        # Process multiple tool calls concurrently
        tasks = [self._process_tool_call(tool_call, agent) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks)

        return [result for result in results if result is not None]

    async def _get_completion_response(
        self,
        agent_messages: list[Message],
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Get completion response for the currently active agent.

        Args:
            agent_messages: The messages to send to the agent

        Yields:
            CompletionResponse objects containing the current delta and finish reason

        Raises:
            ValueError: If there is no active agent
        """
        if not self.active_agent:
            raise ValueError("No active agent")

        messages = [message.model_dump(exclude_none=True) for message in agent_messages]
        tools = [function_to_json(tool) for tool in self.active_agent.tools]
        agent_params = self.active_agent.params or {}

        response_stream = await acompletion(
            model=self.active_agent.model,
            messages=messages,
            stream=True,
            tools=tools,
            tool_choice=self.active_agent.tool_choice,
            parallel_tool_calls=self.active_agent.parallel_tool_calls,
            **agent_params,
        )

        if not isinstance(response_stream, CustomStreamWrapper):
            raise TypeError("Expected a CustomStreamWrapper instance.")

        async for chunk in response_stream:
            choice = chunk.choices[0]
            if not isinstance(choice, StreamingChoices):
                raise TypeError("Expected a StreamingChoices instance.")

            delta = Delta.from_delta(choice.delta)
            finish_reason = choice.finish_reason

            yield CompletionResponse(
                delta=delta,
                finish_reason=finish_reason,
            )

    async def _process_agent_response(
        self,
        agent_messages: list[Message],
    ) -> AsyncGenerator[AgentResponse, None]:
        """Process agent responses and yield updates with accumulated state.

        Streams the agent's responses while maintaining the overall state of the conversation,
        including accumulated content and tool calls.

        Args:
            agent_messages: The messages to send to the agent

        Yields:
            AgentResponse objects containing the current delta and accumulated state

        Raises:
            ValueError: If there is no active agent
            TypeError: If the response stream is not of the expected type
        """
        full_content = ""
        full_tool_calls: list[ChatCompletionDeltaToolCall] = []

        async for completion_response in self._get_completion_response(agent_messages):
            delta = completion_response.delta
            finish_reason = completion_response.finish_reason

            if delta.content:
                full_content += delta.content

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not isinstance(tool_call, ChatCompletionDeltaToolCall):
                        continue

                    if tool_call.id:
                        full_tool_calls.append(tool_call)
                    elif full_tool_calls:
                        last_tool_call = full_tool_calls[-1]
                        last_tool_call.function.arguments += tool_call.function.arguments

            if self.stream_handler:
                await self.stream_handler.on_stream(delta, self.active_agent)

            yield AgentResponse(
                delta=delta,
                finish_reason=finish_reason,
                content=full_content,
                tool_calls=full_tool_calls,
            )

    async def _handle_tool_call_result(
        self,
        result: ToolCallResult,
        agent_messages: list[Message],
    ) -> None:
        """Handle a single tool call result by updating conversation state.

        Updates the message history and agent state based on the tool call result.
        Handles both message results and agent switching.

        Args:
            result: The tool call result to process
            agent_messages: The current conversation messages to update
        """
        match result:
            case ToolCallMessageResult() as message_result:
                self.messages.append(message_result.message)
                agent_messages.append(message_result.message)

            case ToolCallAgentResult() as agent_result:
                tool_call_message = Message(
                    role="tool",
                    content=f"Switching to agent {agent_result.agent.agent_id}",
                    tool_call_id=result.tool_call.id,
                )

                self.messages.append(tool_call_message)
                agent_messages.append(tool_call_message)
                self.agent_queue.append(agent_result.agent)
                self.active_agent = None

    async def _process_assistant_response(
        self,
        content: str | None,
        tool_calls: list[ChatCompletionDeltaToolCall],
        agent_messages: list[Message],
    ) -> None:
        """Process the assistant's complete response and handle any tool calls.

        Creates an assistant message with the response content and processes any tool calls.
        Updates the conversation history with results.

        Args:
            content: The assistant's response content
            tool_calls: List of tool calls made by the assistant
            agent_messages: The current conversation messages to update
        """
        assistant_message = Message(
            role="assistant",
            content=content or None,
        )

        if tool_calls:
            # Add tool calls to the assistant message
            assistant_message.tool_calls = tool_calls
            self.messages.append(assistant_message)
            agent_messages.append(assistant_message)

            # Process tool calls and get results
            results = await self._process_tool_calls(tool_calls)

            # Add tool results to messages
            for result in results:
                await self._handle_tool_call_result(result, agent_messages)
        else:
            self.messages.append(assistant_message)
            agent_messages.append(assistant_message)

    async def _handle_agent_switch(self) -> list[Message]:
        """Handle switching to the next agent in the queue.

        Manages the transition between agents, including context preparation
        and notification of the switch through the stream handler.

        Returns:
            The new agent's prepared context messages

        Raises:
            IndexError: If there are no agents in the queue
        """
        previous_agent = self.active_agent
        next_agent = self.agent_queue.popleft()
        agent_messages = self._prepare_agent_context(next_agent)
        self.active_agent = next_agent

        if self.stream_handler:
            await self.stream_handler.on_agent_switch(previous_agent, next_agent)

        return agent_messages

    async def stream(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        cleanup: bool = True,
    ) -> AsyncGenerator[AgentResponse, None]:
        """Stream thoughts and actions from the agent system.

        Manages the entire conversation flow, including:
        - Initial context setup
        - Agent response processing
        - Tool call handling
        - Agent switching
        - Error handling and cleanup

        Args:
            agent: The initial agent to start the conversation
            prompt: The initial prompt to send to the agent
            messages: Optional list of previous messages for context
            cleanup: Whether to clear agent state after completion. If False,
                maintains the last active agent for subsequent interactions.

        Yields:
            AgentResponse objects containing incremental updates from the agent

        Raises:
            ValueError: If there is no active agent
            Exception: Any errors that occur during processing
        """
        if messages:
            self.messages.extend(messages)

        if self.active_agent is None:
            self.active_agent = agent
            self.messages.extend(self._get_initial_conversation(agent, prompt))
            self.agent_messages = deepcopy(self.messages)
        else:
            user_message = Message(role="user", content=prompt)
            self.messages.append(user_message)
            self.agent_messages.append(user_message)

        try:
            while self.active_agent or self.agent_queue:
                if not self.active_agent and self.agent_queue:
                    self.agent_messages = await self._handle_agent_switch()

                last_content = ""
                last_tool_calls: list[ChatCompletionDeltaToolCall] = []

                async for agent_response in self._process_agent_response(self.agent_messages):
                    yield agent_response
                    last_content = agent_response.content
                    last_tool_calls = agent_response.tool_calls

                await self._process_assistant_response(
                    last_content,
                    last_tool_calls,
                    self.agent_messages,
                )

                if not last_tool_calls and not self.agent_queue:
                    break

        except Exception as e:
            if self.stream_handler:
                await self.stream_handler.on_error(e, self.active_agent)
            raise

        finally:
            if self.stream_handler:
                await self.stream_handler.on_complete(self.messages, self.active_agent)

            if cleanup:
                self.active_agent = None
                self.agent_messages = []
                self.agent_queue.clear()

    async def execute(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        cleanup: bool = True,
    ) -> ConversationState:
        """Execute the agent's task and return the final conversation state.

        A high-level method that manages the entire conversation flow and collects
        the final results. Uses stream() internally but provides a simpler interface
        for cases where streaming updates aren't needed.

        Args:
            agent: The agent to execute the task
            prompt: The prompt describing the task
            messages: Optional list of previous messages for context
            cleanup: Whether to clear agent state after completion. If False,
                maintains the last active agent for subsequent interactions.

        Returns:
            ConversationState containing the final response content and all messages

        Raises:
            ValueError: If there is no active agent
            Exception: Any errors that occur during processing
        """
        full_response = ""
        response_stream = self.stream(agent, prompt, messages, cleanup)

        async for agent_response in response_stream:
            if agent_response.content:
                full_response = agent_response.content

        return ConversationState(
            content=full_response,
            agent=self.active_agent,
            agent_messages=self.agent_messages,
            messages=self.messages,
        )
