import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from copy import deepcopy

import litellm
import orjson
from litellm import CustomStreamWrapper, acompletion
from litellm.types.utils import ChatCompletionDeltaToolCall, StreamingChoices

from liteswarm.summarizer import LiteSummarizer, Summarizer
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
    def __init__(
        self,
        stream_handler: StreamHandler | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.active_agent: Agent | None = None
        self.agent_messages: list[Message] = []
        self.agent_queue: deque[Agent] = deque()
        self.stream_handler = stream_handler
        self.full_history: list[Message] = []
        self.working_history: list[Message] = []
        self.summarizer = summarizer or LiteSummarizer()

    async def _prepare_agent_context(
        self,
        agent: Agent,
        prompt: str | None = None,
    ) -> list[Message]:
        """Prepare the agent's context using the working history.

        Creates initial context for an agent by combining:
        1. Agent's system instructions
        2. Optional user prompt
        3. Filtered working history (excluding system messages)

        Args:
            agent: The agent whose context is being prepared
            prompt: Optional initial user message to add

        Returns:
            List of messages representing the agent's context
        """
        initial_messages = [Message(role="system", content=agent.instructions)]
        if prompt:
            initial_messages.append(Message(role="user", content=prompt))

        filtered_history = [msg for msg in self.working_history if msg.role != "system"]

        return initial_messages + filtered_history

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

    async def _process_tool_call_result(
        self,
        result: ToolCallResult,
    ) -> Message:
        """Process a tool call result and return a message to update the conversation.

        Updates message history and agent state based on the tool call result.
        Handles both message results and agent switching.

        Args:
            result: Tool call result to process

        Returns:
            Message to update the conversation history
        """
        tool_message: Message

        match result:
            case ToolCallMessageResult() as message_result:
                tool_message = message_result.message

            case ToolCallAgentResult() as agent_result:
                tool_message = Message(
                    role="tool",
                    content=f"Switching to agent {agent_result.agent.agent_id}",
                    tool_call_id=result.tool_call.id,
                )

                self.agent_queue.append(agent_result.agent)
                self.active_agent = None

        return tool_message

    async def _process_assistant_response(
        self,
        content: str | None,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> list[Message]:
        """Process the assistant's response and any tool calls.

        Creates messages from the assistant's response and processes any tool calls,
        maintaining proper message order and relationships.

        Args:
            content: Text content of the assistant's response
            tool_calls: List of tool calls to process

        Returns:
            List of messages including assistant response and tool results
        """
        messages: list[Message] = [
            Message(
                role="assistant",
                content=content or None,
                tool_calls=tool_calls if tool_calls else None,
            )
        ]

        if tool_calls:
            tool_call_results = await self._process_tool_calls(tool_calls)
            for tool_call_result in tool_call_results:
                tool_message = await self._process_tool_call_result(tool_call_result)
                messages.append(tool_message)

        return messages

    async def _handle_agent_switch(self) -> None:
        """Handle switching to the next agent in the queue.

        Manages the transition between agents by:
        1. Updating agent state
        2. Preparing new agent's context
        3. Notifying stream handler of the switch

        Raises:
            IndexError: If there are no agents in the queue
        """
        previous_agent = self.active_agent
        next_agent = self.agent_queue.popleft()
        self.agent_messages = await self._prepare_agent_context(next_agent)
        self.active_agent = next_agent

        if self.stream_handler:
            await self.stream_handler.on_agent_switch(previous_agent, next_agent)

    async def _update_working_history(self) -> None:
        """Update working history by summarizing full history when needed.

        Checks if working history needs summarization and updates it using
        the full history while maintaining token limits and context coherence.
        """
        if self.summarizer.needs_summarization(self.working_history):
            self.working_history = await self.summarizer.summarize_history(self.full_history)

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
            self.full_history = messages.copy()
            await self._update_working_history()

        if self.active_agent is None:
            self.active_agent = agent
            initial_context = await self._prepare_agent_context(agent, prompt)
            self.full_history = initial_context.copy()
            self.working_history = initial_context.copy()
            self.agent_messages = initial_context.copy()
        else:
            user_message = Message(role="user", content=prompt)
            self.full_history.append(user_message)
            self.working_history.append(user_message)
            self.agent_messages.append(user_message)

        try:
            while self.active_agent or self.agent_queue:
                if not self.active_agent and self.agent_queue:
                    await self._handle_agent_switch()

                last_content = ""
                last_tool_calls: list[ChatCompletionDeltaToolCall] = []

                async for agent_response in self._process_agent_response(self.agent_messages):
                    yield agent_response
                    last_content = agent_response.content
                    last_tool_calls = agent_response.tool_calls

                new_messages = await self._process_assistant_response(
                    last_content,
                    last_tool_calls,
                )

                self.full_history.extend(new_messages)
                self.working_history.extend(new_messages)
                self.agent_messages.extend(new_messages)

                await self._update_working_history()

                if not last_tool_calls and not self.agent_queue:
                    break

        except Exception as e:
            if self.stream_handler:
                await self.stream_handler.on_error(e, self.active_agent)
            raise

        finally:
            if self.stream_handler:
                await self.stream_handler.on_complete(self.full_history, self.active_agent)

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
            messages=self.full_history,
        )
