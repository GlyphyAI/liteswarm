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
    ConversationState,
    Message,
    StreamHandler,
    ToolCallAgentResult,
    ToolCallMessageResult,
    ToolCallResult,
    TypedDelta,
)
from liteswarm.utils import function_to_json

litellm.modify_params = True


class Swarm:
    def __init__(self, stream_handler: StreamHandler | None = None) -> None:
        self.active_agent: Agent | None = None
        self.agent_queue: deque[Agent] = deque()
        self.stream_handler = stream_handler
        self.messages: list[Message] = []

    def _get_last_messages(
        self,
        limit: int | None = None,
        roles: list[str] | None = None,
    ) -> list[Message]:
        """Get the last messages from the full history.

        Args:
            limit: The maximum number of messages to return
            roles: The roles to filter by

        Returns:
            The last messages from the full history filtered by roles and limit
        """
        # If no roles specified, use all messages
        if not roles:
            return self.messages[-limit:] if limit else self.messages

        # Otherwise filter by roles
        filtered_messages = [
            msg for msg in self.messages if msg.get("role") in roles and msg.get("content")
        ]

        return filtered_messages[-limit:] if limit else filtered_messages

    def _get_initial_conversation(
        self,
        agent: Agent,
        message: str | None = None,
    ) -> list[Message]:
        """Get the initial conversation for a given agent, including relevant context.

        Args:
            agent: The agent to prepare context for
            message: The message to add to the conversation
        """
        # Initial system message
        conversation: list[Message] = [{"role": "system", "content": agent.instructions}]

        # Add the current prompt/context
        if message:
            conversation.extend([{"role": "user", "content": message}])

        return conversation

    def _prepare_agent_context(
        self,
        agent: Agent,
        max_length: int = 6,
        context_size: int = 5,
    ) -> list[Message]:
        """Prepare context for a new agent, maintaining conversation history.

        Args:
            agent: The agent to prepare context for
            max_length: Maximum total messages to include
            context_size: Number of recent messages to keep when truncating
        """
        messages: list[Message] = [{"role": "system", "content": agent.instructions}]

        # Get full relevant history
        relevant_history = self._get_last_messages(roles=["user", "assistant"])

        # Summarize relevant history if necessary
        if len(relevant_history) > max_length:
            # Always include the original task
            first_message = relevant_history[0]

            # Get recent context, ensuring we don't overlap with the first message
            recent_messages = relevant_history[-context_size:]
            if first_message in recent_messages:
                recent_messages.remove(first_message)

            # Create a summary of the previous conversation
            relevant_history = [
                first_message,
                {
                    "role": "assistant",
                    "content": "... Previous conversation summarized ...",
                },
                *recent_messages,
            ]

        # Add relevant history to the messages
        messages.extend(relevant_history)

        # Remove tool calls and ensure user message at end
        for message in messages:
            if "tool_calls" in message:
                del message["tool_calls"]

        # Add a user message if the last message is an assistant message
        if messages[-1]["role"] == "assistant":
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue with the task based on the context above.",
                }
            )

        return messages

    async def _process_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent,
    ) -> ToolCallResult | None:
        """Process a single tool call concurrently.

        Args:
            tool_call: The tool call to process
            agent: The agent that the tool call belongs to

        Returns:
            The tool call result or None if the tool call is not valid
        """
        function_name = tool_call.function.name
        function_tools_map = {tool.__name__: tool for tool in agent.function_tools}

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
                    message={
                        "role": "tool",
                        "content": orjson.dumps(function_result).decode(),
                        "tool_call_id": tool_call.id,
                    },
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
        """Process tool calls concurrently and return messages and new agents.

        Args:
            tool_calls: The tool calls to process

        Returns:
            The tool call results
        """
        agent = self.active_agent
        if not agent:
            raise ValueError("No active agent to process tool calls.")

        # Process all tool calls concurrently
        tasks = [self._process_tool_call(tool_call, agent) for tool_call in tool_calls]

        # Gather results and filter out None values
        results = await asyncio.gather(*tasks)

        return [result for result in results if result is not None]

    async def _get_completion_response(
        self,
        agent_messages: list[Message],
    ) -> AsyncGenerator[TypedDelta, None]:
        """Get completion response for current active agent.

        Args:
            agent_messages: The messages to send to the agent

        Returns:
            An async generator of completion deltas
        """
        if not self.active_agent:
            raise ValueError("No active agent")

        tools = [function_to_json(tool) for tool in self.active_agent.function_tools]
        agent_params = self.active_agent.params or {}

        response_stream = await acompletion(
            model=self.active_agent.model,
            messages=agent_messages,
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

            yield choice.delta

    async def _process_agent_response(
        self,
        agent_messages: list[Message],
    ) -> AsyncGenerator[AgentResponse, None]:
        """Process agent response and yield deltas along with accumulated state.

        Args:
            agent_messages: The messages to send to the agent

        Returns:
            An async generator of agent responses
        """
        full_content = ""
        full_tool_calls: list[ChatCompletionDeltaToolCall] = []

        async for delta in self._get_completion_response(agent_messages):
            # Process partial response content
            if delta.content:
                full_content += delta.content

            # Process partial response tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not isinstance(tool_call, ChatCompletionDeltaToolCall):
                        continue

                    if tool_call.id:
                        full_tool_calls.append(tool_call)
                    elif full_tool_calls:
                        last_tool_call = full_tool_calls[-1]
                        last_tool_call.function.arguments += tool_call.function.arguments

            # Stream the partial response delta
            if self.stream_handler:
                await self.stream_handler.on_stream(delta, self.active_agent)

            # Yield both the delta and the current state
            yield AgentResponse(
                delta=delta,
                content=full_content,
                tool_calls=full_tool_calls,
            )

    async def stream(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
    ) -> AsyncGenerator[TypedDelta, None]:
        """Stream thoughts from the agent system.

        Args:
            agent: The agent to stream from
            prompt: The prompt to send to the agent
            messages: The messages to send to the agent

        Returns:
            An async generator of completion deltas
        """
        if messages:
            self.messages.extend(messages)

        if self.active_agent is None:
            self.active_agent = agent
            self.messages.extend(self._get_initial_conversation(agent, prompt))
            agent_messages = deepcopy(self.messages)
        else:
            self.agent_queue.append(agent)

        try:
            while self.active_agent or self.agent_queue:
                if not self.active_agent and self.agent_queue:
                    previous_agent = self.active_agent
                    next_agent = self.agent_queue.popleft()
                    agent_messages = self._prepare_agent_context(next_agent)
                    self.active_agent = next_agent

                    if self.stream_handler:
                        await self.stream_handler.on_agent_switch(previous_agent, next_agent)

                # Process agent response and stream deltas
                last_content = ""
                last_tool_calls: list[ChatCompletionDeltaToolCall] = []

                async for agent_response in self._process_agent_response(agent_messages):
                    # Immediately yield each delta for streaming
                    yield agent_response.delta

                    # Update our accumulated state
                    last_content = agent_response.content
                    last_tool_calls = agent_response.tool_calls

                # Create assistant message with final content and tool calls
                assistant_message = {
                    "role": "assistant",
                    "content": last_content or None,
                    "tool_calls": [],
                }

                if last_tool_calls:
                    # Add tool calls to the assistant message
                    assistant_message["tool_calls"] = [
                        tool_call.model_dump() for tool_call in last_tool_calls
                    ]

                    # Add the assistant message before processing tool calls
                    self.messages.append(assistant_message)
                    agent_messages.append(assistant_message)

                    # Process tool calls and get results
                    results = await self._process_tool_calls(last_tool_calls)

                    # Add tool results to messages
                    for result in results:
                        match result:
                            case ToolCallMessageResult() as message_result:
                                self.messages.append(message_result.message)
                                agent_messages.append(message_result.message)

                            case ToolCallAgentResult() as agent_result:
                                tool_call_message = {
                                    "role": "tool",
                                    "content": f"Switching to agent {agent_result.agent.agent_id}",
                                    "tool_call_id": result.tool_call.id,
                                }

                                self.messages.append(tool_call_message)
                                agent_messages.append(tool_call_message)
                                self.agent_queue.append(agent_result.agent)
                                self.active_agent = None
                else:
                    # No tool calls - just append the assistant message
                    self.messages.append(assistant_message)
                    agent_messages.append(assistant_message)

                # Break if we're done (no tools used and no agents queued)
                if not last_tool_calls and not self.agent_queue:
                    break

        except Exception as e:
            if self.stream_handler:
                await self.stream_handler.on_error(e, self.active_agent)
            raise

        finally:
            if self.stream_handler:
                await self.stream_handler.on_complete(self.messages, self.active_agent)

            self.active_agent = None
            self.agent_queue.clear()

    async def execute(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
    ) -> ConversationState:
        """Execute agent's primary function and return the final conversation state.

        Args:
            agent: The agent to execute
            prompt: The prompt to send to the agent
            messages: The messages to send to the agent

        Returns:
            The final conversation state including the final content and messages
        """
        full_response = ""
        async for delta in self.stream(agent, prompt, messages):
            if delta.content:
                full_response += delta.content

        return ConversationState(
            content=full_response,
            messages=self.messages,
        )
