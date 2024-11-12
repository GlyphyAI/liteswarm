import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator

import litellm
import orjson
from litellm import CustomStreamWrapper, acompletion
from litellm.exceptions import ContextWindowExceededError
from litellm.types.utils import ChatCompletionDeltaToolCall, StreamingChoices, Usage

from liteswarm.exceptions import CompletionError, ContextLengthError
from liteswarm.stream_handler import LiteStreamHandler, StreamHandler
from liteswarm.summarizer import LiteSummarizer, Summarizer
from liteswarm.types import (
    Agent,
    AgentResponse,
    CompletionResponse,
    ConversationState,
    Delta,
    Message,
    ResponseCost,
    ToolCallAgentResult,
    ToolCallMessageResult,
    ToolCallResult,
)
from liteswarm.utils import (
    calculate_response_cost,
    combine_response_cost,
    combine_usage,
    dump_messages,
    function_to_json,
    history_exceeds_token_limit,
    retry_with_exponential_backoff,
    safe_get_attr,
    trim_messages,
)

litellm.modify_params = True

logger = logging.getLogger(__name__)


class Swarm:
    """A class that manages conversations with AI agents and their interactions.

    The Swarm class handles:
    1. Message history management (both full and working history)
    2. Agent switching and tool execution
    3. Conversation summarization
    4. Streaming responses and tool calls

    Attributes:
        active_agent: Currently active agent handling the conversation
        agent_messages: Messages relevant to current agent's context
        agent_queue: Queue of agents waiting to be activated
        stream_handler: Optional handler for streaming events
        full_history: Complete conversation history
        working_history: Summarized history within token limits
        summarizer: Handles conversation summarization
    """

    def __init__(  # noqa: PLR0913
        self,
        stream_handler: StreamHandler | None = None,
        summarizer: Summarizer | None = None,
        include_usage: bool = False,
        include_cost: bool = False,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 10.0,
        backoff_factor: float = 2.0,
        max_response_continuations: int = 5,
        max_agent_switches: int = 10,
    ) -> None:
        """Initialize the Swarm.

        Args:
            stream_handler: Optional handler for streaming events
            summarizer: Optional summarizer for managing conversation history
            include_usage: Whether to include token usage statistics in responses
            include_cost: Whether to include cost statistics in responses
            max_retries: Maximum number of retry attempts for API calls
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            backoff_factor: Factor to multiply delay by after each retry
            max_response_continuations: Maximum number of response continuations allowed
            max_agent_switches: Maximum number of agent switches allowed in a conversation
        """
        self.active_agent: Agent | None = None
        self.agent_messages: list[Message] = []
        self.agent_queue: deque[Agent] = deque()
        self.stream_handler = stream_handler or LiteStreamHandler()
        self.full_history: list[Message] = []
        self.working_history: list[Message] = []
        self.summarizer = summarizer or LiteSummarizer()
        self.include_usage = include_usage
        self.include_cost = include_cost

        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.backoff_factor = backoff_factor

        # Safety limits
        self.max_response_continuations = max_response_continuations
        self.max_agent_switches = max_agent_switches

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

        Handles the execution of a function call, including error handling
        and result transformation. Supports both regular function results
        and agent switching.

        Args:
            tool_call: Tool call to process, containing function name and arguments
            agent: Agent that initiated the tool call

        Returns:
            ToolCallResult if successful, None if failed or function not found

        Raises:
            ValueError: If function arguments are invalid
            Exception: Any exception raised by the function tool itself
        """
        function_name = tool_call.function.name
        function_tools_map = {tool.__name__: tool for tool in agent.tools}

        if function_name not in function_tools_map:
            return None

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

            await self.stream_handler.on_tool_call_result(tool_call_result, agent)

            return tool_call_result

        except Exception as e:
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
            List of successful tool call results, filtering out failed calls

        Raises:
            ValueError: If there is no active agent to process tool calls
        """
        agent = self.active_agent
        if not agent:
            raise ValueError("No active agent to process tool calls.")

        tasks = [self._process_tool_call(tool_call, agent) for tool_call in tool_calls]

        results: list[ToolCallResult | None]
        match len(tasks):
            case 0:
                results = []
            case 1:
                results = [await tasks[0]]
            case _:
                results = await asyncio.gather(*tasks)

        return [result for result in results if result is not None]

    async def _create_completion(
        self,
        agent: Agent,
        messages: list[Message],
    ) -> CustomStreamWrapper:
        """Create a completion request with the given agent and messages.

        Args:
            agent: The agent to use for completion
            messages: The messages to send

        Returns:
            Response stream from the completion

        Raises:
            ValueError: If agent parameters are invalid
            TypeError: If response is not of expected type
        """
        dict_messages = dump_messages(messages)
        tools = [function_to_json(tool) for tool in agent.tools]
        stream_options = {"include_usage": True} if self.include_usage else None

        completion_kwargs = {
            "model": agent.model,
            "messages": dict_messages,
            "tools": tools,
            "tool_choice": agent.tool_choice,
            "parallel_tool_calls": agent.parallel_tool_calls,
            "stream": True,
            "stream_options": stream_options,
        }

        agent_kwargs = agent.params or {}
        for key, value in agent_kwargs.items():
            if key not in completion_kwargs:
                completion_kwargs[key] = value

        response_stream = await acompletion(**completion_kwargs)

        if not isinstance(response_stream, CustomStreamWrapper):
            raise TypeError("Expected a CustomStreamWrapper instance.")

        return response_stream

    async def _retry_completion_with_trimmed_history(self) -> CustomStreamWrapper:
        """Attempt completion with a trimmed message history.

        This method:
        1. Updates the working history to fit the active agent's token limit
        2. Prepares new messages with reduced context
        3. Attempts completion with reduced context

        Returns:
            Response stream from the completion

        Raises:
            ContextLengthError: If context is still too large after reduction
            ValueError: If there is no active agent
        """
        if not self.active_agent:
            raise ValueError("No active agent")

        await self._update_working_history()

        reduced_messages = await self._prepare_agent_context(self.active_agent)

        try:
            return await self._create_completion(self.active_agent, reduced_messages)
        except ContextWindowExceededError as e:
            raise ContextLengthError(
                "Context window exceeded even after reduction attempt",
                original_error=e,
                current_length=len(reduced_messages),
            ) from e

    async def _continue_generation(
        self,
        previous_content: str,
        agent: Agent,
    ) -> CustomStreamWrapper:
        """Continue generation after hitting output token limit.

        Args:
            previous_content: Content generated so far
            agent: Agent to use for continuation

        Returns:
            Response stream for the continuation

        Raises:
            ValueError: If agent parameters are invalid
        """
        continuation_messages = [
            Message(role="system", content=agent.instructions),
            Message(role="assistant", content=previous_content),
            Message(
                role="user",
                content="Please continue your previous response.",
            ),
        ]

        return await self._create_completion(agent, continuation_messages)

    async def _get_completion_response(  # noqa: PLR0912
        self,
        agent_messages: list[Message],
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Get completion response for the currently active agent.

        This method handles the process of obtaining a completion response from the active agent,
        including handling API errors and managing response continuations when the output token
        limit is reached.

        The method performs the following:
        1. Sends the initial completion request and handles context window errors by retrying
           with a trimmed message history.
        2. Streams the response, yielding each delta as it is received.
        3. Automatically continues the response if the output token limit is reached, ensuring
           seamless conversation flow.
        4. Collects and yields usage statistics and response costs if configured.

        Args:
            agent_messages: The messages to send to the agent for completion.

        Yields:
            CompletionResponse objects containing the current delta, finish reason, usage statistics,
            and response cost.

        Raises:
            ValueError: If there is no active agent.
            CompletionError: If completion fails after all retries.
            ContextLengthError: If context window is exceeded and can't be reduced.
            TypeError: If response is not of expected type.
        """
        agent = self.active_agent
        if not agent:
            raise ValueError("No active agent")

        logger.debug(
            "Sending messages to agent [%s]: %s",
            agent.agent_id,
            agent_messages,
        )

        async def get_initial_response() -> CustomStreamWrapper:
            try:
                return await self._create_completion(agent, agent_messages)
            except ContextWindowExceededError:
                logger.warning("Context window exceeded, attempting to reduce context size")
                return await self._retry_completion_with_trimmed_history()

        accumulated_content: str = ""
        current_stream: CustomStreamWrapper | None = None
        continuation_count = 0

        try:
            while continuation_count < self.max_response_continuations:
                if current_stream is None:
                    current_stream = await retry_with_exponential_backoff(
                        get_initial_response,
                        max_retries=self.max_retries,
                        initial_delay=self.initial_retry_delay,
                        max_delay=self.max_retry_delay,
                        backoff_factor=self.backoff_factor,
                    )

                async for chunk in current_stream:
                    choice = chunk.choices[0]
                    if not isinstance(choice, StreamingChoices):
                        raise TypeError("Expected a StreamingChoices instance.")

                    delta = Delta.from_delta(choice.delta)
                    finish_reason = choice.finish_reason
                    usage = safe_get_attr(chunk, "usage", Usage)
                    response_cost = None

                    if delta.content:
                        accumulated_content += delta.content

                    if usage and self.include_cost:
                        response_cost = calculate_response_cost(model=agent.model, usage=usage)

                    yield CompletionResponse(
                        delta=delta,
                        finish_reason=finish_reason,
                        usage=usage,
                        response_cost=response_cost,
                    )

                    if finish_reason == "length":
                        continuation_count += 1
                        if continuation_count >= self.max_response_continuations:
                            logger.warning(
                                "Maximum response continuations (%d) reached",
                                self.max_response_continuations,
                            )

                            # This will return control back to the method caller
                            # as soon as the maximum number of continuations is reached
                            return

                        logger.info(
                            "Response continuation %d/%d",
                            continuation_count,
                            self.max_response_continuations,
                        )

                        current_stream = await self._continue_generation(
                            accumulated_content,
                            agent,
                        )

                        # This break will exit the `for` loop, but the `while` loop
                        # will continue to process the response continuation
                        break

                else:
                    break

        except CompletionError:
            raise
        except ContextLengthError:
            raise
        except Exception as e:
            raise CompletionError("Failed to get completion response", e) from e

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

            await self.stream_handler.on_stream(delta, self.active_agent)

            yield AgentResponse(
                delta=delta,
                finish_reason=finish_reason,
                content=full_content,
                tool_calls=full_tool_calls,
                usage=completion_response.usage,
                response_cost=completion_response.response_cost,
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

        await self.stream_handler.on_agent_switch(previous_agent, next_agent)

    async def _update_working_history(self) -> None:
        """Update the working history by either summarizing or trimming messages.

        This method maintains the working history within token limits by:
        1. Summarizing the full history if the summarizer determines it's necessary
           (e.g., when history length or complexity exceeds defined thresholds)
        2. Trimming the history if it exceeds the model's token limit but doesn't
           yet need summarization

        The working history is updated in place, while the full history remains
        unchanged to preserve the complete conversation record.

        Note:
            - Summarization takes precedence over trimming
            - Trimming only occurs if there's an active agent (to determine token limits)
            - If neither summarization nor trimming is needed, the working history
              remains unchanged
        """
        if self.summarizer.needs_summarization(self.working_history):
            self.working_history = await self.summarizer.summarize_history(self.full_history)
        elif self.active_agent:
            if history_exceeds_token_limit(self.working_history, self.active_agent.model):
                self.working_history = trim_messages(self.full_history, self.active_agent.model)

    async def stream(  # noqa: PLR0912
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        cleanup: bool = True,
    ) -> AsyncGenerator[AgentResponse, None]:
        """Stream agent responses and manage the conversation flow.

        Handles the complete conversation lifecycle including:
        1. History initialization and management
        2. Agent responses and tool calls
        3. Agent switching
        4. History summarization

        Args:
            agent: Initial agent to handle the conversation
            prompt: Initial user prompt
            messages: Optional previous conversation history
            cleanup: Whether to clear agent state after completion

        Yields:
            AgentResponse objects containing response content and state

        Raises:
            Exception: Any errors during conversation processing
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

        agent_switch_count = 0

        try:
            while self.active_agent or self.agent_queue:
                if not self.active_agent and self.agent_queue:
                    agent_switch_count += 1
                    if agent_switch_count > self.max_agent_switches:
                        logger.warning(
                            "Maximum agent switches (%d) reached",
                            self.max_agent_switches,
                        )

                        break

                    logger.info(
                        "Agent switch %d/%d",
                        agent_switch_count,
                        self.max_agent_switches,
                    )

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
            await self.stream_handler.on_error(e, self.active_agent)
            raise

        finally:
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
            agent: Agent to execute the task
            prompt: Prompt describing the task
            messages: Optional previous messages for context
            cleanup: Whether to clear agent state after completion

        Returns:
            ConversationState containing final response and complete history

        Raises:
            ValueError: If there is no active agent
            Exception: Any errors during processing
        """
        full_response = ""
        full_usage: Usage | None = None
        response_stream = self.stream(agent, prompt, messages, cleanup)
        response_cost: ResponseCost | None = None

        async for agent_response in response_stream:
            if agent_response.content:
                full_response = agent_response.content
            if agent_response.usage:
                full_usage = combine_usage(full_usage, agent_response.usage)
            if agent_response.response_cost:
                response_cost = combine_response_cost(response_cost, agent_response.response_cost)

        return ConversationState(
            content=full_response,
            agent=self.active_agent,
            agent_messages=self.agent_messages,
            messages=self.full_history,
            usage=full_usage,
            response_cost=response_cost,
        )
