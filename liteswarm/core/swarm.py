# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import copy
from collections import deque
from collections.abc import AsyncGenerator

import litellm
import orjson
from litellm import CustomStreamWrapper, acompletion
from litellm.exceptions import ContextWindowExceededError
from litellm.types.utils import ChatCompletionDeltaToolCall, ModelResponse, StreamingChoices, Usage

from liteswarm.core.stream_handler import LiteStreamHandler, StreamHandler
from liteswarm.core.summarizer import LiteSummarizer, Summarizer
from liteswarm.types.exceptions import CompletionError, ContextLengthError
from liteswarm.types.result import Result
from liteswarm.types.swarm import (
    Agent,
    AgentResponse,
    CompletionResponse,
    ContextVariables,
    ConversationState,
    Delta,
    Message,
    ResponseCost,
    ToolCallAgentResult,
    ToolCallFailureResult,
    ToolCallMessageResult,
    ToolCallResult,
    ToolMessage,
)
from liteswarm.utils.function import function_has_parameter, function_to_json
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import dump_messages, history_exceeds_token_limit, trim_messages
from liteswarm.utils.misc import safe_get_attr, unwrap_instructions
from liteswarm.utils.retry import retry_with_exponential_backoff
from liteswarm.utils.usage import calculate_response_cost, combine_response_cost, combine_usage

litellm.modify_params = True


class Swarm:
    """A class that manages conversations with AI agents and their interactions.

    The Swarm class orchestrates complex conversations involving multiple AI agents,
    handling message history, tool execution, and agent switching. It provides both
    streaming and synchronous interfaces for agent interactions.

    Key Features:
        - Message history management with automatic summarization
        - Tool execution and agent switching
        - Response streaming with customizable handlers
        - Token usage and cost tracking
        - Automatic retry with exponential backoff
        - Automatic continuation of responses when hitting length limits
        - Safety limits for response length and agent switches

    Example:
        ```python
        def add(a: float, b: float) -> float:
            \"\"\"Add two numbers together.

            Args:
                a: First number.
                b: Second number.

            Returns:
                Sum of the two numbers.
            \"\"\"
            return a + b

        def multiply(a: float, b: float) -> float:
            \"\"\"Multiply two numbers together.

            Args:
                a: First number.
                b: Second number.

            Returns:
                Product of the two numbers.
            \"\"\"
            return a * b

        agent_instructions = (
            "You are a math assistant. Use tools to perform calculations. "
            "When making tool calls, you must provide valid JSON arguments with correct quotes. "
            "After calculations, output must strictly follow this format: 'The result is <tool_result>'"
        )

        # Create an agent with math tools
        agent = Agent.create(
            id="math",
            model="gpt-4o",
            instructions=agent_instructions,
            tools=[add, multiply],
            tool_choice="auto"
        )

        # Initialize swarm and run calculation
        swarm = Swarm(include_usage=True)
        result = await swarm.execute(
            agent=agent,
            prompt="What is (2 + 3) * 4?"
        )

        # The agent will:
        # 1. Use add(2, 3) to get 5
        # 2. Use multiply(5, 4) to get 20
        print(result.content)  # "The result is 20"
        ```

    Note:
        The class maintains internal state during conversations.
        For concurrent conversations, create separate Swarm instances.
    """  # noqa: D214

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
        """Initialize a new Swarm instance.

        Args:
            stream_handler: Handler for streaming events during conversation.
                Defaults to `LiteStreamHandler`.
            summarizer: Handler for summarizing conversation history.
                Defaults to `LiteSummarizer`.
            include_usage: Whether to include token usage statistics in responses.
            include_cost: Whether to include cost statistics in responses.
            max_retries: Maximum number of retry attempts for failed API calls.
            initial_retry_delay: Initial delay between retries in seconds.
            max_retry_delay: Maximum delay between retries in seconds.
            backoff_factor: Multiplicative factor for retry delay after each attempt.
            max_response_continuations: Maximum times a response can be continued
                when hitting length limits.
            max_agent_switches: Maximum number of agent switches allowed in a
                single conversation.

        Note:
            The retry configuration (max_retries, delays, backoff) applies to
            API calls that fail due to transient errors. Context length errors
            are handled separately through history management.
        """
        # Internal state (private)
        self._active_agent: Agent | None = None
        self._agent_messages: list[Message] = []
        self._agent_queue: deque[Agent] = deque()
        self._full_history: list[Message] = []
        self._working_history: list[Message] = []
        self._context_variables: ContextVariables = ContextVariables()

        # Public configuration
        self.stream_handler = stream_handler or LiteStreamHandler()
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

    # ================================================
    # MARK: Tool Processing
    # ================================================

    async def _process_tool_call(
        self,
        agent: Agent,
        context_variables: ContextVariables,
        tool_call: ChatCompletionDeltaToolCall,
    ) -> ToolCallResult:
        """Process a single tool call and return its result.

        Handles the execution of a function call, including error handling
        and result transformation. Supports both regular function results
        and agent switching.

        Args:
            agent: Agent that initiated the tool call
            context_variables: Context variables to pass to the tool function
            tool_call: Tool call to process, containing function name and arguments

        Returns:
            ToolCallResult indicating success or failure of the tool execution
        """
        tool_call_result: ToolCallResult
        function_name = tool_call.function.name
        function_tools_map = {tool.__name__: tool for tool in agent.tools}

        if function_name not in function_tools_map:
            return ToolCallFailureResult(
                tool_call=tool_call,
                error=ValueError(f"Unknown function: {function_name}"),
            )

        await self.stream_handler.on_tool_call(tool_call, agent)

        try:
            args = orjson.loads(tool_call.function.arguments)
            function_tool = function_tools_map[function_name]
            if function_has_parameter(function_tool, "context_variables"):
                args = {**args, "context_variables": context_variables}

            match function_tool(**args):
                case Agent() as agent:
                    tool_call_result = ToolCallAgentResult(
                        tool_call=tool_call,
                        agent=agent,
                        message=Message(
                            role="tool",
                            content=f"Switched to agent {agent.id}",
                            tool_call_id=tool_call.id,
                        ),
                    )

                case Result() as result:
                    content = orjson.dumps(result.value).decode() if result.value else None

                    if result.error:
                        tool_call_result = ToolCallFailureResult(
                            tool_call=tool_call,
                            error=result.error,
                        )
                    elif result.agent:
                        content = content or f"Switched to agent {result.agent.id}"
                        tool_call_result = ToolCallAgentResult(
                            tool_call=tool_call,
                            agent=result.agent,
                            message=Message(
                                role="tool",
                                content=content,
                                tool_call_id=tool_call.id,
                            ),
                            context_variables=result.context_variables,
                        )
                    else:
                        tool_call_result = ToolCallMessageResult(
                            tool_call=tool_call,
                            message=Message(
                                role="tool",
                                content=content or "",
                                tool_call_id=tool_call.id,
                            ),
                            context_variables=result.context_variables,
                        )

                case _ as content:
                    tool_call_result = ToolCallMessageResult(
                        tool_call=tool_call,
                        message=Message(
                            role="tool",
                            content=orjson.dumps(content).decode(),
                            tool_call_id=tool_call.id,
                        ),
                    )

        except Exception as e:
            await self.stream_handler.on_error(e, agent)
            tool_call_result = ToolCallFailureResult(tool_call=tool_call, error=e)

        return tool_call_result

    async def _process_tool_calls(
        self,
        agent: Agent,
        context_variables: ContextVariables,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> list[ToolCallResult]:
        """Process multiple tool calls efficiently and concurrently.

        This method handles the execution of multiple tool calls, optimizing for both
        single and multiple call scenarios:
        - For a single tool call: Processes directly to avoid concurrency overhead
        - For multiple tool calls: Uses asyncio.gather() for concurrent execution

        Args:
            agent: Agent that initiated the tool calls
            context_variables: Context variables for the agent
            tool_calls: List of tool calls to process, each containing function name and arguments

        Returns:
            List of successful ToolCallResult objects. Failed tool calls are filtered out.
            Each result can be either:
            - ToolCallMessageResult: Contains the function's return value
            - ToolCallAgentResult: Contains a new agent for switching

        Note:
            Tool calls that fail or reference unknown functions are silently filtered out
            of the results rather than raising exceptions.
        """
        tasks = [
            self._process_tool_call(
                agent=agent,
                context_variables=context_variables,
                tool_call=tool_call,
            )
            for tool_call in tool_calls
        ]

        results: list[ToolCallResult]
        match len(tasks):
            case 0:
                results = []
            case 1:
                results = [await tasks[0]]
            case _:
                results = await asyncio.gather(*tasks)

        return results

    async def _process_tool_call_result(
        self,
        result: ToolCallResult,
    ) -> ToolMessage:
        """Process a tool call result and prepare the appropriate message response.

        This method handles the following types of tool call results:
        1. Message results: Function return values that should be added to conversation
        2. Agent results: Requests to switch to a new agent
        3. Failure results: Errors that occurred during execution

        Args:
            result: The tool call result to process, either:
                - ToolCallMessageResult containing a function's return value
                - ToolCallAgentResult containing a new agent to switch to
                - ToolCallFailureResult containing an error that occurred during execution

        Returns:
            ToolMessage containing:
            - message: The message to add to conversation history
            - agent: Optional new agent to switch to (None for message results)

        Raises:
            TypeError: If result is not a recognized ToolCallResult subclass

        Note:
            For agent switches, creates a message indicating the switch is occurring
            while also including the new agent in the return value.
        """
        match result:
            case ToolCallMessageResult() as message_result:
                return ToolMessage(
                    message=message_result.message,
                    context_variables=message_result.context_variables,
                )

            case ToolCallAgentResult() as agent_result:
                message = agent_result.message or Message(
                    role="tool",
                    content=f"Switched to agent {agent_result.agent.id}",
                    tool_call_id=agent_result.tool_call.id,
                )

                return ToolMessage(
                    message=message,
                    agent=agent_result.agent,
                    context_variables=agent_result.context_variables,
                )

            case ToolCallFailureResult() as failure_result:
                return ToolMessage(
                    message=Message(
                        role="tool",
                        content=f"Error executing tool: {str(failure_result.error)}",
                        tool_call_id=result.tool_call.id,
                    ),
                )

            case _:
                raise TypeError("Expected a ToolCallResult instance.")

    # ================================================
    # MARK: Response Handling
    # ================================================

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
        tools = [function_to_json(tool) for tool in agent.tools] if agent.tools else None
        stream_options = {"include_usage": True} if self.include_usage else None

        log_verbose(
            f"Sending messages to agent [{agent.id}]: {dict_messages}",
            level="DEBUG",
        )

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

    async def _continue_generation(
        self,
        agent: Agent,
        previous_content: str,
        context_variables: ContextVariables,
    ) -> CustomStreamWrapper:
        """Continue generation after hitting output token limit.

        Args:
            agent: Agent to use for continuation
            previous_content: Content generated so far
            context_variables: Context variables for the agent

        Returns:
            Response stream for the continuation
        """
        instructions = unwrap_instructions(agent.instructions, context_variables)
        continuation_messages = [
            Message(role="system", content=instructions),
            Message(role="assistant", content=previous_content),
            Message(
                role="user",
                content="Please continue your previous response.",
            ),
        ]

        return await self._create_completion(agent, continuation_messages)

    async def _get_completion_response(
        self,
        agent: Agent,
        agent_messages: list[Message],
        context_variables: ContextVariables,
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Stream completion responses from the agent, handling continuations and errors.

        This method manages the streaming response lifecycle, including:
        1. Getting the initial response stream
        2. Processing stream chunks into completion responses
        3. Handling response continuations when output length limits are reached
        4. Managing errors and retries

        Args:
            agent: Agent to use for completion
            agent_messages: Messages to send to the agent
            context_variables: Context variables for the agent

        Yields:
            CompletionResponse objects containing:
            - Current response delta
            - Finish reason
            - Token usage statistics (if enabled)
            - Response cost (if enabled)

        Raises:
            CompletionError: If completion fails after all retries
            ContextLengthError: If context window is exceeded and can't be reduced
        """
        accumulated_content = ""
        continuation_count = 0
        current_stream: CustomStreamWrapper | None = await self._get_initial_stream(
            agent=agent,
            agent_messages=agent_messages,
            context_variables=context_variables,
        )

        try:
            while continuation_count < self.max_response_continuations:
                if not current_stream:
                    break

                async for chunk in current_stream:
                    response = await self._process_stream_chunk(agent, chunk)
                    if response.delta.content:
                        accumulated_content += response.delta.content

                    yield response

                    if response.finish_reason == "length":
                        continuation_count += 1
                        current_stream = await self._handle_continuation(
                            agent=agent,
                            continuation_count=continuation_count,
                            accumulated_content=accumulated_content,
                            context_variables=context_variables,
                        )

                        # This break will exit the `for` loop, but the `while` loop
                        # will continue to process the response continuation
                        break
                else:
                    break

        except (CompletionError, ContextLengthError):
            raise
        except Exception as e:
            raise CompletionError("Failed to get completion response", e) from e

    async def _get_initial_stream(
        self,
        agent: Agent,
        agent_messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> CustomStreamWrapper:
        """Create initial completion stream with retry and context reduction.

        This method:
        1. Attempts to create a completion stream with the given messages
        2. On context window errors, retries with reduced context
        3. Uses exponential backoff for retries

        Args:
            agent: Agent to use for completion
            agent_messages: Messages to send to the agent
            context_variables: Context variables for the agent

        Returns:
            Stream wrapper for the completion response

        Raises:
            CompletionError: If completion fails after all retries
            ContextLengthError: If context window is exceeded and can't be reduced
        """

        async def get_initial_response() -> CustomStreamWrapper:
            try:
                return await self._create_completion(agent=agent, messages=agent_messages)
            except ContextWindowExceededError:
                log_verbose(
                    "Context window exceeded, attempting to reduce context size",
                    level="WARNING",
                )

                return await self._retry_completion_with_trimmed_history(
                    agent=agent,
                    context_variables=context_variables,
                )

        return await retry_with_exponential_backoff(
            get_initial_response,
            max_retries=self.max_retries,
            initial_delay=self.initial_retry_delay,
            max_delay=self.max_retry_delay,
            backoff_factor=self.backoff_factor,
        )

    async def _process_stream_chunk(
        self,
        agent: Agent,
        chunk: ModelResponse,
    ) -> CompletionResponse:
        """Convert a raw stream chunk into a structured completion response.

        This method:
        1. Extracts delta and finish reason from the chunk
        2. Processes usage statistics if enabled
        3. Calculates response cost if enabled

        Args:
            agent: Agent to use for model and cost calculation
            chunk: Raw response chunk from the model

        Returns:
            Structured completion response with delta, finish reason, and optional stats

        Raises:
            TypeError: If chunk format is invalid
        """
        choice = chunk.choices[0]
        if not isinstance(choice, StreamingChoices):
            raise TypeError("Expected a StreamingChoices instance.")

        delta = Delta.from_delta(choice.delta)
        finish_reason = choice.finish_reason
        usage = safe_get_attr(chunk, "usage", Usage)
        response_cost = None

        if usage and self.include_cost:
            response_cost = calculate_response_cost(
                model=agent.model,
                usage=usage,
            )

        return CompletionResponse(
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            response_cost=response_cost,
        )

    async def _handle_continuation(
        self,
        agent: Agent,
        continuation_count: int,
        accumulated_content: str,
        context_variables: ContextVariables,
    ) -> CustomStreamWrapper | None:
        """Create a continuation stream when response length limit is reached.

        This method:
        1. Checks if maximum continuations are reached
        2. Creates a new completion stream with accumulated content
        3. Logs continuation progress

        Args:
            agent: Agent to use for continuation
            continuation_count: Number of continuations so far
            accumulated_content: Content generated up to this point
            context_variables: Context variables for the agent

        Returns:
            New stream for continuation, or None if max continuations reached
        """
        if continuation_count >= self.max_response_continuations:
            log_verbose(
                f"Maximum response continuations ({self.max_response_continuations}) reached",
                level="WARNING",
            )

            return None

        log_verbose(
            f"Response continuation {continuation_count}/{self.max_response_continuations}",
            level="INFO",
        )

        return await self._continue_generation(
            agent=agent,
            previous_content=accumulated_content,
            context_variables=context_variables,
        )

    async def _process_agent_response(
        self,
        agent: Agent,
        agent_messages: list[Message],
        context_variables: ContextVariables,
    ) -> AsyncGenerator[AgentResponse, None]:
        """Stream agent responses while maintaining conversation state.

        This method:
        1. Streams completion responses from the agent
        2. Accumulates content and tool calls
        3. Notifies the stream handler of updates
        4. Yields responses with the current conversation state

        Args:
            agent: Agent to use for completion
            agent_messages: Messages to send to the agent
            context_variables: Context variables for the agent

        Yields:
            AgentResponse objects containing:
            - Current response delta
            - Accumulated content
            - Accumulated tool calls
            - Token usage statistics (if enabled)
            - Response cost (if enabled)

        Raises:
            CompletionError: If completion fails after all retries
            ContextLengthError: If context window is exceeded and can't be reduced
        """
        full_content = ""
        full_tool_calls: list[ChatCompletionDeltaToolCall] = []

        async for completion_response in self._get_completion_response(
            agent=agent,
            agent_messages=agent_messages,
            context_variables=context_variables,
        ):
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

            await self.stream_handler.on_stream(delta, agent)

            yield AgentResponse(
                delta=delta,
                finish_reason=finish_reason,
                content=full_content,
                tool_calls=full_tool_calls,
                usage=completion_response.usage,
                response_cost=completion_response.response_cost,
            )

    async def _process_assistant_response(
        self,
        agent: Agent,
        content: str | None,
        context_variables: ContextVariables,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> list[Message]:
        """Process the assistant's response and execute any tool calls.

        This method handles the complete assistant response cycle:
        1. Creates an assistant message with the text response and/or tool calls
        2. Processes any tool calls and their results
        3. Handles potential agent switches from tool calls
        4. Maintains proper message ordering and relationships

        Args:
            agent: Agent that generated the response
            content: Text content of the assistant's response, may be None if only tool calls
            context_variables: Context variables for the agent
            tool_calls: List of tool calls to process, may be empty if only text response

        Returns:
            List of messages including:
            - The initial assistant message (with content and/or tool calls)
            - Any tool response messages
            - Any agent switch notification messages

        Note:
            If a tool call results in an agent switch, the current agent's state
            is marked as "stale" and the new agent is added to the queue.
        """
        messages: list[Message] = [
            Message(
                role="assistant",
                content=content or None,
                tool_calls=tool_calls if tool_calls else None,
            )
        ]

        if tool_calls:
            tool_call_results = await self._process_tool_calls(
                agent=agent,
                context_variables=context_variables,
                tool_calls=tool_calls,
            )

            for tool_call_result in tool_call_results:
                tool_message = await self._process_tool_call_result(tool_call_result)
                if tool_message.agent:
                    agent.state = "stale"
                    self._agent_queue.append(tool_message.agent)

                if tool_message.context_variables:
                    self._context_variables = ContextVariables(
                        **self._context_variables,
                        **tool_message.context_variables,
                    )

                messages.append(tool_message.message)

        return messages

    # ================================================
    # MARK: History Management
    # ================================================

    async def _prepare_agent_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        """Prepare the agent's context using the working history.

        Creates initial context for an agent by combining:
        1. Agent's system instructions (resolved with context variables if callable)
        2. Filtered working history (excluding system messages)
        3. Optional user prompt

        Args:
            agent: The agent whose context is being prepared
            prompt: Optional initial user message to add
            context_variables: Optional variables for resolving instructions

        Returns:
            List of messages representing the agent's context
        """
        instructions = unwrap_instructions(agent.instructions, context_variables)
        history = [msg for msg in self._working_history if msg.role != "system"]
        messages = [Message(role="system", content=instructions), *history]

        if prompt:
            messages.append(Message(role="user", content=prompt))

        return messages

    async def _update_working_history(self, agent: Agent) -> None:
        """Update the working history by summarizing or trimming messages as needed.

        This method maintains the working history within token and complexity limits
        through two mechanisms:

        1. Summarization:
           - Triggered when the summarizer determines history needs condensing
           - Typically based on length, complexity, or time thresholds
           - Creates a new condensed history while preserving key information
           - Takes precedence over trimming

        2. Token-based Trimming:
           - Triggered when history exceeds the model's token limit
           - Removes older messages while keeping recent context
           - Only applied if summarization isn't needed
           - Uses the agent's model context limits

        Args:
            agent: The agent whose model context limits will be used for trimming

        Note:
            - The full history remains unchanged while working history is updated
            - This method modifies self._working_history in place
            - The choice between summarization and trimming is automatic
            - If neither is needed, the working history remains unchanged
        """
        if self.summarizer.needs_summarization(self._working_history):
            self._working_history = await self.summarizer.summarize_history(self._full_history)
        elif history_exceeds_token_limit(self._working_history, agent.model):
            self._working_history = trim_messages(self._full_history, agent.model)

    async def _retry_completion_with_trimmed_history(
        self,
        agent: Agent,
        context_variables: ContextVariables | None = None,
    ) -> CustomStreamWrapper:
        """Attempt completion with a trimmed message history.

        This method:
        1. Updates the working history to fit the active agent's token limit
        2. Prepares new messages with reduced context
        3. Attempts completion with reduced context

        Args:
            agent: Agent to use for completion
            context_variables: Context variables for the agent

        Returns:
            Response stream from the completion

        Raises:
            ContextLengthError: If context is still too large after reduction
        """
        await self._update_working_history(agent)

        reduced_messages = await self._prepare_agent_context(
            agent=agent,
            context_variables=context_variables,
        )

        try:
            return await self._create_completion(agent, reduced_messages)
        except ContextWindowExceededError as e:
            raise ContextLengthError(
                "Context window exceeded even after reduction attempt",
                original_error=e,
                current_length=len(reduced_messages),
            ) from e

    # ================================================
    # MARK: Agent Management
    # ================================================

    async def _initialize_conversation(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
    ) -> None:
        """Initialize the conversation state.

        Sets up the initial state for the conversation, including the active agent,
        full history, working history, and agent messages.

        Args:
            agent: Initial agent to handle the conversation
            prompt: Initial user prompt
            messages: Optional previous conversation history
            context_variables: Optional context variables for the agent
        """
        if messages:
            self._full_history = copy.deepcopy(messages)
            await self._update_working_history(agent)

        self._context_variables = context_variables or ContextVariables()

        if self._active_agent is None:
            self._active_agent = agent
            self._active_agent.state = "active"

            initial_context = await self._prepare_agent_context(
                agent=agent,
                prompt=prompt,
                context_variables=context_variables,
            )

            self._full_history = initial_context.copy()
            self._working_history = initial_context.copy()
            self._agent_messages = initial_context.copy()
        else:
            user_message = Message(role="user", content=prompt)
            self._full_history.append(user_message)
            self._working_history.append(user_message)
            self._agent_messages.append(user_message)

    async def _handle_agent_switch(
        self,
        switch_count: int,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> bool:
        """Switch to the next agent in the queue.

        This method:
        1. Gets the next agent from the queue
        2. Updates the active agent and agent messages
        3. Notifies the stream handler of the switch
        4. Preserves conversation history for the new agent

        Args:
            switch_count: Current number of agent switches in the conversation
            prompt: Optional prompt to send to the new agent
            context_variables: Optional context variables for the new agent

        Returns:
            True if the switch was successful, False otherwise
        """
        if switch_count >= self.max_agent_switches:
            log_verbose(
                f"Maximum agent switches ({self.max_agent_switches}) reached",
                level="WARNING",
            )
            return False

        if not self._agent_queue:
            return False

        log_verbose(
            f"Agent switch {switch_count}/{self.max_agent_switches}",
            level="INFO",
        )

        next_agent = self._agent_queue.popleft()
        next_agent.state = "active"

        previous_agent = self._active_agent
        self._active_agent = next_agent
        self._agent_messages = await self._prepare_agent_context(
            agent=next_agent,
            prompt=prompt,
            context_variables=context_variables,
        )

        await self.stream_handler.on_agent_switch(previous_agent, next_agent)

        return True

    # ================================================
    # MARK: Public Interface
    # ================================================

    async def stream(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
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
            context_variables: Optional dictionary of variables accessible to:
                - Agent instructions (if using callable instructions)
                - Tool functions (automatically injected as context_variables parameter)
            cleanup: Whether to clear agent state after completion

        Yields:
            AgentResponse objects containing response content and state

        Example:
            ```python
            def get_instructions(context: ContextVariables) -> str:
                return f"Help {context['user_name']} with math."

            def add(a: float, b: float, context_variables: ContextVariables) -> float:
                print(f"User {context_variables['user_name']} adding {a} + {b}")
                return a + b

            agent = Agent.create(
                id="math",
                model="gpt-4",
                instructions=get_instructions,
                tools=[add]
            )

            async for response in swarm.stream(
                agent=agent,
                prompt="What is 2 + 2?",
                context_variables={"user_name": "Alice"}
            ):
                print(response.content)
            ```

        Raises:
            Exception: Any errors during conversation processing
        """
        await self._initialize_conversation(
            agent=agent,
            prompt=prompt,
            messages=messages,
            context_variables=context_variables,
        )

        try:
            agent_switch_count = 0
            while self._active_agent or self._agent_queue:
                if not self._active_agent:
                    break

                if self._active_agent.state == "stale":
                    agent_switch_count += 1
                    agent_switched = await self._handle_agent_switch(
                        switch_count=agent_switch_count,
                        context_variables=self._context_variables,
                    )

                    if not agent_switched:
                        break

                last_content = ""
                last_tool_calls: list[ChatCompletionDeltaToolCall] = []

                async for agent_response in self._process_agent_response(
                    agent=self._active_agent,
                    agent_messages=self._agent_messages,
                    context_variables=self._context_variables,
                ):
                    yield agent_response
                    last_content = agent_response.content or ""
                    last_tool_calls = agent_response.tool_calls

                new_messages = await self._process_assistant_response(
                    agent=self._active_agent,
                    content=last_content,
                    context_variables=self._context_variables,
                    tool_calls=last_tool_calls,
                )

                self._full_history.extend(new_messages)
                self._working_history.extend(new_messages)
                self._agent_messages.extend(new_messages)

                await self._update_working_history(self._active_agent)

                if not last_tool_calls and not self._agent_queue:
                    break

        except Exception as e:
            await self.stream_handler.on_error(e, self._active_agent)
            raise

        finally:
            await self.stream_handler.on_complete(self._full_history, self._active_agent)

            if cleanup:
                self._active_agent = None
                self._agent_messages = []
                self._agent_queue.clear()

    async def execute(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
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
            context_variables: Optional dictionary of variables accessible to:
                - Agent instructions (if using callable instructions)
                - Tool functions (automatically injected as context_variables parameter)
            cleanup: Whether to clear agent state after completion

        Returns:
            ConversationState containing:
                - Final response content
                - Active agent and message history
                - Token usage and cost statistics (if enabled)

        Example:
            ```python
            def get_instructions(context: ContextVariables) -> str:
                return f"Help {context['user_name']} with their task."

            agent = Agent.create(
                id="helper",
                model="gpt-4",
                instructions=get_instructions
            )

            result = await swarm.execute(
                agent=agent,
                prompt="Hello!",
                context_variables={"user_name": "Bob"}
            )
            print(result.content)  # Response will be personalized for Bob
            ```

        Raises:
            ValueError: If there is no active agent
            Exception: Any errors during processing
        """
        full_response = ""
        full_usage: Usage | None = None
        response_cost: ResponseCost | None = None

        response_stream = self.stream(
            agent=agent,
            prompt=prompt,
            messages=messages,
            context_variables=context_variables,
            cleanup=cleanup,
        )

        async for agent_response in response_stream:
            if agent_response.content:
                full_response = agent_response.content
            if agent_response.usage:
                full_usage = combine_usage(full_usage, agent_response.usage)
            if agent_response.response_cost:
                response_cost = combine_response_cost(response_cost, agent_response.response_cost)

        return ConversationState(
            content=full_response,
            agent=self._active_agent,
            agent_messages=self._agent_messages,
            agent_queue=list(self._agent_queue),
            messages=self._full_history,
            usage=full_usage,
            response_cost=response_cost,
        )
