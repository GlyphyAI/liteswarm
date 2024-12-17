# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import copy
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

import litellm
import orjson
from litellm import CustomStreamWrapper, acompletion, get_supported_openai_params
from litellm.exceptions import ContextWindowExceededError
from litellm.types.utils import ChatCompletionDeltaToolCall, ModelResponse, StreamingChoices, Usage
from pydantic import BaseModel

from liteswarm.core.stream_handler import LiteSwarmStreamHandler, SwarmStreamHandler
from liteswarm.core.summarizer import LiteSummarizer, Summarizer
from liteswarm.types.exceptions import CompletionError, ContextLengthError
from liteswarm.types.swarm import (
    Agent,
    AgentResponse,
    AgentState,
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
    ToolResult,
)
from liteswarm.utils.function import function_has_parameter, functions_to_json
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import dump_messages, history_exceeds_token_limit, trim_messages
from liteswarm.utils.misc import parse_content, safe_get_attr
from liteswarm.utils.retry import retry_with_exponential_backoff
from liteswarm.utils.typing import is_subtype
from liteswarm.utils.unwrap import unwrap_instructions
from liteswarm.utils.usage import calculate_response_cost, combine_response_cost, combine_usage

litellm.modify_params = True


class Swarm:
    """Orchestrator for AI agent conversations and interactions.

    Swarm orchestrates complex conversations involving multiple AI agents,
    handling message history, tool execution, and agent switching. It provides both
    streaming and synchronous interfaces for agent interactions.

    Manages complex conversations with multiple AI agents, handling:
    - Message history and automatic summarization
    - Tool execution and agent switching
    - Response streaming with custom handlers
    - Token usage and cost tracking
    - Automatic retry with exponential backoff
    - Response continuation for length limits
    - Safety limits for responses and switches

    Examples:
        Basic calculation:
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
            agent = Agent(
                id="math",
                instructions=agent_instructions,
                llm=LLM(
                    model="gpt-4o",
                    tools=[add, multiply],
                    tool_choice="auto",
                ),
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

    Notes:
        - Maintains internal state during conversations
        - Create separate instances for concurrent conversations
    """  # noqa: D214

    def __init__(  # noqa: PLR0913
        self,
        stream_handler: SwarmStreamHandler | None = None,
        memory: Memory | None = None,
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
                Defaults to LiteSwarmStreamHandler.
            summarizer: Handler for summarizing conversation history.
                Defaults to LiteSummarizer.
            include_usage: Whether to include token usage statistics.
            include_cost: Whether to include cost statistics.
            max_retries: Maximum retry attempts for failed API calls.
            initial_retry_delay: Initial delay between retries (seconds).
            max_retry_delay: Maximum delay between retries (seconds).
            backoff_factor: Multiplier for retry delay after each attempt.
            max_response_continuations: Maximum times a response can be
                continued when hitting length limits.
            max_agent_switches: Maximum number of agent switches allowed
                in a single conversation.

        Notes:
            The retry configuration (max_retries, delays, backoff) applies
            to API calls that fail due to transient errors. Context length
            errors are handled separately through history management.
        """
        # Internal state (private)
        self._active_agent: Agent | None = None
        self._agent_queue: deque[Agent] = deque()
        self._context_variables: ContextVariables = ContextVariables()

        # Public configuration
        self.stream_handler = stream_handler or LiteSwarmStreamHandler()
        self.memory = memory or LiteMemory()
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

    def _parse_tool_call_result(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        result: Any,
    ) -> ToolCallResult:
        """Parse a tool's return value into an internal result representation.

        Converts various tool return types into appropriate internal result objects
        for framework processing. The method handles three main cases:
        1. Direct Agent returns for immediate agent switching
        2. ToolResult objects for complex tool responses
        3. Simple return values that become message content

        This is an internal method that bridges the public tool API (using ToolResult)
        with the framework's internal processing (using ToolCallResult hierarchy).

        Args:
            tool_call: The original tool call that produced this result.
            result: The raw return value from the tool function, which can be:
                - An Agent instance for direct agent switching.
                - A ToolResult for complex responses with context/agent updates.
                - Any JSON-serializable value for simple responses.

        Returns:
            An internal ToolCallResult subclass instance:
            - ToolCallAgentResult for agent switches.
            - ToolCallMessageResult for data responses.
            - Content is always properly formatted for conversation.

        Examples:
            Agent switching:
                ```python
                # Internal processing of agent switch
                result = self._parse_tool_call_result(
                    tool_call=call,
                    result=Agent(id="expert", instructions="You are an expert"),
                )
                assert isinstance(result, ToolCallAgentResult)
                assert result.agent.id == "expert"
                ```

            Complex tool result:
                ```python
                # Internal processing of tool result with context
                result = self._parse_tool_call_result(
                    tool_call=call,
                    result=ToolResult(
                        content=42,
                        context_variables=ContextVariables(last_result=42),
                    ),
                )
                assert isinstance(result, ToolCallMessageResult)
                assert result.message.content == "42"
                assert result.context_variables["last_result"] == 42
                ```

            Simple value:
                ```python
                # Internal processing of direct return
                result = self._parse_tool_call_result(
                    tool_call=call,
                    result="Calculation complete",
                )
                assert isinstance(result, ToolCallMessageResult)
                assert result.message.content == "Calculation complete"
                ```

        Notes:
            - This is an internal method used by the framework
            - Tool functions should return ToolResult instances
            - See ToolResult documentation for the public API
        """
        match result:
            case Agent() as agent:
                return ToolCallAgentResult(
                    tool_call=tool_call,
                    agent=agent,
                    message=Message(
                        role="tool",
                        content=f"Switched to agent {agent.id}",
                        tool_call_id=tool_call.id,
                    ),
                )

            case ToolResult() as tool_output:
                content = parse_content(tool_output.content)

                if tool_output.agent:
                    return ToolCallAgentResult(
                        tool_call=tool_call,
                        agent=tool_output.agent,
                        message=Message(
                            role="tool",
                            content=content,
                            tool_call_id=tool_call.id,
                        ),
                        context_variables=tool_output.context_variables,
                    )

                return ToolCallMessageResult(
                    tool_call=tool_call,
                    message=Message(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call.id,
                    ),
                    context_variables=tool_output.context_variables,
                )

            case _:
                return ToolCallMessageResult(
                    tool_call=tool_call,
                    message=Message(
                        role="tool",
                        content=parse_content(result),
                        tool_call_id=tool_call.id,
                    ),
                )

    async def _process_tool_call(
        self,
        agent: Agent,
        context_variables: ContextVariables,
        tool_call: ChatCompletionDeltaToolCall,
    ) -> ToolCallResult:
        """Process a single tool call and handle its execution lifecycle.

        Manages the complete execution of a function call, including error handling,
        argument validation, and result transformation. The method supports both
        regular function return values and special cases like agent switching.

        Args:
            agent: Agent that initiated the tool call.
            context_variables: Context variables to pass to the tool function for dynamic resolution.
            tool_call: Tool call details containing function name and arguments to execute.

        Returns:
            ToolCallResult indicating success or failure of the tool execution.

        Notes:
            Tool calls can return different types of results:
            - Regular values that get converted to conversation messages
            - New agents that trigger agent switching behavior
            - Complex result objects for special response handling
        """
        tool_call_result: ToolCallResult
        function_name = tool_call.function.name
        function_tools_map = {tool.__name__: tool for tool in agent.llm.tools or []}

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

            tool_call_result = self._parse_tool_call_result(
                tool_call=tool_call,
                result=function_tool(**args),
            )

        except Exception as error:
            await self.stream_handler.on_error(error, agent)
            tool_call_result = ToolCallFailureResult(tool_call=tool_call, error=error)

        return tool_call_result

    async def _process_tool_calls(
        self,
        agent: Agent,
        context_variables: ContextVariables,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> list[ToolCallResult]:
        """Process multiple tool calls with optimized execution strategies.

        This method intelligently handles tool call execution based on the number of calls:
        - For a single tool call: Uses direct processing to avoid concurrency overhead
        - For multiple tool calls: Leverages asyncio.gather() for efficient parallel execution

        Args:
            agent: Agent that initiated the calls, providing execution context.
            context_variables: Context variables for dynamic resolution in tool functions.
            tool_calls: List of tool calls to process, each with function details (name and arguments).

        Returns:
            List of successful ToolCallResult objects. Failed calls are filtered out.
            Results can be either:
            - ToolCallMessageResult containing function return values
            - ToolCallAgentResult containing new agents for switching

        Notes:
            Tool calls that fail or reference unknown functions are filtered from results
            rather than raising exceptions to maintain conversation flow.
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
        """Process a tool call result into an appropriate conversation message.

        Handles different types of tool call results with specialized processing:
        - Message results: Converts function return values into conversation entries
        - Agent results: Creates switch notifications and requests agent switching
        - Failure results: Generates appropriate error messages for conversation

        Args:
            result: Tool call result to process, which can be:
                - ToolCallMessageResult with function return value
                - ToolCallAgentResult with new agent for switching
                - ToolCallFailureResult with execution error details

        Returns:
            ToolMessage containing:
            - message: Formatted message for conversation history
            - agent: Optional new agent for switching scenarios
            - context_variables: Optional context updates for next steps

        Raises:
            TypeError: If result type is not a recognized ToolCallResult subclass.
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
        """Create a completion request with comprehensive configuration.

        Prepares and sends a completion request with full configuration:
        - Message history with proper formatting
        - Tool configurations and permissions
        - Response format specifications
        - Usage tracking and cost monitoring settings
        - Model-specific parameters from agent

        Args:
            agent: Agent to use for completion, providing model settings.
            messages: Messages to send as conversation context.

        Returns:
            Response stream from the completion API.

        Raises:
            ValueError: If agent parameters are invalid or inconsistent.
            TypeError: If response format is unexpected.
        """
        exclude_keys = {"response_format", "litellm_kwargs"}
        llm_kwargs = agent.llm.model_dump(exclude=exclude_keys, exclude_none=True)
        llm_override_kwargs = {
            "messages": dump_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True} if self.include_usage else None,
            "tools": functions_to_json(agent.llm.tools),
        }

        response_format = agent.llm.response_format
        supported_params = get_supported_openai_params(agent.llm.model) or []
        if "response_format" in supported_params and response_format:
            llm_override_kwargs["response_format"] = response_format

            response_format_str: str | None = None
            if is_subtype(response_format, BaseModel):
                response_format_str = orjson.dumps(response_format.model_json_schema()).decode()
            else:
                response_format_str = orjson.dumps(response_format).decode()

            log_verbose(
                f"Using response format: {response_format_str}",
                level="DEBUG",
            )

        completion_kwargs = {
            **llm_kwargs,
            **llm_override_kwargs,
            **(agent.llm.litellm_kwargs or {}),
        }

        log_verbose(
            f"Sending messages to agent [{agent.id}]: {completion_kwargs.get('messages')}",
            level="DEBUG",
        )

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
        """Continue generation after reaching the output token limit.

        Creates a new completion request optimized for continuation:
        - Includes previous content as meaningful context
        - Maintains original agent instructions and settings
        - Adds continuation-specific prompting
        - Preserves conversation coherence

        Args:
            agent: Agent for continuation, maintaining consistency.
            previous_content: Content generated before hitting limit.
            context_variables: Context for dynamic resolution.

        Returns:
            Response stream for the continuation request.
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
        """Stream agent completion responses handling continuations and errors.

        Manages the complete response lifecycle with advanced features:
        - Initial response stream creation and validation
        - Chunk processing with proper delta extraction
        - Automatic continuation on length limits
        - Error handling with intelligent retries
        - Usage tracking and cost monitoring

        Args:
            agent: Agent for completion, providing model settings.
            agent_messages: Messages forming conversation context.
            context_variables: Context for dynamic resolution.

        Yields:
            CompletionResponse containing:
            - Current response delta with content updates
            - Finish reason for proper flow control
            - Usage statistics for monitoring (if enabled)
            - Cost information for tracking (if enabled)

        Raises:
            CompletionError: If completion fails after all retry attempts.
            ContextLengthError: If context exceeds limits and cannot be reduced.
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
        """Create initial completion stream with robust error handling.

        Implements a comprehensive completion strategy:
        - Multiple retry attempts with exponential backoff
        - Automatic context reduction on length errors
        - Intelligent history trimming when needed
        - Proper error propagation for unrecoverable cases

        Args:
            agent: Agent for completion with model settings.
            agent_messages: Messages forming the conversation context.
            context_variables: Optional context for dynamic resolution.

        Returns:
            Stream wrapper for managing completion response.

        Raises:
            CompletionError: If completion fails after exhausting retries.
            ContextLengthError: If context remains too large after reduction.
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
        """Process a raw stream chunk into a structured completion response.

        Performs stream chunk processing:
        - Extracts and validates response delta
        - Determines appropriate finish reason
        - Calculates detailed usage statistics
        - Computes accurate response cost
        - Handles special token cases

        Args:
            agent: Agent providing model info and cost settings.
            chunk: Raw response chunk from the model API.

        Returns:
            Structured completion response with all metadata.

        Raises:
            TypeError: If chunk format is invalid or unexpected.
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
                model=agent.llm.model,
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
        """Handle response continuation with proper limits and tracking.

        Manages the continuation process with safeguards:
        - Checks for maximum continuation limit
        - Creates properly contextualized completions
        - Maintains conversation coherence
        - Tracks and logs continuation progress
        - Handles resource cleanup

        Args:
            agent: Agent for continuation, maintaining consistency.
            continuation_count: Number of continuations performed.
            accumulated_content: Previously generated content.
            context_variables: Context for dynamic resolution.

        Returns:
            New stream for continuation, or None if max continuations reached.
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
        """Process agent responses with state management and tracking.

        Implements agent response handling:
        - Streams completion responses with proper buffering
        - Accumulates content and tool calls accurately
        - Updates stream handler with progress
        - Maintains conversation state consistency
        - Tracks usage and costs throughout

        Args:
            agent: Agent processing the response.
            agent_messages: Messages providing conversation context.
            context_variables: Context for dynamic resolution.

        Yields:
            AgentResponse containing:
            - Current response delta with updates
            - Accumulated content for context
            - Collected tool calls for execution
            - Usage and cost statistics for monitoring

        Raises:
            CompletionError: If completion fails after retries.
            ContextLengthError: If context exceeds limits after reduction.
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
        """Process complete assistant response, and handle tool calls.

        Handles the full response cycle with proper ordering:
        - Creates formatted assistant message with content
        - Processes tool calls in correct sequence
        - Manages potential agent switches
        - Maintains message relationships
        - Updates conversation state

        Args:
            agent: Agent that generated the response.
            content: Text content of response, may be None.
            context_variables: Context for tool execution.
            tool_calls: Tool calls to process in order.

        Returns:
            Messages including all components:
            - Primary assistant message
            - Tool response messages
            - Switch notification messages
            - Related status updates

        Notes:
            Updates agent state and queue when switches occur by marking the current
            agent as stale and adding the new agent to the queue.
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
                    agent.state = AgentState.STALE
                    self._agent_queue.append(tool_message.agent)

                if tool_message.context_variables:
                    self._context_variables.update(tool_message.context_variables)

                messages.append(tool_message.message)

        return messages

    # ================================================
    # MARK: History Management
    # ================================================

    async def _retry_completion_with_trimmed_history(
        self,
        agent: Agent,
        context_variables: ContextVariables | None = None,
    ) -> CustomStreamWrapper:
        """Retry completion with optimized context reduction.

        Implements intelligent context reduction:
        - Updates working history within limits
        - Prepares minimal but sufficient context
        - Attempts completion with reduced context
        - Maintains conversation coherence

        Args:
            agent: Agent for completion attempt.
            context_variables: Context for resolution.

        Returns:
            Response stream from completion.

        Raises:
            ContextLengthError: If context remains too large after reduction.
        """
        try:
            await self._trim_working_history(agent.llm.model)
            reduced_messages = self.memory.create_agent_context(
                agent=agent,
                context_variables=context_variables,
            )

            return await self._create_completion(agent, reduced_messages)
        except ContextWindowExceededError as e:
            raise ContextLengthError(
                "Context window exceeded even after reduction attempt",
                original_error=e,
                current_length=len(reduced_messages),
            ) from e

    async def _trim_working_history(self, model: str) -> None:
        """Manage working history size through summarization or trimming.

        Maintains history within model context limits by:
        - First attempting summarization if complexity warrants
        - Falling back to token-based trimming if needed
        - Preserving essential conversation context

        Args:
            model: Model identifier to determine context limits.
        """
        messages = self.memory.get_working_history()
        if self.summarizer.needs_summarization(messages):
            messages = await self.summarizer.summarize_history(messages)
            self.memory.set_working_history(messages)
            log_verbose(f"The working history was summarized for model {model}")
        elif history_exceeds_token_limit(messages, model):
            messages = trim_messages(messages, model)
            self.memory.set_working_history(messages)
            log_verbose(f"The working history was trimmed for model {model}")
        else:
            log_verbose(f"The working history does not exceed token limit for model {model}")

    # ================================================
    # MARK: Agent Management
    # ================================================

    async def _initialize_conversation(
        self,
        agent: Agent,
        prompt: str | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        """Initialize the conversation state.

        Sets up complete conversation environment:
        - Configures active agent with settings
        - Initializes message histories
        - Sets up context variables
        - Prepares initial messages
        - Establishes tracking state

        Args:
            agent: Initial agent for conversation.
            prompt: Starting user prompt.
            messages: Optional existing history.
            context_variables: Optional context for dynamic resolution.
        """
        if messages:
            self.memory.set_history(messages)

        if prompt:
            self.memory.append_message(Message(role="user", content=prompt))

        if self._active_agent is None:
            self._active_agent = agent
            self._active_agent.state = AgentState.ACTIVE
        else:
            # We probaby want to queue the agent instead,
            # but to do so we need to support passing an empty agent
            # to the stream method.
            # TODO: implement passing an empty agent to the stream method,
            #       and then queue the agent instead of setting it as active
            self._active_agent.state = AgentState.ACTIVE

    async def _handle_agent_switch(self, switch_count: int) -> bool:
        """Handle transition between agents in the swarm.

        Manages the complete agent switching process, including:
        - Validating switch limits and conditions
        - Managing agent queue operations
        - Updating agent states
        - Notifying stream handlers
        - Maintaining conversation continuity

        The switching process follows these steps:
        - Check switch count against maximum limit
        - Verify availability of next agent
        - Update states of previous and next agents
        - Notify handlers of the transition
        - Establish new agent as active

        Args:
            switch_count: Current number of agent switches performed.
                Used to enforce maximum switch limits.

        Returns:
            True if switch was successful, False otherwise.

        Notes:
            - Switches are limited by max_agent_switches
            - Previous agent is marked as STALE
            - Next agent becomes ACTIVE
            - Queue is updated after successful switch
            - Stream handler is notified of changes
        """
        if switch_count >= self.max_agent_switches:
            log_verbose(
                f"Maximum agent switches ({self.max_agent_switches}) reached",
                level="WARNING",
            )
            return False

        if not self._agent_queue:
            return False

        log_verbose(f"Agent switch {switch_count}/{self.max_agent_switches}")

        previous_agent = self._active_agent
        next_agent = self._agent_queue.popleft()
        next_agent.state = AgentState.ACTIVE
        self._active_agent = next_agent

        await self.stream_handler.on_agent_switch(previous_agent, next_agent)

        return True

    async def _stream_agent_responses(
        self,
        iteration_count: int = 0,
        agent_switch_count: int = 0,
    ) -> AsyncGenerator[AgentResponse, None]:
        """Stream responses from active and subsequent agents in the swarm.

        Manages the core response generation loop of the swarm, handling:
        - Sequential agent responses and transitions
        - Agent state management and switching
        - Response generation and streaming
        - Tool call processing and execution
        - Message history updates

        The method implements the main processing loop that:
        - Checks iteration and switch limits
        - Verifies agent state and handles switches
        - Creates agent context with history
        - Processes agent response and tool calls
        - Updates history and prepares for next iteration
        - Handles agent state transitions

        Args:
            iteration_count: Current number of processing iterations.
                Used to enforce maximum iteration limits. Defaults to 0.
            agent_switch_count: Number of agent switches performed.
                Used to enforce maximum switch limits. Defaults to 0.

        Yields:
            AgentResponse objects containing:
            - Response content and deltas
            - Tool calls and their results
            - Usage statistics if enabled
            - Cost information if enabled

        Notes:
            - Processing continues until no active agents remain
            - Agents can become stale and trigger switches
            - Empty responses are handled gracefully
            - The loop respects max_iterations limit
            - Agent switches respect max_agent_switches limit
        """
        while iteration_count < self.max_iterations:
            iteration_count += 1
            if not self._active_agent:
                break

            if self._active_agent.state == AgentState.STALE:
                agent_switch_count += 1
                agent_switched = await self._handle_agent_switch(agent_switch_count)
                if not agent_switched:
                    log_verbose("No more agents to switch to, stopping")
                    break

                log_verbose(f"Agent {self._active_agent.id} is stale, switching to next agent")
            else:
                log_verbose(f"Agent {self._active_agent.id} is active, processing response")

            agent_messages = self.memory.create_agent_context(
                agent=self._active_agent,
                context_variables=self._context_variables,
            )

            last_agent_response: AgentResponse | None = None
            async for agent_response in self._process_agent_response(
                agent=self._active_agent,
                agent_messages=agent_messages,
                context_variables=self._context_variables,
            ):
                yield agent_response
                last_agent_response = agent_response

            if not last_agent_response:
                continue

            if last_agent_response.content or last_agent_response.tool_calls:
                new_messages = await self._process_assistant_response(
                    agent=self._active_agent,
                    content=last_agent_response.content,
                    context_variables=self._context_variables,
                    tool_calls=last_agent_response.tool_calls,
                )

                self.memory.extend_messages(new_messages)
            else:
                # We might not want to do this, but it's a good fallback
                # Please consider removing this if it leads to unexpected behavior
                self.memory.append_message(Message(role="assistant", content="<empty>"))
                log_verbose(
                    "No content was generated, appending empty message",
                    level="WARNING",
                )

            if not last_agent_response.tool_calls:
                self._active_agent.state = AgentState.STALE

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
        """Stream responses from a swarm of agents.

        This is the main entry point for streaming responses from a swarm of agents.
        The swarm processes the prompt through a series of agents, where each agent
        can contribute to the response or delegate to other agents. The process
        continues until a complete response is generated or an error occurs.

        The swarm maintains conversation state and handles:
        - Agent initialization and switching
        - Message history management
        - Response streaming and accumulation
        - Tool call execution
        - Error recovery and retries

        Args:
            agent: Initial agent to handle the conversation.
            prompt: The user's input prompt to process.
            messages: Optional list of previous conversation messages for context.
                If provided, these messages are used as conversation history.
            context_variables: Optional variables for dynamic instruction resolution
                and tool execution. These variables are passed to agents and tools.
            cleanup: Whether to clear agent state after completion.

        Yields:
            AgentResponse objects containing:
            - Current response delta with content updates
            - Accumulated content so far
            - Tool calls being processed
            - Usage statistics if enabled
            - Cost information if enabled

        Raises:
            SwarmError: If an unrecoverable error occurs during processing.
            ContextLengthError: If the context becomes too large to process.
            MaxAgentSwitchesError: If too many agent switches occur.
            MaxResponseContinuationsError: If response requires too many continuations.

        Notes:
            - The swarm uses a queue of agents to handle complex tasks
            - Each agent can process the input or delegate to other agents
            - The conversation history is preserved across agent switches
            - The stream can be interrupted at any point
            - Errors during processing may be recovered automatically

        Examples:
            Basic usage:
                ```python
                def get_instructions(context: ContextVariables) -> str:
                    return f"Help {context['user_name']} with math."


                def add(a: float, b: float, context_variables: ContextVariables) -> float:
                    return a + b


                agent = Agent(
                    id="math",
                    instructions=get_instructions,
                    llm=LLM(
                        model="gpt-4o",
                        tools=[add],
                    ),
                )

                async for response in swarm.stream(
                    agent=agent,
                    prompt="What is 2 + 2?",
                    context_variables=ContextVariables({"user_name": "Alice"}),
                ):
                    print(response.content)
                ```
        """
        try:
            self._context_variables.update(context_variables or {})
            await self._initialize_conversation(
                agent=agent,
                prompt=prompt,
                messages=messages,
            )

            async for response in self._stream_agent_responses():
                yield response

        except Exception as e:
            await self.stream_handler.on_error(e, self._active_agent)
            raise

        finally:
            await self.stream_handler.on_complete(
                messages=self.memory.get_full_history(),
                agent=self._active_agent,
            )

            if cleanup:
                self.cleanup()

    async def execute(
        self,
        agent: Agent,
        prompt: str,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
        cleanup: bool = True,
    ) -> ConversationState:
        """Execute a prompt and return the complete response.

        This is a convenience method that wraps the stream() method to provide
        a simpler interface when streaming isn't required. It accumulates all
        content from the stream and returns the final response as a string.

        The method provides the same functionality as stream() but with:
        - Automatic response accumulation
        - Simplified error handling
        - Single string return value
        - Blocking execution until completion

        Args:
            agent: Agent to execute the task.
            prompt: The user's input prompt to process.
            messages: Optional list of previous conversation messages for context.
                If provided, these messages are used as conversation history.
            context_variables: Optional variables for dynamic instruction resolution
                and tool execution. These variables are passed to agents and tools.
            cleanup: Whether to clear agent state after completion.

        Returns:
            ConversationState containing:
            - Final response content
            - Active agent and message history
            - Token usage and cost statistics (if enabled)

        Raises:
            SwarmError: If an unrecoverable error occurs during processing.
            ContextLengthError: If the context becomes too large to process.
            MaxAgentSwitchesError: If too many agent switches occur.
            MaxResponseContinuationsError: If response requires too many continuations.

        Notes:
            - This method blocks until the complete response is generated
            - All streaming responses are accumulated internally
            - The same error handling and recovery as stream() applies
            - The conversation history is preserved in the same way

        Examples:
            Basic usage:
                ```python
                def get_instructions(context: ContextVariables) -> str:
                    return f"Help {context['user_name']} with their task."


                agent = Agent(
                    id="helper",
                    instructions=get_instructions,
                    llm=LLM(model="gpt-4o"),
                )

                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello!",
                    context_variables=ContextVariables({"user_name": "Bob"}),
                )
                print(result.content)  # Response will be personalized for Bob
                ```
        """
        last_response: AgentResponse | None = None
        accumulated_content: str | None = None
        accumulated_parsed_content: JSON | BaseModel | None = None
        accumulated_usage: Usage | None = None
        accumulated_response_cost: ResponseCost | None = None

        stream = self.stream(
            agent=agent,
            prompt=prompt,
            messages=messages,
            context_variables=context_variables,
            cleanup=cleanup,
        )

        async for agent_response in stream:
            last_response = agent_response
            accumulated_content = agent_response.content
            accumulated_parsed_content = agent_response.parsed_content

            accumulated_usage = combine_usage(
                accumulated_usage,
                agent_response.usage,
            )

            accumulated_response_cost = combine_response_cost(
                accumulated_response_cost,
                agent_response.response_cost,
            )

        if not last_response:
            raise SwarmError("No response was generated")

        return AgentExecutionResult(
            agent=last_response.agent,
            content=accumulated_content,
            parsed_content=accumulated_parsed_content,
            usage=accumulated_usage,
            response_cost=accumulated_response_cost,
        )

    def cleanup(self, clear_history: bool = False) -> None:
        """Clean up swarm state and reset agents.

        Performs a complete cleanup of the swarm's internal state, including:
        - Resetting active agent to idle state
        - Clearing the agent queue
        - Resetting all queued agents to idle state
        - Clearing context variables
        - Optionally clearing conversation history

        This method is useful for:
        - Preparing the swarm for a new conversation
        - Cleaning up resources after errors
        - Resetting state between independent executions
        - Managing memory usage in long-running applications

        Args:
            clear_history: Whether to clear the conversation history.
                If True, removes all stored messages. Defaults to False.

        Notes:
            - Agent states are reset to IDLE
            - Context variables are cleared
            - Agent queue is emptied
            - History clearing is optional to allow for conversation continuity

        Examples:
            Basic cleanup:
                ```python
                # Reset swarm state but keep history
                swarm.cleanup()
                ```

            Complete reset:
                ```python
                # Reset everything including history
                swarm.cleanup(clear_history=True)
                ```
        """
        if self._active_agent:
            self._active_agent.state = AgentState.IDLE
            self._active_agent = None

        for agent in self._agent_queue:
            agent.state = AgentState.IDLE

        self._agent_queue.clear()
        self._context_variables.clear()

        if clear_history:
            self.memory.clear_history()
