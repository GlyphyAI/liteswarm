# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import sys
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

import json_repair
import litellm
import orjson
from litellm import CustomStreamWrapper
from litellm.exceptions import ContextWindowExceededError
from litellm.types.utils import ChatCompletionDeltaToolCall, ModelResponse, StreamingChoices, Usage
from litellm.utils import token_counter
from pydantic import BaseModel

from liteswarm.core.context_manager import ContextManager, LiteContextManager
from liteswarm.core.event_handler import SwarmEventHandler
from liteswarm.core.message_store import LiteMessageStore, MessageStore
from liteswarm.types.collections import AsyncStream, ReturnItem, YieldItem, returnable
from liteswarm.types.events import (
    AgentResponseChunkEvent,
    AgentSwitchEvent,
    CompleteEvent,
    CompletionResponseChunkEvent,
    ErrorEvent,
    SwarmEventType,
    ToolCallResultEvent,
)
from liteswarm.types.exceptions import (
    CompletionError,
    ContextLengthError,
    MaxAgentSwitchesError,
    MaxResponseContinuationsError,
    SwarmError,
)
from liteswarm.types.llm import ResponseFormat, ResponseFormatJsonSchema, ResponseSchema
from liteswarm.types.misc import JSON
from liteswarm.types.swarm import (
    Agent,
    AgentExecutionResult,
    AgentResponseChunk,
    AgentState,
    CompletionResponseChunk,
    ContextVariables,
    Delta,
    FinishReason,
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
from liteswarm.utils.messages import dump_messages
from liteswarm.utils.misc import parse_content, safe_get_attr
from liteswarm.utils.retry import retry_with_backoff
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
                return a + b


            def multiply(a: float, b: float) -> float:
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
                prompt="What is (2 + 3) * 4?",
            )

            # The agent will:
            # 1. Use add(2, 3) to get 5
            # 2. Use multiply(5, 4) to get 20
            print(result.content)  # "The result is 20"
            ```

    Notes:
        - Maintains internal state during conversations
        - Create separate instances for concurrent conversations
    """

    def __init__(
        self,
        event_handler: SwarmEventHandler | None = None,
        message_store: MessageStore[Any] | None = None,
        context_manager: ContextManager | None = None,
        include_usage: bool = False,
        include_cost: bool = False,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 10.0,
        backoff_factor: float = 2.0,
        max_response_continuations: int = 5,
        max_agent_switches: int = 10,
        max_iterations: int = sys.maxsize,
    ) -> None:
        """Initialize a new Swarm instance with specified configuration.

        Configures swarm behavior with:
        - Event handler for processing swarm events
        - Message storage and history management
        - Context optimization and filtering
        - Usage and cost tracking
        - Retry and safety limits

        Args:
            event_handler: Handler for processing swarm events. Defaults to LiteSwarmEventHandler.
            message_store: Store for conversation history. Defaults to LiteMessageStore.
            context_manager: Manager for context optimization and relevance. Defaults to LiteContextManager.
            include_usage: Whether to track token usage. Defaults to False.
            include_cost: Whether to track response costs. Defaults to False.
            max_retries: Maximum API retry attempts. Defaults to 3.
            initial_retry_delay: Initial retry delay in seconds. Defaults to 1.0.
            max_retry_delay: Maximum retry delay in seconds. Defaults to 10.0.
            backoff_factor: Multiplier for retry delay. Defaults to 2.0.
            max_response_continuations: Maximum response length continuations. Defaults to 5.
            max_agent_switches: Maximum allowed agent switches. Defaults to 10.
            max_iterations: Maximum processing iterations. Defaults to sys.maxsize.
        """
        # Internal state (private)
        self._active_agent: Agent | None = None
        self._agent_queue: deque[Agent] = deque()
        self._context_variables: ContextVariables = ContextVariables()

        # Public configuration
        self.event_handler = event_handler or LiteSwarmEventHandler()
        self.message_store = message_store or LiteMessageStore()
        self.context_manager = context_manager or LiteContextManager(self.message_store)
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
        self.max_iterations = max_iterations

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
                    result=result,
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
                        result=result,
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
                    result=result,
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
                    result=result,
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
                result=None,
                error=ValueError(f"Unknown function: {function_name}"),
            )

        try:
            args = orjson.loads(tool_call.function.arguments)
            function_tool = function_tools_map[function_name]
            if function_has_parameter(function_tool, "context_variables"):
                args = {**args, "context_variables": context_variables}

            function_tool_result = function_tool(**args)
            tool_call_result = self._parse_tool_call_result(
                tool_call=tool_call,
                result=function_tool_result,
            )

        except Exception as error:
            tool_call_result = ToolCallFailureResult(
                tool_call=tool_call,
                result=None,
                error=error,
            )

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

    def _process_tool_call_result(
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

    def _prepare_completion_kwargs(
        self,
        agent: Agent,
        messages: list[Message],
    ) -> dict[str, Any]:
        """Prepare completion kwargs for both sync and async completions.

        Args:
            agent: Agent to use for completion, providing model settings.
            messages: Messages to send as conversation context.

        Returns:
            Dictionary of completion kwargs ready for litellm.completion/acompletion.
        """
        exclude_keys = {"response_format", "litellm_kwargs"}
        llm_messages = dump_messages(messages, exclude_none=True)
        llm_kwargs = agent.llm.model_dump(exclude=exclude_keys, exclude_none=True)
        llm_override_kwargs = {
            "messages": llm_messages,
            "stream": True,
            "stream_options": {"include_usage": True} if self.include_usage else None,
            "tools": functions_to_json(agent.llm.tools),
        }

        response_format = agent.llm.response_format
        supported_params = litellm.get_supported_openai_params(agent.llm.model) or []
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
            f"Sending messages to agent [{agent.id}]: {orjson.dumps(llm_messages).decode()}",
            level="DEBUG",
        )

        return completion_kwargs

    async def _create_completion(
        self,
        agent: Agent,
        messages: list[Message],
    ) -> CustomStreamWrapper:
        """Create an async completion request with comprehensive configuration.

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
            ContextWindowExceededError: If context window is exceeded.
        """
        completion_kwargs = self._prepare_completion_kwargs(agent, messages)
        response_stream = await litellm.acompletion(**completion_kwargs)
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
            Message(role="user", content="Please continue your previous response."),
        ]

        return await self._create_completion(agent, continuation_messages)

    async def _get_completion_response(
        self,
        agent: Agent,
        agent_messages: list[Message],
        context_variables: ContextVariables,
    ) -> AsyncGenerator[CompletionResponseChunk, None]:
        """Stream completion response chunks from the language model.

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
            CompletionResponseChunk containing:
            - Current response delta with content updates
            - Finish reason for proper flow control
            - Usage statistics for monitoring (if enabled)
            - Cost information for tracking (if enabled)

        Raises:
            CompletionError: If completion fails after all retry attempts.
            ContextLengthError: If context exceeds limits and cannot be reduced.
        """
        try:
            accumulated_content: str = ""
            continuation_count: int = 0
            current_stream: CustomStreamWrapper = await self._get_initial_stream(
                agent=agent,
                agent_messages=agent_messages,
                context_variables=context_variables,
            )

            while continuation_count < self.max_response_continuations:
                async for chunk in current_stream:
                    response_chunk = self._process_stream_chunk(agent, chunk)
                    if response_chunk.delta.content:
                        accumulated_content += response_chunk.delta.content

                    yield response_chunk

                    if response_chunk.finish_reason == "length":
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
            raise CompletionError(
                f"Failed to get completion response: {e}",
                original_error=e,
            ) from e

    async def _get_initial_stream(
        self,
        agent: Agent,
        agent_messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> CustomStreamWrapper:
        """Create initial completion stream with robust error handling.

        Creates and manages the initial completion stream with comprehensive features:
        - Automatic context reduction on length errors
        - Intelligent history trimming when needed
        - Proper error propagation for unrecoverable cases
        - Token usage optimization
        - Stream state validation

        Args:
            agent: Agent for completion, providing model settings and tools.
            agent_messages: Messages forming the conversation context.
            context_variables: Optional context for dynamic resolution.

        Returns:
            CustomStreamWrapper managing the completion response stream.

        Raises:
            CompletionError: If completion fails after exhausting retries.
            ContextLengthError: If context remains too large after reduction.
        """
        try:
            return await self._create_completion(
                agent=agent,
                messages=agent_messages,
            )
        except ContextWindowExceededError:
            log_verbose(
                "Context window exceeded, attempting to reduce context size",
                level="WARNING",
            )

            return await self._reduce_context_size(
                agent=agent,
                context_variables=context_variables,
            )

    def _process_stream_chunk(
        self,
        agent: Agent,
        chunk: ModelResponse,
    ) -> CompletionResponseChunk:
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

        if usage is not None and self.include_cost:
            response_cost = calculate_response_cost(
                model=agent.llm.model,
                usage=usage,
            )

        return CompletionResponseChunk(
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
    ) -> CustomStreamWrapper:
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
            New stream for continuation.

        Raises:
            MaxResponseContinuationsError: If maximum continuations reached.
        """
        if continuation_count >= self.max_response_continuations:
            generated_tokens = token_counter(model=agent.llm.model, text=accumulated_content)
            raise MaxResponseContinuationsError(
                message=f"Maximum response continuations ({self.max_response_continuations}) reached",
                continuation_count=continuation_count,
                max_continuations=self.max_response_continuations,
                total_tokens=generated_tokens,
            )

        log_verbose(
            f"Response continuation {continuation_count}/{self.max_response_continuations}",
            level="INFO",
        )

        return await self._continue_generation(
            agent=agent,
            previous_content=accumulated_content,
            context_variables=context_variables,
        )

    def _should_parse_agent_response(
        self,
        model: str,
        custom_llm_provider: str | None = None,
        response_format: ResponseFormat | None = None,
    ) -> bool:
        """Determine if response content requires parsing based on format and model support.

        Args:
            model: Model identifier to check for format support.
            custom_llm_provider: Optional custom provider to check for format support.
            response_format: Format specification to evaluate.

        Returns:
            True if content should be parsed based on format and model capabilities.
        """
        if not response_format:
            return False

        if not litellm.supports_response_schema(model, custom_llm_provider):
            return False

        return (
            is_subtype(response_format, BaseModel)
            or is_subtype(response_format, ResponseFormatJsonSchema)
            or is_subtype(response_format, ResponseSchema)
        )

    def _parse_agent_response_content(
        self,
        full_content: str,
        finish_reason: FinishReason | None,
        response_format: ResponseFormat | None,
        ignore_errors: bool = False,
    ) -> JSON | BaseModel | None:
        """Parse agent response content into specified format when appropriate.

        Args:
            full_content: Complete response content to parse.
            finish_reason: Reason for response completion.
            response_format: Target format specification.
            ignore_errors: Whether to ignore errors and return None.

        Returns:
            Parsed content in specified format, or None if parsing not needed/possible.

        Raises:
            CompletionError: If parsing fails and ignore_errors is False.
        """
        try:
            parsed_content = json_repair.loads(full_content)
            if isinstance(parsed_content, tuple):
                parsed_content = parsed_content[0]

            valid_finish_reasons: set[FinishReason] = {"stop", "tool_calls"}
            if finish_reason in valid_finish_reasons:
                if parsed_content and is_subtype(response_format, BaseModel):
                    return response_format.model_validate(parsed_content)

            return parsed_content

        except Exception as e:
            if ignore_errors:
                return None

            raise CompletionError(
                f"Failed to parse agent response content: {e}",
                original_error=e,
            ) from e

    @returnable
    async def _process_agent_response(
        self,
        agent: Agent,
        agent_messages: list[Message],
        context_variables: ContextVariables,
    ) -> AsyncStream[SwarmEventType, AgentResponseChunk | None]:
        """Process agent response and stream completion events.

        Handles the core response processing from the language model. Streams both raw
        completion chunks and processed agent response chunks, accumulates content and
        tool calls, and handles response parsing. The @returnable decorator enables
        streaming both types of events while returning the final accumulated response
        chunk for further processing.

        Args:
            agent: Agent processing the response.
            agent_messages: Messages providing conversation context.
            context_variables: Context for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding completion and response events,
            returning the final accumulated response chunk or None.

        Raises:
            CompletionError: If completion fails after retries.
            ContextLengthError: If context exceeds limits after reduction.

        Notes:
            - Internal method used by _stream_agent_execution()
            - Streams both raw completion and processed response events
            - Accumulates content and tool calls incrementally
            - Handles response format parsing if specified
            - Uses retry with backoff for completion errors
            - Returns final response for tool call processing
        """
        full_content: str | None = None
        full_tool_calls: list[ChatCompletionDeltaToolCall] = []
        parsed_content: JSON | BaseModel | None = None
        last_response_chunk: AgentResponseChunk | None = None

        should_parse_content = self._should_parse_agent_response(
            model=agent.llm.model,
            response_format=agent.llm.response_format,
        )

        completion_stream = retry_with_backoff(
            self._get_completion_response,
            max_retries=self.max_retries,
            initial_delay=self.initial_retry_delay,
            max_delay=self.max_retry_delay,
            backoff_factor=self.backoff_factor,
        )

        async for completion_chunk in completion_stream(
            agent=agent,
            agent_messages=agent_messages,
            context_variables=context_variables,
        ):
            yield YieldItem(CompletionResponseChunkEvent(agent=agent, chunk=completion_chunk))

            delta = completion_chunk.delta
            finish_reason = completion_chunk.finish_reason
            if delta.content:
                if full_content is None:
                    full_content = delta.content
                else:
                    full_content += delta.content

            if should_parse_content and full_content:
                parsed_content = self._parse_agent_response_content(
                    full_content=full_content,
                    finish_reason=finish_reason,
                    response_format=agent.llm.response_format,
                    ignore_errors=True,
                )

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if not isinstance(tool_call, ChatCompletionDeltaToolCall):
                        continue

                    if tool_call.id:
                        full_tool_calls.append(tool_call)
                    elif full_tool_calls:
                        last_tool_call = full_tool_calls[-1]
                        last_tool_call.function.arguments += tool_call.function.arguments

            last_response_chunk = AgentResponseChunk(
                agent=agent,
                completion=completion_chunk,
                content=full_content,
                parsed_content=parsed_content,
                tool_calls=full_tool_calls,
            )

            yield YieldItem(AgentResponseChunkEvent(chunk=last_response_chunk))

        yield ReturnItem(last_response_chunk)

    @returnable
    async def _process_assistant_response(
        self,
        agent: Agent,
        content: str | None,
        context_variables: ContextVariables,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> AsyncStream[SwarmEventType, list[Message]]:
        """Process assistant response and stream tool call events.

        Handles assistant response processing and tool execution. Creates messages
        for the response and any tool calls, manages agent switching if needed,
        and updates context. The @returnable decorator enables streaming tool call
        events during processing while returning the complete list of generated
        messages at the end.

        Args:
            agent: Agent that generated the response.
            content: Text content of response, may be None.
            context_variables: Context for tool execution.
            tool_calls: Tool calls to process in order.

        Returns:
            ReturnableAsyncGenerator yielding tool call events and returning
            the list of generated messages.

        Notes:
            - Internal method used by _stream_agent_execution()
            - Creates assistant message with content and tool calls
            - Processes tool calls and yields events for each
            - Handles agent switching through tool results
            - Updates context variables from tool results
            - Returns all generated messages for history
        """
        messages: list[Message] = [
            Message(
                role="assistant",
                content=content,
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
                yield YieldItem(ToolCallResultEvent(agent=agent, tool_call_result=tool_call_result))

                tool_message = self._process_tool_call_result(tool_call_result)
                if tool_message.agent:
                    agent.state = AgentState.STALE
                    self._agent_queue.append(tool_message.agent)

                if tool_message.context_variables:
                    self._context_variables.update(tool_message.context_variables)

                messages.append(tool_message.message)

        yield ReturnItem(messages)

    # ================================================
    # MARK: Context Management
    # ================================================

    async def _optimize_context(self, agent: Agent) -> list[Message]:
        """Optimize the message context for an agent's execution.

        Uses the context manager to optimize the conversation history by:
        - Filtering irrelevant messages
        - Summarizing long conversations
        - Removing redundant content
        - Maintaining conversation coherence
        - Preserving critical context

        Args:
            agent: Agent requiring context optimization.

        Returns:
            List of optimized messages ready for agent execution.
        """
        return await self.context_manager.optimize_context(
            model=agent.llm.model,
        )

    async def _create_agent_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        """Prepare execution context for an agent.

        Creates a properly ordered message list for agent execution:
        1. System message with resolved instructions
        2. Filtered conversation history (excluding system messages)
        3. Optional prompt as user message

        Args:
            agent: Agent requiring context creation.
            prompt: Optional user prompt to include.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            List of messages ready for execution.
        """
        return await self.context_manager.create_context(
            agent=agent,
            prompt=prompt,
            context_variables=context_variables,
        )

    async def _reduce_context_size(
        self,
        agent: Agent,
        context_variables: ContextVariables | None = None,
    ) -> CustomStreamWrapper:
        """Reduce context size when it exceeds model limits.

        Uses the context manager to optimize message history when
        the context window is exceeded. Attempts to maintain conversation
        coherence while fitting within model limits.

        Args:
            agent: Agent for completion attempt
            context_variables: Optional context for resolution

        Returns:
            Response stream from completion

        Raises:
            ContextLengthError: If context remains too large after optimization
        """
        optimized_messages = await self._optimize_context(agent)
        agent_messages = await self._create_agent_context(
            agent=agent,
            context_variables=context_variables,
        )

        try:
            return await self._create_completion(agent, agent_messages)
        except ContextWindowExceededError as e:
            raise ContextLengthError(
                message="Context window exceeded even after optimization",
                current_length=len(optimized_messages),
                original_error=e,
            ) from e

    # ================================================
    # MARK: Agent Management
    # ================================================

    async def _activate_agent(
        self,
        agent: Agent,
        context_variables: ContextVariables | None = None,
        add_system_message: bool = True,
    ) -> None:
        """Activate a new agent and handle core state changes.

        Handles the core agent activation process:
        - Sets agent as active
        - Updates agent state to ACTIVE
        - Optionally adds system message with agent instructions

        Args:
            agent: Agent to activate.
            context_variables: Optional context for dynamic instruction resolution.
            add_system_message: Whether to add system message to history. Defaults to True.
        """
        self._active_agent = agent
        self._active_agent.state = AgentState.ACTIVE

        if add_system_message:
            instructions = unwrap_instructions(agent.instructions, context_variables)
            message = Message(role="system", content=instructions)
            await self.message_store.add_message(message)

    async def _initialize_conversation(
        self,
        agent: Agent,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
    ) -> None:
        """Initialize and set up a new conversation session.

        Prepares the conversation environment by:
        - Loading existing messages if provided
        - Activating the agent (adds system message if needed)
        - Adding initial prompt to history if provided

        Args:
            agent: Agent to handle the conversation.
                Will become the active agent.
            prompt: Optional starting message from the user.
                Added to history if provided. Defaults to None.
            messages: Optional pre-existing conversation history.
                Loaded into message store if provided. Defaults to None.
            context_variables: Optional context for dynamic resolution.
                Used to resolve agent instructions. Defaults to None.

        Raises:
            SwarmError: If neither prompt nor messages are provided.

        Notes:
            - At least one of prompt or messages must be provided
            - System message is only added if agent changes
            - Message order is preserved: system -> history -> prompt
        """
        if not messages and not prompt:
            raise SwarmError("Please provide at least one message or prompt")

        if messages:
            await self.message_store.set_messages(messages)

        await self._activate_agent(
            agent=agent,
            context_variables=context_variables,
            add_system_message=agent != self._active_agent,
        )

        if prompt:
            message = Message(role="user", content=prompt)
            await self.message_store.add_message(message)

    @returnable
    async def _stream_agent_execution(
        self,
        iteration_count: int = 0,
        agent_switch_count: int = 0,
    ) -> AsyncStream[SwarmEventType, None]:
        """Stream events from active agent and handle agent switching.

        Core execution loop of the swarm. Manages agent lifecycle, processes responses,
        and handles agent switching when needed. Streams events for agent responses,
        tool calls, and switches. The @returnable decorator provides consistent event
        streaming, with None as the final return value since result accumulation
        happens at a higher level.

        Args:
            iteration_count: Current number of processing iterations.
                Used to enforce maximum iteration limits. Defaults to 0.
            agent_switch_count: Number of agent switches performed.
                Used to enforce maximum switch limits. Defaults to 0.

        Returns:
            ReturnableAsyncGenerator yielding events and returning None.

        Raises:
            SwarmError: If no active agent is available.
            MaxAgentSwitchesError: If maximum number of switches is exceeded.

        Notes:
            - Internal method used by _create_swarm_event_stream()
            - Manages agent state transitions (IDLE -> ACTIVE -> STALE)
            - Handles agent queue for switching
            - Enforces iteration and switch limits
            - Delegates response processing to _process_agent_response()
        """
        if not self._active_agent:
            raise SwarmError("No active agent available to process messages")

        switch_history: list[str] = [self._active_agent.id]
        while iteration_count < self.max_iterations:
            iteration_count += 1
            if self._active_agent.state == AgentState.STALE:
                if agent_switch_count >= self.max_agent_switches:
                    raise MaxAgentSwitchesError(
                        message=f"Maximum number of agent switches ({self.max_agent_switches}) exceeded",
                        switch_count=agent_switch_count,
                        max_switches=self.max_agent_switches,
                        switch_history=switch_history,
                    )

                if not self._agent_queue:
                    log_verbose("No more agents in queue, stopping execution")
                    break

                previous_agent = self._active_agent
                next_agent = self._agent_queue.popleft()
                agent_switch_count += 1
                switch_history.append(next_agent.id)

                log_verbose(f"Switching from agent {previous_agent.id} to {next_agent.id}")
                await self._activate_agent(next_agent, self._context_variables)

                yield YieldItem(AgentSwitchEvent(previous=previous_agent, current=next_agent))

            agent_messages = await self._create_agent_context(
                agent=self._active_agent,
                context_variables=self._context_variables,
            )

            agent_response_stream = self._process_agent_response(
                agent=self._active_agent,
                agent_messages=agent_messages,
                context_variables=self._context_variables,
            )

            async for event in agent_response_stream:
                yield YieldItem(event)

            last_agent_response = await agent_response_stream.get_result()
            if not last_agent_response:
                continue

            if last_agent_response.content or last_agent_response.tool_calls:
                new_messages_stream = self._process_assistant_response(
                    agent=self._active_agent,
                    content=last_agent_response.content,
                    context_variables=self._context_variables,
                    tool_calls=last_agent_response.tool_calls,
                )

                async for event in new_messages_stream:
                    yield YieldItem(event)

                messages = await new_messages_stream.get_result()
                await self.message_store.add_messages(messages)
            else:
                # We might not want to do this, but it's a good fallback
                # Please consider removing this if it leads to unexpected behavior
                empty_message = Message(role="assistant", content="<empty>")
                await self.message_store.add_message(empty_message)
                log_verbose(
                    "Empty response received, appending placeholder message",
                    level="WARNING",
                )

            if not last_agent_response.tool_calls:
                self._active_agent.state = AgentState.STALE

        yield ReturnItem(None)

    @returnable
    async def _create_swarm_event_stream(
        self,
        agent: Agent,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEventType, None]:
        """Create the base event stream for swarm execution.

        Core implementation of swarm's event streaming. Initializes conversation,
        processes agent responses, handles tool calls, and manages agent switches.
        Yields various events during execution but doesn't accumulate a final result.
        The @returnable decorator is used here primarily for consistent event streaming,
        with None as the final return value.

        Args:
            agent: Initial agent for handling conversations.
            prompt: Optional user prompt to process.
            messages: Optional list of previous conversation messages.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning None.

        Raises:
            SwarmError: If neither prompt nor messages are provided.
            ContextLengthError: If context becomes too large.
            MaxAgentSwitchesError: If too many switches occur.
            MaxResponseContinuationsError: If response needs too many continuations.

        Notes:
            - Internal method used by stream()
            - Handles core event generation
            - Manages conversation state
            - Doesn't accumulate final result
        """
        try:
            self._context_variables.update(context_variables or {})
            await self._initialize_conversation(
                agent=agent,
                prompt=prompt,
                messages=messages,
                context_variables=self._context_variables,
            )

            async for event in self._stream_agent_execution():
                yield YieldItem(event)

            yield ReturnItem(None)

        except Exception as e:
            yield YieldItem(ErrorEvent(agent=self._active_agent, error=e))
            raise

        finally:
            all_messages = await self.message_store.get_messages()
            yield YieldItem(CompleteEvent(agent=self._active_agent, messages=all_messages))

    # ================================================
    # MARK: Public Interface
    # ================================================

    @returnable
    async def stream(
        self,
        agent: Agent,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEventType, AgentExecutionResult]:
        """Stream swarm events and return final execution result.

        Main entry point for swarm execution. Processes user input through the agent,
        streaming various events (responses, tool calls, errors) during execution.
        Accumulates content and metadata to produce a final result. The @returnable
        decorator extends the async generator to support both streaming events during
        execution and returning a final result value when complete.

        Args:
            agent: Agent to execute the task.
            prompt: Optional user prompt to process.
            messages: Optional list of previous conversation messages for context.
            context_variables: Optional variables for dynamic instruction resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning final result.

        Raises:
            SwarmError: If neither prompt nor messages are provided.
            ContextLengthError: If context becomes too large to process.
            MaxAgentSwitchesError: If too many agent switches occur.
            MaxResponseContinuationsError: If response requires too many continuations.

        Examples:
            Stream events and get result:
                ```python
                stream = swarm.stream(agent, prompt="Hello")

                # Process events during execution
                async for event in stream:
                    match event.type:
                        case "agent_response_chunk":
                            print(event.chunk.completion.delta.content)
                        case "error":
                            print(f"Error: {event.error}")

                # Get final result after completion
                result = await stream.get_result()
                print(f"Final content: {result.content}")
                ```

            Just get final result:
                ```python
                stream = swarm.stream(agent, prompt="Hello")
                result = await stream.get_result()
                print(result.content)
                ```

        Notes:
            - AsyncStream supports both event streaming and final result
            - Events provide real-time updates during execution
            - Final result contains complete response with metadata
            - Stream can be consumed multiple times if needed
        """
        event_stream = self._create_swarm_event_stream(
            agent=agent,
            prompt=prompt,
            messages=messages,
            context_variables=context_variables,
        )

        last_chunk: AgentResponseChunk | None = None
        accumulated_content: str | None = None
        accumulated_parsed_content: JSON | BaseModel | None = None
        accumulated_usage: Usage | None = None
        accumulated_response_cost: ResponseCost | None = None

        async for event in event_stream:
            yield YieldItem(event)

            if event.type == "agent_response_chunk":
                last_chunk = event.chunk
                accumulated_content = last_chunk.content
                accumulated_parsed_content = last_chunk.parsed_content
                accumulated_usage = combine_usage(
                    accumulated_usage,
                    last_chunk.completion.usage,
                )
                accumulated_response_cost = combine_response_cost(
                    accumulated_response_cost,
                    last_chunk.completion.response_cost,
                )

        if last_chunk is None:
            raise SwarmError("No agent response chunks received")

        execution_result = AgentExecutionResult(
            agent=last_chunk.agent,
            content=accumulated_content,
            parsed_content=accumulated_parsed_content,
            usage=accumulated_usage,
            response_cost=accumulated_response_cost,
        )

        yield ReturnItem(execution_result)

    async def execute(
        self,
        agent: Agent,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
        event_handler: SwarmEventHandler | None = None,
    ) -> AgentExecutionResult:
        """Execute user prompt and return final execution result.

        Convenience method that wraps stream() to provide event handling and result
        collection. Events are processed through the provided handler if specified,
        and the final result is returned.

        Args:
            agent: Agent to execute the task.
            prompt: Optional user prompt to process.
            messages: Optional list of previous conversation messages for context.
            context_variables: Optional variables for dynamic instruction resolution.
            event_handler: Optional handler for processing execution events.

        Returns:
            Complete execution result with final content and metadata.

        Raises:
            SwarmError: If neither prompt nor messages are provided.
            ContextLengthError: If context becomes too large to process.
            MaxAgentSwitchesError: If too many agent switches occur.
            MaxResponseContinuationsError: If response requires too many continuations.

        Examples:
            With event handler:
                ```python
                class LoggingHandler(SwarmEventHandler):
                    async def on_event(self, event: SwarmEventType) -> None:
                        match event.type:
                            case "agent_response_chunk":
                                print(event.chunk.completion.delta.content)
                            case "error":
                                print(f"Error: {event.error}")


                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                    event_handler=LoggingHandler(),
                )
                print(f"Final content: {result.content}")
                ```

            Without event handler:
                ```python
                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                    context_variables={"user": "Alice"},
                )
                print(result.content)
                ```

        Notes:
            - Events are processed through handler if provided
            - Final result is returned after completion
            - Same error handling as stream() applies
            - Conversation history is preserved
        """
        stream = self.stream(
            agent=agent,
            prompt=prompt,
            messages=messages,
            context_variables=context_variables,
        )

        if event_handler:
            async for event in stream:
                await event_handler.on_event(event)

        return await stream.get_result()

    async def cleanup(
        self,
        clear_agents: bool = True,
        clear_context: bool = False,
        clear_messages: bool = False,
    ) -> None:
        """Clean up swarm state and reset components.

        Performs selective cleanup of the swarm's internal state based on flags:
        - Agent states and queue management
        - Context variables clearing
        - Message storage clearing

        Args:
            clear_agents: Whether to reset all agent states and clear queue.
                If True, resets active agent and empties queue. Defaults to True.
            clear_context: Whether to clear context variables.
                If True, removes all stored variables. Defaults to False.
            clear_messages: Whether to clear message store.
                If True, removes all stored messages. Defaults to False.

        Examples:
            Basic cleanup (just agents):
                ```python
                # After executing a conversation
                result = await swarm.execute(agent, prompt="Hello")

                # Reset agent states and queue
                swarm.cleanup()  # Only clears agents by default
                ```

            Full cleanup:
                ```python
                # Clear everything for a fresh start
                swarm.cleanup(
                    clear_agents=True,
                    clear_context=True,
                    clear_messages=True,
                )
                ```

            Selective cleanup:
                ```python
                # Keep conversation history but reset agents and context
                swarm.cleanup(
                    clear_agents=True,
                    clear_context=True,
                    clear_messages=False,  # Preserve message history
                )

                # Start new conversation with same context
                result = await swarm.execute(
                    agent=new_agent,
                    prompt="Continue from previous context",
                )
                ```

            Between conversations:
                ```python
                # First conversation
                result1 = await swarm.execute(agent1, prompt="Task 1")
                swarm.cleanup(clear_messages=True)  # Clear previous conversation

                # Second conversation starts fresh
                result2 = await swarm.execute(agent2, prompt="Task 2")
                ```

        Notes:
            - Agent cleanup resets both active agent and queue
            - Context and message clearing are optional
            - Must be called explicitly when cleanup is needed
            - Partial cleanup allows for state preservation
        """
        if clear_agents:
            if self._active_agent:
                self._active_agent.state = AgentState.IDLE
                self._active_agent = None

            for agent in self._agent_queue:
                agent.state = AgentState.IDLE

            self._agent_queue.clear()

        if clear_context:
            self._context_variables.clear()

        if clear_messages:
            await self.message_store.clear()
