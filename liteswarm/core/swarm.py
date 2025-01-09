# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import sys
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

import litellm
import orjson
from litellm import CustomStreamWrapper
from litellm.exceptions import ContextWindowExceededError
from litellm.types.utils import ChatCompletionDeltaToolCall, ModelResponse, StreamingChoices, Usage
from litellm.utils import token_counter
from pydantic import BaseModel

from liteswarm.core.swarm_stream import SwarmStream
from liteswarm.types.collections import (
    AsyncStream,
    ReturnItem,
    YieldItem,
    returnable,
)
from liteswarm.types.events import (
    AgentActivateEvent,
    AgentCompleteEvent,
    AgentExecutionCompleteEvent,
    AgentExecutionStartEvent,
    AgentResponseChunkEvent,
    AgentResponseEvent,
    AgentStartEvent,
    AgentSwitchEvent,
    CompletionResponseChunkEvent,
    SwarmEvent,
    ToolCallResultEvent,
)
from liteswarm.types.exceptions import (
    CompletionError,
    ContextLengthError,
    MaxAgentSwitchesError,
    MaxResponseContinuationsError,
    SwarmError,
)
from liteswarm.types.llm import (
    ResponseFormat,
    ResponseFormatJsonSchema,
    ResponseFormatPydantic,
    ResponseSchema,
)
from liteswarm.types.swarm import (
    Agent,
    AgentContext,
    AgentContextUpdate,
    AgentExecutionResult,
    AgentResponse,
    AgentResponseChunk,
    AgentState,
    CompletionResponseChunk,
    ContextVariables,
    Delta,
    Message,
    ResponseCost,
    ToolCallResult,
    ToolResult,
)
from liteswarm.types.typing import is_subtype
from liteswarm.utils.function import function_has_parameter, functions_to_json
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import dump_messages, get_max_tokens
from liteswarm.utils.misc import (
    parse_content,
    parse_response,
    resolve_agent_instructions,
    safe_get_attr,
)
from liteswarm.utils.retry import retry_wrapper
from liteswarm.utils.usage import calculate_response_cost

litellm.modify_params = True


class Swarm:
    """Provider-agnostic orchestrator for AI agent interactions.

    Swarm orchestrates conversations with AI agents by processing messages,
    executing tools, and managing agent transitions. It provides async streaming
    interfaces for agent interactions, powered by 100+ supported LLMs. Each
    execution is independent and maintains its own isolated state.

    Swarm is designed as a lightweight execution client for agents, allowing
    developers to build higher-level abstractions while using Swarm as the
    underlying execution engine.

    Key features:
        - Tool execution and agent switching
        - Response validation and continuation
        - Automatic retries with backoff
        - Usage and cost tracking

    Examples:
        Basic usage:
            ```python
            from liteswarm.types.swarm import Agent, Message
            from liteswarm.types.llm import LLM


            def add(a: int, b: int) -> int:
                return a + b


            def multiply(a: int, b: int) -> int:
                return a * b


            # Create agent with tools
            agent = Agent(
                id="math",
                instructions="You are a math assistant.",
                llm=LLM(
                    model="gpt-4o",
                    tools=[add, multiply],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
            )

            # Execute with messages
            messages = [Message(role="user", content="Calculate 2 + 2")]
            result = await swarm.execute(agent, messages=messages)
            print(result.content)  # "The result is 4"
            ```

        Stream events:
            ```python
            messages = [Message(role="user", content="Calculate (2 + 3) * 4")]
            stream = swarm.stream(agent, messages=messages)

            async for event in stream:
                if event.type == "agent_response_chunk":
                    print(event.chunk.completion.delta.content)
                elif event.type == "tool_call_result":
                    print(f"Tool result: {event.tool_call_result.result}")

            result = await stream.get_return_value()
            ```

        Agent switching:
            ```python
            def switch_to_expert(domain: str) -> ToolResult:
                return ToolResult.switch_agent(
                    agent=Agent(
                        id=f"{domain}-expert",
                        instructions=f"You are a {domain} expert.",
                        llm=LLM(model="gpt-4o"),
                    ),
                    content=f"Switching to {domain} expert",
                )


            router = Agent(
                id="router",
                instructions="Route questions to experts.",
                llm=LLM(
                    model="gpt-4o",
                    tools=[switch_to_expert],
                ),
            )

            messages = [Message(role="user", content="Explain quantum physics")]
            async for event in swarm.stream(router, messages=messages):
                if event.type == "agent_switch":
                    print(f"Switched to {event.current.id}")
            ```

    Notes:
        - Each execution maintains isolated conversation state
        - Create separate instances for concurrent conversations
        - Safety limits prevent infinite loops and recursion
    """

    def __init__(
        self,
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
        """Initialize a new Swarm instance.

        Creates a swarm instance with specified configuration for usage tracking,
        error recovery, and safety limits. Each execution maintains its own
        isolated conversation state.

        Args:
            include_usage: Whether to track token usage.
            include_cost: Whether to track response costs.
            max_retries: Maximum API retry attempts.
            initial_retry_delay: Initial retry delay in seconds.
            max_retry_delay: Maximum retry delay in seconds.
            backoff_factor: Multiplier for retry delay.
            max_response_continuations: Maximum response length continuations.
            max_agent_switches: Maximum allowed agent switches.
            max_iterations: Maximum processing iterations.

        Notes:
            - Each execution maintains isolated conversation state
            - Retry configuration uses exponential backoff
            - Safety limits prevent infinite loops and recursion
        """
        # Internal state (private)
        self._active_agent: Agent | None = None
        self._pending_agent_contexts: deque[AgentContext] = deque()
        self._conversation_messages: list[Message] = []
        self._context_variables: ContextVariables = ContextVariables()

        # Public configuration
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
        tool_return_value: Any,
    ) -> ToolCallResult:
        """Parse a tool's return value into a framework result representation.

        Converts tool return values into ToolCallResult instances for internal processing.
        The method processes three types of returns: direct Agent returns for switching,
        ToolResult objects for complex responses, and simple values that become message content.

        Args:
            tool_call: Original tool call details.
            tool_return_value: Raw return value from the tool function:
                - Agent instance for switching
                - ToolResult for complex responses
                - JSON-serializable value for simple responses

        Returns:
            ToolCallResult with processed result and any context updates.
        """

        def _create_tool_message(content: str) -> Message:
            return Message(
                role="tool",
                content=content,
                tool_call_id=tool_call.id,
            )

        match tool_return_value:
            case Agent() as agent:
                return ToolCallResult(
                    tool_call=tool_call,
                    tool_return_value=tool_return_value,
                    tool_message=_create_tool_message(f"Switched to agent {agent.id}"),
                    context_update=AgentContextUpdate(
                        agent_context=AgentContext(agent=agent),
                    ),
                )

            case ToolResult() as tool_result:
                content = parse_content(tool_result.content)
                return ToolCallResult(
                    tool_call=tool_call,
                    tool_return_value=tool_return_value,
                    tool_message=_create_tool_message(content),
                    context_update=tool_result.context_update,
                )

            case _:
                content = parse_content(tool_return_value)
                return ToolCallResult(
                    tool_call=tool_call,
                    tool_return_value=tool_return_value,
                    tool_message=_create_tool_message(content),
                )

    async def _process_tool_call(
        self,
        agent: Agent,
        context_variables: ContextVariables,
        tool_call: ChatCompletionDeltaToolCall,
    ) -> ToolCallResult:
        """Process a single tool call execution.

        Manages the complete lifecycle of a tool call by validating the tool exists,
        executing it with proper context, handling errors gracefully, and processing
        the result. The method supports both regular function return values and
        special cases like agent switching.

        Args:
            agent: Agent that initiated the tool call.
            context_variables: Context for dynamic resolution.
            tool_call: Tool call details with function name and arguments.

        Returns:
            ToolCallResult containing the execution result.
        """
        tool_call_result: ToolCallResult
        tools = agent.llm.tools or []
        function_name = tool_call.function.name
        function_tools_map = {tool.__name__: tool for tool in tools if hasattr(tool, "__name__")}

        if function_name not in function_tools_map:
            return ToolCallResult(
                tool_call=tool_call,
                tool_return_value=None,
                tool_error=ValueError(f"Unknown function: {function_name}"),
                tool_message=Message(
                    role="tool",
                    content=f"Unknown function: {function_name}",
                    tool_call_id=tool_call.id,
                ),
            )

        try:
            args = orjson.loads(tool_call.function.arguments)
            function_tool = function_tools_map[function_name]
            if function_has_parameter(function_tool, "context_variables"):
                args = {**args, "context_variables": context_variables}

            tool_return_value = function_tool(**args)
            tool_call_result = self._parse_tool_call_result(
                tool_call=tool_call,
                tool_return_value=tool_return_value,
            )

        except Exception as error:
            tool_call_result = ToolCallResult(
                tool_call=tool_call,
                tool_return_value=None,
                tool_error=error,
                tool_message=Message(
                    role="tool",
                    content=f"Error executing tool: {str(error)}",
                    tool_call_id=tool_call.id,
                ),
            )

        return tool_call_result

    async def _process_tool_calls(
        self,
        agent: Agent,
        context_variables: ContextVariables,
        tool_calls: list[ChatCompletionDeltaToolCall],
    ) -> list[ToolCallResult]:
        """Process multiple tool calls with optimized execution.

        Handles tool call execution using two strategies for optimal performance:
        direct execution for single calls to minimize overhead, and parallel
        execution with asyncio.gather for multiple calls. The execution order
        is preserved in the results.

        Args:
            agent: Agent that initiated the calls.
            context_variables: Context for dynamic resolution.
            tool_calls: List of tool calls to process.

        Returns:
            List of ToolCallResult objects containing execution results.
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
            "tools": functions_to_json(agent.llm.tools) if agent.llm.tools else None,
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
        """Create a completion request with agent's configuration.

        Prepares and sends a completion request by configuring message history,
        tool settings, response format, and usage tracking based on agent's
        configuration.

        Args:
            agent: Agent to use for completion.
            messages: Messages for conversation context.

        Returns:
            Response stream from the completion API.

        Raises:
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
        """Continue generation after reaching output token limit.

        Creates a new completion request that continues from the previous content
        while maintaining the original agent's instructions and settings.

        Args:
            agent: Agent for continuation.
            previous_content: Content generated before hitting limit.
            context_variables: Context for dynamic resolution.

        Returns:
            Response stream for the continuation request.
        """
        instructions = resolve_agent_instructions(agent, context_variables)
        continuation_messages = [
            Message(role="system", content=instructions),
            Message(role="assistant", content=previous_content),
            Message(role="user", content="Please continue your previous response."),
        ]

        return await self._create_completion(agent, continuation_messages)

    async def _get_completion_response(
        self,
        agent: Agent,
        messages: list[Message],
        context_variables: ContextVariables,
    ) -> AsyncGenerator[CompletionResponseChunk, None]:
        """Stream completion response chunks from the language model.

        Manages the complete response lifecycle including continuation handling,
        error recovery, and usage tracking. Uses automatic retries with backoff
        for transient errors.

        Args:
            agent: Agent for completion.
            messages: Messages for conversation context.
            context_variables: Context for dynamic resolution.

        Yields:
            CompletionResponseChunk containing content updates and metadata.

        Raises:
            CompletionError: If completion fails after all retry attempts.
            ContextLengthError: If context exceeds limits and cannot be reduced.
        """
        try:
            accumulated_content: str = ""
            continuation_count: int = 0
            current_stream: CustomStreamWrapper = await self._get_initial_completion(
                agent=agent,
                messages=messages,
            )

            while continuation_count < self.max_response_continuations:
                async for chunk in current_stream:
                    response_chunk = self._process_completion_chunk(agent, chunk)
                    if response_chunk.delta.content:
                        accumulated_content += response_chunk.delta.content

                    yield response_chunk

                    if response_chunk.finish_reason == "length":
                        continuation_count += 1
                        current_stream = await self._handle_response_continuation(
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

    async def _get_initial_completion(
        self,
        agent: Agent,
        messages: list[Message],
    ) -> CustomStreamWrapper:
        """Create initial completion stream with error handling.

        Creates the first completion stream attempt, handling context length
        errors by providing clear error messages with token counts.

        Args:
            agent: Agent for completion.
            messages: Messages for conversation context.

        Returns:
            CustomStreamWrapper managing the completion response stream.

        Raises:
            CompletionError: If completion fails after exhausting retries.
            ContextLengthError: If context remains too large after reduction.
        """
        try:
            return await self._create_completion(
                agent=agent,
                messages=messages,
            )
        except ContextWindowExceededError as e:
            model = agent.llm.model
            token_count = token_counter(model=model, messages=messages)
            max_tokens = agent.llm.max_tokens or get_max_tokens(model)
            err_message = f"Context window exceeded: {token_count}/{max_tokens} tokens for {model}"

            log_verbose(err_message, level="ERROR")

            raise ContextLengthError(
                message=err_message,
                model=model,
                current_length=token_count,
                max_length=max_tokens,
                original_error=e,
            ) from e

    def _process_completion_chunk(
        self,
        agent: Agent,
        chunk: ModelResponse,
    ) -> CompletionResponseChunk:
        """Process completion stream chunk into a structured response.

        Extracts response delta, determines finish reason, and calculates usage
        statistics from the raw chunk data.

        Args:
            agent: Agent providing model info and cost settings.
            chunk: Raw response chunk from the model API.

        Returns:
            Structured completion response with metadata.

        Raises:
            TypeError: If chunk format is invalid.
        """
        choice = chunk.choices[0]
        if not isinstance(choice, StreamingChoices):
            raise TypeError("Expected a StreamingChoices instance.")

        delta = Delta.from_delta(choice.delta)
        finish_reason = choice.finish_reason
        usage: Usage | None = safe_get_attr(chunk, "usage", Usage)
        response_cost: ResponseCost | None = None

        if usage is not None and self.include_cost:
            response_cost = calculate_response_cost(
                model=agent.llm.model,
                usage=usage,
            )

        return CompletionResponseChunk(
            id=chunk.id,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            response_cost=response_cost,
        )

    async def _handle_response_continuation(
        self,
        agent: Agent,
        continuation_count: int,
        accumulated_content: str,
        context_variables: ContextVariables,
    ) -> CustomStreamWrapper:
        """Handle response continuation with proper limits.

        Creates a new completion stream that continues the previous response
        while enforcing maximum continuation limits.

        Args:
            agent: Agent for continuation.
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
        """Determine if response content requires parsing.

        Checks if the model supports structured output and if a response format
        is specified. Only enables parsing when both conditions are met.

        Args:
            model: Model identifier to check for format support.
            custom_llm_provider: Optional custom provider to check.
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

    @returnable
    async def _stream_agent_response(
        self,
        agent: Agent,
        messages: list[Message],
        context_variables: ContextVariables,
    ) -> AsyncStream[SwarmEvent, AgentResponse]:
        """Stream agent response and process completion events.

        Streams raw completion chunks and agent response chunks from the language model.
        Accumulates content and tool calls incrementally, with optional response parsing
        based on the agent response format configuration.

        Args:
            agent: Agent whose response is being streamed.
            messages: Messages providing conversation context.
            context_variables: Context for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding completion and response events,
            returning the final accumulated agent response.

        Raises:
            CompletionError: If completion fails after retries.
            ContextLengthError: If context exceeds limits after reduction.
        """
        snapshot_content: str | None = None
        snapshot_tool_calls: list[ChatCompletionDeltaToolCall] = []

        should_parse_content = self._should_parse_agent_response(
            model=agent.llm.model,
            response_format=agent.llm.response_format,
        )

        create_completion_stream = retry_wrapper(
            self._get_completion_response,
            max_retries=self.max_retries,
            initial_delay=self.initial_retry_delay,
            max_delay=self.max_retry_delay,
            backoff_factor=self.backoff_factor,
        )

        completion_stream = create_completion_stream(
            agent=agent,
            messages=messages,
            context_variables=context_variables,
        )

        async for completion_chunk in completion_stream:
            yield YieldItem(CompletionResponseChunkEvent(response_chunk=completion_chunk))

            delta = completion_chunk.delta
            if delta.content:
                if snapshot_content is None:
                    snapshot_content = delta.content
                else:
                    snapshot_content += delta.content

            chunk_parsed: object | None = None
            if should_parse_content and snapshot_content:
                chunk_parsed = parse_response(snapshot_content)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.id:
                        snapshot_tool_calls.append(tool_call)
                    elif snapshot_tool_calls:
                        last_tool_call = snapshot_tool_calls[-1]
                        last_tool_call.function.arguments += tool_call.function.arguments

            response_chunk = AgentResponseChunk(
                completion=completion_chunk,
                content=snapshot_content,
                parsed=chunk_parsed,
                tool_calls=snapshot_tool_calls,
            )

            yield YieldItem(AgentResponseChunkEvent(agent=agent, response_chunk=response_chunk))

        snapshot_parsed: BaseModel | None = None
        if should_parse_content and snapshot_content:
            response_format: type[BaseModel] | None = None
            if is_subtype(agent.llm.response_format, BaseModel):
                response_format = agent.llm.response_format

            snapshot_parsed = parse_response(
                snapshot_content,
                response_format=response_format,
            )

        agent_response = AgentResponse(
            id=completion_chunk.id,
            role=completion_chunk.delta.role,
            finish_reason=completion_chunk.finish_reason,
            content=snapshot_content,
            parsed=snapshot_parsed,
            tool_calls=snapshot_tool_calls,
            usage=completion_chunk.usage,
            response_cost=completion_chunk.response_cost,
        )

        yield ReturnItem(agent_response)

    @returnable
    async def _process_agent_response(
        self,
        agent: Agent,
        content: str | None,
        context_variables: ContextVariables,
        tool_calls: list[ChatCompletionDeltaToolCall] | None = None,
    ) -> AsyncStream[SwarmEvent, list[Message]]:
        """Process agent response and stream tool call events.

        Creates messages for the response and processes any tool calls in order.
        The method handles agent switching through tool results and updates context
        variables based on tool execution results. All generated messages are collected
        and returned for conversation history.

        Args:
            agent: Agent that generated the response.
            content: Text content of response, may be None.
            context_variables: Context for tool execution.
            tool_calls: Tool calls to process in order.

        Returns:
            ReturnableAsyncGenerator yielding tool call events and returning
            the list of generated messages.
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

                if tool_call_result.context_update is not None:
                    if tool_call_result.context_update.type == "agent":
                        agent.state = AgentState.STALE
                        agent_context = tool_call_result.context_update.agent_context
                        self._pending_agent_contexts.append(agent_context)
                    elif tool_call_result.context_update.type == "context_variables":
                        self._context_variables = tool_call_result.context_update.context_variables

                messages.append(tool_call_result.tool_message)

        yield ReturnItem(messages)

    # ================================================
    # MARK: Agent Management
    # ================================================

    def _create_agent_messages(
        self,
        agent: Agent,
        messages: list[Message],
        instructions: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        """Create ordered list of messages for agent execution.

        Combines a system message containing agent instructions with the
        filtered conversation history. If instructions are not provided,
        they will be resolved from the agent's template.

        Args:
            agent: Agent requiring message preparation.
            messages: Conversation history to include.
            instructions: Optional pre-resolved instructions.
            context_variables: Optional variables for resolving instructions.

        Returns:
            List of messages with system message first, followed by conversation history.
        """
        if instructions is None:
            instructions = resolve_agent_instructions(agent, context_variables)

        system_message = Message(role="system", content=instructions)
        agent_messages = [message for message in messages if message.role != "system"]
        return [system_message, *agent_messages]

    def _activate_agent(self, agent: Agent) -> None:
        """Activate an agent for execution.

        Updates the swarm's state by setting the provided agent as active
        and changing its state to ACTIVE. This is a required step before
        any agent execution can begin.

        Args:
            agent: Agent to activate.
        """
        self._active_agent = agent
        self._active_agent.state = AgentState.ACTIVE

    @returnable
    async def _stream_agent_execution(
        self,
        iteration_count: int = 0,
        agent_switch_count: int = 0,
    ) -> AsyncStream[SwarmEvent, AgentExecutionResult]:
        """Stream events from active agent and handle agent switching.

        Core execution loop of the swarm. Manages agent lifecycle, processes responses,
        and handles agent switching when needed. Streams events for agent responses,
        tool calls, and switches.

        Args:
            iteration_count: Current number of processing iterations.
            agent_switch_count: Number of agent switches performed.

        Returns:
            ReturnableAsyncGenerator yielding events and returning execution result.

        Raises:
            SwarmError: If execution encounters internal state inconsistencies.
            MaxAgentSwitchesError: If maximum number of switches is exceeded.
        """
        if not self._active_agent:
            raise SwarmError("No active agent available to process messages")

        agent_response: AgentResponse | None = None
        agent_responses: list[AgentResponse] = []
        current_agent_messages: list[Message] = []
        current_execution_messages: list[Message] = []
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

                if not self._pending_agent_contexts:
                    log_verbose("No more pending agents, stopping execution")
                    break

                prev_agent = self._active_agent
                next_agent_context = self._pending_agent_contexts.popleft()
                next_agent = next_agent_context.agent

                if next_agent_context.messages is not None:
                    self._conversation_messages = next_agent_context.messages
                if next_agent_context.context_variables is not None:
                    self._context_variables = next_agent_context.context_variables

                agent_switch_count += 1
                current_agent_messages.clear()
                switch_history.append(next_agent.id)

                log_verbose(f"Switching from agent {prev_agent.id} to {next_agent.id}")
                self._activate_agent(next_agent)

                yield YieldItem(AgentActivateEvent(agent=self._active_agent))
                yield YieldItem(AgentSwitchEvent(prev_agent=prev_agent, next_agent=next_agent))

            agent_instructions = resolve_agent_instructions(
                agent=self._active_agent,
                context_variables=self._context_variables,
            )

            agent_context_messages = self._create_agent_messages(
                agent=self._active_agent,
                messages=self._conversation_messages,
                instructions=agent_instructions,
                context_variables=self._context_variables,
            )

            agent_response_stream = self._stream_agent_response(
                agent=self._active_agent,
                messages=agent_context_messages,
                context_variables=self._context_variables,
            )

            yield YieldItem(
                AgentStartEvent(
                    agent=self._active_agent,
                    agent_instructions=agent_instructions,
                    messages=agent_context_messages,
                )
            )

            async for event in agent_response_stream:
                yield YieldItem(event)

            agent_response = await agent_response_stream.get_return_value()
            agent_responses.append(agent_response)

            yield YieldItem(
                AgentResponseEvent(
                    agent=self._active_agent,
                    response=agent_response,
                )
            )

            if agent_response.content or agent_response.tool_calls:
                new_messages_stream = self._process_agent_response(
                    agent=self._active_agent,
                    content=agent_response.content,
                    context_variables=self._context_variables,
                    tool_calls=agent_response.tool_calls,
                )

                async for event in new_messages_stream:
                    yield YieldItem(event)

                new_messages = await new_messages_stream.get_return_value()
            else:
                # We might not want to do this, but it's a good fallback
                # Please consider removing this if it leads to unexpected behavior
                new_messages = [Message(role="assistant", content="<empty>")]
                log_verbose(
                    "Empty response received, appending placeholder message",
                    level="WARNING",
                )

            current_agent_messages.extend(new_messages)
            current_execution_messages.extend(new_messages)
            self._conversation_messages.extend(new_messages)

            # If agent response contains tool calls, we continue processing
            # because most llms will expect an assistant response after a tool message
            if not agent_response.tool_calls:
                self._active_agent.state = AgentState.STALE

            # Response processing can also make the agent stale,
            # so we need to check if the agent is stale at the end of the loop
            if self._active_agent.state == AgentState.STALE:
                yield YieldItem(
                    AgentCompleteEvent(
                        agent=self._active_agent,
                        agent_instructions=agent_instructions,
                        response=agent_response,
                        messages=current_agent_messages,
                    )
                )

        if agent_response is None:
            raise SwarmError("No agent response received")

        result = AgentExecutionResult(
            agent=self._active_agent,
            agent_response=agent_response,
            agent_responses=agent_responses,
            new_messages=current_execution_messages,
            all_messages=self._conversation_messages,
            context_variables=self._context_variables,
        )

        yield ReturnItem(result)

    @returnable
    async def _create_swarm_event_stream(
        self,
        agent: Agent,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEvent, AgentExecutionResult]:
        """Create the base event stream for swarm execution.

        Core implementation of swarm's event streaming. Initializes conversation,
        processes agent responses, handles tool calls, and manages agent switches.

        Args:
            agent: Initial agent for handling conversations.
            messages: List of conversation messages for context.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning execution result.

        Raises:
            SwarmError: If execution encounters internal state inconsistencies.
            ContextLengthError: If context becomes too large.
            MaxAgentSwitchesError: If too many switches occur.
            MaxResponseContinuationsError: If response needs too many continuations.
        """
        if not messages:
            raise SwarmError("Messages list is empty")

        yield YieldItem(
            AgentExecutionStartEvent(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
            )
        )

        self._context_variables = context_variables or ContextVariables()
        self._activate_agent(agent)
        yield YieldItem(AgentActivateEvent(agent=agent))

        self._conversation_messages = self._create_agent_messages(
            agent=agent,
            messages=messages,
            context_variables=self._context_variables,
        )

        agent_execution_stream = self._stream_agent_execution()
        async for event in agent_execution_stream:
            yield YieldItem(event)

        result = await agent_execution_stream.get_return_value()
        yield YieldItem(
            AgentExecutionCompleteEvent(
                agent=self._active_agent,
                execution_result=result,
            )
        )

        yield ReturnItem(result)

    # ================================================
    # MARK: Public Interface
    # ================================================

    def stream(
        self,
        agent: Agent,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
        response_format: type[ResponseFormatPydantic] | None = None,
    ) -> SwarmStream[ResponseFormatPydantic]:
        """Start agent execution with provided context and stream execution events.

        Main entry point for swarm execution. Starts agent execution with the provided
        messages and streams various events (responses, tool calls, errors) during
        execution. Accumulates content and conversation history to produce a final result.

        Args:
            agent: Agent that will process the messages.
            messages: List of conversation messages for context.
            context_variables: Optional variables for dynamic instruction resolution.
            response_format: Optional type to parse the response into.

        Returns:
            SwarmStream yielding events and returning final result.

        Raises:
            SwarmError: If execution encounters internal state inconsistencies.
            ContextLengthError: If context becomes too large to process.
            MaxAgentSwitchesError: If too many agent switches occur.
            MaxResponseContinuationsError: If response requires too many continuations.
            TypeError: If the response format does not match the parsed response.

        Examples:
            Stream events and get result:
                ```python
                from liteswarm.types.swarm import Agent, Message
                from liteswarm.types.llm import LLM


                def add(a: int, b: int) -> int:
                    return a + b


                def multiply(a: int, b: int) -> int:
                    return a * b


                # Create agent with tools
                agent = Agent(
                    id="math",
                    instructions="You are a math assistant. Use tools for calculations.",
                    llm=LLM(
                        model="gpt-4o",
                        tools=[add, multiply],
                        tool_choice="auto",
                    ),
                )

                messages = [Message(role="user", content="Calculate (2 + 3) * 4")]
                stream = swarm.stream(agent, messages=messages)

                # Process events during execution
                async for event in stream:
                    if event.type == "agent_response_chunk":
                        print(event.chunk.completion.delta.content)
                    elif event.type == "tool_call_result":
                        print(f"Tool result: {event.tool_call_result.result}")

                # Get final result after completion
                result = await stream.get_return_value()
                print(f"Final result: {result.content}")  # "The result is 20"
                ```

            Just get final result:
                ```python
                messages = [Message(role="user", content="Calculate 2 + 2")]
                stream = swarm.stream(agent, messages=messages)
                result = await stream.get_return_value()
                print(result.content)  # "The result is 4"
                ```

        Notes:
            - Process events in real-time or await the final result
            - Events provide visibility into execution progress
            - Final result includes complete conversation history
        """
        event_stream = self._create_swarm_event_stream(
            agent=agent,
            messages=messages,
            context_variables=context_variables,
        )

        return SwarmStream(event_stream, response_format)

    async def execute(
        self,
        agent: Agent,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
        response_format: type[ResponseFormatPydantic] | None = None,
    ) -> AgentExecutionResult[ResponseFormatPydantic]:
        """Start agent execution and return final execution result.

        Convenience method that wraps stream() to provide direct result collection
        without event handling. Starts agent execution with the provided messages
        and returns the final result when execution is complete.

        Args:
            agent: Agent that will process the messages.
            messages: List of conversation messages for context.
            context_variables: Optional variables for dynamic instruction resolution.
            response_format: Optional type to parse the response into.

        Returns:
            Complete execution result with final content and metadata.

        Raises:
            SwarmError: If execution encounters internal state inconsistencies.
            ContextLengthError: If context becomes too large to process.
            MaxAgentSwitchesError: If too many agent switches occur.
            MaxResponseContinuationsError: If response requires too many continuations.
            TypeError: If the response format does not match the parsed response.

        Examples:
            Basic usage:
                ```python
                from liteswarm.types.swarm import Agent, Message
                from liteswarm.types.llm import LLM

                agent = Agent(
                    id="assistant",
                    instructions="You are a helpful assistant.",
                    llm=LLM(model="gpt-4o"),
                )

                messages = [Message(role="user", content="What is 2 + 2?")]
                result = await swarm.execute(agent, messages=messages)
                print(result.content)  # "2 + 2 equals 4"
                ```

            With context variables:
                ```python
                from liteswarm.types.swarm import ContextVariables


                def build_agent_prompt(context_variables: ContextVariables) -> str:
                    return f"You're a helpful assistant for {context_variables.user_name}."


                agent = Agent(
                    id="assistant",
                    instructions=build_agent_prompt,
                    llm=LLM(model="gpt-4o"),
                )

                messages = [Message(role="user", content="What's my name?")]
                context = ContextVariables(user_name="Alice")
                result = await swarm.execute(
                    agent=agent,
                    messages=messages,
                    context_variables=context,
                )
                print(result.content)  # "Your name is Alice"
                ```

        Notes:
            - Simplified interface for getting just the final result
            - Internally uses stream() but handles events for you
            - Maintains the same error handling as stream()
        """
        stream = self.stream(
            agent=agent,
            messages=messages,
            context_variables=context_variables,
            response_format=response_format,
        )

        return await stream.get_return_value()
