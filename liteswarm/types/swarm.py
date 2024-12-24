# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TypeAlias

from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    FunctionCall,
    Usage,
)
from litellm.types.utils import Delta as LiteDelta
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from liteswarm.types.context import ContextVariables
from liteswarm.types.llm import LLM
from liteswarm.types.misc import JSON

AgentInstructions: TypeAlias = str | Callable[[ContextVariables], str]
"""Instructions for defining agent behavior.

Can be either a static string or a function that generates instructions
dynamically based on context.

Examples:
    Static instructions:
        ```python
        instructions: AgentInstructions = '''
            You are a helpful assistant.
            Follow these guidelines:
            1. Be concise and clear.
            2. Ask for clarification when needed.
            '''
        ```

    Dynamic instructions:
        ```python
        def generate_instructions(context: ContextVariables) -> str:
            return f'''
                You are helping {context.get('user_name')}.
                Your expertise is in {context.get('domain')}.
                Use {context.get('preferred_language')} when possible.
            '''
        ```
"""

FinishReason: TypeAlias = Literal["stop", "length", "tool_calls", "function_call", "content_filter"]
"""OpenAI supported finish reasons for streaming responses.

Indicates why the model stopped generating tokens during a streaming response.

Supported finish reasons:
- stop: Model reached a natural stopping point or hit a provided stop sequence.
- length: Maximum allowed tokens were reached.
- tool_calls: Model called a tool.
- function_call: Deprecated equivalent of tool_calls.
- content_filter: Content was omitted due to safety or moderation filters.
"""

MessageRole: TypeAlias = Literal["assistant", "user", "system", "tool", "function", "developer"]
"""Role of the message author in a conversation.

Supported roles:
- assistant: Model-generated responses to user messages.
- user: End-user messages containing prompts or context.
- system: System-level instructions that guide model behavior.
- tool: Results from tool/function executions.
- function: Legacy alias for tool responses [deprecated].
- developer: Developer-provided instructions for reasoning models.
"""

ToolCall: TypeAlias = ChatCompletionDeltaToolCall
"""Tool call in a streaming response from the model.

Represents a function or tool invocation request with:
- id: Unique identifier for tracking the call.
- function: Function name and arguments to execute.
- type: Optional tool type identifier (e.g., 'function').
- index: Position in the sequence of tool calls.

Used in both complete messages and streaming deltas.
"""

AudioResponse: TypeAlias = ChatCompletionAudioResponse
"""Audio response generated by the model.

Contains audio data and metadata:
- id: Unique identifier for the response.
- data: Base64 encoded audio bytes in requested format.
- expires_at: Unix timestamp when audio will be purged.
- transcript: Text transcription of the audio content.

Used in conversations with audio capabilities.
"""

DYNAMIC_INSTRUCTIONS = "<dynamic_instructions>"
"""Constant indicating that agent instructions are dynamic (callable)."""


class Message(BaseModel):
    """Message in a conversation between users, assistants, and tools.

    Represents all types of communication including system instructions,
    user inputs, assistant responses, and tool results.

    Examples:
        Create different message types:
            ```python
            # System instructions
            system_msg = Message(
                role="system",
                content="You are a helpful assistant.",
            )

            # User input
            user_msg = Message(
                role="user",
                content="Calculate 2 + 2",
            )

            # Assistant response with tool
            assistant_msg = Message(
                role="assistant",
                content="Let me calculate that.",
                tool_calls=[
                    ToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2, "b": 2}'},
                    )
                ],
            )

            # Tool result
            tool_msg = Message(
                role="tool",
                content="4",
                tool_call_id="calc_1",
            )
            ```
    """

    role: MessageRole
    """Role of the message author."""

    content: str | None = None
    """Text content of the message."""

    tool_calls: list[ToolCall] | None = None
    """Tool calls made in this message."""

    tool_call_id: str | None = None
    """ID of the tool call this message responds to."""

    audio: AudioResponse | None = None
    """Audio response data if available."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ToolMessage(BaseModel):
    """Message resulting from a tool call.

    Contains the tool's response and optionally includes a new agent
    or context updates.

    Examples:
        Simple tool response:
            ```python
            msg = ToolMessage(
                message=Message(
                    role="tool",
                    content="4",
                    tool_call_id="calc_1",
                )
            )
            ```

        Agent switch with context:
            ```python
            msg = ToolMessage(
                message=Message(
                    role="tool",
                    content="Switching to expert",
                    tool_call_id="switch_1",
                ),
                agent=Agent(
                    id="math-expert",
                    instructions="You are a math expert.",
                    llm=LLM(model="gpt-4o"),
                ),
                context_variables=ContextVariables(
                    specialty="mathematics",
                    confidence=0.9,
                ),
            )
            ```
    """

    message: Message
    """Tool's response message."""

    agent: "Agent | None" = None
    """Optional new agent for switching."""

    context_variables: ContextVariables | None = None
    """Optional context updates."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class Delta(BaseModel):
    """Partial update in a streaming response.

    Represents incremental updates during streaming, including content
    chunks, role changes, and tool calls.

    Examples:
        Different types of updates:
            ```python
            # Content chunk
            content_delta = Delta(role="assistant", content="Hello, ")

            # Tool call start
            tool_delta = Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2'},
                    )
                ]
            )

            # Tool call completion
            completion_delta = Delta(
                tool_calls=[
                    ToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": ', "b": 2}'},
                    )
                ]
            )
            ```
    """

    content: str | None = None
    """Text content in this update."""

    role: MessageRole | None = None
    """Role of the message author."""

    function_call: FunctionCall | dict[str, Any] | None = None
    """Function call information (deprecated)."""

    tool_calls: list[ToolCall | dict[str, Any]] | None = None
    """Tool calls being made."""

    audio: AudioResponse | None = None
    """Audio response data."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @classmethod
    def from_delta(cls, delta: LiteDelta) -> "Delta":
        """Create a Delta from a LiteLLM delta object.

        Args:
            delta: LiteLLM delta to convert.

        Returns:
            New Delta instance with copied attributes.
        """
        return cls(
            content=delta.content,
            role=delta.role,
            function_call=delta.function_call,
            tool_calls=delta.tool_calls,
            audio=delta.audio,
        )


class ResponseCost(BaseModel):
    """Cost information for a model response.

    Tracks token costs for both input prompts and model completions.

    Examples:
        Calculate total cost:
            ```python
            cost = ResponseCost(
                prompt_tokens_cost=0.001,  # $0.001 for input
                completion_tokens_cost=0.002,  # $0.002 for output
            )
            total = cost.prompt_tokens_cost + cost.completion_tokens_cost
            ```
    """

    prompt_tokens_cost: float
    """Cost of tokens in the prompt."""

    completion_tokens_cost: float
    """Cost of tokens in the completion."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentState(str, Enum):
    """State of an agent in the conversation lifecycle.

    Tracks whether an agent is ready for tasks, actively processing,
    or needs replacement.
    """

    IDLE = "idle"
    """Agent is ready to handle new tasks."""

    ACTIVE = "active"
    """Agent is currently processing a task."""

    STALE = "stale"
    """Agent needs to be replaced or refreshed."""


class Agent(BaseModel):
    """AI agent that participates in conversations and uses tools.

    Represents an AI participant with specific instructions, capabilities,
    and state management.

    Examples:
        Create a specialized agent:
            ```python
            # Define tool functions
            def search_docs(query: str) -> str:
                \"\"\"Search documentation.\"\"\"
                return f"Results for: {query}"

            def generate_code(spec: str) -> str:
                \"\"\"Generate code from spec.\"\"\"
                return f"Code for: {spec}"

            # Create coding assistant
            agent = Agent(
                id="coding-assistant",
                instructions='''
                    You are a coding assistant.
                    1. Search docs for relevant info.
                    2. Generate code solutions.
                    3. Explain your changes.
                ''',
                llm=LLM(
                    model="gpt-4o",
                    tools=[search_docs, generate_code],
                    tool_choice="auto",
                    temperature=0.7
                )
            )
            ```

        Dynamic instructions:
            ```python
            def get_instructions(context: ContextVariables) -> str:
                return f'''
                    You are helping {context.get('user_name')}.
                    Expertise: {context.get('domain')}.
                    Language: {context.get('language')}.
                '''


            agent = Agent(
                id="expert",
                instructions=get_instructions,
                llm=LLM(model="gpt-4o"),
            )
            ```
    """

    id: str
    """Unique identifier for the agent."""

    instructions: AgentInstructions
    """Behavior definition (static or dynamic)."""

    llm: LLM
    """Language model configuration."""

    state: AgentState = AgentState.IDLE
    """Current agent state."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @field_serializer("instructions")
    def serialize_instructions(self, instructions: AgentInstructions) -> str:
        """Serialize agent instructions for storage or transmission.

        Static instructions are serialized as-is. Dynamic instructions (functions)
        are serialized as a special constant string.

        Returns:
            Original instructions string or DYNAMIC_INSTRUCTIONS constant.
        """
        if callable(instructions):
            return DYNAMIC_INSTRUCTIONS
        return instructions


class ToolResult(BaseModel):
    """Public API wrapper for tool execution results.

    This class is part of the public API and should be used to wrap results from
    your tool functions. It allows you to:
    - Return data from your tools
    - Switch to a different agent
    - Update context variables

    The content must be JSON serializable to be used as a tool result. For complex
    return types, consider serializing them to a JSON-compatible format.

    Examples:
        Simple tool result:
            ```python
            # In your tool function
            def calculate(a: int, b: int) -> ToolResult:
                result = a + b
                return ToolResult(content=result)
            ```

        Tool result with context updates:
            ```python
            def fetch_data(url: str) -> ToolResult:
                data = requests.get(url).json()
                return ToolResult(
                    content=data["summary"],
                    context_variables=ContextVariables(
                        full_data=data,
                        last_updated=datetime.now().isoformat(),
                    ),
                )
            ```

        Agent switching with context:
            ```python
            def analyze_complexity(code: str) -> ToolResult:
                if "machine learning" in code:
                    return ToolResult(
                        content="Switching to ML expert",
                        agent=Agent(
                            id="ml-expert",
                            instructions="You are an ML expert.",
                            llm=LLM(model="gpt-4o"),
                        ),
                        context_variables=ContextVariables(
                            domain="machine_learning",
                            complexity="high",
                        ),
                    )
                return ToolResult(content="Standard code review needed")
            ```

    Note:
        This is the recommended way to return results from your tool functions.
        Do not use internal classes like ToolCallResult and its subclasses.
    """

    content: Any
    """Content of the tool result. Must be JSON serializable."""

    agent: Agent | None = None
    """Optional agent to switch to."""

    context_variables: ContextVariables | None = None
    """Optional context updates to apply."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ToolCallResult(BaseModel):
    """Internal base class for tool call results.

    This is an internal class used by the framework to handle tool execution results.
    It should not be used directly in user code. For wrapping tool results in your
    application code, use the ToolResult class instead.

    The ToolCallResult hierarchy (ToolCallMessageResult, ToolCallAgentResult,
    ToolCallFailureResult) is designed for internal event processing and should only
    be referenced in generic event handlers if needed.

    For public API usage:
        - Use ToolResult to wrap your tool's return values
        - Use ToolResult when you need to switch agents or update context
        - See ToolResult documentation for examples

    Internal framework usage only:
        ```python
        # Framework internal processing
        class ToolEventHandler:
            async def handle_tool_call(self, result: ToolCallResult):
                match result:
                    case ToolCallMessageResult():
                        # Process message result
                        pass
                    case ToolCallAgentResult():
                        # Process agent switch
                        pass
                    case ToolCallFailureResult():
                        # Handle failure
                        pass
        ```
    """

    tool_call: ToolCall
    """Tool call that produced this result."""

    result: Any
    """Result of the tool call."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ToolCallMessageResult(ToolCallResult):
    """Internal class for tool call results that produced a message.

    This is an internal class used by the framework to handle tool message results.
    It should not be used directly in user code. For wrapping tool results in your
    application code, use the ToolResult class instead.

    Used internally for tools that return data or text responses, optionally
    with context updates.

    For public API usage:
        - Use ToolResult to wrap your tool's return values
        - See ToolResult documentation for examples

    Internal framework usage only:
        ```python
        # Framework internal processing
        async def handle_tool_message(result: ToolCallMessageResult):
            # Process the tool's message
            message = result.message
            if result.context_variables:
                # Apply context updates
                pass
        ```
    """

    message: Message
    """Tool's response message."""

    context_variables: ContextVariables | None = None
    """Optional context updates."""


class ToolCallAgentResult(ToolCallResult):
    """Internal class for tool call results that produced a new agent.

    This is an internal class used by the framework to handle agent switching.
    It should not be used directly in user code. For switching agents in your
    application code, use the ToolResult class instead.

    Used internally for agent-switching tools that return a new agent with
    optional transition message and context.

    For public API usage:
        - Use ToolResult to switch agents or update context
        - See ToolResult documentation for examples

    Internal framework usage only:
        ```python
        # Framework internal processing
        async def handle_agent_switch(result: ToolCallAgentResult):
            # Switch to new agent
            new_agent = result.agent
            if result.message:
                # Handle transition message
                pass
            if result.context_variables:
                # Apply context updates
                pass
        ```
    """

    agent: Agent
    """New agent to switch to."""

    message: Message | None = None
    """Optional transition message."""

    context_variables: ContextVariables | None = None
    """Optional context updates."""


class ToolCallFailureResult(ToolCallResult):
    """Internal class for failed tool call results.

    This is an internal class used by the framework to handle tool execution failures.
    It should not be used directly in user code. For handling tool errors in your
    application code, use standard Python exception handling instead.

    Used internally to capture errors during tool execution for proper framework-level
    handling.

    For public API usage:
        - Use try/except blocks to handle tool errors
        - See tool function documentation for specific exceptions

    Internal framework usage only:
        ```python
        # Framework internal processing
        async def handle_tool_failure(result: ToolCallFailureResult):
            # Log the error
            logger.error(f"Tool failed: {result.error}")
            # Update execution state
            execution.mark_failed(result.tool_call.id)
        ```
    """

    error: Exception
    """Error that occurred during execution."""


class CompletionResponseChunk(BaseModel):
    """Single chunk in a streaming completion response.

    Represents an incremental update from the language model including content
    delta and metadata. Each chunk contains the latest changes and running
    statistics.

    Examples:
        Process chunk:
            ```python
            chunk = CompletionResponseChunk(
                delta=Delta(content="Hello"),
                finish_reason=None,
                usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
                response_cost=ResponseCost(
                    prompt_tokens_cost=0.0001,
                    completion_tokens_cost=0.0002,
                ),
            )
            ```
    """

    delta: Delta
    """Content update in this chunk."""

    finish_reason: FinishReason | None = None
    """Reason for response generation stopping."""

    usage: Usage | None = None
    """Token usage statistics."""

    response_cost: ResponseCost | None = None
    """Cost information."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentResponseChunk(BaseModel):
    """Single chunk in a streaming agent response.

    Contains both the completion response chunk and accumulated state:
    - Completion chunk (delta, finish reason, usage, cost)
    - Accumulated content from previous chunks
    - Collected tool calls
    - Parsed content when format specified

    Each chunk represents a streaming step with:
    - Latest completion response
    - Progressive content accumulation
    - Real-time parsing when applicable

    Examples:
        Content update:
            ```python
            chunk = AgentResponseChunk(
                agent=agent,
                completion=CompletionResponseChunk(
                    delta=Delta(content=" world"),
                    usage=Usage(prompt_tokens=10, completion_tokens=2),
                ),
                content="Hello world",  # Full content so far
                parsed_content=None,
            )
            print(chunk.completion.delta.content)  # Latest: " world"
            print(chunk.content)  # Total: "Hello world"
            ```

        Tool call update:
            ```python
            chunk = AgentResponseChunk(
                agent=agent,
                completion=CompletionResponseChunk(
                    delta=Delta(tool_calls=[new_call]),
                    usage=Usage(prompt_tokens=15, completion_tokens=5),
                    response_cost=ResponseCost(
                        prompt_tokens_cost=0.0001,
                        completion_tokens_cost=0.0002,
                    ),
                ),
                content="Let me calculate that",
                parsed_content=None,
                tool_calls=[previous_call, new_call],  # All calls so far
            )
            ```

    Notes:
        - Completion contains the latest update with delta and metadata
        - Content and tool_calls maintain running state
        - Usage and costs are tracked in completion
        - Parsed content updates with format rules
    """

    agent: Agent
    """Agent that produced this chunk."""

    completion: CompletionResponseChunk
    """Completion response chunk."""

    content: str | None = None
    """Accumulated content so far."""

    parsed_content: JSON | BaseModel | None = None
    """Parsed content if response format was specified."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    """Accumulated tool calls."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentExecutionResult(BaseModel):
    """Final result of an agent's execution.

    Contains the complete execution outcome including:
    - Final content and parsed results
    - Active agent information
    - Usage statistics and costs
    - Parsed content if format was specified

    This class represents the final state after:
    - All content has been generated
    - All tool calls have completed
    - All statistics have been accumulated
    - All costs have been calculated

    Examples:
        Basic result:
            ```python
            result = AgentExecutionResult(
                agent=agent,
                content="Final response text",
                usage=Usage(total_tokens=100),
            )
            print(result.content)
            ```

        Result with parsed content:
            ```python
            result = AgentExecutionResult(
                agent=agent,
                content='{"answer": 42}',
                parsed_content={"answer": 42},
                usage=Usage(total_tokens=50),
                response_cost=ResponseCost(
                    prompt_tokens_cost=0.001,
                    completion_tokens_cost=0.002,
                ),
            )
            print(result.parsed_content["answer"])  # 42
            ```

    Notes:
        - Content may be None if no text was generated
        - Parsed content is available if response format was specified
        - Usage and cost are optional but typically present
        - Agent reference is always available
    """

    agent: Agent
    """Currently active agent."""

    content: str | None = None
    """Final conversation content."""

    parsed_content: JSON | BaseModel | None = None
    """Parsed conversation content if response format was specified."""

    usage: Usage | None = None
    """Total token usage."""

    response_cost: ResponseCost | None = None
    """Total cost information."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )
