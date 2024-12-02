# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from enum import Enum
from typing import Literal, TypeAlias

from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    FunctionCall,
    Usage,
)
from litellm.types.utils import Delta as LiteDelta
from pydantic import BaseModel, ConfigDict, Field

from liteswarm.types.context import ContextVariables
from liteswarm.types.llm import LLM

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
                content="You are a helpful assistant."
            )

            # User input
            user_msg = Message(
                role="user",
                content="Calculate 2 + 2"
            )

            # Assistant response with tool
            assistant_msg = Message(
                role="assistant",
                content="Let me calculate that.",
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2, "b": 2}'}
                    )
                ]
            )

            # Tool result
            tool_msg = Message(
                role="tool",
                content="4",
                tool_call_id="calc_1"
            )
            ```
    """

    role: Literal["assistant", "user", "system", "tool"]
    """Role of the message sender."""

    content: str | None = None
    """Text content of the message."""

    tool_calls: list[ChatCompletionDeltaToolCall] | None = None
    """Tool calls made in this message."""

    tool_call_id: str | None = None
    """ID of the tool call this message responds to."""

    audio: ChatCompletionAudioResponse | None = None
    """Audio response data if available."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="allow",
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
                    tool_call_id="calc_1"
                )
            )
            ```

        Agent switch with context:
            ```python
            msg = ToolMessage(
                message=Message(
                    role="tool",
                    content="Switching to expert",
                    tool_call_id="switch_1"
                ),
                agent=Agent(
                    id="math-expert",
                    instructions="You are a math expert.",
                    llm=LLM(model="gpt-4o")
                ),
                context_variables=ContextVariables(
                    specialty="mathematics",
                    confidence=0.9
                )
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
    )


class Delta(BaseModel):
    """Partial update in a streaming response.

    Represents incremental updates during streaming, including content
    chunks, role changes, and tool calls.

    Examples:
        Different types of updates:
            ```python
            # Content chunk
            content_delta = Delta(
                role="assistant",
                content="Hello, "
            )

            # Tool call start
            tool_delta = Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2'}
                    )
                ]
            )

            # Tool call completion
            completion_delta = Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": ', "b": 2}'}
                    )
                ]
            )
            ```
    """

    content: str | None = None
    """Text content in this update."""

    role: str | None = None
    """Role of the message being updated."""

    function_call: FunctionCall | dict | None = None
    """Function call information."""

    tool_calls: list[ChatCompletionDeltaToolCall | dict] | None = None
    """Tool calls being made."""

    audio: ChatCompletionAudioResponse | None = None
    """Audio response data."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
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
                completion_tokens_cost=0.002  # $0.002 for output
            )
            total = cost.prompt_tokens_cost + cost.completion_tokens_cost
            ```
    """

    prompt_tokens_cost: float
    """Cost of tokens in the prompt."""

    completion_tokens_cost: float
    """Cost of tokens in the completion."""


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
                llm=LLM(model="gpt-4o")
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
    )


class ToolCallResult(BaseModel):
    """Base class for tool call results.

    Provides common structure for all tool execution results.
    """

    tool_call: ChatCompletionDeltaToolCall
    """Tool call that produced this result."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ToolCallMessageResult(ToolCallResult):
    """Result of a tool call that produced a message.

    Used for tools that return data or text responses, optionally
    with context updates.

    Examples:
        Tool response with context:
            ```python
            result = ToolCallMessageResult(
                tool_call=calc_call,
                message=Message(
                    role="tool",
                    content="42",
                    tool_call_id="calc_1"
                ),
                context_variables=ContextVariables(
                    last_result=42,
                    calculation_type="simple"
                )
            )
            ```
    """

    message: Message
    """Tool's response message."""

    context_variables: ContextVariables | None = None
    """Optional context updates."""


class ToolCallAgentResult(ToolCallResult):
    """Result of a tool call that produced a new agent.

    Used for agent-switching tools that return a new agent with
    optional transition message and context.

    Examples:
        Switch to expert agent:
            ```python
            result = ToolCallAgentResult(
                tool_call=switch_call,
                agent=Agent(
                    id="math-expert",
                    instructions="You are a math expert.",
                    llm=LLM(model="gpt-4o")
                ),
                message=Message(
                    role="tool",
                    content="Switching to math expert",
                    tool_call_id="switch_1"
                ),
                context_variables=ContextVariables(
                    expertise="mathematics",
                    difficulty="advanced"
                )
            )
            ```
    """

    agent: Agent
    """New agent to switch to."""

    message: Message | None = None
    """Optional transition message."""

    context_variables: ContextVariables | None = None
    """Optional context updates."""


class ToolCallFailureResult(ToolCallResult):
    """Result of a failed tool call.

    Captures errors during tool execution for proper handling.

    Examples:
        Handle tool failure:
            ```python
            result = ToolCallFailureResult(
                tool_call=failed_call,
                error=ValueError("Invalid input: negative number")
            )
            ```
    """

    error: Exception
    """Error that occurred during execution."""


class CompletionResponse(BaseModel):
    """Response chunk from the language model.

    Represents a single chunk in a streaming response with
    content and metadata.

    Examples:
        Process response chunk:
            ```python
            response = CompletionResponse(
                delta=Delta(content="Hello"),
                finish_reason=None,
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=1,
                    total_tokens=11
                ),
                response_cost=ResponseCost(
                    prompt_tokens_cost=0.0001,
                    completion_tokens_cost=0.0002
                )
            )
            ```
    """

    delta: Delta
    """Content update in this chunk."""

    finish_reason: str | None = None
    """Reason for response completion."""

    usage: Usage | None = None
    """Token usage statistics."""

    response_cost: ResponseCost | None = None
    """Cost information."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class AgentResponse(BaseModel):
    """Processed response from an agent with accumulated state.

    Combines current update with accumulated content and tool calls
    from previous updates.

    Examples:
        Track response progress:
            ```python
            response = AgentResponse(
                delta=Delta(content=" world"),
                content="Hello world",  # Accumulated
                tool_calls=[calc_call],  # Accumulated
                finish_reason=None,
                usage=Usage(prompt_tokens=10, completion_tokens=2),
                response_cost=ResponseCost(
                    prompt_tokens_cost=0.0001,
                    completion_tokens_cost=0.0002
                )
            )
            ```
    """

    delta: Delta
    """Current content update."""

    finish_reason: str | None = None
    """Reason for response completion."""

    content: str | None = None
    """Accumulated content so far."""

    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)
    """Accumulated tool calls."""

    usage: Usage | None = None
    """Token usage statistics."""

    response_cost: ResponseCost | None = None
    """Cost information."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ConversationState(BaseModel):
    """Complete state of a conversation.

    Captures all aspects of a conversation including content,
    agents, history, and metrics.

    Examples:
        Track conversation state:
            ```python
            state = ConversationState(
                content="Final response",
                agent=current_agent,
                agent_messages=[  # Current context
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi")
                ],
                agent_queue=[backup_agent],  # Queued agents
                messages=[],  # Full history
                usage=Usage(total_tokens=100),
                response_cost=ResponseCost(
                    prompt_tokens_cost=0.001,
                    completion_tokens_cost=0.002
                )
            )
            ```
    """

    content: str | None = None
    """Final conversation content."""

    agent: Agent | None = None
    """Currently active agent."""

    agent_messages: list[Message] = Field(default_factory=list)
    """Current agent's message context."""

    agent_queue: list[Agent] = Field(default_factory=list)
    """Queue of pending agents."""

    messages: list[Message] = Field(default_factory=list)
    """Complete conversation history."""

    usage: Usage | None = None
    """Total token usage."""

    response_cost: ResponseCost | None = None
    """Total cost information."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
