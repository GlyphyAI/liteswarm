# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from litellm.types.utils import Delta as LiteDelta
from litellm.types.utils import FunctionCall, Usage
from pydantic import BaseModel, ConfigDict, Discriminator, field_serializer

from liteswarm.types.context import ContextVariables
from liteswarm.types.llm import LLM, AudioResponse, FinishReason, MessageRole, ToolCall
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

DYNAMIC_INSTRUCTIONS = "<dynamic_instructions>"
"""Constant indicating that agent instructions are dynamic (callable)."""


class Message(BaseModel):
    """Message in a conversation between user, assistant, and tools.

    Represents a message in a conversation between participants
    with content and optional tool interactions. Each message has
    a specific role and may include tool calls or responses.

    Examples:
        System message:
            ```python
            system_msg = Message(
                role="system",
                content="You are a helpful assistant.",
            )
            ```

        Assistant with tool:
            ```python
            assistant_msg = Message(
                role="assistant",
                content="Let me calculate that.",
                tool_calls=[
                    ToolCall(
                        id="calc_1",
                        function={"name": "add", "arguments": '{"a": 2, "b": 2}'},
                        type="function",
                        index=0,
                    )
                ],
            )
            ```

        Tool response:
            ```python
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


class AgentState(str, Enum):
    """State of an agent in the conversation lifecycle.

    Tracks an agent's readiness and activity status during execution.
    State transitions occur automatically during processing.
    """

    IDLE = "idle"
    """Agent is ready for new tasks."""

    ACTIVE = "active"
    """Agent is processing a task."""

    STALE = "stale"
    """Agent needs replacement."""


class Agent(BaseModel):
    """Configuration for an AI conversation participant.

    Defines an agent's identity, behavior, and capabilities through
    instructions and language model settings. Instructions can be
    static text or dynamically generated based on context variables.

    Examples:
        Basic agent:
            ```python
            agent = Agent(
                id="assistant",
                instructions="You are a helpful assistant.",
                llm=LLM(model="gpt-4o"),
            )
            ```

        Tool-enabled agent:
            ```python
            def search_docs(query: str) -> ToolResult:
                return ToolResult(content="Search results for " + query)


            def generate_code(spec: str) -> ToolResult:
                return ToolResult(content="Generated code for " + spec)


            agent = Agent(
                id="coder",
                instructions="You are a coding assistant.",
                llm=LLM(
                    model="gpt-4o",
                    tools=[search_docs, generate_code],
                    tool_choice="auto",
                ),
            )
            ```

        Dynamic instructions:
            ```python
            def get_instructions(context: ContextVariables) -> str:
                return f"Help {context.user_name} with {context.task}."


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
    """Behavior definition (static text or dynamic function)."""

    llm: LLM
    """Language model and tool configuration."""

    state: AgentState = AgentState.IDLE
    """Current execution state."""

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


class AgentContext(BaseModel):
    """Internal configuration for agent execution control.

    Used during agent switching to optionally override the execution
    context of the new agent. This allows resetting conversation
    history or injecting specific variables for the new agent.

    When switching agents:
    - If messages is provided, replaces conversation history
    - If context_variables provided, resets variables
    - If either is None, maintains current context
    """

    agent: Agent
    """Agent configuration and state."""

    messages: list[Message] | None = None
    """Optional message history to reset with."""

    context_variables: ContextVariables | None = None
    """Optional variables to reset with."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentContextUpdate(BaseModel):
    """Context update for agent switching.

    Indicates that execution should transition to a new agent with
    optional message history and context variables.
    """

    type: Literal["agent"] = "agent"
    """Type of context update."""

    agent_context: AgentContext
    """New agent context to switch to."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ContextVariablesUpdate(BaseModel):
    """Context update for runtime variables.

    Updates the execution context with new variables that affect
    instruction resolution and tool behavior.
    """

    type: Literal["context_variables"] = "context_variables"
    """Type of context update."""

    context_variables: ContextVariables
    """New context variables to update with."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


ContextUpdate = Annotated[
    AgentContextUpdate | ContextVariablesUpdate,
    Discriminator("type"),
]
"""Union type for all context update variants."""


class Delta(BaseModel):
    """Streaming update from language model generation.

    Contains new content and changes since the last update during
    streaming. Updates can include text content, role changes,
    tool calls, or audio responses.

    Note:
        Any field may be empty or partially complete.
    """

    content: str | None = None
    """New text content."""

    role: MessageRole | None = None
    """Role of the message author."""

    function_call: FunctionCall | None = None
    """Function call update (deprecated)."""

    tool_calls: list[ToolCall] | None = None
    """Tool calls being made."""

    audio: AudioResponse | None = None
    """Audio response if available."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @property
    def is_empty(self) -> bool:
        """Check if the delta is empty.

        Returns:
            True if the delta is empty, False otherwise.
        """
        return all(not getattr(self, attr) for attr in self.model_fields)

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
    """

    prompt_tokens_cost: float
    """Cost of input tokens."""

    completion_tokens_cost: float
    """Cost of output tokens."""

    total_tokens_cost: float
    """Total cost of the response."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ToolResult(BaseModel):
    """Result from a tool execution.

    Wraps tool return values with optional context updates for
    agent switching or variable updates. All content must be
    JSON serializable.

    Examples:
        Basic result:
            ```python
            def get_weather(location: str) -> ToolResult:
                return ToolResult(content=f"Weather in {location}: 20°C")
            ```

        Context update:
            ```python
            def fetch_user(id: str) -> ToolResult:
                user = db.get_user(id)
                return ToolResult.update_context(
                    content=f"Found user {user.name}",
                    context_variables=ContextVariables(user=user),
                )
            ```

        Agent switch:
            ```python
            def route_request(domain: str) -> ToolResult:
                return ToolResult.switch_agent(
                    agent=get_expert(domain),
                    content=f"Switching to {domain} expert",
                )
            ```
    """

    content: Any
    """Tool execution result (must be JSON serializable)."""

    context_update: ContextUpdate | None = None
    """Optional update to runtime context."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    @classmethod
    def switch_agent(
        cls,
        agent: Agent,
        content: Any,
        messages: list[Message] | None = None,
        context_variables: ContextVariables | None = None,
    ) -> "ToolResult":
        """Create a tool result that switches to a new agent.

        Convenience method for creating a result that transitions
        execution to a different agent. Optionally resets conversation
        history and context variables for the new agent.

        Args:
            agent: Agent to switch execution to.
            content: Result content to include.
            messages: Optional new conversation history.
            context_variables: Optional new context variables.

        Returns:
            ToolResult configured for agent switching.
        """
        return cls(
            content=content,
            context_update=AgentContextUpdate(
                agent_context=AgentContext(
                    agent=agent,
                    messages=messages,
                    context_variables=context_variables,
                ),
            ),
        )

    @classmethod
    def update_context(
        cls,
        content: Any,
        context_variables: ContextVariables | None = None,
    ) -> "ToolResult":
        """Create a tool result that updates context variables.

        Convenience method for creating a result that overrides the
        runtime context variables. If provided, the new variables
        completely replace the existing context variables.

        Args:
            content: Result content to include.
            context_variables: Optional new context variables.

        Returns:
            ToolResult configured for context update.
        """
        return cls(
            content=content,
            context_update=ContextVariablesUpdate(
                context_variables=context_variables,
            ),
        )


class ToolCallResult(BaseModel):
    """Internal result of a tool call execution.

    Contains the complete execution outcome including the original
    call, return value, response message, and any errors or context
    updates.

    Note:
        This is an internal class. Use ToolResult in your code.
    """

    tool_call: ToolCall
    """Original tool call details."""

    tool_return_value: Any
    """Raw return value from the tool."""

    tool_message: Message
    """Tool's response message."""

    tool_error: Exception | None = None
    """Error that occurred during execution."""

    context_update: ContextUpdate | None = None
    """Optional update to runtime context."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class CompletionResponseChunk(BaseModel):
    """Streaming chunk from model completion.

    Contains content delta and optional usage statistics. When usage
    tracking is enabled, the final chunk will have an empty delta
    and include token usage and cost calculations.

    Notes:
        - Usage and cost are only present in the final chunk when
          tracking is enabled in Swarm (include_usage and include_cost).
        - Cost is calculated based on model-specific token pricing.
    """

    delta: Delta
    """Content and tool updates."""

    finish_reason: FinishReason | None = None
    """Reason for response generation stopping."""

    usage: Usage | None = None
    """Token usage statistics."""

    response_cost: ResponseCost | None = None
    """Cost calculation."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentResponseChunk(BaseModel):
    """Processed chunk of agent streaming response.

    Contains both raw completion data and accumulated content from
    previous chunks. Maintains running state of content, tool calls,
    and any parsed structured output.
    """

    completion: CompletionResponseChunk
    """Raw completion chunk from model."""

    content: str | None = None
    """Accumulated response content."""

    parsed_content: JSON | BaseModel | None = None
    """Parsed content if response format specified."""

    tool_calls: list[ToolCall] | None = None
    """Accumulated tool calls."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentResponse(BaseModel):
    """Final response collected after agent execution completes.

    Accumulates streaming chunks into a complete response, including
    content, tool calls, and execution statistics. Used to track the
    full output of a single agent interaction.
    """

    role: MessageRole | None = None
    """Role in the conversation."""

    finish_reason: FinishReason | None = None
    """Reason for response generation stopping."""

    content: str | None = None
    """Final response content."""

    parsed_content: JSON | BaseModel | None = None
    """Parsed content if response format specified."""

    tool_calls: list[ToolCall] | None = None
    """Tool calls made during execution."""

    usage: Usage | None = None
    """Token usage statistics."""

    response_cost: ResponseCost | None = None
    """Cost calculation."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentExecutionResult(BaseModel):
    """Complete result of agent execution.

    Contains the final state after all processing, including responses,
    messages, and context updates. Preserves the complete execution
    history and final agent state.
    """

    last_agent: Agent
    """Agent that produced final response."""

    last_agent_response: AgentResponse
    """Final response from agent."""

    agent_responses: list[AgentResponse]
    """Agent responses collected during execution."""

    new_messages: list[Message]
    """Output messages generated during execution."""

    all_messages: list[Message]
    """Complete message history of execution."""

    context_variables: ContextVariables | None = None
    """Final context variables."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )
