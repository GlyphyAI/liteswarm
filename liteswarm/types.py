# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, Generic, Literal, Self, TypeVar

from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    FunctionCall,
    Usage,
)
from litellm.types.utils import Delta as LiteDelta
from pydantic import BaseModel, Field

T = TypeVar("T")
"""Generic type placeholder."""

Tool = Callable[..., Any]
"""A tool that can be called by an agent."""

AgentState = Literal["idle", "active", "stale"]
"""The state of an agent."""

ContextVariables = dict[str, Any]
"""Context variables for an agent."""

Instructions = str | Callable[[ContextVariables], str]
"""Agent instructions - either a string or a function that takes context variables."""


class Message(BaseModel):
    """A message in the conversation between users, assistants, and tools."""

    role: Literal["assistant", "user", "system", "tool"]
    """The role of the message sender ("assistant", "user", "system", or "tool")."""
    content: str | None = None
    """The text content of the message."""
    tool_calls: list[ChatCompletionDeltaToolCall] | None = None
    """List of tool calls made in this message."""
    tool_call_id: str | None = None
    """ID of the tool call this message is responding to."""
    audio: ChatCompletionAudioResponse | None = None
    """Audio response data, if any."""

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        extra = "allow"


class ToolMessage(BaseModel):
    """A message resulting from a tool call, optionally including a new agent."""

    message: Message
    """The message containing the tool's response."""
    agent: "Agent | None" = None
    """Optional new agent to switch to (for agent-switching tools)."""
    context_variables: ContextVariables | None = None
    """Context variables to pass to the next agent."""


class Delta(BaseModel):
    """A partial update in a streaming response."""

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

    @classmethod
    def from_delta(cls, delta: LiteDelta) -> "Delta":
        """Create a Delta instance from a LiteLLM delta object.

        Args:
            delta: The LiteLLM delta to convert

        Returns:
            New Delta instance with copied attributes
        """
        return cls(
            content=delta.content,
            role=delta.role,
            function_call=delta.function_call,
            tool_calls=delta.tool_calls,
            audio=delta.audio,
        )


class ResponseCost(BaseModel):
    """Cost information for a model response."""

    prompt_tokens_cost: float
    """Cost of tokens in the prompt."""
    completion_tokens_cost: float
    """Cost of tokens in the completion."""


class Agent(BaseModel):
    """An AI agent that can participate in conversations and use tools."""

    id: str
    """Unique identifier for the agent."""
    model: str
    """The language model to use."""
    instructions: Instructions
    """System prompt defining the agent's behavior. Can be string or function."""
    tools: list[Tool] = Field(default_factory=list)
    """List of functions the agent can call."""
    tool_choice: str | None = None
    """How the agent should choose tools ("auto", "none", etc.)."""
    parallel_tool_calls: bool | None = None
    """Whether multiple tools can be called simultaneously."""
    state: AgentState = "idle"
    """Current state of the agent."""
    params: dict[str, Any] | None = Field(default_factory=dict)
    """Additional parameters for the language model."""

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        extra = "allow"

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        id: str,
        model: str,
        instructions: Instructions,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        state: AgentState = "idle",
        **params: Any,
    ) -> Self:
        """Create a new Agent instance with the given configuration.

        Args:
            id: Unique identifier for the agent
            model: The language model to use
            instructions: System prompt defining the agent's behavior
            tools: List of functions the agent can call
            tool_choice: How the agent should choose tools
            parallel_tool_calls: Whether multiple tools can be called simultaneously
            state: Initial state of the agent
            **params: Additional parameters for the language model

        Returns:
            New Agent instance
        """
        return cls(
            id=id,
            model=model,
            instructions=instructions,
            tools=tools or [],
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            state=state,
            params=params,
        )


class Result(BaseModel, Generic[T]):
    """A generic wrapper for operation results in the agentic system.

    This class provides a standardized way to return results from any operation
    (agents, functions, tools, etc.) with support for:
    - Success values of any type
    - Error information
    - Agent switching
    - Context variable updates

    Args:
        T: The type of the value field.

    Example:
        ```python
        # Simple value result
        Result[float](value=42.0)

        # Error result
        Result[str](error=ValueError("Invalid input"))

        # Agent switch with context
        Result[None](
            agent=new_agent,
            context_variables={"user": "Alice"}
        )
        ```
    """

    value: T | None = None
    """The operation's result value, if any."""
    error: Exception | None = None
    """Any error that occurred during the operation."""
    agent: Agent | None = None
    """Optional new agent to switch to."""
    context_variables: ContextVariables | None = None
    """Optional context variables to update."""

    class Config:  # noqa: D106
        arbitrary_types_allowed = True


class ToolCallResult(BaseModel):
    """Base class for results of tool calls."""

    tool_call: ChatCompletionDeltaToolCall
    """The tool call that produced this result."""


class ToolCallMessageResult(ToolCallResult):
    """Result of a tool call that produced a message."""

    message: Message
    """The message containing the tool's response."""
    context_variables: ContextVariables | None = None
    """Context variables to pass to the next agent."""


class ToolCallAgentResult(ToolCallResult):
    """Result of a tool call that produced a new agent."""

    agent: Agent
    """The new agent to switch to."""
    message: Message | None = None
    """Optional message to add to the conversation."""
    context_variables: ContextVariables | None = None
    """Context variables to pass to the next agent."""


class ToolCallFailureResult(ToolCallResult):
    """Result of a failed tool call."""

    error: Exception
    """The exception that occurred during tool execution."""

    class Config:  # noqa: D106
        arbitrary_types_allowed = True


class CompletionResponse(BaseModel):
    """A response chunk from the language model."""

    delta: Delta
    """The content update in this chunk."""
    finish_reason: str | None = None
    """Why the response ended (if it did)."""
    usage: Usage | None = None
    """Token usage statistics."""
    response_cost: ResponseCost | None = None
    """Cost information for this response."""


class AgentResponse(BaseModel):
    """A processed response from an agent, including accumulated state."""

    delta: Delta
    """The content update in this response."""
    finish_reason: str | None = None
    """Why the response ended (if it did)."""
    content: str | None = None
    """Accumulated content so far."""
    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)
    """Accumulated tool calls."""
    usage: Usage | None = None
    """Token usage statistics."""
    response_cost: ResponseCost | None = None
    """Cost information for this response."""


class ConversationState(BaseModel):
    """Complete state of a conversation."""

    content: str | None = None
    """Final content of the conversation."""
    agent: Agent | None = None
    """Currently active agent."""
    agent_messages: list[Message] = Field(default_factory=list)
    """Messages for the current agent."""
    agent_queue: list[Agent] = Field(default_factory=list)
    """Queue of agents waiting to be activated."""
    messages: list[Message] = Field(default_factory=list)
    """Complete conversation history."""
    usage: Usage | None = None
    """Total token usage statistics."""
    response_cost: ResponseCost | None = None
    """Total cost information."""


class FunctionDocstring(BaseModel):
    """Parsed documentation for a function tool."""

    description: str | None = None
    """Description of what the function does."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """Documentation for each parameter."""
