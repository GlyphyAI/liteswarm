from collections.abc import Callable
from typing import Any, Literal, Self

from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    FunctionCall,
    Usage,
)
from litellm.types.utils import Delta as LiteDelta
from pydantic import BaseModel, Field

Tool = Callable[..., Any]

AgentState = Literal["idle", "active", "stale"]


class Message(BaseModel):
    role: Literal["assistant", "user", "system", "tool"]
    content: str | None = None
    tool_calls: list[ChatCompletionDeltaToolCall] | None = None
    tool_call_id: str | None = None
    audio: ChatCompletionAudioResponse | None = None

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        extra = "allow"


class Delta(BaseModel):
    content: str | None = None
    role: str | None = None
    function_call: FunctionCall | dict | None = None
    tool_calls: list[ChatCompletionDeltaToolCall | dict] | None = None
    audio: ChatCompletionAudioResponse | None = None

    @classmethod
    def from_delta(cls, delta: LiteDelta) -> "Delta":
        return cls(
            content=delta.content,
            role=delta.role,
            function_call=delta.function_call,
            tool_calls=delta.tool_calls,
            audio=delta.audio,
        )


class ResponseCost(BaseModel):
    prompt_tokens_cost: float
    completion_tokens_cost: float


class Agent(BaseModel):
    id: str
    model: str
    instructions: str
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None
    state: AgentState = "idle"
    params: dict[str, Any] | None = Field(default_factory=dict)

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        extra = "allow"

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        id: str,
        model: str,
        instructions: str,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        state: AgentState = "idle",
        **params: Any,
    ) -> Self:
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


class ToolCallResult(BaseModel):
    tool_call: ChatCompletionDeltaToolCall


class ToolCallMessageResult(ToolCallResult):
    message: Message


class ToolCallAgentResult(ToolCallResult):
    agent: Agent


class CompletionResponse(BaseModel):
    delta: Delta
    finish_reason: str | None = None
    usage: Usage | None = None
    response_cost: ResponseCost | None = None


class AgentResponse(BaseModel):
    delta: Delta
    finish_reason: str | None = None
    content: str | None = None
    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    response_cost: ResponseCost | None = None


class ConversationState(BaseModel):
    content: str | None = None
    agent: Agent | None = None
    agent_messages: list[Message] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    usage: Usage | None = None
    response_cost: ResponseCost | None = None
