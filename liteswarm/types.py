from collections.abc import Callable
from typing import Any, Literal, Optional, Self

from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    FunctionCall,
)
from pydantic import BaseModel, Field
from typing_extensions import Protocol, TypedDict

FunctionTool = Callable[..., Any]


class Message(TypedDict, total=False):
    role: Literal["assistant", "user", "system", "tool"]
    content: str | None
    tool_calls: list[ChatCompletionDeltaToolCall] | None
    tool_call_id: str | None
    audio: ChatCompletionAudioResponse | None


class Delta(BaseModel):
    content: str | None = None
    role: str | None = None
    function_call: FunctionCall | dict | None = None
    tool_calls: list[ChatCompletionDeltaToolCall | dict] | None = None
    audio: ChatCompletionAudioResponse | None = None

    @classmethod
    def from_delta(cls, delta: Any) -> "Delta":
        return cls(
            content=delta.content,
            role=delta.role,
            function_call=delta.function_call,
            tool_calls=delta.tool_calls,
            audio=delta.audio,
        )


class StreamHandler(Protocol):
    async def on_stream(
        self,
        chunk: Delta,
        agent: Optional["Agent"],
    ) -> None: ...

    async def on_error(
        self,
        error: Exception,
        agent: Optional["Agent"],
    ) -> None: ...

    async def on_agent_switch(
        self,
        previous_agent: Optional["Agent"],
        next_agent: "Agent",
    ) -> None: ...

    async def on_complete(
        self,
        messages: list[Message],
        agent: Optional["Agent"],
    ) -> None: ...

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Optional["Agent"],
    ) -> None: ...

    async def on_tool_call_result(
        self,
        tool_call_result: "ToolCallResult",
        agent: Optional["Agent"],
    ) -> None: ...


class Agent(BaseModel):
    agent_id: str
    model: str
    instructions: str
    function_tools: list[FunctionTool] = Field(default_factory=list)
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None
    params: dict[str, Any] | None = Field(default_factory=dict)

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        extra = "allow"

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        agent_id: str,
        model: str,
        instructions: str,
        function_tools: list[FunctionTool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        **params: Any,
    ) -> Self:
        return cls(
            agent_id=agent_id,
            model=model,
            instructions=instructions,
            function_tools=function_tools or [],
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            params=params,
        )


class ToolCallResult(BaseModel):
    tool_call: ChatCompletionDeltaToolCall


class ToolCallMessageResult(ToolCallResult):
    message: Message


class ToolCallAgentResult(ToolCallResult):
    agent: Agent


class StreamResponse(BaseModel):
    messages: list[Message]
    new_agents: list[Agent]
    content: str | None = None
    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)


class AgentResponse(BaseModel):
    delta: Delta
    content: str | None = None
    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)


class ConversationState(BaseModel):
    content: str | None = None
    messages: list[Message]
