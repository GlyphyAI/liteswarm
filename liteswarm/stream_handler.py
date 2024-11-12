from litellm.types.utils import ChatCompletionDeltaToolCall
from typing_extensions import Protocol

from liteswarm.types import Agent, Delta, Message, ToolCallResult


class StreamHandler(Protocol):
    async def on_stream(
        self,
        chunk: Delta,
        agent: Agent | None,
    ) -> None: ...

    async def on_error(
        self,
        error: Exception,
        agent: Agent | None,
    ) -> None: ...

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None: ...

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None: ...

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None: ...

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None: ...

