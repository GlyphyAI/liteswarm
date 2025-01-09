# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Generic, cast

from typing_extensions import override

from liteswarm.types.collections import ReturnableAsyncGenerator
from liteswarm.types.events import SwarmEvent
from liteswarm.types.llm import ResponseFormatPydantic
from liteswarm.types.swarm import AgentExecutionResult


class SwarmStream(
    ReturnableAsyncGenerator[SwarmEvent, AgentExecutionResult[ResponseFormatPydantic]],
    Generic[ResponseFormatPydantic],
):
    """A wrapper around Swarm event stream that adds response parsing capabilities.

    SwarmStream provides a type-safe interface for streaming agent execution events
    and retrieving the final result. It supports automatic response format validation
    and parsing, ensuring that the returned result matches the expected format.

    Type Parameters:
        ResponseFormatPydantic: The type of the parsed response format.

    Examples:
        Basic usage:
            ```python
            from liteswarm.types.swarm import Agent, Message
            from liteswarm.types.llm import LLM

            # Create agent
            agent = Agent(
                id="assistant",
                instructions="You are a helpful assistant.",
                llm=LLM(model="gpt-4o"),
            )

            # Stream events
            stream = swarm.stream(agent, messages=messages)
            async for event in stream:
                if event.type == "agent_response_chunk":
                    print(event.chunk.completion.delta.content, end="", flush=True)

            # Get final result
            result = await stream.get_return_value()
            ```

        With response format:
            ```python
            from pydantic import BaseModel


            class MathResult(BaseModel):
                result: int
                explanation: str


            stream = swarm.stream(
                agent=agent,
                messages=messages,
                response_format=MathResult,
            )

            result = await stream.get_return_value()
            if result.agent_response.parsed:
                print(result.agent_response.parsed.result)  # Type-safe access
            ```

    Notes:
        - Preserves all events from the original stream
        - Validates response format at completion
        - Provides type-safe access to parsed results
    """

    def __init__(
        self,
        agen: ReturnableAsyncGenerator[SwarmEvent, AgentExecutionResult],
        response_format: type[ResponseFormatPydantic] | None = None,
    ) -> None:
        """Initialize a SwarmStream with an event stream and response format.

        Args:
            agen: The underlying event stream to wrap.
            response_format: Optional type to validate and parse the response into.
        """
        super().__init__(agen._agen)  # type: ignore
        self._response_format = response_format

    @override
    async def get_return_value(self) -> AgentExecutionResult[ResponseFormatPydantic]:
        """Get the final execution result with format validation.

        Retrieves the final result from the event stream and validates that it
        matches the expected response format if one was specified. The validation
        ensures type safety when accessing parsed fields of the result.

        Returns:
            Complete execution result with validated format.

        Raises:
            TypeError: If the response format does not match the parsed response.
            RuntimeError: If the stream completes without a return value.

        Example:
            ```python
            class MathResult(BaseModel):
                result: int
                explanation: str


            stream = swarm.stream(
                agent=agent,
                messages=messages,
                response_format=MathResult,
            )

            result = await stream.get_return_value()
            if result.agent_response.parsed:
                print(result.agent_response.parsed.result)  # Type-safe access
            ```
        """
        result = await super().get_return_value()
        if self._response_format is None:
            return cast(AgentExecutionResult[ResponseFormatPydantic], result)

        if not isinstance(result.agent_response.parsed, self._response_format):
            raise TypeError(
                f"Response format type '{self._response_format}' does not match expected type '{type(result.agent_response.parsed)}'"
            )

        return cast(AgentExecutionResult[ResponseFormatPydantic], result)
