# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING

from liteswarm.types.swarm import AgentExecutionResult, AgentResponse, ResponseCost, Usage
from liteswarm.utils.usage import combine_response_cost, combine_usage

if TYPE_CHECKING:
    from pydantic import BaseModel

    from liteswarm.types.misc import JSON


class SwarmStream(AsyncIterator[AgentResponse]):
    """Wrapper for Swarm stream that provides execution result capabilities.

    SwarmStream enhances the base stream functionality by:
    - Providing both streaming and result execution modes
    - Accumulating response content and metadata
    - Tracking usage and cost statistics
    - Managing parsed content handling
    - Supporting flexible consumption patterns

    The wrapper maintains internal state to:
    - Track the last received response
    - Accumulate complete content
    - Combine usage statistics
    - Aggregate cost information
    - Handle parsed content formats

    Notes:
        - Supports both streaming and final result retrieval
        - Preserves all original stream functionality
        - Maintains response metadata and statistics
        - Thread-safe for async operations
        - Memory efficient with minimal overhead

    Examples:
        Streaming mode:
            ```python
            stream = SwarmStream(swarm.stream(agent, prompt="Hello"))
            async for response in stream:
                print(response.content)  # Print content as it arrives
            ```

        Result execution:
            ```python
            stream = SwarmStream(swarm.stream(agent, prompt="Hello"))
            result = await stream.get_result()  # Wait for complete response
            print(result.content)  # Print final content
            ```

        Combined usage:
            ```python
            stream = SwarmStream(swarm.stream(agent, prompt="Hello"))

            # Stream partial responses
            async for response in stream:
                print("Partial:", response.content)

            # Get final result
            result = await stream.get_result()
            print("Final:", result.content)
            ```
    """

    def __init__(self, stream: AsyncGenerator[AgentResponse, None]) -> None:
        """Initialize stream wrapper with base generator.

        Args:
            stream: Base AsyncGenerator[AgentResponse] to wrap.
                Must yield AgentResponse objects.
        """
        self._stream = stream
        self._last_response: AgentResponse | None = None
        self._accumulated_content: str | None = None
        self._accumulated_parsed_content: JSON | BaseModel | None = None
        self._accumulated_usage: Usage | None = None
        self._accumulated_response_cost: ResponseCost | None = None

    def __aiter__(self) -> AsyncIterator[AgentResponse]:
        """Provide async iterator interface.

        Returns:
            Self as AsyncIterator for response streaming.
        """
        return self

    async def __anext__(self) -> AgentResponse:
        """Get next response from stream with accumulation.

        Retrieves the next response while maintaining:
        - Content accumulation
        - Usage statistics
        - Cost tracking
        - Parsed content handling

        Returns:
            Next AgentResponse from the stream.

        Raises:
            StopAsyncIteration: When stream is exhausted.
        """
        response = await self._stream.__anext__()
        self._accumulate_agent_response(response)
        return response

    async def get_result(self) -> AgentExecutionResult:
        """Get complete execution result from stream.

        Processes the stream to completion and returns:
        - Final accumulated content
        - Complete usage statistics
        - Total cost information
        - Final parsed content
        - Last active agent

        Returns:
            AgentExecutionResult containing complete response data.

        Raises:
            ValueError: If no responses were received from stream.

        Notes:
            - Consumes any remaining stream content
            - Blocks until stream is complete
            - Safe to call multiple times
            - Returns consistent result after completion
        """
        await self._wait_until_complete()
        if not self._last_response:
            raise ValueError("No responses received from stream")

        return AgentExecutionResult(
            agent=self._last_response.agent,
            content=self._accumulated_content,
            parsed_content=self._accumulated_parsed_content,
            usage=self._accumulated_usage,
            response_cost=self._accumulated_response_cost,
        )

    async def _wait_until_complete(self) -> None:
        """Wait for stream completion while accumulating results.

        Consumes remaining stream content to ensure:
        - Complete content accumulation
        - Final statistics collection
        - Proper resource cleanup
        """
        async for _ in self:
            pass

    def _accumulate_agent_response(self, agent_response: AgentResponse) -> None:
        """Accumulate response data from stream.

        Updates internal state with:
        - Latest response reference
        - Accumulated content
        - Combined usage statistics
        - Aggregated cost information
        - Latest parsed content

        Args:
            agent_response: New response to accumulate.
        """
        self._last_response = agent_response
        self._accumulated_content = agent_response.content
        self._accumulated_parsed_content = agent_response.parsed_content

        self._accumulated_usage = combine_usage(
            self._accumulated_usage,
            agent_response.usage,
        )

        self._accumulated_response_cost = combine_response_cost(
            self._accumulated_response_cost,
            agent_response.response_cost,
        )
