# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent

ResultValue = TypeVar("ResultValue")
"""Type variable representing the value type in a Result."""


class Result(BaseModel, Generic[ResultValue]):
    """Generic wrapper for operation results in the agentic system.

    Provides a standardized way to handle results from any operation (agents,
    functions, tools) with support for success values, errors, agent switching,
    and context updates.

    Examples:
        Simple success result:
            ```python
            # Return calculation result
            def calculate_average(numbers: list[float]) -> Result[float]:
                try:
                    avg = sum(numbers) / len(numbers)
                    return Result(value=avg)
                except ZeroDivisionError as e:
                    return Result(error=e)
            ```

        Error handling:
            ```python
            # Handle validation error
            def validate_input(data: dict) -> Result[dict]:
                if "required_field" not in data:
                    return Result(
                        error=ValueError("Missing required_field")
                    )
                return Result(value=data)
            ```

        Agent switching:
            ```python
            # Switch to specialized agent
            def handle_complex_query(query: str) -> Result[str]:
                if "math" in query.lower():
                    return Result(
                        agent=Agent(
                            id="math-expert",
                            instructions="You are a math expert.",
                            llm=LLM(model="gpt-4o")
                        ),
                        context_variables=ContextVariables(
                            domain="mathematics",
                            complexity="advanced"
                        )
                    )
                return Result(value="I can help with that.")
            ```
    """

    value: ResultValue | None = None
    """Operation's success value, if available."""

    error: Exception | None = None
    """Error that occurred during operation, if any."""

    agent: Agent | None = None
    """New agent to switch to, if needed."""

    context_variables: ContextVariables | None = None
    """Context updates to apply."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
