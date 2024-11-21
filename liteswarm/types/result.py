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
    """A generic wrapper for operation results in the agentic system.

    This class provides a standardized way to return results from any operation
    (agents, functions, tools, etc.) with support for:
    - Success values of any type
    - Error information
    - Agent switching
    - Context variable updates

    Args:
        ResultValue: The type of the value field.

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

    value: ResultValue | None = None
    """The operation's result value, if any."""

    error: Exception | None = None
    """Any error that occurred during the operation."""

    agent: Agent | None = None
    """Optional new agent to switch to."""

    context_variables: ContextVariables | None = None
    """Optional context variables to update."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
