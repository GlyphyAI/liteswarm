# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, TypeVar

from liteswarm.types import ContextVariables
from liteswarm.types.swarm import AgentInstructions

_GenericType = TypeVar("_GenericType")
"""Type variable representing a generic type."""


def unwrap_callable(
    value: _GenericType | Callable[..., _GenericType],
    *args: Any,
    **kwargs: Any,
) -> _GenericType:
    """Unwrap a value that might be wrapped in a callable.

    If the value is callable, calls it with the provided arguments.
    Otherwise, returns the value as-is.

    Example:
    ```python
    # Direct value
    value = unwrap_callable(42)  # Returns 42

    # Function value
    def get_value(x: int) -> int:
        return x * 2

    value = unwrap_callable(get_value, 21)  # Returns 42
    ```

    Args:
        value: The value or callable to unwrap
        *args: Positional arguments to pass if value is callable
        **kwargs: Keyword arguments to pass if value is callable

    Returns:
        The unwrapped value
    """
    return value(*args, **kwargs) if callable(value) else value


def unwrap_instructions(
    instructions: AgentInstructions,
    context_variables: ContextVariables | None = None,
) -> str:
    """Unwrap instructions if they are a callable.

    If instructions is a callable, it will be called with the provided context
    variables. Otherwise, the instructions string will be returned as-is.

    Args:
        instructions: The instructions to unwrap, either a string or a callable
            that takes context_variables and returns a string
        context_variables: Optional dictionary of context variables to pass to
            the instructions callable

    Returns:
        The unwrapped instructions string

    Example:
        ```python
        def get_instructions(context_variables: ContextVariables) -> str:
            user = context_variables.get("user_name", "user")
            return f"Help {user} with their task."

        # With callable instructions
        instructions = unwrap_instructions(
            get_instructions,
            context_variables={"user_name": "Alice"}
        )
        # Returns: "Help Alice with their task."

        # With string instructions
        instructions = unwrap_instructions("Help the user.")
        # Returns: "Help the user."
        ```
    """
    return unwrap_callable(
        instructions,
        context_variables or ContextVariables(),
    )
