# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from liteswarm.types import ContextVariables
from liteswarm.types.swarm import Instructions
from liteswarm.types.swarm_team import TaskOutput

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
    instructions: Instructions,
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
        context_variables=context_variables or ContextVariables(),
    )


def unwrap_task_output_type(output_schema: TaskOutput) -> type[BaseModel]:
    """Get the Pydantic model type from a TaskOutput.

    Handles both direct model classes and callable output parsers,
    ensuring they return BaseModel types.

    Example:
    ```python
    class OutputModel(BaseModel):
        value: int

    # Direct model
    model_type = unwrap_task_output_type(OutputModel)
    assert model_type == OutputModel

    # Callable parser
    def parse_output(content: str, context: ContextVariables) -> OutputModel:
        return OutputModel(value=42)

    model_type = unwrap_task_output_type(parse_output)
    assert model_type == OutputModel
    ```

    Args:
        output_schema: TaskOutput to analyze

    Returns:
        The underlying Pydantic model type

    Raises:
        TypeError: If output_schema isn't a BaseModel or doesn't return one
    """
    if isinstance(output_schema, type):
        if issubclass(output_schema, BaseModel):
            return output_schema
        else:
            raise TypeError("TaskOutput is not a BaseModel.")

    try:
        dummy_output = output_schema("", ContextVariables())
        if isinstance(dummy_output, BaseModel):
            return dummy_output.__class__
        else:
            raise TypeError("Callable did not return a BaseModel instance.")
    except Exception as e:
        raise TypeError(f"TaskOutput is not a callable or a BaseModel: {e}") from e
