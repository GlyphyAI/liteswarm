# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from liteswarm.types import ContextVariables
from liteswarm.types.swarm import AgentInstructions
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


def unwrap_task_output_type(output_type: TaskOutput) -> type[BaseModel]:
    """Extract the Pydantic model type from a TaskOutput.

    This function handles:
    1. Direct BaseModel subclasses
    2. Callable output parsers (using return type annotations)
    3. Union types (extracting first non-None type)

    Note:
        This function only analyzes type annotations and does not execute
        any callables. All parser functions must have explicit return type
        annotations that are BaseModel subclasses.

    Args:
        output_type: The output type to unwrap. Can be:
            - A Pydantic model class
            - A callable that returns a Pydantic model
            - A Union containing the above types

    Returns:
        The unwrapped Pydantic model class

    Raises:
        TypeError: If output_type is not a BaseModel or doesn't return one
        TypeError: If callable lacks proper return type annotation

    Example:
    ```python
    class ReviewOutput(BaseModel):
        approved: bool
        comments: list[str]

    # Direct model class
    model_type = unwrap_task_output_type(ReviewOutput)
    assert model_type == ReviewOutput

    # Callable parser
    def parse_review(content: str, context: ContextVariables) -> ReviewOutput:
        return ReviewOutput.model_validate_json(content)

    model_type = unwrap_task_output_type(parse_review)
    assert model_type == ReviewOutput

    # Optional type
    from typing import Optional
    model_type = unwrap_task_output_type(Optional[ReviewOutput])
    assert model_type == ReviewOutput
    ```

    Note:
        This function only analyzes type annotations and does not execute
        any callables. All parser functions must have explicit return type
        annotations that are BaseModel subclasses.
    """
    if get_origin(output_type) is Union:
        args = get_args(output_type)
        for arg in args:
            if arg is not type(None):
                return unwrap_task_output_type(arg)

    if isinstance(output_type, type):
        if issubclass(output_type, BaseModel):
            return output_type
        else:
            raise TypeError(
                f"TaskOutput type '{output_type.__name__}' is not a subclass of BaseModel."
            )

    try:
        function_type_hints = get_type_hints(output_type)
    except NameError as e:
        raise TypeError(
            f"Unable to resolve type hints for the callable '{output_type.__name__}': {e}"
        ) from e

    return_type = function_type_hints.get("return")
    if not return_type:
        raise TypeError(
            f"Callable '{output_type.__name__}' must have a return type annotation "
            "that is a subclass of BaseModel."
        )

    return unwrap_task_output_type(return_type)
