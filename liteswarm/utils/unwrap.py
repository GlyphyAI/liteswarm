# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, TypeVar

from liteswarm.types import ContextVariables
from liteswarm.types.swarm import AgentInstructions

_GenericType = TypeVar("_GenericType")
"""Type variable for generic value types.

Used to preserve type information when unwrapping values
that could be either direct values or callables returning
those values.
"""


def unwrap_callable(
    value: _GenericType | Callable[..., _GenericType],
    *args: Any,
    **kwargs: Any,
) -> _GenericType:
    """Extract value from callable or return as-is.

    Handles values that might be either direct values or functions
    that return values. If the input is callable, executes it with
    provided arguments.

    Args:
        value: Direct value or callable to unwrap.
        *args: Positional arguments for callable.
        **kwargs: Keyword arguments for callable.

    Returns:
        Extracted or original value.

    Examples:
        Direct values:
            ```python
            value = unwrap_callable(42)
            assert value == 42

            text = unwrap_callable("hello")
            assert text == "hello"
            ```

        Function values:
            ```python
            def multiply(x: int, y: int) -> int:
                return x * y

            value = unwrap_callable(multiply, 6, 7)
            assert value == 42

            value = unwrap_callable(
                lambda x: x * 2,
                21
            )
            assert value == 42
            ```

        With kwargs:
            ```python
            def format_greeting(
                name: str,
                formal: bool = False
            ) -> str:
                prefix = "Dear" if formal else "Hello"
                return f"{prefix} {name}"

            greeting = unwrap_callable(
                format_greeting,
                name="Alice",
                formal=True
            )
            assert greeting == "Dear Alice"
            ```
    """
    return value(*args, **kwargs) if callable(value) else value


def unwrap_instructions(
    instructions: AgentInstructions,
    context_variables: ContextVariables | None = None,
) -> str:
    """Convert agent instructions to string format.

    Processes instructions that can be either a direct string or
    a function that generates instructions based on context.
    Automatically handles context variables.

    Args:
        instructions: String or function that generates instructions.
        context_variables: Optional context for dynamic instructions.

    Returns:
        Final instruction string.

    Examples:
        Static instructions:
            ```python
            text = unwrap_instructions(
                "You are a helpful assistant."
            )
            assert text == "You are a helpful assistant."
            ```

        Dynamic instructions:
            ```python
            def get_instructions(
                context: ContextVariables
            ) -> str:
                name = context.get("user", "friend")
                return f"Help {name} with their task."

            # With context
            text = unwrap_instructions(
                get_instructions,
                ContextVariables(user="Alice")
            )
            assert text == "Help Alice with their task."

            # Without context
            text = unwrap_instructions(get_instructions)
            assert text == "Help friend with their task."
            ```

        Template system:
            ```python
            def create_expert(
                domain: str
            ) -> AgentInstructions:
                def get_instructions(
                    context: ContextVariables
                ) -> str:
                    return f"You are an expert in {domain}."

                return get_instructions

            python_expert = create_expert("Python")
            text = unwrap_instructions(python_expert)
            assert text == "You are an expert in Python."
            ```
    """
    return unwrap_callable(
        instructions,
        context_variables or ContextVariables(),
    )
