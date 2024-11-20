# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import operator
import re
from collections.abc import Callable
from functools import reduce
from textwrap import dedent
from typing import Any, TypeVar

import orjson
from pydantic import BaseModel

from liteswarm.types import JSON, ContextVariables
from liteswarm.types.swarm import Instructions
from liteswarm.types.swarm_team import Plan, Task, TaskDefinition, TaskOutput

_GenericType = TypeVar("_GenericType")
_AttributeType = TypeVar("_AttributeType")
_AttributeDefaultType = TypeVar("_AttributeDefaultType")


def safe_get_attr(
    obj: Any,
    attr: str,
    expected_type: type[_AttributeType],
    default: _AttributeDefaultType = None,  # type: ignore
) -> _AttributeType | _AttributeDefaultType:
    """Safely retrieves and validates an attribute of an object.

    This function attempts to access the specified attribute from the given object.
    If the attribute exists and its value matches the expected type, the value is returned.
    Otherwise, the `default` value is returned.

    If the `default` is not provided, it defaults to `None`. The return type will be inferred
    as a union of the expected type and the type of the default value.

    Args:
        obj: The object from which to retrieve the attribute.
        attr: The name of the attribute to retrieve.
        expected_type: The expected type of the attribute's value.
        default: The value to return if the attribute does not exist
            or its value does not match the expected type. Defaults to `None`.

    Returns:
        The attribute's value if it exists and matches the expected type,
        or the `default` value otherwise.

    Example:
        ```python
        class Example:
            attribute: int = 42

        instance = Example()

        # Attribute exists and matches expected type
        value1 = safe_get_attr(instance, "attribute", int, default=0)
        print(value1)  # Output: 42

        # Attribute exists but does not match expected type
        value2 = safe_get_attr(instance, "attribute", str, default="default_value")
        print(value2)  # Output: "default_value"

        # Attribute does not exist, returns default
        value3 = safe_get_attr(instance, "nonexistent", int, default=100)
        print(value3)  # Output: 100

        # Attribute does not exist, no default provided
        value4 = safe_get_attr(instance, "nonexistent", int)
        print(value4)  # Output: None
        ```
    """
    value = getattr(obj, attr, default)
    if isinstance(value, expected_type):
        return value

    return default


def extract_json(content: str) -> JSON:
    """Extract a JSON object from a string.

    Args:
        content: The string to extract the JSON from

    Returns:
        The JSON object, or None if no valid JSON is found
    """
    code_block_pattern = r"```(?:json)?\n?(.*?)```"
    matches = re.findall(code_block_pattern, content, re.DOTALL)

    if matches:
        for match in matches:
            try:
                return orjson.loads(match.strip())
            except Exception:
                continue

    try:
        return orjson.loads(content.strip())
    except orjson.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


def dedent_prompt(prompt: str) -> str:
    return dedent(prompt).strip()


def unwrap_callable(
    value: _GenericType | Callable[..., _GenericType],
    *args: Any,
    **kwargs: Any,
) -> _GenericType:
    """Unwrap a callable if it's wrapped in a callable.

    Args:
        value: The value to unwrap
        *args: Arguments to pass to the callable
        **kwargs: Keyword arguments to pass to the callable

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
    """Generalizes the unpacking of TaskOutput objects to return their JSON schema."""
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


def create_union_type(types: list[_GenericType]) -> _GenericType:
    if not types:
        raise ValueError("No types provided for Union.")
    elif len(types) == 1:
        return types[0]
    else:
        return reduce(operator.or_, types)


def create_plan_schema(task_definitions: list[TaskDefinition]) -> type[Plan[Task]]:
    task_schemas = [td.task_schema for td in task_definitions]
    task_schemas_union = create_union_type(task_schemas)
    return Plan[task_schemas_union].create(model_name=Plan.__name__)  # type: ignore
