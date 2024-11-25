# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re
from textwrap import dedent
from typing import Any, TypeVar

import orjson

from liteswarm.types.misc import JSON

_AttributeType = TypeVar("_AttributeType")
"""Type variable representing the expected type of an attribute."""

_AttributeDefaultType = TypeVar("_AttributeDefaultType")
"""Type variable representing the default type of an attribute."""


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

    Attempts to find and parse JSON from:
    1. Code blocks (with or without language identifier)
    2. Raw string content

    Example:
    ```python
    # From code block
    content = '''Here's the config:
    ```json
    {"name": "test", "value": 42}
    ```
    '''
    config = extract_json(content)  # {"name": "test", "value": 42}

    # From raw string
    data = extract_json('{"x": 1, "y": 2}')  # {"x": 1, "y": 2}
    ```

    Args:
        content: String containing JSON data

    Returns:
        Parsed JSON object

    Raises:
        ValueError: If no valid JSON can be found or parsed
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
    """Remove common leading whitespace from every line in a prompt.

    Useful for maintaining readable code while creating clean prompts.
    Removes both indentation and leading/trailing whitespace.

    Example:
    ```python
    prompt = dedent_prompt('''
        You are a helpful assistant.
        Follow these rules:
            1. Be concise
            2. Be clear
            3. Be accurate
    ''')
    # Returns:
    # "You are a helpful assistant.
    #  Follow these rules:
    #      1. Be concise
    #      2. Be clear
    #      3. Be accurate"
    ```

    Args:
        prompt: The prompt string to clean

    Returns:
        The prompt with common leading whitespace removed
    """
    return dedent(prompt).strip()
