# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re
from collections.abc import Awaitable, Callable
from functools import wraps
from textwrap import dedent
from typing import Any, Concatenate, ParamSpec, TypeVar, overload

import orjson
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from pydantic import BaseModel

from liteswarm.types.misc import JSON

_AttributeType = TypeVar("_AttributeType")
"""Type variable for attribute value type.

Used to preserve type information when safely accessing
and validating object attributes.
"""

_AttributeDefaultType = TypeVar("_AttributeDefaultType")
"""Type variable for attribute default value type.

Used to preserve type information for default values when
the attribute doesn't exist or has wrong type.
"""

TParams = ParamSpec("TParams")
"""Type variable for parameter specification.

Used to preserve type information for function parameters.
"""


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

    Examples:
        Basic usage:
            ```python
            class Example:
                attribute: int = 42


            instance = Example()

            # Attribute exists and matches expected type
            value1: int = safe_get_attr(instance, "attribute", int, default=0)
            print(value1)  # Output: 42

            # Attribute exists but does not match expected type
            value2: str = safe_get_attr(instance, "attribute", str, default="default_value")
            print(value2)  # Output: "default_value"

            # Attribute does not exist, returns default
            value3: int = safe_get_attr(instance, "nonexistent", int, default=100)
            print(value3)  # Output: 100

            # Attribute does not exist, no default provided
            value4: int | None = safe_get_attr(instance, "nonexistent", int, default=100)
            print(value4)  # Output: 100
            ```
    """
    value = getattr(obj, attr, default)
    if isinstance(value, expected_type):
        return value

    return default


def extract_json(content: str) -> JSON:
    """Extract and parse JSON data from text content.

    Attempts to find and parse JSON from multiple formats:
    - Markdown code blocks (with or without language tag)
    - Raw JSON strings
    - Indented or formatted JSON

    Args:
        content: Text that may contain JSON data.

    Returns:
        Parsed JSON data as a Python object.

    Raises:
        ValueError: If no valid JSON found or parsing fails.

    Examples:
        Code block:
            ```python
            text = \"\"\"
            Here's the configuration:
            ```json
            {
                "api_key": "abc123",
                "settings": {
                    "timeout": 30,
                    "retry": true
                }
            }
            ```
            \"\"\"
            config = extract_json(text)
            assert config["settings"]["timeout"] == 30
            ```

        Raw JSON:
            ```python
            # Simple object
            data = extract_json('{"x": 1, "y": 2}')
            assert data == {"x": 1, "y": 2}

            # Array
            items = extract_json("[1, 2, 3]")
            assert items == [1, 2, 3]
            ```

        Error handling:
            ```python
            try:
                extract_json("Invalid JSON")
            except ValueError as e:
                print(f"Failed to parse: {e}")
            ```
    """
    code_block_pattern = r"```(?:json)?\n?(.*?)```"
    matches = re.findall(code_block_pattern, content, re.DOTALL)

    if matches:
        for match in matches:
            try:
                if not isinstance(match, str):
                    continue
                return orjson.loads(match.strip())
            except Exception:
                continue

    try:
        return orjson.loads(content.strip())
    except orjson.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


def dedent_prompt(prompt: str) -> str:
    """Clean and format multiline prompt text.

    Removes common leading whitespace and trims surrounding
    whitespace while preserving relative indentation.

    Args:
        prompt: Raw prompt text to clean.

    Returns:
        Cleaned and formatted prompt text.

    Examples:
        Basic cleaning:
            ```python
            text = dedent_prompt(\"\"\"
                Hello!
                This is a multiline prompt.
                    - With indented items
                    - And proper spacing
            \"\"\")
            print(text)
            # Hello!
            # This is a multiline prompt.
            #     - With indented items
            #     - And proper spacing
            ```

        Code examples:
            ```python
            code = dedent_prompt(\"\"\"
                ```python
                def my_function():
                    return "Hello, world!"
                ```
            \"\"\")

            print(code)
            # ```python
            # def my_function():
            #     return "Hello, world!"
            # ```
            ```

        System prompts:
            ```python
            system = dedent_prompt(\"\"\"
                You are an AI assistant that:
                1. Responds concisely
                2. Uses markdown
                3. Includes examples
            \"\"\")
            ```
    """
    return dedent(prompt).strip()


def find_tag_content(text: str, tag: str) -> str | None:
    """Find and extract content from XML-style tagged sections in text.

    Searches for content enclosed between opening and closing tags of the specified name
    using regex pattern matching. The search is case-sensitive and supports multiline content.

    Args:
        text: The source text containing tagged sections to search through.
        tag: The name of the tag to find (without angle brackets).

    Returns:
        The content between the opening and closing tags if found,
        or `None` if no matching tags are found in the text.

    Examples:
        Basic usage:
            ```python
            text = "<data>Important content</data>"
            content = find_tag_content(text, "data")
            print(content)  # Output: "Important content"

            # No matching tags
            result = find_tag_content(text, "missing")
            print(result)  # Output: None
            ```

        Multiline content:
            ```python
            text = \"\"\"
            <config>
                setting1: value1
                setting2: value2
            </config>
            \"\"\"
            content = find_tag_content(text, "config")
            print(content)
            # setting1: value1
            # setting2: value2
            ```
    """
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else None


def parse_content(value: Any) -> str:
    """Parse any value to a string representation suitable for content.

    Handles different types appropriately:
    - Strings are returned as-is
    - Pydantic models are JSON serialized
    - Other objects are JSON serialized with special handling

    Args:
        value: Any value to convert to string.

    Returns:
        String representation of the value.

    Examples:
        ```python
        to_content_string("hello")  # Returns: hello
        to_content_string(42)  # Returns: 42
        to_content_string({"x": 1})  # Returns: {"x": 1}
        to_content_string(MyModel(field=1))  # Returns: {"field": 1}
        ```
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return orjson.dumps(value).decode()


@overload
def with_prompt_session(
    func: Callable[Concatenate[PromptSession[str], TParams], Awaitable[str]],
) -> Callable[TParams, Awaitable[str]]: ...


@overload
def with_prompt_session(
    func: None = None,
    *,
    session: PromptSession[str],
) -> Callable[
    [Callable[Concatenate[PromptSession[str], TParams], Awaitable[str]]],
    Callable[TParams, Awaitable[str]],
]: ...


def with_prompt_session(
    func: Callable[Concatenate[PromptSession[str], TParams], Awaitable[str]] | None = None,
    *,
    session: PromptSession[str] | None = None,
) -> Any:
    """Decorator that provides a shared prompt session for user input handling.

    Creates and maintains a single PromptSession instance with history support,
    command completion, and proper terminal handling. The session is shared
    across all decorated functions to maintain consistent state and history.

    Can be used either as a simple decorator or with a custom session:
    - @with_prompt_session  # Uses default session
    - @with_prompt_session(session=custom_session)  # Uses provided session

    Args:
        func: Async function that takes a PromptSession as its first argument.
        session: Optional custom PromptSession instance to use instead of default.

    Returns:
        Wrapped function that automatically receives the shared session.

    Examples:
        Basic usage:
            ```python
            @with_prompt_session
            async def get_input(session: PromptSession[str], prefix: str) -> str:
                return await session.prompt_async(f"{prefix}> ")


            # Use the decorated function
            response = await get_input("user")  # Shows: user>
            ```

        Custom session:
            ```python
            custom_session = PromptSession[str](
                history=FileHistory(".history"),
                completer=my_completer,
            )


            @with_prompt_session(session=custom_session)
            async def get_input(session: PromptSession[str], prefix: str) -> str:
                return await session.prompt_async(f"{prefix}> ")
            ```

    Notes:
        Default session includes:
        - Command history (in memory)
        - History search
        - Terminal state handling
        - Command completion
        - Suspend functionality (Ctrl+Z)
    """

    def decorator(
        f: Callable[Concatenate[PromptSession[str], TParams], Awaitable[str]],
    ) -> Callable[TParams, Awaitable[str]]:
        prompt_session = session or PromptSession(
            history=InMemoryHistory(),
            enable_history_search=True,
            complete_while_typing=True,
            enable_suspend=True,
        )

        @wraps(f)
        async def wrapper(*args: TParams.args, **kwargs: TParams.kwargs) -> str:
            with patch_stdout():
                return await f(prompt_session, *args, **kwargs)

        return wrapper

    if func is None:
        return decorator

    return decorator(func)


@with_prompt_session
async def prompt(session: PromptSession[str], message: str, **kwargs: Any) -> str:
    """Get user input with a prompt message using a shared session.

    Uses the shared PromptSession to display a prompt and get user input.
    The session provides command history, search, and proper terminal handling.

    Args:
        session: Shared prompt session (injected by decorator).
        message: Prompt message to display.
        **kwargs: Additional arguments passed to prompt_async.

    Returns:
        User input with surrounding whitespace removed.

    Examples:
        ```python
        # Simple prompt
        name = await prompt("Enter your name: ")

        # Password input
        pwd = await prompt("Password: ", is_password=True)

        # With auto-completion
        cmd = await prompt("$ ", completer=my_completer)

        # Multiline input
        text = await prompt("... ", multiline=True)
        ```

    Notes:
        - Supports command history (up/down arrows)
        - Enables history search (Ctrl+R)
        - Handles terminal state properly
        - Preserves suspend functionality (Ctrl+Z)
    """
    return await session.prompt_async(message, **kwargs)
