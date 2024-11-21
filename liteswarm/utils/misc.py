# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re
from textwrap import dedent
from typing import Any, TypeVar, Unpack

import orjson
from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import _FromFieldInfoInputs
from pydantic_core import PydanticUndefined

from liteswarm.types.misc import JSON

_AttributeType = TypeVar("_AttributeType")
"""Type variable representing the expected type of an attribute."""

_AttributeDefaultType = TypeVar("_AttributeDefaultType")
"""Type variable representing the default type of an attribute."""

_PydanticModel = TypeVar("_PydanticModel", bound=BaseModel)
"""Type variable representing a Pydantic model."""


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


def change_field_type(
    model_type: type[_PydanticModel],
    field_name: str,
    new_type: Any,
    new_model_name: str | None = None,
    default: Any = PydanticUndefined,
    **kwargs: Unpack[_FromFieldInfoInputs],
) -> type[_PydanticModel]:
    """Create a new Pydantic model with a modified field type.

    Creates a copy of the original model with one field's type changed,
    preserving all other fields and model configuration. Useful for:
    - Dynamically modifying model schemas
    - Creating variants of existing models
    - Adjusting field validation rules

    Example:
    ```python
    class User(BaseModel):
        id: int
        name: str
        age: int

    # Change age to float and add validation
    UserFloat = change_field_type(
        model_type=User,
        field_name="age",
        new_type=float,
        new_model_name="UserFloat",
        default=0.0,
        ge=0.0,  # Field validation: greater or equal to 0
        description="User's age in years"
    )

    # Original model still uses int
    user1 = User(id=1, name="Alice", age=30)

    # New model uses float with validation
    user2 = UserFloat(id=2, name="Bob", age=30.5)

    # Validation error if age < 0
    user3 = UserFloat(id=3, name="Charlie", age=-1.0)  # Raises ValidationError
    ```

    Args:
        model_type: The original Pydantic model to modify
        field_name: Name of the field to change
        new_type: New type for the field
        new_model_name: Optional name for the new model (defaults to "Updated" + original name)
        default: Optional default value for the field
        **kwargs: Additional field configuration (validation rules, descriptions, etc.)

    Returns:
        A new Pydantic model class with the modified field

    Raises:
        TypeError: If the default value doesn't match the new field type
    """
    fields: dict[str, Any] = {}
    for name, field in model_type.model_fields.items():
        if name == field_name:
            field_kwargs: dict[str, Any] = {}
            for attr_name in _FromFieldInfoInputs.__annotations__.keys():
                if attr_value := getattr(field, attr_name, None):
                    field_kwargs[attr_name] = attr_value

            field_kwargs.update(kwargs)
            field_kwargs.pop("annotation")
            field_info = field.from_field(default=default, **field_kwargs)

            fields[name] = (new_type, field_info)
        else:
            fields[name] = (field.annotation, field)

    if default is not PydanticUndefined and kwargs.get("validate_default"):
        try:
            fields = {field_name: (new_type, default)}
            temp_model = create_model("TempModel", **fields)
            temp_model(**{field_name: default})
        except ValidationError as e:
            raise TypeError(
                f"Default value {default!r} is not valid for field '{field_name}' of type {new_type}"
            ) from e

    updated_model_name = new_model_name or f"Updated{model_type.__name__}"

    new_model = create_model(
        updated_model_name,
        __base__=model_type,
        **fields,
    )

    return new_model
