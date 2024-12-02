# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Number: TypeAlias = int | float
"""Type alias for JSON-compatible numeric values.

Represents both integer and floating-point numbers for use in:
- Function parameters
- API responses
- Configuration values

Examples:
    Basic usage:
        ```python
        def calculate_area(width: Number, height: Number) -> Number:
            return width * height

        # Integer inputs
        area1 = calculate_area(5, 4)  # Returns 20

        # Mixed inputs
        area2 = calculate_area(3.5, 2)  # Returns 7.0
        ```

    Type validation:
        ```python
        from pydantic import BaseModel

        class Rectangle(BaseModel):
            width: Number
            height: Number

        # Both valid
        rect1 = Rectangle(width=10, height=5)
        rect2 = Rectangle(width=3.14, height=2.5)
        ```
"""

JSON: TypeAlias = str | bool | Number | list["JSON"] | dict[str, "JSON"] | None
"""Type alias for JSON-compatible data structures.

Represents any valid JSON value type:
- Strings: Unicode text
- Booleans: true/false
- Numbers: integers and floats
- Arrays: ordered lists of JSON values
- Objects: key-value maps with string keys
- Null: None/null value

Examples:
    Configuration data:
        ```python
        config: JSON = {
            "name": "api-service",
            "version": "2.1.0",
            "settings": {
                "timeout": 30,
                "retries": 3,
                "features": ["auth", "cache"],
                "debug": True,
                "optional": None
            }
        }
        ```

    API responses:
        ```python
        def fetch_data() -> JSON:
            \"\"\"Fetch data from external API.\"\"\"
            return {
                "status": "success",
                "data": [
                    {"id": 1, "value": 42.5},
                    {"id": 2, "value": 98.1}
                ],
                "metadata": {
                    "count": 2,
                    "page": 1
                }
            }
        ```
"""


class FunctionDocstring(BaseModel):
    """Documentation parser for function tools.

    Extracts and structures function documentation into a format
    suitable for tool registration and API schema generation.

    Supports standard docstring sections:
    - Description: What the function does
    - Arguments: Parameter descriptions
    - Returns: Output description
    - Examples: Usage examples

    Examples:
        Basic function:
            ```python
            def greet(name: str, formal: bool = False) -> str:
                \"\"\"Generate a greeting message.

                Args:
                    name: Person's name to greet.
                    formal: Whether to use formal greeting.

                Returns:
                    Formatted greeting message.
                \"\"\"
                prefix = "Dear" if formal else "Hello"
                return f"{prefix} {name}"

            docstring = FunctionDocstring(
                description="Generate a greeting message.",
                parameters={
                    "name": "Person's name to greet",
                    "formal": "Whether to use formal greeting"
                }
            )
            ```

        Complex function:
            ```python
            def process_data(
                data: dict,
                options: list[str] | None = None
            ) -> JSON:
                \"\"\"Process input data with given options.

                Args:
                    data: Input data to process.
                    options: Processing options to apply.

                Returns:
                    Processed data in JSON format.
                \"\"\"
                return {"processed": data}

            docstring = FunctionDocstring(
                description="Process input data with given options.",
                parameters={
                    "data": "Input data to process",
                    "options": "Processing options to apply"
                }
            )
            ```
    """  # noqa: D214

    description: str | None = None
    """Main description of the function's purpose."""

    parameters: dict[str, Any] = Field(default_factory=dict)
    """Documentation for each function parameter."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
