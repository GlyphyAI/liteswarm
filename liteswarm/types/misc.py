# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Number: TypeAlias = int | float
"""A numeric type that can be either integer or float.

Used for JSON-compatible numeric values in:
- Function arguments
- Tool responses
- Model parameters

Example:
```python
def calculate_area(width: Number, height: Number) -> Number:
    return width * height

area = calculate_area(5, 3.5)  # Returns 17.5
```
"""

JSON: TypeAlias = str | bool | Number | list["JSON"] | dict[str, "JSON"] | None
"""A JSON-compatible type for structured data.

Represents any valid JSON value:
- Primitive types (str, bool, numbers, null)
- Arrays (lists) of JSON values
- Objects (dicts) with string keys and JSON values

Example:
```python
# Valid JSON values
config: JSON = {
    "name": "test",
    "enabled": True,
    "threshold": 0.5,
    "tags": ["a", "b", "c"],
    "metadata": {
        "version": "1.0",
        "nested": {
            "value": None
        }
    }
}

def process_json(data: JSON) -> JSON:
    \"\"\"Process any JSON-compatible data.\"\"\"
    return {"processed": data}
```
"""


class FunctionDocstring(BaseModel):
    """Parsed documentation for a function tool.

    Extracts and structures Python docstring information for:
    - Function description
    - Parameter documentation
    - Return value documentation

    Used to generate JSON schemas and documentation for tools
    that agents can use.

    Example:
    ```python
    def add_numbers(a: float, b: float) -> float:
        \"\"\"Add two numbers together.

        Args:
            a: First number to add
            b: Second number to add

        Returns:
            Sum of the two numbers
        \"\"\"
        return a + b

    docstring = FunctionDocstring(
        description="Add two numbers together.",
        parameters={
            "a": "First number to add",
            "b": "Second number to add"
        }
    )
    ```
    """  # noqa: D214

    description: str | None = None
    """Description of what the function does"""

    parameters: dict[str, Any] = Field(default_factory=dict)
    """Documentation for each parameter"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
