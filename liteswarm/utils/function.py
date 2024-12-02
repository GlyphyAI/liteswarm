# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, get_type_hints

from griffe import Docstring, DocstringSectionKind

from liteswarm.types.misc import FunctionDocstring
from liteswarm.utils.logging import disable_logging, log_verbose

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    None: "null",
    Any: "object",
}


def function_to_json(
    func: Callable[..., Any],
    description: str | None = None,
) -> dict[str, Any]:
    """Convert Python function to OpenAI function schema.

    Automatically generates an OpenAI-compatible function description,
    excluding internal parameters like 'context_variables'. Handles
    type hints, docstrings, and parameter validation.

    Args:
        func: Function to convert.
        description: Optional override for function description.

    Returns:
        OpenAI function schema as a dictionary.

    Raises:
        ValueError: If function conversion fails.

    Examples:
        Basic function:
            ```python
            def calculate_sum(a: int, b: int) -> int:
                \"\"\"Add two numbers together.

                Args:
                    a: First number.
                    b: Second number.

                Returns:
                    Sum of the numbers.
                \"\"\"
                return a + b

            schema = function_to_json(calculate_sum)
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "calculate_sum",
            #         "description": "Add two numbers together.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "a": {
            #                     "type": "integer",
            #                     "description": "First number."
            #                 },
            #                 "b": {
            #                     "type": "integer",
            #                     "description": "Second number."
            #                 }
            #             },
            #             "required": ["a", "b"]
            #         }
            #     }
            # }
            ```

        Enum parameter:
            ```python
            class Color(Enum):
                RED = "red"
                BLUE = "blue"

            def paint(color: Color) -> str:
                \"\"\"Paint in specified color.

                Args:
                    color: Color to use.
                \"\"\"
                return f"Painting in {color.value}"

            schema = function_to_json(paint)
            # Parameters will include enum values:
            # "color": {
            #     "type": "string",
            #     "enum": ["red", "blue"],
            #     "description": "Color to use."
            # }
            ```
    """  # noqa: D214
    try:
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        type_hints = get_type_hints(func)

        # Parse docstring
        func_docstring = parse_docstring_params(docstring)
        func_description = description or func_docstring.description
        func_param_docs = func_docstring.parameters

        # Process parameters
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Skip context_variables (reserved for internal use)
            if param_name == "context_variables":
                continue

            param_type = type_hints.get(param_name, type(Any))
            param_desc = func_param_docs.get(param_name, "")

            # Build parameter schema
            param_schema: dict[str, Any] = {
                "type": TYPE_MAP.get(param_type, "string"),
                "description": param_desc if param_desc else f"Parameter: {param_name}",
            }

            # Handle enums
            if isinstance(param_type, type) and issubclass(param_type, Enum):
                param_schema["type"] = "string"
                param_schema["enum"] = [e.value for e in param_type]

            properties[param_name] = param_schema

            # Add to required if no default value
            if param.default == param.empty:
                required.append(param_name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }

        if required:
            schema["function"]["parameters"]["required"] = required

        return schema

    except Exception as e:
        log_verbose(f"Failed to convert function {func.__name__}: {str(e)}", level="ERROR")
        raise ValueError(f"Failed to convert function {func.__name__}: {str(e)}") from e


def parse_docstring_params(docstring: str) -> FunctionDocstring:
    """Parse function docstring into structured format.

    Uses Griffe to extract description and parameter documentation
    from docstrings in various styles (Google, Sphinx, NumPy).

    Args:
        docstring: Raw docstring text to parse.

    Returns:
        Structured docstring information.

    Examples:
        Google style:
            ```python
            def example(name: str) -> str:
                \"\"\"Greet someone.

                Args:
                    name: Person to greet.

                Returns:
                    Greeting message.
                \"\"\"
                return f"Hello {name}"

            info = parse_docstring_params(example.__doc__)
            # info.description = "Greet someone."
            # info.parameters = {"name": "Person to greet."}
            ```

        Sphinx style:
            ```python
            def example(name: str) -> str:
                \"\"\"Greet someone.

                :param name: Person to greet.
                :return: Greeting message.
                \"\"\"
                return f"Hello {name}"

            info = parse_docstring_params(example.__doc__)
            # Same structured output
            ```
    """  # noqa: D214
    if not docstring:
        return FunctionDocstring()

    try:
        with disable_logging():
            style = detect_docstring_style(docstring)
            docstring_parser = Docstring(docstring)
            parsed_docstring = docstring_parser.parse(parser=style)

        description = ""
        parameters: dict[str, str] = {}

        for section in parsed_docstring:
            match section.kind:
                case DocstringSectionKind.text:
                    section_dict = section.as_dict()
                    description = section_dict.get("value", "")

                case DocstringSectionKind.parameters:
                    section_dict = section.as_dict()
                    param_list = section_dict.get("value", [])

                    for param in param_list:
                        param_name = getattr(param, "name", None)
                        param_desc = getattr(param, "description", "")
                        if param_name:
                            parameters[param_name] = param_desc

                case _:
                    continue

        return FunctionDocstring(
            description=description,
            parameters=parameters,
        )

    except Exception as e:
        log_verbose(f"Failed to parse docstring: {e}", level="WARNING")
        return FunctionDocstring()


def detect_docstring_style(docstring: str) -> Literal["google", "sphinx", "numpy"]:
    """Detect docstring format using pattern matching.

    Analyzes docstring content to determine its style based on
    common patterns and section markers.

    Args:
        docstring: Raw docstring text to analyze.

    Returns:
        Detected style: "google", "sphinx", or "numpy".

    Examples:
        Google style detection:
            ```python
            style = detect_docstring_style(\"\"\"
                Do something.

                Args:
                    x: Input value.

                Returns:
                    Modified value.
            \"\"\")
            assert style == "google"
            ```

        Sphinx style detection:
            ```python
            style = detect_docstring_style(\"\"\"
                Do something.

                :param x: Input value.
                :return: Modified value.
            \"\"\")
            assert style == "sphinx"
            ```
    """  # noqa: D214
    if not docstring:
        return "google"  # default to google style

    # Google style indicators
    if "Args:" in docstring or "Returns:" in docstring or "Raises:" in docstring:
        return "google"

    # Sphinx style indicators
    if ":param" in docstring or ":return:" in docstring or ":rtype:" in docstring:
        return "sphinx"

    # NumPy style indicators
    if (
        "Parameters\n" in docstring
        or "Returns\n" in docstring
        or "Parameters\r\n" in docstring
        or "Returns\r\n" in docstring
    ):
        return "numpy"

    return "google"


def function_has_parameter(func: Callable[..., Any], param: str) -> bool:
    """Check if function accepts specific parameter.

    Inspects function's type hints to determine if it accepts
    the given parameter name.

    Args:
        func: Function to inspect.
        param: Parameter name to check.

    Returns:
        True if parameter exists, False otherwise.

    Examples:
        Parameter check:
            ```python
            def greet(name: str) -> str:
                return f"Hello {name}"

            assert function_has_parameter(greet, "name")
            assert not function_has_parameter(greet, "age")
            ```

        Type hints required:
            ```python
            def greet(name):  # No type hint
                return f"Hello {name}"

            # Returns False (no type hints)
            result = function_has_parameter(greet, "name")
            ```
    """
    return param in get_type_hints(func)


def functions_to_json(functions: list[Callable[..., Any]] | None) -> list[dict[str, Any]] | None:
    """Convert multiple functions to OpenAI schemas.

    Batch converts Python functions to OpenAI-compatible function
    descriptions, filtering out None values.

    Args:
        functions: List of functions to convert.

    Returns:
        List of function schemas, or None if input is None.

    Examples:
        Multiple functions:
            ```python
            def add(a: int, b: int) -> int:
                \"\"\"Add numbers.\"\"\"
                return a + b

            def multiply(a: int, b: int) -> int:
                \"\"\"Multiply numbers.\"\"\"
                return a * b

            schemas = functions_to_json([add, multiply])
            # Returns list of two function schemas
            ```

        Empty input:
            ```python
            assert functions_to_json(None) is None
            assert functions_to_json([]) is None
            ```
    """
    if not functions:
        return None

    return [function_to_json(func) for func in functions]
