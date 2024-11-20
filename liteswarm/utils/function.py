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
    """Convert a Python function to an OpenAI-compatible function description.

    The function will automatically exclude the `context_variables` parameter from
    the generated schema, as it's handled internally by the framework.

    Args:
        func: The function to convert
        description: Optional function description

    Returns:
        Dict containing the function schema in OpenAI's format

    Example:
        ```python
        def greet(name: str, context_variables: ContextVariables) -> str:
            \"\"\"Greet someone by name.

            Args:
                name: The name to greet
                context_variables: Context variables (handled by framework)
            \"\"\"
            return f"Hello, {name}!"

        schema = function_to_json(greet)
        # Schema will only include the 'name' parameter
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
    """Extract parameter descriptions from docstring using Griffe.

    Args:
        docstring: The docstring to parse.

    Returns:
        FunctionDocstring: Parsed docstring information.
    """
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
    """Detect the style of a docstring using heuristics.

    Args:
        docstring: The docstring to analyze.

    Returns:
        str: The detected style ("google", "sphinx", or "numpy").
    """
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
    """Check if a function has a specific parameter.

    Args:
        func: The function to check
        param: The parameter to check for

    Returns:
        True if the function has the parameter, False otherwise
    """
    return param in get_type_hints(func)
