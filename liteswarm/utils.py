import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, get_type_hints


@dataclass
class Parameter:
    type: str
    description: str | None
    required: bool
    enum_values: list[str] | None = None
    default: Any = None


class FunctionConverter:
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

    @classmethod
    def function_to_json(
        cls,
        func: Callable[..., Any],
        description: str | None = None,
    ) -> dict[str, Any]:
        """Convert a Python function to an OpenAI-compatible function description.

        Args:
            func: The function to convert
            description: Optional function description

        Returns:
            Dict containing the function schema wrapped in the expected structure
        """
        try:
            signature = inspect.signature(func)
            doc = inspect.getdoc(func) or ""
            type_hints = get_type_hints(func)

            parameters = cls._process_parameters(signature, type_hints, doc)

            # Construct the inner function schema
            function_schema = {
                "name": func.__name__,
                "description": description or doc.split("\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": parameters["properties"],
                    "required": parameters["required"],
                },
            }

            # Remove empty required list
            if not function_schema["parameters"]["required"]:
                del function_schema["parameters"]["required"]

            # Wrap with the outer structure
            wrapped_schema = {"type": "function", "function": function_schema}

            return wrapped_schema

        except ValueError as e:
            raise ValueError(
                f"Failed to convert function {func.__name__}: {str(e)}"
            ) from e

    @classmethod
    def _process_parameters(
        cls,
        signature: inspect.Signature,
        type_hints: dict[str, type],
        docstring: str,
    ) -> dict[str, Any]:
        """Process function parameters and extract their schemas.

        Args:
            signature: The function signature
            type_hints: The function type hints
            docstring: The function docstring

        Returns:
            Dict containing the parameter schema
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Parse docstring for parameter descriptions
        param_docs = cls._parse_docstring_params(docstring)

        for param_name, param in signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            param_type = type_hints.get(param_name, type(Any))
            param_schema = cls._get_parameter_schema(
                param_name,
                param_type,
                param,
                param_docs.get(param_name, ""),
            )

            properties[param_name] = param_schema

            # Add to required list if no default value
            if param.default == param.empty:
                required.append(param_name)

        return {"properties": properties, "required": required}

    @classmethod
    def _get_parameter_schema(
        cls,
        name: str,
        param_type: type,
        param: inspect.Parameter,
        description: str,
    ) -> dict[str, Any]:
        """Convert a single parameter to its JSON schema.

        Args:
            name: The parameter name
            param_type: The parameter type
            param: The parameter object
            description: The parameter description

        Returns:
            Dict containing the parameter schema
        """
        schema: dict[str, Any] = {"description": description} if description else {}

        # Handle Union types (e.g., Optional[type])
        if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
            types = [t for t in param_type.__args__ if t is not type(None)]
            if len(types) == 1:
                param_type = types[0]
            else:
                # Multiple possible types
                schema["type"] = [cls.TYPE_MAP.get(t, "string") for t in types]
                return schema

        # Handle Enum types
        if isinstance(param_type, type) and issubclass(param_type, Enum):
            schema["type"] = "string"
            schema["enum"] = [e.value for e in param_type]
            return schema

        # Handle List types
        if hasattr(param_type, "__origin__") and param_type.__origin__ is list:
            schema["type"] = "array"
            item_type = param_type.__args__[0] if param_type.__args__ else Any
            schema["items"] = {"type": cls.TYPE_MAP.get(item_type, "string")}
            return schema

        # Handle Dict types
        if hasattr(param_type, "__origin__") and param_type.__origin__ is dict:
            schema["type"] = "object"
            return schema

        # Handle basic types
        schema["type"] = cls.TYPE_MAP.get(param_type, "string")

        # Add default value if exists
        if param.default != param.empty:
            schema["default"] = param.default

        return schema

    @staticmethod
    def _parse_docstring_params(docstring: str) -> dict[str, str]:
        """Extract parameter descriptions from docstring.

        Args:
            docstring: The docstring to parse

        Returns:
            Dict containing the parameter descriptions
        """
        param_docs: dict[str, str] = {}
        if not docstring:
            return param_docs

        lines = docstring.split("\n")
        current_param = None
        current_desc = []

        for line in lines:
            line = line.strip()  # noqa: PLW2901
            if line.startswith(":param"):
                if current_param:
                    param_docs[current_param] = " ".join(current_desc).strip()
                current_desc = []
                parts = line.split(":", 2)
                if len(parts) >= 3:  # noqa: PLR2004
                    current_param = parts[1].replace("param ", "").strip()
                    current_desc.append(parts[2].strip())
            elif current_param and line and not line.startswith(":"):
                current_desc.append(line)

        if current_param:
            param_docs[current_param] = " ".join(current_desc).strip()

        return param_docs


def function_to_json(
    func: Callable[..., Any],
    description: str | None = None,
) -> dict[str, Any]:
    """Convert a Python function to an OpenAI-compatible function description.

    Args:
        func: The function to convert
        description: Optional function description

    Returns:
        Dict containing the function schema
    """
    return FunctionConverter.function_to_json(func, description)
