import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Any, TypeVar, Union, get_type_hints

from litellm.exceptions import RateLimitError, ServiceUnavailableError

from liteswarm.exceptions import CompletionError
from liteswarm.types import Message

T = TypeVar("T")

logger = logging.getLogger(__name__)


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
            raise ValueError(f"Failed to convert function {func.__name__}: {str(e)}") from e

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


def filter_tool_call_pairs(messages: list[Message]) -> list[Message]:
    """Filter messages to maintain only complete tool call/result pairs.

    This utility function ensures that:
    1. Tool calls have corresponding tool results
    2. Tool results have corresponding tool calls
    3. Orphaned tool calls or results are filtered out

    Args:
        messages: List of messages to filter

    Returns:
        List of messages with only complete tool call/result pairs
    """
    # Find valid tool call/result pairs
    tool_call_ids = set()
    tool_result_ids = set()

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.id:
                    tool_call_ids.add(tool_call.id)
        elif message.role == "tool" and message.tool_call_id:
            tool_result_ids.add(message.tool_call_id)

    valid_tool_ids = tool_call_ids.intersection(tool_result_ids)

    # Filter messages to maintain valid tool call/result pairs
    filtered_messages = []

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            filtered_tool_calls = [
                tool_call for tool_call in message.tool_calls if tool_call.id in valid_tool_ids
            ]

            msg = Message(
                role=message.role,
                content=message.content,
                tool_calls=filtered_tool_calls or None,
            )

            filtered_messages.append(msg)
        elif message.role == "tool":
            if message.tool_call_id in valid_tool_ids:
                filtered_messages.append(message)
        else:
            filtered_messages.append(message)

    return filtered_messages


async def retry_with_exponential_backoff(
    operation: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
) -> T:
    """Execute an operation with exponential backoff retry logic.

    Args:
        operation: Async operation to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry

    Returns:
        The result of the operation if successful

    Raises:
        The last error encountered if all retries fail
    """
    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except (RateLimitError, ServiceUnavailableError) as e:
            last_error = e
            if attempt == max_retries:
                break

            # Calculate next delay with exponential backoff
            delay = min(delay * backoff_factor, max_delay)

            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1f seconds...",
                attempt + 1,
                max_retries + 1,
                str(e),
                delay,
            )

            await asyncio.sleep(delay)

    if last_error:
        error_type = last_error.__class__.__name__
        raise CompletionError(
            f"Operation failed after {max_retries + 1} attempts: {error_type}",
            last_error,
        )

    raise CompletionError("Operation failed with unknown error", Exception("Unknown error"))


def safe_get_attr(
    obj: Any,
    attr: str,
    expected_type: type[T],
    default: T | None = None,
) -> T | None:
    """Safely get and validate an attribute of an object.

    Args:
        obj: Object to get attribute from
        attr: Name of the attribute to get
        expected_type: Expected type of the attribute
        default: Default value to return if the attribute does not exist or is not of the expected type

    Returns:
        The attribute value if it exists and matches the expected type, None otherwise

    Example:
        usage = safe_get_attr(chunk, "usage", Usage)
        tool_call = safe_get_attr(delta, "tool_call", ChatCompletionDeltaToolCall)
    """
    value = getattr(obj, attr, default)
    return value if isinstance(value, expected_type) else default


def combine_dicts(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine two dictionaries by adding their numeric values and preserving non-numeric ones.

    Args:
        left: First dictionary (or None)
        right: Second dictionary (or None)

    Returns:
        Combined dictionary where:
        - Numeric values are added together
        - Non-numeric values from left dict are preserved
        - Keys unique to right dict are included
        Returns None if both inputs are None

    Example:
        >>> combine_dicts({"a": 1, "b": "text"}, {"a": 2, "c": 3.5})
        {"a": 3, "b": "text", "c": 3.5}
    """
    if left is None:
        return right

    if right is None:
        return left

    result = {}

    all_keys = set(left) | set(right)

    for key in all_keys:
        left_value = left.get(key)
        right_value = right.get(key)

        if isinstance(left_value, Number) and isinstance(right_value, Number):
            result[key] = left_value + right_value  # type: ignore
        elif key in left:
            result[key] = left_value
        else:
            result[key] = right_value

    return result
