import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Any, Literal, TypeVar, get_type_hints

from griffe import Docstring, DocstringSectionKind
from litellm import Usage
from litellm.cost_calculator import cost_per_token
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from litellm.utils import get_max_tokens, token_counter
from litellm.utils import trim_messages as litellm_trim_messages

from liteswarm.exceptions import CompletionError
from liteswarm.logging import log_verbose
from liteswarm.types import FunctionDocstring, Message, ResponseCost

T = TypeVar("T")


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

        Follows OpenAI's function calling format:
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }

        Args:
            func: The function to convert
            description: Optional function description

        Returns:
            Dict containing the OpenAI-compatible function schema
        """
        try:
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            type_hints = get_type_hints(func)

            # Parse docstring
            func_docstring = cls._parse_docstring_params(docstring)
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
                    "type": cls.TYPE_MAP.get(param_type, "string"),
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

            schema = {
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
        func_docstring = cls._parse_docstring_params(docstring)
        func_param_docs = func_docstring.parameters

        for param_name, param in signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            param_type = type_hints.get(param_name, type(Any))
            param_schema = cls._get_parameter_schema(
                param_name,
                param_type,
                param,
                func_param_docs.get(param_name, ""),
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

    @classmethod
    def _parse_docstring_params(cls, docstring: str) -> FunctionDocstring:
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
                style = cls._detect_docstring_style(docstring)
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

    @classmethod
    def _detect_docstring_style(cls, docstring: str) -> Literal["google", "sphinx", "numpy"]:
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


def function_has_parameter(func: Callable[..., Any], param: str) -> bool:
    """Check if a function has a specific parameter.

    Args:
        func: The function to check
        param: The parameter to check for

    Returns:
        True if the function has the parameter, False otherwise
    """
    return param in get_type_hints(func)


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


def dump_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Dump messages to a list of dictionaries.

    Args:
        messages: List of messages to dump

    Returns:
        List of dictionaries
    """
    return [message.model_dump(exclude_none=True) for message in messages]


def load_messages(dicts: list[dict[str, Any]], strict: bool = False) -> list[Message]:
    """Load messages from a list of dictionaries.

    Args:
        dicts: List of dictionaries to load
        strict: Whether to use strict validation

    Returns:
        List of messages
    """
    return [Message.model_validate(dict, strict=strict) for dict in dicts]


def trim_messages(messages: list[Message], model: str | None = None) -> list[Message]:
    """Trim messages to the maximum token limit for the model.

    Args:
        messages: List of messages to trim
        model: The model to use for trimming

    Returns:
        List of trimmed messages
    """
    dict_messages = dump_messages(messages)
    trimmed_messages = litellm_trim_messages(dict_messages, model)
    if isinstance(trimmed_messages, tuple):
        trimmed_messages = trimmed_messages[0]

    return load_messages(trimmed_messages)


def history_exceeds_token_limit(messages: list[Message], model: str) -> bool:
    """Check if the history exceeds the token limit for the model.

    Args:
        messages: List of messages to check
        model: The model to check against

    Returns:
        True if the history exceeds the token limit, False otherwise
    """
    max_tokens = get_max_tokens(model)
    if max_tokens is None:
        return False

    dict_messages = dump_messages(messages)
    history_tokens = token_counter(model, messages=dict_messages)

    return history_tokens > max_tokens


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

            log_verbose(
                "Attempt %d/%d failed: %s. Retrying in %.1f seconds...",
                attempt + 1,
                max_retries + 1,
                str(e),
                delay,
                level="WARNING",
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


def combine_usage(left: Usage | None, right: Usage | None) -> Usage | None:
    """Combine two Usage objects by adding their token counts.

    This function handles:
    1. Cases where either usage is None
    2. Addition of all token counts
    3. Preservation of token details if present

    Args:
        left: First Usage object (or None)
        right: Second Usage object (or None)

    Returns:
        Combined Usage object, or None if both inputs are None

    Example:
        >>> total = combine_usage(response1.usage, response2.usage)
        >>> print(f"Total tokens: {total.total_tokens if total else 0}")
    """
    if left is None:
        return right

    if right is None:
        return left

    prompt_tokens = (left.prompt_tokens or 0) + (right.prompt_tokens or 0)
    completion_tokens = (left.completion_tokens or 0) + (right.completion_tokens or 0)
    total_tokens = (left.total_tokens or 0) + (right.total_tokens or 0)

    lhs_reasoning_tokens = safe_get_attr(left, "reasoning_tokens", int) or 0
    rhs_reasoning_tokens = safe_get_attr(right, "reasoning_tokens", int) or 0
    reasoning_tokens = lhs_reasoning_tokens + rhs_reasoning_tokens

    lhs_completion_tokens_details = safe_get_attr(left, "completion_tokens_details", dict)
    rhs_completion_tokens_details = safe_get_attr(right, "completion_tokens_details", dict)
    completion_tokens_details = combine_dicts(
        lhs_completion_tokens_details,
        rhs_completion_tokens_details,
    )

    lhs_prompt_tokens_details = safe_get_attr(left, "prompt_tokens_details", dict)
    rhs_prompt_tokens_details = safe_get_attr(right, "prompt_tokens_details", dict)
    prompt_tokens_details = combine_dicts(
        lhs_prompt_tokens_details,
        rhs_prompt_tokens_details,
    )

    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        completion_tokens_details=completion_tokens_details,
        prompt_tokens_details=prompt_tokens_details,
    )


def combine_response_cost(
    left: ResponseCost | None,
    right: ResponseCost | None,
) -> ResponseCost | None:
    """Combine two ResponseCost objects by adding their costs.

    Args:
        left: First ResponseCost object (or None)
        right: Second ResponseCost object (or None)

    Returns:
        Combined ResponseCost object, or None if both inputs are None
    """
    if left is None:
        return right

    if right is None:
        return left

    return ResponseCost(
        prompt_tokens_cost=left.prompt_tokens_cost + right.prompt_tokens_cost,
        completion_tokens_cost=left.completion_tokens_cost + right.completion_tokens_cost,
    )


def calculate_response_cost(model: str, usage: Usage) -> ResponseCost:
    """Calculate the cost of a response based on the usage object.

    Args:
        model: The model used for the response
        usage: The usage object for the response

    Returns:
        The cost of the response
    """
    prompt_tokens_cost, completion_tokens_cost = cost_per_token(
        model=model,
        usage_object=usage,
    )

    return ResponseCost(
        prompt_tokens_cost=prompt_tokens_cost,
        completion_tokens_cost=completion_tokens_cost,
    )


@contextmanager
def disable_logging() -> Generator[None, None, None]:
    """Disable logging for the duration of the context manager.

    Example:
        ```python
        with disable_logging():
            logging.info("This will not be printed")

        logging.info("This will be printed")
        ```
    """
    old_level = logging.root.getEffectiveLevel()
    logging.root.setLevel(logging.CRITICAL + 1)
    yield
    logging.root.setLevel(old_level)
