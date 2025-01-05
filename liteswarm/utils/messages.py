# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Sequence
from typing import Any

import litellm
from litellm.utils import token_counter

from liteswarm.types.swarm import Message
from liteswarm.types.utils import MessageAdapter, MessageT, TrimMessagesResult
from liteswarm.utils.logging import log_verbose


def filter_tool_call_pairs(messages: list[MessageT]) -> list[MessageT]:
    """Filter message history to maintain valid tool interactions.

    Ensures message history contains only complete tool interactions by:
    - Keeping tool calls with matching results
    - Keeping tool results with matching calls
    - Removing orphaned tool calls or results
    - Preserving non-tool messages
    - Handling multiple tool calls in a single message
    - Preserving message order and relationships

    Args:
        messages: List of conversation messages to filter.

    Returns:
        Filtered message list with only complete tool interactions.

    Examples:
        Single Tool Call with Matching Result:
            ```python
            # Example 1: Single Tool Call with Matching Result
            messages = [
                Message(
                    role="user",
                    content="Calculate 2+2",
                ),
                Message(
                    role="assistant",
                    content="I will call the 'calculator' tool now.",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="calculator",
                            args={"expression": "2+2"},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="call_1",
                    content="4",
                ),
                Message(
                    role="assistant",
                    content="The result is 4.",
                ),
            ]
            filtered = filter_tool_call_pairs2(messages)
            # All messages are preserved because the tool call "call_1" has a matching result.
            ```

        Single Tool Call Missing a Result:
            ```python
            messages = [
                Message(
                    role="user",
                    content="Calculate 2+2",
                ),
                Message(
                    role="assistant",
                    content="I will call the 'calculator' tool now.",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="calculator",
                            args={"expression": "2+2"},
                        )
                    ],
                ),
                # No tool message for call_1
                Message(
                    role="assistant",
                    content="Done.",
                ),
            ]
            filtered = filter_tool_call_pairs2(messages)
            # The assistant message remains, but the orphaned tool call is pruned.
            ```

        Multiple Tool Calls (Partial Completion):
            ```python
            messages = [
                Message(
                    role="user",
                    content="Calculate 2+2 and then multiply 4 by 3",
                ),
                Message(
                    role="assistant",
                    content="Let me call two tools: add and multiply",
                    tool_calls=[
                        ToolCall(id="call_1", name="add", args={"x": 2, "y": 2}),
                        ToolCall(id="call_2", name="multiply", args={"x": 4, "y": 3}),
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="call_1",
                    content="4",
                ),
                # No tool result for call_2
                Message(
                    role="assistant",
                    content="The sum is 4. I haven't gotten the multiplication result yet, but let's move on.",
                ),
            ]
            filtered = filter_tool_call_pairs2(messages)
            # Only 'call_1' is kept. 'call_2' is orphaned and removed.
            ```

        Tool Result with No Matching Call:
            ```python
            messages = [
                Message(
                    role="assistant",
                    content="No tools called yet.",
                ),
                Message(
                    role="tool",
                    tool_call_id="call_999",
                    content="Orphaned result for call_999",
                ),
                Message(
                    role="assistant",
                    content="Something else.",
                ),
            ]
            filtered = filter_tool_call_pairs2(messages)
            # The orphaned tool message referencing 'call_999' is removed, preserving conversation flow.
            ```
    """
    if not messages:
        return []

    # First pass: identify valid tool call/result pairs
    tool_call_map: dict[str, MessageT] = {}
    tool_result_map: dict[str, MessageT] = {}

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.id:
                    tool_call_map[tool_call.id] = message
        elif message.role == "tool" and message.tool_call_id:
            tool_result_map[message.tool_call_id] = message

    # Find valid pairs
    valid_tool_ids = set(tool_call_map.keys()) & set(tool_result_map.keys())

    # Second pass: build filtered message list
    filtered_messages: list[MessageT] = []
    processed_tool_calls: set[str] = set()

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            # Filter tool calls to only include those with results
            valid_calls = [
                tool_call
                for tool_call in message.tool_calls
                if tool_call.id and tool_call.id in valid_tool_ids
            ]

            if valid_calls:
                # Create new message with only valid tool calls
                filtered_messages.append(
                    message.model_copy(
                        update={
                            "role": message.role,
                            "content": message.content,
                            "tool_calls": valid_calls,
                        },
                    )
                )

                # Track which tool calls we've processed
                processed_tool_calls.update(call.id for call in valid_calls if call.id)
            elif message.content:
                # Keep messages that have content even if their tool calls were invalid
                filtered_messages.append(
                    message.model_copy(
                        update={
                            "role": message.role,
                            "content": message.content,
                        },
                    )
                )

        elif message.role == "tool" and message.tool_call_id:
            # Only include tool results that match a valid and processed tool call
            tool_call_id = message.tool_call_id
            if tool_call_id in valid_tool_ids and tool_call_id in processed_tool_calls:
                filtered_messages.append(message)

        else:
            # Keep all non-tool-related messages
            filtered_messages.append(message)

    return filtered_messages


def trim_messages(
    messages: list[MessageT],
    model: str | None = None,
    trim_ratio: float = 0.75,
    max_tokens: int | None = None,
) -> TrimMessagesResult[MessageT]:
    """Trim message history to fit within model's context window.

    Implements intelligent message trimming that:
    - Preserves system messages by combining them
    - Maintains tool call/response pairs at the end
    - Keeps most recent messages when trimming
    - Respects model token limits
    - Provides available response tokens

    Args:
        messages: Message history to trim.
        model: Target model identifier (e.g., "gpt-4o").
        trim_ratio: Proportion of max tokens to target (0.0-1.0).
        max_tokens: Optional override for model's max tokens.

    Returns:
        TrimmedMessages containing the trimmed message list and available response tokens.

    Examples:
        Basic trimming:
            ```python
            # Trim with default settings (0.75 ratio)
            result = trim_messages(messages, "gpt-4o")
            trimmed_messages = result.messages
            response_tokens = result.response_tokens
            ```

        Custom ratio:
            ```python
            # More aggressive trimming
            result = trim_messages(
                messages=messages,
                model="gpt-4o",
                trim_ratio=0.5,  # Use only 50% of limit
            )
            ```

        Custom token limit:
            ```python
            # Trim to specific token count
            result = trim_messages(
                messages=messages,
                max_tokens=1000,  # Exact limit
                trim_ratio=1.0,  # Use full limit
            )
            ```
    """
    if not messages:
        return TrimMessagesResult(messages=[], response_tokens=0)

    try:
        # Get target token limit
        if max_tokens is None and model:
            model_max_tokens = get_max_tokens(model)
            if model_max_tokens:
                max_tokens = model_max_tokens

        if not max_tokens:
            return TrimMessagesResult(messages=messages, response_tokens=0)

        max_tokens = int(max_tokens * trim_ratio)

        # Extract and combine system messages
        system_messages: list[MessageT] = []
        non_system_messages: list[MessageT] = []
        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        combined_system: MessageT | None = None
        if system_messages:
            # Combine system messages
            combined_content = "\n".join(msg.content for msg in system_messages if msg.content)
            if combined_content:
                # Create new system message with combined content
                combined_system = system_messages[0].model_copy(
                    update={"content": combined_content}
                )

        # Handle tool messages at the end
        tool_messages: list[MessageT] = []
        remaining_messages: list[MessageT] = non_system_messages.copy()

        # Collect consecutive tool messages from the end
        while remaining_messages and remaining_messages[-1].role == "tool":
            tool_messages.insert(0, remaining_messages.pop())

        # Calculate current token usage
        messages_to_check: list[MessageT] = []
        if combined_system:
            messages_to_check.append(combined_system)
        messages_to_check.extend(remaining_messages)
        messages_to_check.extend(tool_messages)

        current_tokens = token_counter(
            model=model or "",
            messages=dump_messages(messages_to_check),
        )

        log_verbose(f"Current tokens: {current_tokens}, max tokens: {max_tokens}")

        # Return as is if under limit
        if current_tokens <= max_tokens:
            result = messages_to_check
            response_tokens = max_tokens - current_tokens if model else 0
            return TrimMessagesResult(messages=result, response_tokens=response_tokens)

        log_verbose(f"Trimming messages: current={current_tokens}, max={max_tokens}")

        # Keep removing oldest non-system messages until we're under limit
        while remaining_messages and current_tokens > max_tokens:
            # Remove oldest message
            remaining_messages.pop(0)

            # Recalculate tokens
            messages_to_check = []
            if combined_system:
                messages_to_check.append(combined_system)
            messages_to_check.extend(remaining_messages)
            messages_to_check.extend(tool_messages)

            current_tokens = token_counter(
                model=model or "",
                messages=dump_messages(messages_to_check),
            )

        # Build final message list
        result = messages_to_check
        response_tokens = max_tokens - current_tokens if model else 0
        return TrimMessagesResult(messages=result, response_tokens=response_tokens)

    except Exception as e:
        log_verbose(f"Error during message trimming: {str(e)}")
        return TrimMessagesResult(messages=messages, response_tokens=0)


def validate_messages(
    messages: Any,
    strict: bool = False,
) -> list[Message]:
    """Validate and convert message-compatible objects to Message objects.

    Performs type validation and conversion of message data using Pydantic.
    Handles various message formats including tool calls and responses.
    Ensures data consistency and type safety for the messaging system.

    Args:
        messages: Raw message data (JSON-like structures) to validate.
        strict: Whether to enforce strict validation rules.

    Returns:
        List of validated Message objects with proper typing.

    Raises:
        ValidationError: If message data fails validation rules.

    Examples:
        Basic messages:
            ```python
            messages = validate_messages(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            )
            assert all(isinstance(msg, Message) for msg in messages)
            ```

        Tool interactions:
            ```python
            messages = validate_messages(
                [
                    {
                        "role": "assistant",
                        "content": "Let me calculate that",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "calculator", "arguments": '{"x": 10, "y": 5}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "15", "tool_call_id": "call_1"},
                ]
            )
            ```

        Strict validation:
            ```python
            # This will raise ValidationError due to extra field
            messages = validate_messages(
                [
                    {
                        "role": "user",
                        "content": "Hello",
                        "extra_field": "value",  # Not allowed in strict mode
                    }
                ],
                strict=True,
            )
            ```
    """
    return MessageAdapter.validate_python(messages, strict=strict)


def dump_messages(
    messages: Sequence[Message],
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
) -> list[dict[str, Any]]:
    """Convert Message objects to dictionary format.

    Serializes Message objects into dictionary format suitable for:
    - API requests
    - Token counting
    - Storage/serialization
    - Cross-system compatibility

    Args:
        messages: Sequence of Message objects to convert.
        exclude_unset: Whether to exclude unset fields.
        exclude_defaults: Whether to exclude default fields.
        exclude_none: Whether to exclude fields with None values.

    Returns:
        List of message dictionaries with all relevant fields.

    Raises:
        ValidationError: If any message fails validation.

    Examples:
        Basic conversion:
            ```python
            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ]
            dumped = dump_messages(messages)
            assert all(isinstance(msg, dict) for msg in dumped)
            ```

        With tool calls:
            ```python
            messages = [
                Message(
                    role="assistant",
                    content="Calculating",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                        }
                    ],
                )
            ]
            dumped = dump_messages(messages)
            assert "tool_calls" in dumped[0]
            ```
    """
    return MessageAdapter.dump_python(
        list(messages),
        exclude_unset=exclude_unset,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
    )


def exceeds_token_limit(
    messages: Sequence[Message],
    model: str | None = None,
    max_tokens: int | None = None,
) -> bool:
    """Check if messages exceed model's token limit.

    Calculates total tokens in message history and compares
    against the model's maximum context window size.

    Args:
        messages: Message history to check.
        model: Model identifier to check against.
        max_tokens: Optional override for model's max tokens.

    Returns:
        True if history exceeds limit, False otherwise.

    Examples:
        Basic check:
            ```python
            history = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ]
            # Short conversation
            assert not exceeds_token_limit(history, "gpt-4")
            ```

        Long conversation:
            ```python
            long_history = [
                Message(role="user", content="Very long text..."),
                # ... many messages ...
            ]
            if exceeds_token_limit(long_history, "gpt-4"):
                # Trim history or switch to larger model
                trimmed = trim_messages(long_history, "gpt-4")
            ```
    """
    if max_tokens is None and model:
        max_tokens = get_max_tokens(model)

    if max_tokens is None:
        raise ValueError("Please provide a max_tokens value or a model name to check against")

    current_tokens = token_counter(
        model=model,
        messages=dump_messages(messages),
    )

    return current_tokens > max_tokens


def get_max_tokens(model: str) -> int:
    """Get the maximum token limit for a language model.

    Retrieves the maximum context window size for the specified model from
    litellm's model configuration. Handles different formats of token limits
    and ensures consistent integer output.

    Args:
        model: Model identifier (e.g., "gpt-4o", "text-embedding-3-small").

    Returns:
        Maximum number of tokens the model can process.

    Raises:
        ValueError: If the model is not registered with litellm or max_tokens is invalid.

    Examples:
        Basic usage:
            ```python
            max_tokens = get_max_tokens("gpt-4o")
            print(f"GPT-4o can handle up to {max_tokens} tokens")
            ```

        Error handling:
            ```python
            try:
                max_tokens = get_max_tokens("unknown-model")
            except ValueError as e:
                print(f"Invalid model: {e}")
            ```
    """
    model_cost: dict[str, Any] = litellm.model_cost
    model_info: dict[str, Any] = model_cost.get(model, {})
    if not model_info:
        raise ValueError(f"Model {model} is not registered with litellm")

    max_tokens = model_info.get("max_input_tokens", model_info["max_tokens"])
    if max_tokens is None:
        return 0
    if isinstance(max_tokens, str):
        return int(max_tokens)
    if isinstance(max_tokens, int):
        return max_tokens

    raise ValueError(f"Invalid max tokens value for model {model}: {max_tokens}")
