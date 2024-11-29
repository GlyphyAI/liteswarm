# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from litellm.utils import get_max_tokens, token_counter
from litellm.utils import trim_messages as litellm_trim_messages

from liteswarm.types.swarm import Message


def filter_tool_call_pairs(messages: list[Message]) -> list[Message]:
    """Filter message history to maintain valid tool interactions.

    Ensures message history contains only complete tool interactions by:
    - Keeping tool calls with matching results
    - Keeping tool results with matching calls
    - Removing orphaned tool calls or results
    - Preserving non-tool messages

    Args:
        messages: List of conversation messages to filter.

    Returns:
        Filtered message list with only complete tool interactions.

    Examples:
        Complete interaction:
            ```python
            messages = [
                Message(role="user", content="Calculate 2+2"),
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="call_1", name="calculate")
                    ]
                ),
                Message(
                    role="tool",
                    tool_call_id="call_1",
                    content="4"
                )
            ]
            filtered = filter_tool_call_pairs(messages)
            # All messages preserved (complete interaction)
            ```

        Orphaned call:
            ```python
            messages = [
                Message(role="user", content="Calculate 2+2"),
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="call_1", name="calculate")
                    ]
                )
                # No tool result
            ]
            filtered = filter_tool_call_pairs(messages)
            # Tool call removed (incomplete interaction)
            ```
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
    """Convert message objects to serializable dictionaries.

    Transforms Message objects into plain dictionaries suitable for:
    - JSON serialization
    - API requests
    - Storage/persistence

    Args:
        messages: List of Message objects to convert.

    Returns:
        List of serializable dictionaries.

    Examples:
        Basic conversion:
            ```python
            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!")
            ]
            dicts = dump_messages(messages)
            # [
            #     {"role": "user", "content": "Hello"},
            #     {"role": "assistant", "content": "Hi!"}
            # ]
            ```

        Tool messages:
            ```python
            messages = [
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="123", name="search")
                    ]
                ),
                Message(
                    role="tool",
                    tool_call_id="123",
                    content="Results found"
                )
            ]
            dicts = dump_messages(messages)
            # Converts to API-compatible format
            ```
    """
    return [message.model_dump(exclude_none=True) for message in messages]


def load_messages(dicts: list[dict[str, Any]], strict: bool = False) -> list[Message]:
    """Convert dictionaries back to Message objects.

    Parses dictionaries into Message objects with optional
    strict validation of the input format.

    Args:
        dicts: List of message dictionaries to convert.
        strict: Whether to enforce strict schema validation.

    Returns:
        List of Message objects.

    Examples:
        Basic loading:
            ```python
            dicts = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
            messages = load_messages(dicts)
            # [
            #     Message(role="user", content="Hello"),
            #     Message(role="assistant", content="Hi!")
            # ]
            ```

        Strict validation:
            ```python
            dicts = [
                {"role": "invalid", "content": "Bad"}
            ]
            try:
                messages = load_messages(dicts, strict=True)
            except ValidationError:
                print("Invalid message format")
            ```
    """
    return [Message.model_validate(dict, strict=strict) for dict in dicts]


def trim_messages(messages: list[Message], model: str | None = None) -> list[Message]:
    """Trim message history to fit model's context window.

    Reduces message history to fit within the model's maximum token
    limit while preserving the most recent and relevant messages.

    Args:
        messages: Message history to trim.
        model: Target model identifier (e.g., "gpt-4o").

    Returns:
        Trimmed message list that fits the model's limits.

    Examples:
        Basic trimming:
            ```python
            history = [
                Message(role="user", content="Long conversation..."),
                # ... many messages ...
                Message(role="user", content="Latest message")
            ]
            trimmed = trim_messages(history, "gpt-4o")
            # Returns subset of messages fitting GPT-4o's context
            ```

        Model-specific:
            ```python
            # GPT-4o (larger context)
            gpt4o_msgs = trim_messages(history, "gpt-4o")

            # GPT-3.5 (smaller context)
            gpt3_5_msgs = trim_messages(history, "gpt-3.5-turbo")
            assert len(gpt3_5_msgs) <= len(gpt4o_msgs)
            ```
    """
    dict_messages = dump_messages(messages)
    trimmed_messages = litellm_trim_messages(dict_messages, model)
    if isinstance(trimmed_messages, tuple):
        trimmed_messages = trimmed_messages[0]

    return load_messages(trimmed_messages)


def history_exceeds_token_limit(messages: list[Message], model: str) -> bool:
    """Check if message history exceeds model's token limit.

    Calculates total tokens in message history and compares
    against the model's maximum context window size.

    Args:
        messages: Message history to check.
        model: Model identifier to check against.

    Returns:
        True if history exceeds limit, False otherwise.

    Examples:
        Basic check:
            ```python
            history = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!")
            ]
            # Short conversation
            assert not history_exceeds_token_limit(
                history,
                "gpt-4o"
            )
            ```

        Long conversation:
            ```python
            long_history = [
                Message(role="user", content="Very long text..."),
                # ... many messages ...
            ]
            if history_exceeds_token_limit(long_history, "gpt-3.5-turbo"):
                # Trim history or switch to larger model
                trimmed = trim_messages(long_history, "gpt-3.5-turbo")
            ```
    """
    max_tokens = get_max_tokens(model)
    if max_tokens is None:
        return False

    dict_messages = dump_messages(messages)
    history_tokens = token_counter(model, messages=dict_messages)

    return history_tokens > max_tokens
