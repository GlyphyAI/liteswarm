# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from litellm.utils import get_max_tokens, token_counter
from litellm.utils import trim_messages as litellm_trim_messages

from liteswarm.types.swarm import Message


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
