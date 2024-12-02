# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from numbers import Number
from typing import Any

from litellm import Usage
from litellm.cost_calculator import cost_per_token

from liteswarm.types.swarm import ResponseCost
from liteswarm.utils.misc import safe_get_attr


def combine_dicts(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge dictionaries with special handling for numbers.

    Creates a new dictionary by combining two input dictionaries:
    - Adds numeric values when keys overlap
    - Preserves non-numeric values from left dictionary
    - Includes unique keys from both dictionaries

    Args:
        left: First dictionary to merge.
        right: Second dictionary to merge.

    Returns:
        Combined dictionary or None if both inputs are None.

    Examples:
        Basic merge:
            ```python
            result = combine_dicts(
                {"a": 1, "b": "text"},
                {"a": 2, "c": 3}
            )
            assert result == {
                "a": 3,        # Numbers added
                "b": "text",   # Non-numeric preserved
                "c": 3         # Unique key included
            }
            ```

        None handling:
            ```python
            assert combine_dicts(None, None) is None
            assert combine_dicts({"a": 1}, None) == {"a": 1}
            assert combine_dicts(None, {"b": 2}) == {"b": 2}
            ```

        Mixed types:
            ```python
            result = combine_dicts(
                {"count": 1, "name": "test"},
                {"count": 2, "name": "other"}
            )
            assert result == {
                "count": 3,     # Numbers added
                "name": "test"  # Left value preserved
            }
            ```
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
    """Merge two LiteLLM usage statistics.

    Combines token counts and details from two Usage objects:
    - Adds all token counts (prompt, completion, total)
    - Merges token details dictionaries
    - Handles optional fields and None values

    Args:
        left: First Usage object to merge.
        right: Second Usage object to merge.

    Returns:
        Combined Usage object or None if both inputs are None.

    Examples:
        Basic merge:
            ```python
            usage1 = Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
            usage2 = Usage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30
            )
            total = combine_usage(usage1, usage2)
            assert total.prompt_tokens == 30
            assert total.completion_tokens == 15
            assert total.total_tokens == 45
            ```

        With details:
            ```python
            usage1 = Usage(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details={
                    "system": 3,
                    "user": 7
                }
            )
            usage2 = Usage(
                prompt_tokens=15,
                completion_tokens=8,
                prompt_tokens_details={
                    "system": 5,
                    "user": 10
                }
            )
            total = combine_usage(usage1, usage2)
            assert total.prompt_tokens_details == {
                "system": 8,
                "user": 17
            }
            ```
    """
    if left is None:
        return right

    if right is None:
        return left

    prompt_tokens = (left.prompt_tokens or 0) + (right.prompt_tokens or 0)
    completion_tokens = (left.completion_tokens or 0) + (right.completion_tokens or 0)
    total_tokens = (left.total_tokens or 0) + (right.total_tokens or 0)

    lhs_reasoning_tokens = safe_get_attr(left, "reasoning_tokens", int, default=0)
    rhs_reasoning_tokens = safe_get_attr(right, "reasoning_tokens", int, default=0)
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
    """Merge cost information from two responses.

    Combines prompt and completion token costs from two
    ResponseCost objects, handling None values appropriately.

    Args:
        left: First ResponseCost to merge.
        right: Second ResponseCost to merge.

    Returns:
        Combined ResponseCost or None if both inputs are None.

    Examples:
        Basic merge:
            ```python
            cost1 = ResponseCost(
                prompt_tokens_cost=0.001,
                completion_tokens_cost=0.002
            )
            cost2 = ResponseCost(
                prompt_tokens_cost=0.003,
                completion_tokens_cost=0.004
            )
            total = combine_response_cost(cost1, cost2)
            assert total.prompt_tokens_cost == 0.004
            assert total.completion_tokens_cost == 0.006
            ```

        None handling:
            ```python
            assert combine_response_cost(None, None) is None

            cost = ResponseCost(
                prompt_tokens_cost=0.001,
                completion_tokens_cost=0.002
            )
            assert combine_response_cost(cost, None) == cost
            assert combine_response_cost(None, cost) == cost
            ```
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
    """Calculate API cost for model usage.

    Computes the cost of API usage based on the model type
    and token counts, using LiteLLM's pricing data.

    Args:
        model: Model identifier (e.g., "gpt-4o").
        usage: Token usage statistics.

    Returns:
        Cost breakdown for prompt and completion tokens.

    Examples:
        GPT-4o cost:
            ```python
            usage = Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
            cost = calculate_response_cost("gpt-4o", usage)
            # Costs based on current pricing:
            # prompt: $2.50/1M tokens
            # completion: $10.00/1M tokens
            assert cost.prompt_tokens_cost == 0.00025
            assert cost.completion_tokens_cost == 0.0005
            ```

        GPT-4o-mini cost:
            ```python
            usage = Usage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500
            )
            cost = calculate_response_cost(
                "gpt-4o-mini",
                usage
            )
            # Costs based on current pricing:
            # prompt: $0.150/1M tokens
            # completion: $0.600/1M tokens
            assert cost.prompt_tokens_cost == 0.00015
            assert cost.completion_tokens_cost == 0.0003
            ```
    """
    prompt_tokens_cost, completion_tokens_cost = cost_per_token(
        model=model,
        usage_object=usage,
    )

    return ResponseCost(
        prompt_tokens_cost=prompt_tokens_cost,
        completion_tokens_cost=completion_tokens_cost,
    )
