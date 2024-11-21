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
