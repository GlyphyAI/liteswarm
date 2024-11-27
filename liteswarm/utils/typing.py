# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Sequence
from typing import Any, TypeVar, Union

T = TypeVar("T")


def union(types: Sequence[T]) -> Union[T]:  # noqa: UP007
    """Create a Union type from a sequence of types dynamically.

    This utility function creates a Union type from a sequence of types at runtime.
    It's useful when you need to create Union types dynamically based on a collection
    of types rather than specifying them statically.

    Args:
        types: A sequence of types to be combined into a Union type.
            The sequence can contain any valid Python types (classes, built-in types, etc.).

    Returns:
        A Union type combining all the provided types.

    Example:
        ```python
        # Create a Union type for int, str, and float
        number_types = [int, str, float]
        NumberUnion = union_type(number_types)  # Union[int, str, float]

        # Use in type hints
        def process_number(value: NumberUnion) -> None:
            pass

        # Create a Union type for custom classes
        class A: pass
        class B: pass
        custom_union = union_type([A, B])  # Union[A, B]
        ```

    Note:
        This function is particularly useful when working with dynamic type systems
        or when the set of types needs to be determined at runtime. For static type
        unions, it's recommended to use the standard `Union[T1, T2, ...]` syntax directly.
    """
    union: Any = Union[tuple(types)]  # noqa: UP007
    return union
