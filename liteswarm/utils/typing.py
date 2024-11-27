# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable, Sequence
from typing import Any, TypeGuard, TypeVar, Union, get_origin

from typing_extensions import TypeIs

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
        NumberUnion = union(number_types)  # Union[int, str, float]

        # Use in type hints
        def process_number(value: NumberUnion) -> None:
            pass

        # Create a Union type for custom classes
        class A: pass
        class B: pass
        custom_union = union([A, B])  # Union[A, B]
        ```

    Note:
        This function is particularly useful when working with dynamic type systems
        or when the set of types needs to be determined at runtime. For static type
        unions, it's recommended to use the standard `Union[T1, T2, ...]` syntax directly.
    """
    union: Any = Union[tuple(types)]  # noqa: UP007
    return union


def is_callable(obj: Any) -> TypeIs[Callable[..., Any]]:
    """Type guard to check if an object is a callable (function or method), excluding classes.

    Args:
        obj: Object to check.

    Returns:
        True if the object is a callable but not a class, False otherwise.
    """
    return callable(obj) and not isinstance(obj, type)


def is_subtype(obj: Any, obj_type: type[T]) -> TypeGuard[type[T]]:
    """Type guard to check if an object is a valid subclass of a target type.

    Args:
        obj: Object to check.
        obj_type: Target type to check against.

    Returns:
        True if the object is a valid subclass of the target type, False otherwise.
    """
    return (
        obj is not None
        and not get_origin(obj)
        and isinstance(obj, type)
        and issubclass(obj, obj_type)
    )
