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
    """Type guard for identifying callable objects, excluding class types.

    This function checks if an object is callable (like functions or methods) while
    specifically excluding class types, which are also technically callable.

    Args:
        obj: Object to check for callability.

    Returns:
        True if the object is a callable but not a class type, False otherwise.

    Example:
        ```python
        def my_func(): pass
        class MyClass: pass

        is_callable(my_func)  # Returns True
        is_callable(MyClass)  # Returns False
        is_callable(print)    # Returns True
        ```
    """
    return callable(obj) and not isinstance(obj, type)


def is_subtype(obj: Any, obj_type: type[T]) -> TypeGuard[type[T]]:
    """Type guard for validating subclass relationships.

    This function performs a comprehensive check to ensure an object is a valid
    subclass of a target type, handling edge cases like None values and generic types.

    Args:
        obj: Object to check for subclass relationship.
        obj_type: Target type to validate against.

    Returns:
        True if obj is a valid subtype of obj_type, False otherwise.

    Example:
        ```python
        class Animal: pass
        class Dog(Animal): pass

        is_subtype(Dog, Animal)     # Returns True
        is_subtype(str, Animal)     # Returns False
        is_subtype(None, Animal)    # Returns False
        ```
    """
    return (
        obj is not None
        and not get_origin(obj)
        and isinstance(obj, type)
        and issubclass(obj, obj_type)
    )
