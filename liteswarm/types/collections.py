# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import KeysView
from typing import Protocol, TypeVar

_KeyType = TypeVar("_KeyType")
"""Type variable representing the type of keys in a mapping-like structure.

This type variable is used to specify the type of keys that can be used
to access values within objects adhering to the `SupportsKeysAndGetItem` protocol.
"""

_ValueType_co = TypeVar("_ValueType_co", covariant=True)
"""Covariant type variable representing the type of values in a mapping-like structure.

This type variable is used to specify the type of values associated with keys
in objects adhering to the `SupportsKeysAndGetItem` protocol. The covariance
allows for more flexible type relationships, enabling subclasses to be used
where superclasses are expected.
"""


class SupportsKeysAndGetItem(Protocol[_KeyType, _ValueType_co]):
    """Protocol for objects that support key-based access similar to mappings.

    This protocol defines the minimal interface required for an object to be
    considered as supporting key-based retrieval of items. It is intended to
    represent objects that behave like dictionaries or other mapping types,
    providing access to keys and corresponding values.

    Args:
        _KeyType: The type of keys used in the mapping. This allows for flexibility
            in the types of keys that can be used (e.g., `str`, `int`, etc.).
        _ValueType_co: The type of values associated with the keys. The covariant
            nature of this type variable allows for subtype compatibility, meaning
            that a protocol expecting a certain value type can accept a more specific
            subtype.
    """

    def keys(self) -> KeysView[_KeyType]:
        """Return a view object of the mapping's keys.

        Returns:
            A view object containing all keys in the mapping.
        """
        ...

    def __getitem__(self, key: _KeyType) -> _ValueType_co:
        """Retrieve the value associated with the given key.

        Args:
            key: The key whose associated value is to be returned.

        Raises:
            KeyError: If the key does not exist in the mapping.

        Returns:
            The value associated with the specified key.
        """
        ...
