# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Generator, ItemsView, Iterable, KeysView
from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, PrivateAttr

ReservedContextKey: TypeAlias = Literal["output_format"]
"""Type for reserved context keys that have special meaning."""

RESERVED_CONTEXT_KEYS: set[ReservedContextKey] = set(get_args(ReservedContextKey))
"""Set of reserved context keys that cannot be used by agents."""


class ContextVariables(BaseModel):
    """Type for context variables passed to agents.

    This class provides a safe way to work with context variables while protecting
    reserved keys that have special meaning in the system.
    """

    _data: dict[str, Any] = PrivateAttr(default_factory=dict)
    _reserved: dict[str, Any] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize context variables.

        Args:
            **kwargs: Arbitrary keyword arguments to store as context variables.

        Raises:
            ValueError: If any of the provided keys are reserved system keys.
        """
        super().__init__()
        self._data = {}
        self._reserved = {}

        input_keys = set(kwargs.keys())
        reserved_overlap = input_keys & RESERVED_CONTEXT_KEYS
        if reserved_overlap:
            raise ValueError(f"Context variables cannot use reserved keys: {reserved_overlap}")

        self._data.update(kwargs)

    def __getitem__(self, key: str) -> Any:
        """Get a value from context variables.

        Args:
            key: The key to retrieve.

        Raises:
            ValueError: If the key is reserved.
            KeyError: If the key does not exist.

        Returns:
            The value associated with the key.
        """
        if key in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"Cannot access reserved key: {key}")

        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value in context variables.

        Args:
            key: The key to set.
            value: The value to associate with the key.

        Raises:
            ValueError: If the key is reserved.
        """
        if key in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"Cannot set reserved key: {key}")

        self._data[key] = value

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        """Iterate over non-reserved keys only.

        This allows the ContextVariables to be unpacked with **, excluding reserved keys.

        Returns:
            An iterator over non-reserved keys.
        """
        yield from self._data.items()

    def __len__(self) -> int:
        """Get the number of context variables (excluding reserved).

        Returns:
            The number of non-reserved context variables.
        """
        return len(self._data)

    def __str__(self) -> str:
        """Return a string representation of the context variables.

        Returns:
            A string representation of the context variables.
        """
        return str(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the context variables.

        Returns:
            A string representation of the context variables.
        """
        return str(self._data)

    def keys(self) -> KeysView[str]:
        """Return a view of non-reserved keys.

        Returns:
            A KeysView of non-reserved keys.
        """
        return self._data.keys()

    def items(self) -> ItemsView[str, Any]:
        """Return a view of non-reserved items.

        Returns:
            An ItemsView of non-reserved key-value pairs.
        """
        return self._data.items()

    def all_items(self) -> ItemsView[str, Any]:
        """Get all items including reserved keys.

        Useful for scenarios where both data and reserved keys are needed.

        Returns:
            An ItemsView of all key-value pairs.
        """
        combined = self._data | self._reserved
        return combined.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default.

        Args:
            key: The key to retrieve.
            default: The default value if the key is not found.

        Returns:
            The value associated with the key or the default value.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def set_reserved(self, key: ReservedContextKey, value: Any) -> None:
        """Set a reserved context key.

        This method allows system code to set reserved keys directly in the context.

        Args:
            key: The reserved key to set.
            value: The value to set for the reserved key.

        Raises:
            ValueError: If the key is not a reserved key.
        """
        if key not in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"Key must be a reserved key: {key}")

        self._reserved[key] = value

    def get_reserved(self, key: ReservedContextKey, default: Any = None) -> Any:
        """Get a reserved context key value.

        Args:
            key: The reserved key to retrieve.
            default: The default value if the key is not found.

        Returns:
            The value of the reserved key or the default value.
        """
        return self._reserved.get(key, default)

    def update(
        self,
        other: Iterable[tuple[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update multiple non-reserved context variables at once.

        Args:
            other: A mapping of key-value pairs to update.
            **kwargs: Additional key-value pairs to update.

        Raises:
            ValueError: If any keys in `other` or `kwargs` are reserved.
        """
        if other:
            for key, _ in other:
                if key in RESERVED_CONTEXT_KEYS:
                    raise ValueError(f"Cannot set reserved key through update: {key}")

            self._data.update(other)

        if kwargs:
            input_keys = set(kwargs.keys())
            reserved_overlap = input_keys & RESERVED_CONTEXT_KEYS
            if reserved_overlap:
                raise ValueError(f"Context variables cannot use reserved keys: {reserved_overlap}")

            self._data.update(kwargs)

    def update_reserved(
        self,
        other: Iterable[tuple[ReservedContextKey, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update multiple reserved context variables at once.

        Args:
            other: A mapping of reserved key-value pairs to update.
            **kwargs: Additional reserved key-value pairs to update.

        Raises:
            ValueError: If any keys in `other` or `kwargs` are not reserved.
        """
        if other:
            for key, _ in other:
                if key not in RESERVED_CONTEXT_KEYS:
                    raise ValueError(
                        f"Only reserved keys can be set through update_reserved: {key}"
                    )

            self._reserved.update(other)

        if kwargs:
            input_keys = set(kwargs.keys())
            non_reserved = input_keys - RESERVED_CONTEXT_KEYS
            if non_reserved:
                raise ValueError(
                    f"Only reserved keys can be set through update_reserved: {non_reserved}"
                )

            self._reserved.update(kwargs)
