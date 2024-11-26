# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, MutableMapping
from typing import Any, Literal, TypeAlias, get_args

from liteswarm.types.collections import SupportsKeysAndGetItem

ReservedContextKey: TypeAlias = Literal["response_format"]
"""Literal type for reserved context keys with special system significance."""

RESERVED_CONTEXT_KEYS: set[ReservedContextKey] = set(get_args(ReservedContextKey))
"""Set of keys reserved by the system from unintended use by agents."""


class ContextVariables(MutableMapping[str, Any]):
    """Manages context variables with protection for reserved system keys.

    This class provides a secure way to handle context variables, ensuring that
    reserved keys are protected from unintended modifications by agents. It behaves
    like a standard mutable mapping for non-reserved keys while allowing controlled
    access and modification of reserved keys through dedicated methods.

    Examples:
        **Basic Usage:**

        ```python
        ctx = ContextVariables()
        ctx["user"] = "Alice"
        ctx["session_id"] = "XYZ123"

        print(ctx)        # Output: {'user': 'Alice', 'session_id': 'XYZ123'}
        print(ctx.all())  # Output: {'user': 'Alice', 'session_id': 'XYZ123'}

        # Attempting to set a reserved key directly raises an error
        try:
            ctx["output_format"] = "json"
        except ValueError as e:
            print(e)  # Output: Cannot set reserved key: output_format
        ```

        **Initializing with Mappings and Iterables:**

        ```python
        initial_data = {"theme": "dark", "language": "en"}
        additional_data = [("timezone", "UTC"), ("notifications", True)]
        ctx = ContextVariables(initial_data, additional_data, user_id=42)

        print(ctx)        # Output: {'theme': 'dark', 'language': 'en', 'timezone': 'UTC', 'notifications': True, 'user_id': 42}
        print(ctx.all())  # Output: {'theme': 'dark', 'language': 'en', 'timezone': 'UTC', 'notifications': True, 'user_id': 42}
        ```

        **Updating Context Variables:**

        ```python
        ctx = ContextVariables()
        ctx.update({"user": "Bob", "role": "admin"})
        ctx.update([("access_level", "high"), ("department", "IT")])
        ctx.update(status="active", last_login="2024-04-01")

        print(ctx)  # Output: {'user': 'Bob', 'role': 'admin', 'access_level': 'high', 'department': 'IT', 'status': 'active', 'last_login': '2024-04-01'}
        ```

        **Handling Reserved Keys:**

        ```python
        ctx = ContextVariables()
        ctx.set_reserved("output_format", {"type": "json", "version": 1.0})

        print(ctx.all())  # Output: {'output_format': {'type': 'json', 'version': 1.0}}

        # Accessing reserved keys through dedicated methods
        output_format = ctx.get_reserved("output_format")
        print(output_format)  # Output: {'type': 'json', 'version': 1.0}

        # Attempting to access reserved keys directly raises an error
        try:
            print(ctx["output_format"])
        except ValueError as e:
            print(e)  # Output: Cannot access reserved key: output_format
        ```

        **Updating Reserved Context Variables:**

        ```python
        ctx = ContextVariables()
        ctx.set_reserved("output_format", {"type": "xml"})

        # Update reserved keys using update_reserved
        ctx.update_reserved({"output_format": {"type": "yaml"}})
        ctx.update_reserved(output_format={"type": "csv"})

        print(ctx.get_reserved("output_format"))  # Output: {'type': 'csv'}

        # Attempting to update reserved keys with non-reserved keys raises an error
        try:
            ctx.update_reserved({"output_format": "html", "user": "Charlie"})
        except ValueError as e:
            print(e)  # Output: Only reserved keys can be set through update_reserved: {'user'}
        ```
    """

    _data: dict[str, Any]
    _reserved: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize context variables from mappings or key-value pairs.

        Args:
            *args: Mappings or iterables of key-value pairs to initialize the context variables.
            **kwargs: Additional key-value pairs to initialize the context variables.

        Raises:
            TypeError: If any positional argument is neither a Mapping nor an iterable of key-value pairs.
            ValueError: If any provided key is reserved.
        """
        self._data = {}
        self._reserved = {}
        self._process_initial_args(args, kwargs)

    def _process_initial_args(self, args: Iterable[Any], kwargs: Any) -> None:
        """Process and initialize context variables from positional and keyword arguments.

        Args:
            args: Positional arguments to process.
            kwargs: Keyword arguments to process.

        Raises:
            TypeError: If any positional argument is invalid.
            ValueError: If any key in args or kwargs is reserved.
        """
        for arg in args:
            items = self._extract_items(arg)
            self._validate_keys(items, reserved=False)
            self._data.update(items)

        if kwargs:
            self._validate_keys(kwargs, reserved=False)
            self._data.update(kwargs)

    def _extract_items(self, arg: Any) -> dict[str, Any]:
        """Extract key-value pairs from a positional argument.

        Args:
            arg: A Mapping or an Iterable of key-value pairs.

        Raises:
            TypeError: If arg is neither a Mapping nor an Iterable of key-value pairs.

        Returns:
            A dictionary of extracted key-value pairs.
        """
        if isinstance(arg, Mapping):
            return dict(arg)
        elif isinstance(arg, Iterable):
            try:
                return dict(arg)
            except (ValueError, TypeError) as e:
                raise TypeError("Iterable arguments must contain key-value pairs.") from e
        else:
            raise TypeError(
                "All positional arguments must be mappings or iterables of key-value pairs."
            )

    def _validate_keys(self, data: SupportsKeysAndGetItem[str, Any], reserved: bool) -> None:
        """Validate keys based on reservation status.

        Args:
            data: The key-value mapping to validate.
            reserved: If True, ensures all keys are reserved. If False, ensures no keys are reserved.

        Raises:
            ValueError: If validation fails based on the reserved flag.
        """
        keys = set(data.keys())
        if reserved:
            non_reserved = keys - RESERVED_CONTEXT_KEYS
            if non_reserved:
                raise ValueError(f"Only reserved keys can be set: {non_reserved}")
        else:
            reserved_found = keys & RESERVED_CONTEXT_KEYS
            if reserved_found:
                raise ValueError(f"Reserved keys cannot be used: {reserved_found}")

    def __getitem__(self, key: str) -> Any:
        """Retrieve the value associated with a non-reserved key.

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
        """Assign a value to a non-reserved key.

        Args:
            key: The key to set.
            value: The value to associate with the key.

        Raises:
            ValueError: If the key is reserved.
        """
        if key in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"Cannot set reserved key: {key}")
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a non-reserved key-value pair.

        Args:
            key: The key to delete.

        Raises:
            ValueError: If the key is reserved.
            KeyError: If the key does not exist.
        """
        if key in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"Cannot delete reserved key: {key}")
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over non-reserved keys.

        This enables unpacking with `**`, excluding reserved keys.

        Yields:
            Non-reserved keys.
        """
        yield from (key for key in self._data.keys() if key not in RESERVED_CONTEXT_KEYS)

    def __len__(self) -> int:
        """Get the number of non-reserved context variables.

        Returns:
            The count of non-reserved context variables.
        """
        return len(self._data)

    def __str__(self) -> str:
        """Provide a string representation of non-reserved context variables.

        Returns:
            A string representing the non-reserved context variables.
        """
        return str(self._data)

    def __repr__(self) -> str:
        """Provide an official string representation of the context variables.

        Returns:
            A string in the format ContextVariables({...}).
        """
        return f"ContextVariables({self._data})"

    def all(self) -> dict[str, Any]:
        """Retrieve all context variables, including reserved keys.

        Returns:
            A dictionary containing all key-value pairs.
        """
        return {**self._data, **self._reserved}

    def keys(self) -> KeysView[str]:
        """Retrieve a view of non-reserved keys.

        Returns:
            A KeysView object with non-reserved keys.
        """
        return self._data.keys()

    def all_keys(self) -> KeysView[str]:
        """Retrieve a view of all keys, including reserved ones.

        Returns:
            A KeysView object with all keys.
        """
        return self.all().keys()

    def items(self) -> ItemsView[str, Any]:
        """Retrieve a view of non-reserved key-value pairs.

        Returns:
            An ItemsView object with non-reserved key-value pairs.
        """
        return self._data.items()

    def all_items(self) -> ItemsView[str, Any]:
        """Retrieve a view of all key-value pairs, including reserved ones.

        Useful when both data and reserved keys are needed.

        Returns:
            An ItemsView object with all key-value pairs.
        """
        return self.all().items()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve the value for a key with an optional default.

        Args:
            key: The key to retrieve.
            default: The value to return if the key is not found.

        Returns:
            The value associated with the key or the default if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def set_reserved(self, key: ReservedContextKey, value: Any) -> None:
        """Set a value for a reserved context key.

        Args:
            key: The reserved key to set.
            value: The value to assign to the reserved key.

        Raises:
            ValueError: If the key is not reserved.
        """
        if key not in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"Key must be reserved: {key}")
        self._reserved[key] = value

    def get_reserved(self, key: ReservedContextKey, default: Any = None) -> Any:
        """Retrieve the value of a reserved context key with an optional default.

        Args:
            key: The reserved key to retrieve.
            default: The value to return if the key is not found.

        Returns:
            The value associated with the reserved key or the default if not found.
        """
        return self._reserved.get(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update non-reserved context variables.

        This method mirrors the `MutableMapping.update` signature, allowing updates
        from mappings, iterables of key-value pairs, and keyword arguments.

        Args:
            *args: Mappings or iterables of key-value pairs to update.
            **kwargs: Additional key-value pairs to update.

        Raises:
            TypeError: If arguments are not mappings or iterables of key-value pairs.
            ValueError: If any keys in args or kwargs are reserved.
        """
        for arg in args:
            items = self._extract_items(arg)
            self._validate_keys(items, reserved=False)
            self._data.update(items)

        if kwargs:
            self._validate_keys(kwargs, reserved=False)
            self._data.update(kwargs)

    def update_reserved(self, *args: Any, **kwargs: Any) -> None:
        """Update reserved context variables.

        Allows updating reserved keys exclusively through mappings, iterables of
        key-value pairs, and keyword arguments.

        Args:
            *args: Mappings or iterables of reserved key-value pairs to update.
            **kwargs: Additional reserved key-value pairs to update.

        Raises:
            TypeError: If arguments are not mappings or iterables of key-value pairs.
            ValueError: If any keys in args or kwargs are not reserved.
        """
        for arg in args:
            items = self._extract_items(arg)
            self._validate_keys(items, reserved=True)
            self._reserved.update(items)

        if kwargs:
            self._validate_keys(kwargs, reserved=True)
            self._reserved.update(kwargs)
