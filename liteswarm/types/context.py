# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any

from typing_extensions import override


class ContextVariables(dict[str, Any]):
    """Manages context variables with attribute-style access.

    This class behaves like a standard dictionary but allows accessing and setting
    context variables using attribute syntax (e.g., `ctx.user = "Alice"`).

    Examples:
        **Basic Usage:**

        ```python
        ctx = ContextVariables()
        ctx["user"] = "Alice"
        ctx.session_id = "XYZ123"

        print(ctx)
        # Output: {'user': 'Alice', 'session_id': 'XYZ123'}
        ```

        **Initializing with Mappings and Iterables:**

        ```python
        initial_data = {"theme": "dark", "language": "en"}
        additional_data = [("timezone", "UTC"), ("notifications", True)]
        ctx = ContextVariables(initial_data, additional_data, user_id=42)

        print(ctx)
        # Output: {'theme': 'dark', 'language': 'en', 'timezone': 'UTC', 'notifications': True, 'user_id': 42}
        ```

        **Updating Context Variables:**

        ```python
        ctx = ContextVariables()
        ctx.update({"user": "Bob", "role": "admin"})
        ctx.update([("access_level", "high"), ("department", "IT")])
        ctx.update(status="active", last_login="2024-04-01")

        print(ctx)
        # Output: {'user': 'Bob', 'role': 'admin', 'access_level': 'high', 'department': 'IT', 'status': 'active', 'last_login': '2024-04-01'}
        ```

        **Using Attribute Access:**

        ```python
        ctx = ContextVariables()
        ctx.output_format = "json"
        print(ctx.output_format)
        # Output: json

        # Modifying via attribute
        ctx.output_format = "xml"
        print(ctx["output_format"])
        # Output: xml
        ```

        **Handling Non-Existent Attributes:**

        ```python
        ctx = ContextVariables()
        try:
            print(ctx.non_existent)
        except AttributeError as e:
            print(e)  # Output: 'ContextVariables' object has no attribute 'non_existent'
        ```
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize context variables from mappings or key-value pairs.

        Args:
            *args: Mappings or iterables of key-value pairs to initialize the context variables.
            **kwargs: Additional key-value pairs to initialize the context variables.

        Raises:
            TypeError: If any positional argument is neither a Mapping nor an iterable of key-value pairs.
        """
        super().__init__()
        for arg in args:
            self.update(arg)
        self.update(**kwargs)

    def __getattr__(self, key: str) -> Any:
        """Retrieve the value associated with a key using attribute access.

        Args:
            key: The key to retrieve.

        Raises:
            AttributeError: If the key does not exist.

        Returns:
            The value associated with the key.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ContextVariables' object has no attribute '{key}'") from None

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        """Assign a value to a key using attribute access.

        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        self[key] = value

    @override
    def __delattr__(self, key: str) -> None:
        """Delete a key-value pair using attribute access.

        Args:
            key: The key to delete.

        Raises:
            AttributeError: If the key does not exist.
        """
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ContextVariables' object has no attribute '{key}'") from None
