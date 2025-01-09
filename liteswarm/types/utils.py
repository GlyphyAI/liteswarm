# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from liteswarm.types.swarm import Message

MessageT = TypeVar("MessageT", bound=Message)
"""Type variable for Message-based types."""

MessageAdapter = TypeAdapter(list["Message"])
"""Type adapter for Message list serialization."""


class TrimMessagesResult(BaseModel, Generic[MessageT]):
    """Result of message trimming operation.

    Contains messages that fit within model context limits and
    the remaining tokens available for response. Used for context
    window management and token optimization.

    Example:
        ```python
        result = TrimmedMessages(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            response_tokens=1000,
        )
        ```
    """

    messages: list[MessageT]
    """Messages that fit within context limits."""

    response_tokens: int
    """Tokens available for model response."""


class FunctionDocstring(BaseModel):
    """Documentation parser for function tools.

    Extracts and structures function documentation into a format
    suitable for tool registration and API schema generation.

    Supports standard docstring sections:
    - Description: What the function does
    - Arguments: Parameter descriptions
    - Returns: Output description
    - Examples: Usage examples

    Examples:
        Basic function:
            ```python
            def greet(name: str, formal: bool = False) -> str:
                \"\"\"Generate a greeting message.

                Args:
                    name: Person's name to greet.
                    formal: Whether to use formal greeting.

                Returns:
                    Formatted greeting message.
                \"\"\"
                prefix = "Dear" if formal else "Hello"
                return f"{prefix} {name}"

            docstring = FunctionDocstring(
                description="Generate a greeting message.",
                parameters={
                    "name": "Person's name to greet",
                    "formal": "Whether to use formal greeting"
                }
            )
            ```

        Complex function:
            ```python
            def process_data(
                data: dict,
                options: list[str] | None = None
            ) -> JSON:
                \"\"\"Process input data with given options.

                Args:
                    data: Input data to process.
                    options: Processing options to apply.

                Returns:
                    Processed data in JSON format.
                \"\"\"
                return {"processed": data}

            docstring = FunctionDocstring(
                description="Process input data with given options.",
                parameters={
                    "data": "Input data to process",
                    "options": "Processing options to apply"
                }
            )
            ```
    """  # noqa: D214

    description: str | None = None
    """Main description of the function's purpose."""

    parameters: dict[str, Any] = Field(default_factory=dict)
    """Documentation for each function parameter."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
