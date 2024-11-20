# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, TypeAlias

from pydantic import BaseModel, Field

Number: TypeAlias = int | float
"""A number."""

JSON: TypeAlias = str | bool | Number | list["JSON"] | dict[str, "JSON"] | None
"""A JSON type."""


class FunctionDocstring(BaseModel):
    """Parsed documentation for a function tool."""

    description: str | None = None
    """Description of what the function does."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """Documentation for each parameter."""
