# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, Any] | list[Any] | str | float | int | bool | None
"""Type alias for JSON-compatible data structures."""
