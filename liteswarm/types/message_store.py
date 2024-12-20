# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class LiteMessageStoreFilter(BaseModel):
    """Filter criteria for message retrieval in LiteMessageStore.

    Provides filtering by role, time range, metadata, and message count. Combines
    criteria using AND logic, with unspecified fields being ignored. Filters are
    applied in sequence: role, time range, metadata, and finally message count
    limit.

    Examples:
        Filter by role:
            ```python
            filter = LiteMessageStoreFilter(role="user")
            ```

        Filter by time range:
            ```python
            filter = LiteMessageStoreFilter(
                after=datetime(2024, 1, 1),
                before=datetime(2024, 2, 1),
            )
            ```

        Filter by metadata:
            ```python
            filter = LiteMessageStoreFilter(
                metadata_filters={"tag": "greeting"},
            )
            ```

        Get last N messages:
            ```python
            filter = LiteMessageStoreFilter(last_n=5)
            ```

        Combined filters:
            ```python
            filter = LiteMessageStoreFilter(
                role="assistant",
                last_n=10,
                metadata_filters={"important": True},
            )
            ```
    """

    role: str | None = None
    """Filter by message role."""

    before: datetime | None = None
    """Include messages before this time."""

    after: datetime | None = None
    """Include messages after this time."""

    metadata_filters: dict[str, Any] | None = None
    """Filter by metadata key/value pairs."""

    last_n: int | None = None
    """Return only the last N messages."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )
