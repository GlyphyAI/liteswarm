# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from litellm.exceptions import RateLimitError, ServiceUnavailableError

from liteswarm.types.exceptions import CompletionError
from liteswarm.utils.logging import log_verbose

_RetryReturnType = TypeVar("_RetryReturnType")


async def retry_with_exponential_backoff(
    operation: Callable[..., Awaitable[_RetryReturnType]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
) -> _RetryReturnType:
    """Execute an operation with exponential backoff retry logic.

    Args:
        operation: Async operation to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry

    Returns:
        The result of the operation if successful

    Raises:
        The last error encountered if all retries fail
    """
    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except (RateLimitError, ServiceUnavailableError) as e:
            last_error = e
            if attempt == max_retries:
                break

            # Calculate next delay with exponential backoff
            delay = min(delay * backoff_factor, max_delay)

            log_verbose(
                "Attempt %d/%d failed: %s. Retrying in %.1f seconds...",
                attempt + 1,
                max_retries + 1,
                str(e),
                delay,
                level="WARNING",
            )

            await asyncio.sleep(delay)

    if last_error:
        error_type = last_error.__class__.__name__
        raise CompletionError(
            f"Operation failed after {max_retries + 1} attempts: {error_type}",
            last_error,
        )

    raise CompletionError("Operation failed with unknown error", Exception("Unknown error"))
