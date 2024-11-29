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
"""Type variable for retry operation's return type.

Used to preserve type information when retrying operations
that return different types of values.
"""


async def retry_with_exponential_backoff(
    operation: Callable[..., Awaitable[_RetryReturnType]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
) -> _RetryReturnType:
    """Retry async operation with exponential backoff.

    Executes an async operation with configurable retry behavior:
    - Exponentially increasing delays between retries
    - Maximum retry count and delay limits
    - Automatic handling of common API errors
    - Detailed logging of retry attempts

    Args:
        operation: Async function to execute.
        max_retries: Maximum retry attempts (default: 3).
        initial_delay: Starting delay in seconds (default: 1.0).
        max_delay: Maximum delay in seconds (default: 10.0).
        backoff_factor: Delay multiplier after each retry (default: 2.0).

    Returns:
        Result from the successful operation execution.

    Raises:
        CompletionError: If all retry attempts fail, wraps the last error.

    Examples:
        Basic retry:
            ```python
            async def make_api_call() -> dict:
                response = await api.request()
                return response.json()

            try:
                result = await retry_with_exponential_backoff(
                    make_api_call,
                    max_retries=3
                )
            except CompletionError as e:
                print(f"API call failed: {e}")
            ```

        Custom configuration:
            ```python
            async def unstable_operation() -> str:
                if random.random() < 0.8:
                    raise ServiceUnavailableError("Server busy")
                return "success"

            result = await retry_with_exponential_backoff(
                unstable_operation,
                max_retries=5,
                initial_delay=0.1,
                max_delay=5.0,
                backoff_factor=3.0
            )
            # Retries with delays: 0.1s, 0.3s, 0.9s, 2.7s, 5.0s
            ```

        Rate limiting:
            ```python
            async def rate_limited_call() -> Response:
                try:
                    return await api.request()
                except RateLimitError:
                    # Will be caught and retried with backoff
                    raise

            response = await retry_with_exponential_backoff(
                rate_limited_call,
                max_retries=3,
                initial_delay=2.0  # Start with longer delay
            )
            ```
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
