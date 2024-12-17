# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import inspect
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from liteswarm.types.exceptions import CompletionError
from liteswarm.utils.logging import log_verbose

P = ParamSpec("P")
"""Type variable capturing function parameter types."""

T = TypeVar("T")
"""Type variable capturing function return type."""


def _calculate_next_delay(
    current_delay: float,
    backoff_factor: float,
    max_delay: float,
) -> float:
    """Calculate the next delay using exponential backoff.

    Args:
        current_delay: Current delay in seconds.
        backoff_factor: Multiplier for exponential backoff.
        max_delay: Maximum allowed delay in seconds.

    Returns:
        Next delay value capped at max_delay.
    """
    return min(current_delay * backoff_factor, max_delay)


def _log_retry_attempt(
    attempts: int,
    max_retries: int,
    error: Exception,
    next_delay: float,
    is_error: bool = False,
) -> None:
    """Log retry attempt or final error with appropriate level and message.

    Args:
        attempts: Current attempt number.
        max_retries: Maximum number of retries allowed.
        error: Exception that triggered the retry.
        next_delay: Time until next retry attempt.
        is_error: Whether this is a final error log.
    """
    if is_error:
        error_type = error.__class__.__name__
        log_verbose(
            f"Operation failed after {max_retries + 1} attempts: {error_type}",
            level="ERROR",
        )
    else:
        log_verbose(
            "Attempt %d/%d failed: %s. Retrying in %.1f seconds...",
            attempts,
            max_retries + 1,
            str(error),
            next_delay,
            level="WARNING",
        )


def _handle_retry_error(error: Exception, max_retries: int) -> CompletionError:
    """Create a CompletionError with context about retry attempts.

    Args:
        error: The last error that occurred.
        max_retries: Maximum number of retries that were attempted.

    Returns:
        CompletionError wrapping the original error with retry context.
    """
    error_type = error.__class__.__name__
    return CompletionError(
        f"Operation failed after {max_retries + 1} attempts: {error_type}",
        error,
    )


async def _async_wrapper(
    func: Callable[P, Awaitable[T]],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    exception: type[Exception] | tuple[type[Exception], ...],
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> T:
    """Retry wrapper for async functions with exponential backoff.

    Args:
        func: Async function to retry.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        exception: Exception types that trigger retries.
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for exponential backoff.

    Returns:
        Result from the successful function execution.

    Raises:
        CompletionError: If all retry attempts fail.
    """
    attempts = 0
    last_error: Exception | None = None
    delay = initial_delay

    while attempts <= max_retries:
        try:
            return await func(*args, **kwargs)

        except exception as e:
            last_error = e
            attempts += 1
            if attempts > max_retries:
                break

            delay = _calculate_next_delay(delay, backoff_factor, max_delay)
            _log_retry_attempt(attempts, max_retries, e, delay)
            await asyncio.sleep(delay)

    if last_error:
        _log_retry_attempt(attempts, max_retries, last_error, 0, is_error=True)
        raise _handle_retry_error(last_error, max_retries)

    raise CompletionError("Operation failed with unknown error", Exception("Unknown error"))


def _sync_wrapper(
    func: Callable[P, T],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    exception: type[Exception] | tuple[type[Exception], ...],
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> T:
    """Retry wrapper for synchronous functions with exponential backoff.

    Args:
        func: Synchronous function to retry.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        exception: Exception types that trigger retries.
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for exponential backoff.

    Returns:
        Result from the successful function execution.

    Raises:
        CompletionError: If all retry attempts fail.
    """
    attempts = 0
    last_error: Exception | None = None
    delay = initial_delay

    while attempts <= max_retries:
        try:
            return func(*args, **kwargs)

        except exception as e:
            last_error = e
            attempts += 1
            if attempts > max_retries:
                break

            delay = _calculate_next_delay(delay, backoff_factor, max_delay)
            _log_retry_attempt(attempts, max_retries, e, delay)
            time.sleep(delay)

    if last_error:
        _log_retry_attempt(attempts, max_retries, last_error, 0, is_error=True)
        raise _handle_retry_error(last_error, max_retries)

    raise CompletionError("Operation failed with unknown error", Exception("Unknown error"))


async def _async_gen_wrapper(
    func: Callable[P, AsyncGenerator[T, None] | Awaitable[AsyncGenerator[T, None]]],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    exception: type[Exception] | tuple[type[Exception], ...],
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> AsyncGenerator[T, None]:
    """Retry wrapper for async generators with exponential backoff.

    If any error occurs during generation or iteration, retries the entire
    generator from the beginning. This is particularly useful for non-deterministic
    generators like LLM streaming.

    Args:
        func: Async generator function to retry.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        exception: Exception types that trigger retries.
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for exponential backoff.

    Yields:
        Items from the generator.

    Raises:
        CompletionError: If all retry attempts fail.
    """
    attempts = 0
    last_error: Exception | None = None
    delay = initial_delay

    while attempts <= max_retries:
        try:
            if inspect.iscoroutinefunction(func):
                gen = await func(*args, **kwargs)
            else:
                gen = func(*args, **kwargs)

            async for item in gen:
                yield item

            return

        except exception as e:
            last_error = e
            attempts += 1
            if attempts > max_retries:
                break

            delay = _calculate_next_delay(delay, backoff_factor, max_delay)
            _log_retry_attempt(attempts, max_retries, e, delay)
            await asyncio.sleep(delay)

    if last_error:
        _log_retry_attempt(attempts, max_retries, last_error, 0, is_error=True)
        raise _handle_retry_error(last_error, max_retries)

    raise CompletionError("Operation failed with unknown error", Exception("Unknown error"))


def _sync_gen_wrapper(
    func: Callable[P, Generator[T, None, None]],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    exception: type[Exception] | tuple[type[Exception], ...],
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> Generator[T, None, None]:
    """Retry wrapper for synchronous generators with exponential backoff.

    If any error occurs during generation or iteration, retries the entire
    generator from the beginning.

    Args:
        func: Generator function to retry.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        exception: Exception types that trigger retries.
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for exponential backoff.

    Yields:
        Items from the generator.

    Raises:
        CompletionError: If all retry attempts fail.
    """
    attempts = 0
    last_error: Exception | None = None
    delay = initial_delay

    while attempts <= max_retries:
        try:
            gen = func(*args, **kwargs)
            yield from gen
            return

        except exception as e:
            last_error = e
            attempts += 1
            if attempts > max_retries:
                break

            delay = _calculate_next_delay(delay, backoff_factor, max_delay)
            _log_retry_attempt(attempts, max_retries, e, delay)
            time.sleep(delay)

    if last_error:
        _log_retry_attempt(attempts, max_retries, last_error, 0, is_error=True)
        raise _handle_retry_error(last_error, max_retries)

    raise CompletionError("Operation failed with unknown error", Exception("Unknown error"))


def retry_with_backoff(
    func: Callable[P, T],
    exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
) -> Callable[P, T]:
    """Wraps a function with exponential backoff retry logic.

    Supports:
      - synchronous functions
      - synchronous generators
      - async functions (async coroutines)
      - async generator functions

    For generator functions (both sync and async), the wrapper will retry the entire
    operation from the beginning if any error occurs during generation or iteration.

    Args:
        func: The function to apply backoff to.
        exception: The exception(s) to trigger a retry upon.
        max_retries: The maximum number of retries.
        initial_delay: The initial delay between retries in seconds.
        max_delay: The maximum possible delay between retries in seconds.
        backoff_factor: Exponential backoff multiplier factor.

    Returns:
        A callable that has the same signature as `func` but with retry logic.

    Raises:
        TypeError: If the function type is not supported.

    Examples:
        Sync function retry:
            ```python
            @retry_with_backoff(exception=ConnectionError, max_retries=3)
            def fetch_data() -> dict:
                return requests.get("https://api.example.com/data").json()
            ```

        Sync generator retry:
            ```python
            @retry_with_backoff(exception=IOError, max_retries=3)
            def read_file_chunks(path: str) -> Generator[str, None, None]:
                with open(path, "r") as f:
                    while chunk := f.read(1024):
                        yield chunk
            ```

        Async function retry:
            ```python
            @retry_with_backoff(
                exception=(aiohttp.ClientError, asyncio.TimeoutError),
                max_retries=5,
                initial_delay=0.1,
            )
            async def fetch_async_data() -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://api.example.com/data") as response:
                        return await response.json()
            ```

        Async generator retry:
            ```python
            @retry_with_backoff(exception=Exception, max_retries=3)
            async def stream_data() -> AsyncGenerator[str, None]:
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://api.example.com/stream") as response:
                        async for chunk in response.content.iter_chunked(1024):
                            yield chunk.decode()
            ```
    """
    if inspect.isasyncgenfunction(func):

        async def async_gen_wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[T, None]:
            async for item in _async_gen_wrapper(
                func=func,
                args=args,
                kwargs=kwargs,
                exception=exception,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            ):
                yield item

        return async_gen_wrapper  # type: ignore

    if inspect.iscoroutinefunction(func):

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await _async_wrapper(
                func=func,
                args=args,
                kwargs=kwargs,
                exception=exception,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            )

        return async_wrapper  # type: ignore

    if inspect.isgeneratorfunction(func):

        def gen_wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[T, None, None]:
            return _sync_gen_wrapper(
                func=func,
                args=args,
                kwargs=kwargs,
                exception=exception,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            )

        return gen_wrapper  # type: ignore

    if inspect.isfunction(func):

        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return _sync_wrapper(
                func=func,
                args=args,
                kwargs=kwargs,
                exception=exception,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
            )

        return sync_wrapper

    raise TypeError(
        f"Unsupported callable type: {type(func)}. "
        "The retry decorator supports:\n"
        "  - Regular functions (def func)\n"
        "  - Async functions (async def func)\n"
        "  - Generator functions (def func with yield)\n"
        "  - Async generator functions (async def func with yield)\n"
        "Did you pass a function instead of calling it?"
    )
