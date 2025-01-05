# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from abc import abstractmethod
from collections.abc import AsyncGenerator, Callable
from types import TracebackType
from typing import Any, Generic, NoReturn, ParamSpec, Self, TypeAlias, TypeVar, final

from typing_extensions import TypeIs, override

YieldType = TypeVar("YieldType")
"""Type variable for the type of values yielded by the stream."""

ReturnType = TypeVar("ReturnType")
"""Type variable for the type of the final return value of the stream."""

P = ParamSpec("P")
"""Parameter specification for function parameters."""


@final
class UnsetType:
    """A singleton type to represent an unset value.

    This is used internally to distinguish between None as a valid return value
    and the absence of a return value. The singleton instance of this class
    is used as a sentinel value to indicate that no return value has been set.

    Attributes:
        __slots__: Empty tuple to prevent instance attribute creation.

    Example:
        ```python
        result: str | UnsetType = Unset
        if isinstance(result, UnsetType):
            print("No value has been set")
        ```
    """

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "Unset"

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UnsetType)

    def __bool__(self) -> bool:
        return False


Unset = UnsetType()
"""A singleton instance of UnsetType used to represent unset/missing values.

This sentinel value is used internally to distinguish between None as a valid
return value and the absence of a return value.
"""


class StreamItemBase(Generic[YieldType, ReturnType]):
    """Base class for stream items that provides common functionality.

    This abstract base class defines the interface for items that can be yielded
    from a returnable async generator. It provides methods to check whether an item
    represents a yield or return value, and to safely unwrap these values.

    Type Parameters:
        YieldType: The type of values that can be yielded.
        ReturnType: The type of the final return value.

    Example:
        ```python
        class CustomItem(StreamItemBase[int, str]):
            @property
            def is_return(self) -> bool:
                return False

            def unwrap_yield(self) -> int:
                return 42

            def unwrap_return(self) -> str:
                raise ValueError("Not a return value")
        ```
    """

    @property
    @abstractmethod
    def value(self) -> YieldType | ReturnType:
        """Get the underlying value, regardless of whether it's a yield or return value.

        Returns:
            The wrapped value, either a yield value or a return value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_return(self) -> bool:
        """Check if this item represents a return value.

        This property helps type checkers narrow down the type of the item
        and its value. When True, the value property will contain a ReturnType.

        Returns:
            True if this item contains a return value, False if it contains a yield value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_yield(self) -> bool:
        """Check if this item represents a yielded value.

        This property helps type checkers narrow down the type of the item
        and its value. When True, the value property will contain a YieldType.

        Returns:
            True if this item contains a yield value, False if it contains a return value.
        """
        raise NotImplementedError

    @abstractmethod
    def unwrap_yield(self) -> YieldType:
        """Extract the yielded value from this item.

        Returns:
            The yielded value if this item contains one.

        Raises:
            ValueError: If this item contains a return value instead of a yield value.
        """
        raise NotImplementedError

    @abstractmethod
    def unwrap_return(self) -> ReturnType:
        """Extract the return value from this item.

        Returns:
            The return value if this item contains one.

        Raises:
            ValueError: If this item contains a yield value instead of a return value.
        """
        raise NotImplementedError


@final
class YieldItem(StreamItemBase[YieldType, ReturnType]):
    """A type that represents a yielded value in a stream.

    This class wraps values that are yielded during iteration of an async generator.
    It provides type-safe access to the yielded value and implements the StreamItemBase
    interface.

    Type Parameters:
        YieldType: The type of the wrapped yield value.
        ReturnType: The type of return values (for interface compatibility).

    Example:
        ```python
        item = YieldItem[int, str](42)
        assert item.is_yield
        assert item.unwrap_yield() == 42
        ```
    """

    def __init__(self, value: YieldType) -> None:
        """Initialize a YieldItem with a yielded value.

        Args:
            value: The value to be yielded.
        """
        self._value = value

    @property
    @override
    def value(self) -> YieldType:
        return self._value

    @property
    @override
    def is_return(self) -> bool:
        return False

    @property
    @override
    def is_yield(self) -> bool:
        return True

    @override
    def unwrap_yield(self) -> YieldType:
        return self._value

    @override
    def unwrap_return(self) -> ReturnType:
        raise ValueError("Called unwrap_return on a yielded value")


@final
class ReturnItem(StreamItemBase[YieldType, ReturnType]):
    """A type that represents a return value in a stream.

    This class wraps the final return value of an async generator. It provides
    type-safe access to the return value and implements the StreamItemBase interface.

    Type Parameters:
        YieldType: The type of yield values (for interface compatibility).
        ReturnType: The type of the wrapped return value.

    Example:
        ```python
        item = ReturnItem[int, str]("done")
        assert item.is_return
        assert item.unwrap_return() == "done"
        ```
    """

    def __init__(self, value: ReturnType) -> None:
        """Initialize a ReturnItem with a return value.

        Args:
            value: The value to be returned.
        """
        self._value = value

    @property
    @override
    def value(self) -> ReturnType:
        return self._value

    @property
    @override
    def is_return(self) -> bool:
        return True

    @property
    @override
    def is_yield(self) -> bool:
        return False

    @override
    def unwrap_yield(self) -> YieldType:
        raise ValueError("Called unwrap_yield on a return value")

    @override
    def unwrap_return(self) -> ReturnType:
        return self._value


StreamItem = YieldItem[YieldType, ReturnType] | ReturnItem[YieldType, ReturnType]
"""A type that represents either a yielded value or a return value in a stream.

This union type combines YieldItem and ReturnItem to represent all possible values
that can come from a returnable async generator. It provides a type-safe way to
handle both regular yielded values and the final return value.

Type Parameters:
    YieldType: The type of regular values yielded by the generator.
    ReturnType: The type of the final return value.

Example:
    ```python
    def handle_item(item: StreamItem[int, str]) -> None:
        if item.is_yield:
            print(f"Got yield value: {item.unwrap_yield()}")
        else:
            print(f"Got return value: {item.unwrap_return()}")
    ```
"""


AsyncStream: TypeAlias = AsyncGenerator[StreamItem[YieldType, ReturnType], Any]
"""Type alias for async generators that can yield values and return a final value.

This type represents an async generator that yields StreamItem instances which
can either contain regular yielded values or a final return value. It's designed
to be used with the @returnable decorator to create generators that can both
yield values and return a final result.

Type Parameters:
    YieldType: The type of regular values yielded by the generator.
    ReturnType: The type of the final return value.

Example:
    ```python
    @returnable
    async def count_to(n: int) -> AsyncStream[int, str]:
        for i in range(n):
            yield YieldItem(i)
        yield ReturnItem("done")
    ```
"""


def is_return(item: StreamItem[YieldType, ReturnType]) -> TypeIs[ReturnItem[YieldType, ReturnType]]:
    """Type guard to check if an item is a ReturnItem.

    This function helps type checkers narrow down the type of a StreamItem
    to ReturnItem when the check returns True.

    Args:
        item: The stream item to check.

    Returns:
        True if the item is a ReturnItem, False otherwise.
    """
    return item.is_return


def is_yield(item: StreamItem[YieldType, ReturnType]) -> TypeIs[YieldItem[YieldType, ReturnType]]:
    """Type guard to check if an item is a YieldItem.

    This function helps type checkers narrow down the type of a StreamItem
    to YieldItem when the check returns True.

    Args:
        item: The stream item to check.

    Returns:
        True if the item is a YieldItem, False otherwise.
    """
    return item.is_yield


class ReturnableAsyncGenerator(Generic[YieldType, ReturnType]):
    """A wrapper for async generators that preserves return values.

    This class wraps an async generator that yields StreamItem instances and provides
    a way to access both the yielded values and the final return value. It implements
    the async generator protocol and adds a get_return_value() method to retrieve the
    return value.

    Type Parameters:
        YieldType: The type of values yielded by the generator.
        ReturnType: The type of the final return value.

    Example:
        ```python
        async def numbers() -> AsyncStream[int, str]:
            yield YieldItem(1)
            yield YieldItem(2)
            yield ReturnItem("done")


        gen = ReturnableAsyncGenerator(numbers())
        async for num in gen:
            print(num)  # Prints: 1, 2
        result = await gen.get_return_value()  # Gets: "done"
        ```
    """

    def __init__(self, agen: AsyncGenerator[StreamItem[YieldType, ReturnType], Any]) -> None:
        """Initialize a ReturnableAsyncGenerator with an async generator.

        Args:
            agen: The async generator to wrap.
        """
        self._agen = agen
        self._final_result: ReturnType | UnsetType = Unset
        self._iter_finished = False
        self._close_called = False

    # ================================================
    # MARK: Private Helpers
    # ================================================

    async def _handle_next_item(self, item: StreamItem[YieldType, ReturnType]) -> YieldType:
        """Handle the next item from the generator, managing return values and iteration state.

        This method processes each item yielded from the underlying generator. If the
        item is a return value, it stores it and ends iteration. If it's a yield value,
        it passes it through.

        Args:
            item: The next item from the generator.

        Returns:
            The unwrapped yield value.

        Raises:
            StopAsyncIteration: If the item is a return value.
        """
        if is_return(item):
            self._final_result = item.value
            self._iter_finished = True
            self._close_called = True

            try:
                await self._agen.athrow(GeneratorExit)
            except (GeneratorExit, StopAsyncIteration):
                pass

            raise StopAsyncIteration

        return item.value

    async def _handle_stop_iteration(self) -> NoReturn:
        """Handle StopAsyncIteration by updating the generator state and always raising.

        This method updates the internal state to mark the generator as finished and
        closed, then raises StopAsyncIteration. It's used to handle both normal
        completion and early termination.

        Raises:
            StopAsyncIteration: Always raised after updating state.
        """
        self._iter_finished = True
        self._close_called = True
        raise StopAsyncIteration

    # ================================================
    # MARK: Async Generator Protocol
    # ================================================

    def __aiter__(self) -> Self:
        """Return self as an async iterator, preserving type information."""
        return self

    async def __anext__(self) -> YieldType:
        """Get next value, preserving the exact YieldType.

        Returns:
            The next yielded value from the generator.

        Raises:
            StopAsyncIteration: When iteration is complete or a return value is encountered.
        """
        try:
            return await self._handle_next_item(await self._agen.__anext__())
        except StopAsyncIteration:
            return await self._handle_stop_iteration()

    async def asend(self, value: Any) -> YieldType:
        """Send a value into the underlying generator.

        Args:
            value: The value to send into the generator.

        Returns:
            The next yielded value from the generator.

        Raises:
            StopAsyncIteration: When iteration is complete or a return value is encountered.
        """
        if self._iter_finished:
            return await self._handle_stop_iteration()

        try:
            return await self._handle_next_item(await self._agen.asend(value))
        except StopAsyncIteration:
            return await self._handle_stop_iteration()

    async def athrow(
        self,
        typ: type[BaseException],
        val: BaseException | object = None,
        tb: TracebackType | None = None,
    ) -> YieldType:
        """Throw an exception into the underlying generator.

        Args:
            typ: The type of exception to throw.
            val: The exception instance or value to throw.
            tb: Optional traceback to associate with the exception.

        Returns:
            The next yielded value from the generator.

        Raises:
            StopAsyncIteration: When iteration is complete or a return value is encountered.
            BaseException: The exception thrown into the generator if it's not handled.
        """
        try:
            return await self._handle_next_item(await self._agen.athrow(typ, val, tb))
        except StopAsyncIteration:
            return await self._handle_stop_iteration()

    async def aclose(self) -> None:
        """Close the underlying generator.

        This method ensures that the generator is properly closed and cleaned up.
        It's safe to call multiple times.
        """
        if not self._close_called:
            self._close_called = True
            self._iter_finished = True
            await self._agen.aclose()

    # ================================================
    # MARK: Public API
    # ================================================

    async def get_return_value(self) -> ReturnType:
        """Retrieve the generator's return value.

        If iteration is not complete, this method will consume the remaining
        items in the generator until it finds a return value. If iteration is
        already complete, it returns the previously captured return value.

        Returns:
            The final return value from the generator.

        Raises:
            RuntimeError: If the generator completes without yielding a return value
                        and iteration is already finished.
            TypeError: If the generator completes without yielding a return value
                      during this call.

        Notes:
            - Safe to call multiple times
            - Will complete iteration if not already finished
            - Subsequent calls after completion return cached result
            - None is a valid return value
        """
        if self._iter_finished:
            if isinstance(self._final_result, UnsetType):
                raise RuntimeError("Generator completed without yielding a return value")
            return self._final_result

        try:
            async for _ in self:
                pass
        except RuntimeError:
            # Generator might have been closed without a return value
            if isinstance(self._final_result, UnsetType):
                raise TypeError("Generator completed without yielding a return value") from None

        if isinstance(self._final_result, UnsetType):
            raise TypeError("Generator completed without yielding a return value")

        return self._final_result


def returnable(
    func: Callable[P, AsyncStream[YieldType, ReturnType]],
) -> Callable[P, ReturnableAsyncGenerator[YieldType, ReturnType]]:
    """Decorator that wraps an async generator to make its return value accessible.

    This decorator transforms an async generator that yields StreamItem instances into
    a ReturnableAsyncGenerator, making it easy to access both yielded values through
    iteration and the final return value through get_return_value().

    Type Parameters:
        P: The parameters of the decorated function.
        YieldType: The type of values yielded by the generator.
        ReturnType: The type of the final return value.

    Args:
        func: An async generator function that yields StreamItem instances.

    Returns:
        A wrapped function that returns a ReturnableAsyncGenerator instance.

    Example:
        ```python
        @returnable
        async def count_to(n: int) -> AsyncStream[int, str]:
            for i in range(n):
                yield YieldItem(i)
            yield ReturnItem("done")


        # Usage:
        counter = count_to(3)
        async for num in counter:
            print(num)  # Prints: 0, 1, 2
        result = await counter.get_return_value()  # Gets: "done"
        ```
    """

    def wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> ReturnableAsyncGenerator[YieldType, ReturnType]:
        return ReturnableAsyncGenerator(func(*args, **kwargs))

    return wrapper
