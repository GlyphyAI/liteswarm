# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import Agent

ResultValue = TypeVar("ResultValue")
"""Type variable representing the value type in a Result."""


class Result(BaseModel, Generic[ResultValue]):
    """Generic wrapper for operation results in the agentic system.

    Provides a standardized way to handle results from any operation (agents,
    functions, tools) with support for success values, errors, agent switching,
    and context updates.

    Examples:
        Simple success result:
            ```python
            # Return calculation result
            def calculate_average(numbers: list[float]) -> Result[float]:
                try:
                    avg = sum(numbers) / len(numbers)
                    return Result(value=avg)
                except ZeroDivisionError as e:
                    return Result(error=e)
            ```

        Error handling:
            ```python
            # Handle validation error
            def validate_input(data: dict) -> Result[dict]:
                if "required_field" not in data:
                    return Result(
                        error=ValueError("Missing required_field")
                    )
                return Result(value=data)
            ```

        Agent switching:
            ```python
            # Switch to specialized agent
            def handle_complex_query(query: str) -> Result[str]:
                if "math" in query.lower():
                    return Result(
                        agent=Agent(
                            id="math-expert",
                            instructions="You are a math expert.",
                            llm=LLM(model="gpt-4o")
                        ),
                        context_variables=ContextVariables(
                            domain="mathematics",
                            complexity="advanced"
                        )
                    )
                return Result(value="I can help with that.")
            ```
    """

    value: ResultValue | None = None
    """Operation's success value, if available."""

    error: Exception | None = None
    """Error that occurred during operation, if any."""

    agent: Agent | None = None
    """New agent to switch to, if needed."""

    context_variables: ContextVariables | None = None
    """Context updates to apply."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def unwrap(self, error_message: str = "No value in result") -> ResultValue:
        """Unwrap the result value or raise an error.

        Attempts to extract the value from the Result. If an error is present in the
        Result, that error is raised. If no value is present and no error is set,
        raises ValueError with the provided message.

        Args:
            error_message: Custom error message to use when no value is present.
                Defaults to "No value in result".

        Returns:
            ResultValue: The value stored in the Result.

        Raises:
            ValueError: If no value is present and no error is set.
            Exception: Whatever error is stored in the Result, if one exists.

        Examples:
            Successful unwrap:
                ```python
                result = Result(value=42)
                value = result.unwrap()  # Returns 42
                ```

            Custom error message:
                ```python
                result = Result[int]()
                try:
                    value = result.unwrap("Missing calculation result")
                except ValueError as e:
                    print(e)  # Prints: Missing calculation result
                ```

            Propagating stored error:
                ```python
                result = Result(error=ValueError("Invalid input"))
                try:
                    value = result.unwrap()
                except ValueError as e:
                    print(e)  # Prints: Invalid input
                ```
        """
        if self.error:
            raise self.error

        if not self.value:
            raise ValueError(error_message)

        return self.value

    def unwrap_or(self, default: ResultValue) -> ResultValue:
        """Unwrap the result value or return a default.

        Similar to unwrap(), but returns a default value instead of raising an error
        when no value is present. This provides a safe way to handle Results that
        might not have a value.

        Args:
            default: The value to return if the Result has no value.
                Must be of the same type as the Result's value type.

        Returns:
            ResultValue: Either the value from the Result or the provided default.

        Examples:
            With value present:
                ```python
                result = Result(value=42)
                value = result.unwrap_or(0)  # Returns 42
                ```

            With no value:
                ```python
                result = Result[int]()
                value = result.unwrap_or(0)  # Returns 0
                ```

            With error:
                ```python
                result = Result(error=ValueError("Invalid"))
                value = result.unwrap_or(0)  # Returns 0
                ```
        """
        try:
            return self.unwrap()
        except ValueError:
            return default

    def success(self) -> bool:
        """Check if the result represents a successful operation.

        A result is considered successful if it contains a value and no error.
        This is useful for conditional logic and error handling flows.

        Returns:
            bool: True if the result has a value and no error, False otherwise.

        Examples:
            Success cases:
                ```python
                # With value, no error
                result = Result(value=42)
                assert result.success() == True

                # With value and context (still success)
                result = Result(
                    value="data",
                    context_variables=ContextVariables(domain="test")
                )
                assert result.success() == True
                ```

            Failure cases:
                ```python
                # No value
                result = Result[str]()
                assert result.success() == False

                # Has error
                result = Result(error=ValueError("Failed"))
                assert result.success() == False
                ```
        """
        return self.error is None and self.value is not None

    def failure(self) -> bool:
        """Check if the result represents a failed operation.

        A result is considered a failure if it either contains an error
        or lacks a value. This is the logical opposite of success().

        Returns:
            bool: True if the result has an error or no value, False otherwise.

        Examples:
            Failure cases:
                ```python
                # With error
                result = Result(error=ValueError("Invalid input"))
                assert result.failure() == True

                # No value or error
                result = Result[int]()
                assert result.failure() == True
                ```

            Success case:
                ```python
                # With value, no error
                result = Result(value="success")
                assert result.failure() == False
                ```

        Notes:
            - A result with neither error nor value is considered a failure.
            - This method is equivalent to `not result.success()`.
        """
        return self.error is not None or self.value is None
