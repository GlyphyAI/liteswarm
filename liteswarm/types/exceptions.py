# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


class SwarmError(Exception):
    """Base exception class for all Swarm-related errors.

    Provides a common ancestor for all custom exceptions in the system,
    enabling unified error handling and logging of Swarm operations.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await swarm.execute(prompt)
            except SwarmError as e:
                logger.error(f"Swarm operation failed: {e}")
            ```

        Custom exception:
            ```python
            class ValidationError(SwarmError):
                \"\"\"Raised when agent input validation fails.\"\"\"
                def __init__(self, field: str, value: Any) -> None:
                    super().__init__(
                        f"Invalid value {value} for field {field}"
                    )
            ```
    """


class CompletionError(SwarmError):
    """Exception raised when LLM completion fails permanently.

    Indicates that the language model API call failed and exhausted
    all retry attempts. Preserves the original error for debugging
    and error reporting.

    Examples:
        Basic handling:
            ```python
            try:
                response = await agent.complete(prompt)
            except CompletionError as e:
                logger.error(
                    f"API call failed: {e}",
                    extra={
                        "error_type": type(e.original_error).__name__,
                        "details": str(e.original_error)
                    }
                )
            ```

        Fallback strategy:
            ```python
            try:
                response = await primary_agent.complete(prompt)
            except CompletionError:
                # Switch to backup model
                backup_agent = Agent(
                    id="backup",
                    llm=LLM(model="gpt-3.5-turbo")
                )
                response = await backup_agent.complete(prompt)
            ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception,
    ) -> None:
        """Initialize a new CompletionError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused the failure.
        """
        super().__init__(message)
        self.original_error = original_error


class ContextLengthError(SwarmError):
    """Exception raised when input exceeds model's context limit.

    Occurs when the combined length of conversation history and new
    input exceeds the model's maximum context window, even after
    attempting context reduction strategies.

    Examples:
        Basic handling:
            ```python
            try:
                response = await agent.complete(prompt)
            except ContextLengthError as e:
                logger.warning(
                    "Context length exceeded",
                    extra={
                        "current_length": e.current_length,
                        "error": str(e.original_error)
                    }
                )
            ```

        Automatic model upgrade:
            ```python
            async def complete_with_fallback(
                prompt: str,
                agent: Agent
            ) -> str:
                try:
                    return await agent.complete(prompt)
                except ContextLengthError:
                    # Switch to larger context model
                    large_agent = Agent(
                        id="large-context",
                        instructions=agent.instructions,
                        llm=LLM(
                            model="claude-3-opus",
                            max_tokens=100000
                        )
                    )
                    return await large_agent.complete(prompt)
            ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception,
        current_length: int,
    ) -> None:
        """Initialize a new ContextLengthError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused the failure.
            current_length: Current context length that exceeded the limit.
        """
        super().__init__(message)
        self.original_error = original_error
        self.current_length = current_length
