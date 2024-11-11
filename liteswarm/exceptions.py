class SwarmError(Exception):
    """Base class for Swarm-specific exceptions."""


class CompletionError(SwarmError):
    """Raised when completion fails after all retries."""

    def __init__(self, message: str, original_error: Exception) -> None:
        super().__init__(message)
        self.original_error = original_error


class ContextLengthError(SwarmError):
    """Raised when the context length exceeds the model's limit."""

    def __init__(
        self,
        message: str,
        original_error: Exception,
        current_length: int,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.current_length = current_length
