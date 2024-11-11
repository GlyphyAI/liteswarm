class SwarmError(Exception):
    """Base class for Swarm-specific exceptions."""


class CompletionError(SwarmError):
    """Raised when completion fails after all retries."""

    def __init__(self, message: str, original_error: Exception) -> None:
        super().__init__(message)
        self.original_error = original_error
