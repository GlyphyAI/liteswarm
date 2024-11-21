# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


class SwarmError(Exception):
    """Base class for Swarm-specific exceptions.

    All custom exceptions in the Swarm system inherit from this class
    to allow for specific error handling of Swarm-related issues.

    Example:
    ```python
    try:
        result = await swarm.execute(agent, prompt)
    except SwarmError as e:
        print(f"Swarm operation failed: {e}")
    ```
    """


class CompletionError(SwarmError):
    """Raised when completion fails after all retries.

    This exception indicates that the language model API call failed
    repeatedly and exhausted all retry attempts. The original error
    is preserved for debugging.

    Example:
    ```python
    try:
        response = await swarm.execute(agent, prompt)
    except CompletionError as e:
        print(f"Completion failed: {e}")
        print(f"Original error: {e.original_error}")
    ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception,
    ) -> None:
        """Initialize a CompletionError.

        Args:
            message: Human-readable error description
            original_error: The underlying exception that caused the failure
        """
        super().__init__(message)
        self.original_error = original_error


class ContextLengthError(SwarmError):
    """Raised when the context length exceeds the model's limit.

    This exception occurs when the conversation history is too long
    for the model's context window, even after attempting to reduce
    it through summarization or trimming.

    Example:
    ```python
    try:
        response = await swarm.execute(agent, prompt)
    except ContextLengthError as e:
        print(f"Context too long: {e}")
        print(f"Current length: {e.current_length}")
        print(f"Original error: {e.original_error}")

        # Maybe switch to a model with larger context
        new_agent = Agent(
            id="large-context",
            instructions=agent.instructions,
            llm=LLMConfig(model="claude-3-5-sonnet-20241022")
        )
    ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception,
        current_length: int,
    ) -> None:
        """Initialize a ContextLengthError.

        Args:
            message: Human-readable error description
            original_error: The underlying exception that caused the failure
            current_length: The current context length that exceeded the limit
        """
        super().__init__(message)
        self.original_error = original_error
        self.current_length = current_length
