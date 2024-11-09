from typing import Protocol

from litellm import acompletion
from litellm.types.utils import Choices, ModelResponse

from liteswarm.types import Message


class Summarizer(Protocol):
    """Protocol for conversation summarizers."""

    async def summarize(self, messages: list[Message]) -> str:
        """Summarize the given messages."""
        ...


class LiteSummarizer:
    """Summarizes conversations using LiteLLM."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        system_prompt: str | None = None,
        summarize_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that creates clear and concise summaries of conversations."
        )
        self.summarize_prompt = summarize_prompt or (
            "Please provide a concise summary of the following conversation. "
            "Focus on key decisions, outcomes, and important context needed for continuing the conversation. "
            "Format the summary as a clear list of key points."
        )

    async def summarize(self, messages: list[Message]) -> str:
        """Create a concise summary of the conversation history.

        Args:
            messages: List of messages to summarize

        Returns:
            A string containing the summarized conversation

        Raises:
            TypeError: If the response from the LLM is not of the expected type
            ValueError: If the response from the LLM is empty
        """
        summary_messages = [
            Message(role="system", content=self.system_prompt),
            *messages,
            Message(role="user", content=self.summarize_prompt),
        ]

        response = await acompletion(
            model=self.model,
            messages=[msg.model_dump(exclude_none=True) for msg in summary_messages],
            stream=False,
        )

        if not isinstance(response, ModelResponse):
            raise TypeError("Expected a CompletionResponse instance.")

        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise TypeError("Expected a StreamingChoices instance.")

        summary_content = choice.message.content
        if not summary_content:
            raise ValueError("Failed to summarize conversation.")

        return f"Previous conversation summary:\n{summary_content}"
