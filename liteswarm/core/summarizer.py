# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
from collections.abc import Sequence
from typing import Protocol

from litellm import acompletion
from litellm.types.utils import Choices, ModelResponse

from liteswarm.types.swarm import Message
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.messages import filter_tool_call_pairs

GroupedMessages = tuple[list[Message], list[Message]]


class Summarizer(Protocol):
    """Protocol for conversation summarizers.

    This protocol defines the interface that all summarizer implementations must follow.
    Summarizers are responsible for managing conversation history length while preserving
    context and maintaining tool call/result relationships.
    """

    def needs_summarization(self, messages: Sequence[Message]) -> bool:
        """Determine if the conversation history needs summarization.

        Args:
            messages: The messages to check

        Returns:
            True if the messages exceed length limits and need summarization
        """
        ...

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Create a concise summary of the conversation history.

        Args:
            messages: The messages to summarize

        Returns:
            A string containing the summarized conversation

        Raises:
            NotImplementedError: If the summarizer doesn't support direct summarization
        """
        ...

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Summarize conversation history while preserving important context.

        Args:
            messages: The complete list of messages to process

        Returns:
            A list of Message objects containing the processed history
        """
        ...


class LiteSummarizer:
    """Summarizes conversations using LiteLLM while preserving important context.

    This summarizer:
    1. Maintains recent messages unchanged
    2. Summarizes older messages in chunks
    3. Preserves tool call/result relationships
    4. Processes chunks concurrently for efficiency
    5. Excludes system messages from summarization
    6. Supports fallback to another summarizer on failure

    Attributes:
        model: The LLM model to use for summarization
        system_prompt: Instructions for the summarization model
        summarize_prompt: The prompt template for summarization requests
        max_history_length: Maximum number of messages before summarization
        preserve_recent: Number of recent messages to keep unchanged
        chunk_size: Number of messages to summarize in each chunk
        fallback_summarizer: Optional summarizer to use if LLM summarization fails
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str = "claude-3-5-haiku-20241022",
        system_prompt: str | None = None,
        summarize_prompt: str | None = None,
        max_history_length: int = 50,
        preserve_recent: int = 25,
        chunk_size: int = 25,
        fallback_summarizer: Summarizer | None = None,
    ) -> None:
        """Initialize the LiteLLM-based summarizer.

        Args:
            model: The LLM model to use for summarization
            system_prompt: Custom system prompt for the summarization model
            summarize_prompt: Custom prompt template for summarization requests
            max_history_length: Maximum number of messages before summarization
            preserve_recent: Number of recent messages to keep unchanged
            chunk_size: Number of messages to summarize in each chunk
            fallback_summarizer: Optional summarizer to use if LLM summarization fails
        """
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that creates clear and concise summaries of conversations. "
            "Focus on capturing key information, decisions, and context that would be important "
            "for continuing the conversation effectively."
        )
        self.summarize_prompt = summarize_prompt or (
            "Please summarize the above conversation segment. Focus on:\n"
            "1. Key decisions and outcomes\n"
            "2. Important context and information discovered\n"
            "3. Any pending questions or unresolved issues\n"
            "Keep the summary concise but informative."
        )
        self.max_history_length = max_history_length
        self.preserve_recent = preserve_recent
        self.chunk_size = chunk_size
        self.fallback_summarizer = fallback_summarizer or TruncationSummarizer(
            max_history_length=max_history_length,
            preserve_recent=preserve_recent,
        )

    def _group_messages_for_summary(self, messages: list[Message]) -> GroupedMessages:
        """Group messages into those to preserve and those to summarize.

        This method:
        1. Filters out system messages
        2. Preserves the most recent messages
        3. Ensures tool call/result pairs stay together
        4. Prepares older messages for chunked summarization

        Args:
            messages: List of messages to process

        Returns:
            A tuple of (messages_to_preserve, messages_to_summarize)
        """
        if not self.needs_summarization(messages):
            return messages, []

        non_system_messages = [msg for msg in messages if msg.role != "system"]

        if not non_system_messages:
            return [], []

        to_preserve = non_system_messages[-self.preserve_recent :]
        to_summarize = non_system_messages[: -self.preserve_recent]

        if not to_summarize:
            return filter_tool_call_pairs(to_preserve), []

        filtered_to_preserve = filter_tool_call_pairs(to_preserve)
        filtered_to_summarize = filter_tool_call_pairs(to_summarize)

        return filtered_to_preserve, filtered_to_summarize

    async def _summarize_message_chunk(self, messages: Sequence[Message]) -> str:
        """Summarize a chunk of messages using the LLM.

        Args:
            messages: The chunk of messages to summarize

        Returns:
            A string containing the chunk's summary

        Raises:
            TypeError: If the LLM response is not of the expected type
            ValueError: If the LLM fails to generate a summary
        """
        summary_messages = [
            Message(role="system", content=self.system_prompt),
            *messages,
            Message(role="user", content=self.summarize_prompt),
        ]

        log_verbose(f"Summarizing messages: {summary_messages}", level="DEBUG")

        response = await acompletion(
            model=self.model,
            messages=[msg.model_dump(exclude_none=True) for msg in summary_messages],
            stream=False,
        )

        log_verbose(f"Summarization response: {response}", level="DEBUG")

        if not isinstance(response, ModelResponse):
            raise TypeError("Expected a CompletionResponse instance.")

        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise TypeError("Expected a StreamingChoices instance.")

        summary_content = choice.message.content
        if not summary_content:
            raise ValueError("Failed to summarize conversation.")

        return summary_content

    def _create_message_chunks(self, messages: list[Message]) -> list[list[Message]]:
        """Create chunks of messages while preserving tool call/result pairs.

        This method:
        1. Respects the chunk size limit when possible
        2. Keeps tool calls and their results in the same chunk
        3. Filters each chunk to maintain tool call integrity
        4. Handles pending tool calls across chunk boundaries

        Args:
            messages: List of messages to chunk

        Returns:
            List of message chunks, each containing complete tool call/result pairs
        """
        if not messages:
            return []

        chunks: list[list[Message]] = []
        current_chunk: list[Message] = []
        pending_tool_calls: dict[str, Message] = {}

        def add_chunk() -> None:
            """Add a chunk of messages to the list of chunks."""
            if current_chunk:
                filtered_chunk = filter_tool_call_pairs(current_chunk)
                if filtered_chunk:
                    chunks.append(filtered_chunk)
                current_chunk.clear()
                pending_tool_calls.clear()

        def add_chunk_if_needed() -> None:
            """Add a chunk if the current chunk is full or has no pending tool calls."""
            if len(current_chunk) >= self.chunk_size and not pending_tool_calls:
                add_chunk()

        for message in messages:
            add_chunk_if_needed()

            if message.role == "assistant" and message.tool_calls:
                current_chunk.append(message)
                for tool_call in message.tool_calls:
                    if tool_call.id:
                        pending_tool_calls[tool_call.id] = message

            elif message.role == "tool" and message.tool_call_id:
                current_chunk.append(message)
                pending_tool_calls.pop(message.tool_call_id, None)
                add_chunk_if_needed()

            else:
                current_chunk.append(message)
                add_chunk_if_needed()

        if current_chunk:
            add_chunk()

        return chunks

    def needs_summarization(self, messages: Sequence[Message]) -> bool:
        """Check if the message history exceeds the length limit.

        Args:
            messages: The messages to check

        Returns:
            True if the number of messages exceeds max_history_length
        """
        return len(messages) > self.max_history_length

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Create a direct summary of the provided messages.

        This is a convenience method that summarizes a single chunk of messages.
        For full history management, use summarize_history instead.

        Args:
            messages: The messages to summarize

        Returns:
            A string containing the summarized conversation
        """
        return await self._summarize_message_chunk(messages)

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Summarize conversation history while preserving important context.

        This method attempts to summarize using the LLM first, and if that fails
        and fallback is enabled, uses the fallback summarizer.

        Args:
            messages: The complete list of messages to process

        Returns:
            A list of Message objects containing summarized history and recent messages

        Raises:
            TypeError: If LLM responses are not of the expected type
            ValueError: If summarization fails and fallback is disabled or not available
        """
        if not messages:
            return []

        try:
            # Split messages into those to preserve and those to summarize
            to_preserve, to_summarize = self._group_messages_for_summary(messages)
            if not to_summarize:
                return to_preserve

            # Create chunks that preserve tool call/result pairs
            chunks = self._create_message_chunks(to_summarize)
            tasks = [self._summarize_message_chunk(chunk) for chunk in chunks]

            # Run summarization tasks concurrently
            summaries: list[str]
            match len(tasks):
                case 0:
                    summaries = []
                case 1:
                    summaries = [await tasks[0]]
                case _:
                    summaries = await asyncio.gather(*tasks)

            # Combine summaries and preserved messages chronologically
            final_messages = []

            if summaries:
                combined_summary = "\n\n".join(summaries)
                final_messages.append(
                    Message(
                        role="assistant",
                        content=f"Previous conversation history:\n{combined_summary}",
                    )
                )

            final_messages.extend(to_preserve)

            log_verbose(
                "Final message count: %d (preserved=%d, summaries=%d)",
                len(final_messages),
                len(to_preserve),
                len(summaries),
                level="DEBUG",
            )

            return final_messages

        except Exception as e:
            if self.fallback_summarizer:
                log_verbose(
                    "LLM summarization failed, falling back to %s: %s",
                    self.fallback_summarizer.__class__.__name__,
                    str(e),
                    level="WARNING",
                )

                return await self.fallback_summarizer.summarize_history(messages)

            log_verbose(
                f"Summarization failed and no fallback available: {str(e)}",
                level="ERROR",
            )

            raise


class TruncationSummarizer:
    """A simple summarizer that truncates old messages and keeps only recent ones.

    This summarizer is useful when:
    1. You want to minimize API calls and avoid LLM costs
    2. Exact history preservation isn't critical
    3. You need predictable token usage
    4. You want faster processing
    5. Recent context is more important than historical context

    The summarizer maintains tool call/result relationships within the preserved messages
    but does not attempt to preserve historical context through summarization.
    """

    def __init__(
        self,
        max_history_length: int = 50,
        preserve_recent: int = 25,
    ) -> None:
        """Initialize the truncation summarizer.

        Args:
            max_history_length: Maximum number of messages to keep before truncating
            preserve_recent: Number of recent messages to preserve when truncating
        """
        self.max_history_length = max_history_length
        self.preserve_recent = preserve_recent

    def needs_summarization(self, messages: Sequence[Message]) -> bool:
        """Check if the message history needs truncation.

        Args:
            messages: The messages to check

        Returns:
            True if the number of messages exceeds max_history_length
        """
        return len(messages) > self.max_history_length

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Direct summarization is not supported by this summarizer.

        Raises:
            NotImplementedError: Always, as this summarizer only supports truncation
        """
        raise NotImplementedError("Truncation summarizer does not support summarization.")

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Truncate conversation history to keep only recent messages.

        This method:
        1. Removes system messages (they should be added by the agent)
        2. Keeps only the most recent messages up to preserve_recent
        3. Ensures tool call/result pairs stay together
        4. Maintains chronological order

        Args:
            messages: The complete list of messages to process

        Returns:
            A list of Message objects containing only the most recent messages
        """
        if not messages:
            return []

        # Filter out system messages
        non_system_messages = [msg for msg in messages if msg.role != "system"]

        if len(non_system_messages) <= self.preserve_recent:
            return filter_tool_call_pairs(non_system_messages)

        # Keep only the most recent messages
        recent_messages = non_system_messages[-self.preserve_recent :]

        # Filter tool calls to maintain pairs
        filtered_messages = filter_tool_call_pairs(recent_messages)

        log_verbose(
            "Truncated message count: %d (from=%d, preserved=%d)",
            len(filtered_messages),
            len(messages),
            self.preserve_recent,
            level="DEBUG",
        )

        return filtered_messages
