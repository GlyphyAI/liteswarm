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
"""Tuple used to group messages into two lists:
- Messages to preserve unchanged
- Messages to summarize
"""


class Summarizer(Protocol):
    """Protocol for conversation summarizers.

    This protocol defines the interface that all summarizer implementations must follow.
    Summarizers are responsible for managing conversation history length while preserving
    context and maintaining tool call/result relationships.

    The protocol ensures that all summarizers provide consistent functionality for:
    - Determining when summarization is needed
    - Creating concise summaries of conversations
    - Processing complete conversation histories

    Notes:
        Implementations should focus on maintaining conversation coherence
        and preserving important context while reducing history length.
    """

    def needs_summarization(self, messages: Sequence[Message]) -> bool:
        """Determine if the conversation history needs summarization.

        Args:
            messages: The conversation messages to evaluate for summarization.

        Returns:
            True if the messages exceed length limits and need summarization,
            False otherwise.
        """
        ...

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Create a concise summary of the conversation history.

        Args:
            messages: The conversation messages to summarize.

        Returns:
            A string containing the summarized conversation content.

        Raises:
            NotImplementedError: If the summarizer doesn't support direct summarization.
        """
        ...

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Summarize conversation history while preserving important context.

        Args:
            messages: The complete list of messages to process and summarize.

        Returns:
            A list of Message objects containing the processed history with
            important context preserved.
        """
        ...


class LiteSummarizer:
    """Summarizes conversations using LiteLLM while preserving important context.

    This summarizer implements an intelligent conversation management strategy:
    - Maintains recent messages unchanged for immediate context
    - Summarizes older messages in chunks for efficiency
    - Preserves tool call/result relationships for coherence
    - Processes chunks concurrently for better performance
    - Excludes system messages from summarization
    - Supports fallback to another summarizer on failure

    Example:
        ```python
        summarizer = LiteSummarizer(
            model="gpt-4o",
            max_history_length=50,
            preserve_recent=25
        )

        # Check if summarization is needed
        if summarizer.needs_summarization(messages):
            # Summarize while preserving recent context
            summarized = await summarizer.summarize_history(messages)
        ```
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str = "gpt-4o",
        system_prompt: str | None = None,
        summarize_prompt: str | None = None,
        max_history_length: int = 50,
        preserve_recent: int = 25,
        chunk_size: int = 25,
        fallback_summarizer: Summarizer | None = None,
    ) -> None:
        """Initialize the LiteLLM-based summarizer.

        Args:
            model: The LLM model identifier for summarization.
            system_prompt: Custom system prompt for the summarization model.
                If None, uses a default prompt focused on clear and concise summaries.
            summarize_prompt: Custom prompt template for summarization requests.
                If None, uses a default template focusing on key information.
            max_history_length: Maximum number of messages before triggering summarization.
            preserve_recent: Number of recent messages to keep unchanged.
            chunk_size: Number of messages to include in each summarization chunk.
            fallback_summarizer: Optional backup summarizer to use if LLM fails.
                If None, creates a TruncationSummarizer with the same parameters.
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

        Implements a careful grouping strategy:
        - Filters out system messages as they're handled separately
        - Preserves the most recent messages for immediate context
        - Ensures tool call/result pairs remain together
        - Prepares older messages for efficient chunked summarization

        Args:
            messages: List of messages to process and group.

        Returns:
            A tuple containing:
            - List of messages to preserve unchanged
            - List of messages to summarize
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

        Handles the LLM-based summarization process:
        - Prepares messages with system and user prompts
        - Sends request to the LLM
        - Validates and processes the response
        - Handles potential errors

        Args:
            messages: The chunk of messages to summarize.

        Returns:
            A string containing the chunk's summary.

        Raises:
            TypeError: If the LLM response format is unexpected.
            ValueError: If the LLM fails to generate a valid summary.
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

        Implements a sophisticated chunking strategy:
        - Respects the configured chunk size limit when possible
        - Keeps related tool calls and results in the same chunk
        - Filters each chunk to maintain tool call integrity
        - Handles pending tool calls across chunk boundaries

        Args:
            messages: List of messages to divide into chunks.

        Returns:
            List of message chunks, each containing complete tool call/result pairs.
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
            messages: The conversation messages to evaluate.

        Returns:
            True if the number of messages exceeds max_history_length,
            False otherwise.
        """
        return len(messages) > self.max_history_length

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Create a direct summary of the provided messages.

        This is a convenience method for summarizing a single chunk of messages.
        For complete history management, use summarize_history instead.

        Args:
            messages: The conversation messages to summarize.

        Returns:
            A string containing the summarized conversation.
        """
        return await self._summarize_message_chunk(messages)

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Summarize conversation history while preserving important context.

        Implements a comprehensive summarization strategy:
        - Attempts LLM-based summarization first
        - Falls back to alternative summarizer if LLM fails
        - Preserves recent messages and tool call relationships
        - Processes older messages in parallel chunks
        - Maintains chronological order in the final output

        Args:
            messages: The complete list of messages to process.

        Returns:
            A list of Message objects containing:
            - Summarized older messages (if any)
            - Preserved recent messages
            - Complete tool call/result pairs

        Raises:
            TypeError: If LLM responses are not in the expected format.
            ValueError: If summarization fails and no fallback is available.

        Example:
            ```python
            summarizer = LiteSummarizer()

            # Process complete conversation history
            summarized = await summarizer.summarize_history(messages)

            # Use summarized messages in next conversation turn
            agent.process_messages(summarized)
            ```
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

    This summarizer provides a lightweight alternative when:
    - You want to minimize API calls and avoid LLM costs
    - Exact history preservation isn't critical
    - You need predictable token usage
    - You want faster processing
    - Recent context is more important than historical context

    The summarizer maintains tool call/result relationships within the preserved
    messages but does not attempt to preserve historical context through summarization.

    Example:
        ```python
        summarizer = TruncationSummarizer(
            max_history_length=50,
            preserve_recent=25
        )

        # Truncate history while preserving recent messages
        truncated = await summarizer.summarize_history(messages)
        ```
    """

    def __init__(
        self,
        max_history_length: int = 50,
        preserve_recent: int = 25,
    ) -> None:
        """Initialize the truncation summarizer.

        Args:
            max_history_length: Maximum messages to keep before truncating.
            preserve_recent: Number of recent messages to preserve when truncating.
        """
        self.max_history_length = max_history_length
        self.preserve_recent = preserve_recent

    def needs_summarization(self, messages: Sequence[Message]) -> bool:
        """Check if the message history needs truncation.

        Args:
            messages: The conversation messages to evaluate.

        Returns:
            True if the number of messages exceeds max_history_length,
            False otherwise.
        """
        return len(messages) > self.max_history_length

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Direct summarization is not supported by this summarizer.

        Args:
            messages: The messages that would be summarized.

        Raises:
            NotImplementedError: Always, as this summarizer only supports truncation.
        """
        raise NotImplementedError("Truncation summarizer does not support summarization.")

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Truncate conversation history to keep only recent messages.

        Implements a simple but effective truncation strategy:
        - Removes system messages (they should be added by the agent)
        - Keeps only the most recent messages up to preserve_recent
        - Ensures tool call/result pairs stay together
        - Maintains chronological order

        Args:
            messages: The complete list of messages to process.

        Returns:
            A list of Message objects containing only the most recent messages
            with complete tool call/result pairs.

        Example:
            ```python
            summarizer = TruncationSummarizer(preserve_recent=25)

            # Keep only the 25 most recent messages
            recent = await summarizer.summarize_history(messages)
            ```
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
