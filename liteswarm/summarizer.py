import asyncio
from collections.abc import Sequence
from typing import Protocol

from litellm import acompletion
from litellm.types.utils import Choices, ModelResponse

from liteswarm.types import Message

GroupedMessages = tuple[list[Message], list[Message]]


class Summarizer(Protocol):
    """Protocol for conversation summarizers."""

    def needs_summarization(self, messages: Sequence[Message]) -> bool:
        """Determine if the conversation history needs summarization."""
        ...

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Create a concise summary of the conversation history."""
        ...

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Summarize conversation history while preserving important context."""
        ...


class LiteSummarizer:
    """Summarizes conversations using LiteLLM while preserving important context."""

    def __init__(  # noqa: PLR0913
        self,
        model: str = "claude-3-5-haiku-20241022",
        system_prompt: str | None = None,
        summarize_prompt: str | None = None,
        max_history_length: int = 50,
        preserve_recent: int = 25,
        chunk_size: int = 25,
    ) -> None:
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

    def _filter_tool_call_pairs(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Filter messages to maintain only complete tool call/result pairs.

        Args:
            messages: List of messages to filter

        Returns:
            List of messages with only complete tool call/result pairs
        """
        # Find valid tool call/result pairs
        tool_call_ids = set()
        tool_result_ids = set()

        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.id:
                        tool_call_ids.add(tool_call.id)
            elif message.role == "tool" and message.tool_call_id:
                tool_result_ids.add(message.tool_call_id)

        valid_tool_ids = tool_call_ids.intersection(tool_result_ids)

        # Filter messages to maintain valid tool call/result pairs
        filtered_messages = []

        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                filtered_tool_calls = [
                    tool_call for tool_call in message.tool_calls if tool_call.id in valid_tool_ids
                ]

                msg = Message(
                    role=message.role,
                    content=message.content,
                    tool_calls=filtered_tool_calls or None,
                )

                filtered_messages.append(msg)
            elif message.role == "tool":
                if message.tool_call_id in valid_tool_ids:
                    filtered_messages.append(message)
            else:
                filtered_messages.append(message)

        return filtered_messages

    def _group_messages_for_summary(self, messages: list[Message]) -> GroupedMessages:
        """Group messages into those that should be preserved and those that can be summarized.

        The method preserves the most recent messages and ensures tool call/result pairs
        stay together. Messages are grouped chronologically to maintain conversation flow.

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
            return self._filter_tool_call_pairs(to_preserve), []

        filtered_to_preserve = self._filter_tool_call_pairs(to_preserve)
        filtered_to_summarize = self._filter_tool_call_pairs(to_summarize)

        return filtered_to_preserve, filtered_to_summarize

    async def _summarize_message_chunk(self, messages: Sequence[Message]) -> str:
        """Summarize a chunk of messages."""
        summary_messages = [
            Message(role="system", content=self.system_prompt),
            *messages,
            Message(role="user", content=self.summarize_prompt),
        ]

        # print(f"\n\n[DEBUG] Summarizing messages: {summary_messages}\n\n")

        response = await acompletion(
            model=self.model,
            messages=[msg.model_dump(exclude_none=True) for msg in summary_messages],
            stream=False,
        )

        # print(f"\n\n[DEBUG] Summarization response: {response}\n\n")

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

        Args:
            messages: List of messages to chunk

        Returns:
            List of message chunks, where each chunk maintains complete tool call/result pairs
        """
        if not messages:
            return []

        chunks: list[list[Message]] = []
        current_chunk: list[Message] = []
        pending_tool_calls: dict[str, Message] = {}

        def add_chunk() -> None:
            """Add a chunk of messages to the list of chunks."""
            if current_chunk:
                filtered_chunk = self._filter_tool_call_pairs(current_chunk)
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
        """Determine if the conversation history needs summarization."""
        return len(messages) > self.max_history_length

    async def summarize(self, messages: Sequence[Message]) -> str:
        """Create a concise summary of the conversation history."""
        return await self._summarize_message_chunk(messages)

    async def summarize_history(self, messages: list[Message]) -> list[Message]:
        """Summarize conversation history while preserving important context.

        This method:
        1. Preserves recent messages and those with important context (tool calls, etc.)
        2. Summarizes older messages in chunks concurrently to maintain coherent context
        3. Combines preserved messages and summaries in chronological order

        Args:
            messages: The complete list of messages to process

        Returns:
            A list of Message objects containing preserved messages and summaries
        """
        if not messages:
            return []

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

        return final_messages
