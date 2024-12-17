# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
from typing import Protocol

from liteswarm.types.swarm import Agent, ContextVariables, Message
from liteswarm.utils.unwrap import unwrap_instructions


class Memory(Protocol):
    """Protocol for managing conversation history and context.

    Defines the interface for conversation memory management, handling:
    - Full conversation history preservation
    - Working context maintenance
    - Agent context preparation
    - Message manipulation operations

    The protocol supports multiple memory strategies:
    - Complete history tracking
    - Dynamic context management
    - Efficient state updates
    - Safe concurrent access
    - History restoration

    Notes:
        - All methods should be thread-safe
        - History modifications should be atomic
        - Deep copies prevent external state mutation
        - Full history is always preserved

    Examples:
        Basic usage:
            ```python
            class CustomMemory(Memory):
                def append_message(self, message: Message) -> None:
                    # Add to both histories
                    self._full_history.append(copy.deepcopy(message))
                    self._working_history.append(copy.deepcopy(message))

                def get_working_history(self) -> list[Message]:
                    # Return current context
                    return copy.deepcopy(self._working_history)
            ```

        Custom context management:
            ```python
            class SlidingMemory(Memory):
                def append_message(self, message: Message) -> None:
                    self._full_history.append(copy.deepcopy(message))
                    # Keep last N messages for working context
                    if len(self._working_history) >= self._window_size:
                        self._working_history.pop(0)
                    self._working_history.append(copy.deepcopy(message))
            ```
    """

    def get_full_history(self) -> list[Message]:
        """Return the complete conversation history.

        Returns a copy of the full conversation history, including all messages
        from all agents and tools. This history represents the complete state
        of the conversation from start to finish.

        Returns:
            List of all messages in chronological order.

        Notes:
            - Returns a deep copy to prevent external modifications
            - Includes system messages, user inputs, and agent responses
            - Contains tool calls and their results
            - Preserves message order and relationships
        """
        ...

    def get_working_history(self) -> list[Message]:
        """Return the current working history used for context.

        Returns a copy of the working history, which is the history that is
        currently being used for context. This history is typically a subset
        of the full history, and may be updated by summarization or trimming
        if necessary.

        Returns:
            List of messages currently being used for LLM context.

        Notes:
            - Returns a deep copy to prevent external modifications
            - May be summarized or trimmed version of full history
            - Optimized for context window constraints
            - Maintains essential conversation flow
        """
        ...

    def set_history(self, messages: list[Message]) -> None:
        """Set the conversation history.

        Replaces the current conversation history with the provided messages.
        This method is useful for restoring a previous conversation state or
        initializing a new conversation with existing context.

        Args:
            messages: List of messages to set as the conversation history.

        Notes:
            - Replaces both full and working history
            - Creates a deep copy of provided messages
            - Clears any existing history
            - Should be called before starting a new conversation
        """
        ...

    def set_working_history(self, messages: list[Message]) -> None:
        """Set the working history.

        Updates the current working history while preserving the full history.
        Useful for applying summarization or context management strategies
        without losing the complete conversation record.

        Args:
            messages: List of messages to set as the working history.

        Notes:
            - Only updates working history
            - Creates deep copy of messages
            - Preserves full history intact
            - Useful for context management
        """
        ...

    def pop_last_message(self) -> Message | None:
        """Remove and return the last message from history.

        Removes the most recent message from both histories, enabling:
        - Message correction or removal
        - Conversation state rollback
        - Error recovery
        - History manipulation

        Returns:
            Last message if history exists, None if empty.

        Notes:
            - Affects both full and working history
            - Returns None if history is empty
            - Operation is atomic
            - Maintains history consistency
        """
        ...

    def append_message(self, message: Message) -> None:
        """Add a new message to conversation history.

        Appends message to both histories, ensuring:
        - Consistent state updates
        - Safe concurrent access
        - Proper message copying
        - History synchronization

        Args:
            message: Message to append to history.

        Notes:
            - Updates both histories
            - Creates deep copy of message
            - Thread-safe operation
            - May trigger summarization
        """
        ...

    def extend_messages(self, messages: list[Message]) -> None:
        """Add multiple messages to conversation history.

        Extends both histories with provided messages, ensuring:
        - Atomic batch updates
        - Consistent state
        - Safe message copying
        - History synchronization

        Args:
            messages: List of messages to add to history.

        Notes:
            - Updates both histories
            - Creates deep copies
            - Maintains order
            - Thread-safe operation
        """
        ...

    def clear_history(self) -> None:
        """Reset conversation history to initial state.

        Clears both full and working history, providing:
        - Clean state for new conversations
        - Memory cleanup
        - Resource management
        - History reset capability

        Notes:
            - Clears both histories
            - Operation is irreversible
            - Thread-safe operation
            - Resets to initial state
        """
        ...

    def create_agent_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        """Create complete context for agent execution.

        Builds agent context by combining:
        - System instructions with variable resolution
        - Filtered working history for relevance
        - Optional user prompt for direction
        - Required context for processing

        Args:
            agent: Agent requiring context preparation.
            prompt: Optional user prompt to include.
            context_variables: Variables for dynamic resolution.

        Returns:
            Complete message list for agent execution.

        Notes:
            - Resolves dynamic instructions
            - Filters irrelevant messages
            - Maintains conversation flow
            - Optimizes context relevance
        """
        ...


class LiteMemory(Memory):
    """Simple in-memory implementation of the Memory protocol.

    Provides basic memory management with:
    - Separate full and working history tracking
    - Deep copying for thread safety
    - Efficient state management
    - Basic context handling

    This implementation:
    - Maintains two message lists
    - Uses deep copies for safety
    - Provides atomic operations
    - Keeps histories synchronized

    Examples:
        Basic usage:
            ```python
            memory = LiteMemory()

            # Add some messages
            memory.append_message(Message(role="user", content="Hello"))
            memory.append_message(Message(role="assistant", content="Hi"))

            # Get current context
            context = memory.get_working_history()
            ```

    Notes:
        - Thread-safe through copying
        - Memory efficient for most uses
        - Suitable for standard conversations
        - Extendable for custom needs
    """

    def __init__(self) -> None:
        self._full_history: list[Message] = []
        self._working_history: list[Message] = []

    def get_full_history(self) -> list[Message]:
        return copy.deepcopy(self._full_history)

    def get_working_history(self) -> list[Message]:
        return copy.deepcopy(self._working_history)

    def set_history(self, messages: list[Message]) -> None:
        self._full_history = copy.deepcopy(messages)
        self._working_history = copy.deepcopy(messages)

    def set_working_history(self, messages: list[Message]) -> None:
        self._working_history = copy.deepcopy(messages)

    def pop_last_message(self) -> Message | None:
        if not self._full_history:
            return None

        last_message = self._full_history.pop()
        if self._working_history and self._working_history[-1] == last_message:
            self._working_history.pop()

        return last_message

    def append_message(self, message: Message) -> None:
        message_copy = copy.deepcopy(message)
        self._full_history.append(message_copy)
        self._working_history.append(message_copy)

    def extend_messages(self, messages: list[Message]) -> None:
        messages_copy = [copy.deepcopy(msg) for msg in messages]
        self._full_history.extend(messages_copy)
        self._working_history.extend(messages_copy)

    def clear_history(self) -> None:
        self._full_history.clear()
        self._working_history.clear()

    def create_agent_context(
        self,
        agent: Agent,
        prompt: str | None = None,
        context_variables: ContextVariables | None = None,
    ) -> list[Message]:
        instructions = unwrap_instructions(agent.instructions, context_variables)
        history = [msg for msg in self._working_history if msg.role != "system"]
        messages = [Message(role="system", content=instructions), *history]

        if prompt:
            messages.append(Message(role="user", content=prompt))

        return messages
