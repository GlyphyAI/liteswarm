# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from enum import Enum
from typing import Literal, TypeAlias

from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    FunctionCall,
    Usage,
)
from litellm.types.utils import Delta as LiteDelta
from pydantic import BaseModel, ConfigDict, Field

from liteswarm.types.context import ContextVariables
from liteswarm.types.llm import LLM

AgentInstructions: TypeAlias = str | Callable[[ContextVariables], str]
"""Agent instructions - either a string or a function that takes context variables.

Can be either:
- A static string defining the agent's behavior
- A function that generates instructions using context

Example:
```python
# Static instructions
instructions: Instructions = "You are a helpful assistant."

# Dynamic instructions
AGENT_INSTRUCTIONS = \"\"\"
You are helping {user_name}.
Focus on their {interests}.
Use {preferred_language}.
\"\"\".strip()

def generate_instructions(context: ContextVariables) -> str:
    return AGENT_INSTRUCTIONS.format(**context)
```
"""


class AgentState(str, Enum):
    """The state of an agent in the conversation lifecycle.

    States indicate whether an agent is:
    - Ready to handle tasks (IDLE)
    - Currently processing a task (ACTIVE)
    - Needs to be replaced (STALE)
    """

    IDLE = "idle"
    """The agent is idle and waiting for a task"""

    ACTIVE = "active"
    """The agent is actively working on a task"""

    STALE = "stale"
    """The agent is stale and needs to be replaced"""


class Message(BaseModel):
    """A message in the conversation between users, assistants, and tools.

    Messages represent all communication in a conversation, including:
    - System instructions
    - User inputs
    - Assistant responses
    - Tool call results

    Example:
    ```python
    # System message with instructions
    system_msg = Message(
        role="system",
        content="You are a helpful assistant."
    )

    # User question
    user_msg = Message(
        role="user",
        content="What's 2 + 2?"
    )

    # Assistant response with tool call
    assistant_msg = Message(
        role="assistant",
        content="Let me calculate that.",
        tool_calls=[
            ChatCompletionDeltaToolCall(
                id="call_1",
                function={"name": "add", "arguments": '{"a": 2, "b": 2}'}
            )
        ]
    )

    # Tool response
    tool_msg = Message(
        role="tool",
        content="4",
        tool_call_id="call_1"
    )
    ```
    """

    role: Literal["assistant", "user", "system", "tool"]
    """The role of the message sender ("assistant", "user", "system", or "tool")"""

    content: str | None = None
    """The text content of the message"""

    tool_calls: list[ChatCompletionDeltaToolCall] | None = None
    """List of tool calls made in this message"""

    tool_call_id: str | None = None
    """ID of the tool call this message is responding to"""

    audio: ChatCompletionAudioResponse | None = None
    """Audio response data, if any"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="allow",
    )


class ToolMessage(BaseModel):
    """A message resulting from a tool call, optionally including a new agent.

    Tool messages can:
    - Return simple responses from tool execution
    - Trigger agent switches with new context
    - Update conversation context variables

    Example:
    ```python
    # Simple tool response
    calc_msg = ToolMessage(
        message=Message(
            role="tool",
            content="4",
            tool_call_id="calc_1"
        )
    )

    # Agent switch with context
    switch_msg = ToolMessage(
        message=Message(
            role="tool",
            content="Switching to math expert",
            tool_call_id="switch_1"
        ),
        agent=Agent(
            id="math-expert",
            instructions="You are a math expert.",
            llm=LLMConfig(model="gpt-4o")
        ),
        context_variables=ContextVariables(specialty="mathematics")
    )
    ```
    """

    message: Message
    """The message containing the tool's response"""

    agent: "Agent | None" = None
    """Optional new agent to switch to (for agent-switching tools)"""

    context_variables: ContextVariables | None = None
    """Context variables to pass to the next agent"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class Delta(BaseModel):
    """A partial update in a streaming response.

    Deltas represent incremental updates during streaming, containing:
    - Text content chunks
    - Role updates
    - Tool/function calls
    - Audio response data

    Example:
    ```python
    # Content update
    content_delta = Delta(
        role="assistant",
        content="Hello, "
    )

    # Tool call update
    tool_delta = Delta(
        tool_calls=[
            ChatCompletionDeltaToolCall(
                id="calc_1",
                function={"name": "add", "arguments": '{"a": 2'}
            )
        ]
    )

    # Function completion
    completion_delta = Delta(
        tool_calls=[
            ChatCompletionDeltaToolCall(
                id="calc_1",
                function={"name": "add", "arguments": ', "b": 2}'}
            )
        ]
    )
    ```
    """

    content: str | None = None
    """Text content in this update"""

    role: str | None = None
    """Role of the message being updated"""

    function_call: FunctionCall | dict | None = None
    """Function call information"""

    tool_calls: list[ChatCompletionDeltaToolCall | dict] | None = None
    """Tool calls being made"""

    audio: ChatCompletionAudioResponse | None = None
    """Audio response data"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )

    @classmethod
    def from_delta(cls, delta: LiteDelta) -> "Delta":
        """Create a Delta instance from a LiteLLM delta object.

        Args:
            delta: The LiteLLM delta to convert

        Returns:
            New Delta instance with copied attributes
        """
        return cls(
            content=delta.content,
            role=delta.role,
            function_call=delta.function_call,
            tool_calls=delta.tool_calls,
            audio=delta.audio,
        )


class ResponseCost(BaseModel):
    """Cost information for a model response.

    Tracks the cost of tokens used in:
    - The input prompt
    - The model's completion

    Example:
    ```python
    cost = ResponseCost(
        prompt_tokens_cost=0.001,  # Cost for input tokens
        completion_tokens_cost=0.002  # Cost for output tokens
    )

    total_cost = cost.prompt_tokens_cost + cost.completion_tokens_cost
    ```
    """

    prompt_tokens_cost: float
    """Cost of tokens in the prompt"""

    completion_tokens_cost: float
    """Cost of tokens in the completion"""


class Agent(BaseModel):
    """An AI agent that can participate in conversations and use tools.

    Agents are the core participants in conversations, with capabilities for:
    - Following system instructions
    - Using tools to perform actions
    - Maintaining conversation state
    - Supporting parallel tool execution

    Example:
    ```python
    def search_docs(query: str) -> str:
        \"\"\"Search documentation for information.\"\"\"
        return f"Results for: {query}"

    def generate_code(spec: str) -> str:
        \"\"\"Generate code based on specification.\"\"\"
        return f"Code implementing: {spec}"

    # Create a coding assistant agent
    coding_agent = Agent(
        id="coding-assistant",
        instructions='''You are a coding assistant.
                       Use search_docs to find relevant information.
                       Use generate_code to implement solutions.''',
        llm=LLMConfig(
            model="gpt-4o",
            tools=[search_docs, generate_code],
            tool_choice="auto",
            parallel_tool_calls=True,
            temperature=0.7,
            response_format=CodeOutput  # Optional Pydantic model
        )
    )

    # Create an agent with dynamic instructions
    def get_instructions(context: ContextVariables) -> str:
        return f'''You are helping {context["user_name"]}.
                  Your expertise is in {context["domain"]}.'''

    expert_agent = Agent(
        id="domain-expert",
        instructions=get_instructions,
        llm=LLMConfig(
            model="gpt-4o",
            max_tokens=2000,
            temperature=0.3
        )
    )
    ```
    """

    id: str
    """Unique identifier for the agent"""

    instructions: AgentInstructions
    """System prompt defining behavior (string or function)"""

    llm: LLM
    """LLM configuration for the agent's behavior and capabilities"""

    state: AgentState = AgentState.IDLE
    """Current state of the agent (idle, active, or stale)"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ToolCallResult(BaseModel):
    """Base class for results of tool calls.

    Provides common structure for all tool call results:
    - The original tool call that produced this result
    - Subclasses add specific result types
    """

    tool_call: ChatCompletionDeltaToolCall
    """The tool call that produced this result"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ToolCallMessageResult(ToolCallResult):
    """Result of a tool call that produced a message.

    Used for tool calls that return data or text responses.
    Can optionally update context variables.

    Example:
    ```python
    result = ToolCallMessageResult(
        tool_call=calc_call,
        message=Message(
            role="tool",
            content="42",
            tool_call_id="calc_1"
        ),
        context_variables=ContextVariables(last_result=42)
    )
    ```
    """

    message: Message
    """The message containing the tool's response"""

    context_variables: ContextVariables | None = None
    """Context variables to pass to the next agent"""


class ToolCallAgentResult(ToolCallResult):
    """Result of a tool call that produced a new agent.

    Used for agent-switching tools that return a new agent
    to handle the conversation. Can include a transition
    message and context updates.

    Example:
    ```python
    result = ToolCallAgentResult(
        tool_call=switch_call,
        agent=Agent(
            id="expert",
            instructions="You are an expert...",
            llm=LLMConfig(model="gpt-4o")
        ),
        message=Message(
            role="tool",
            content="Switching to expert",
            tool_call_id="switch_1"
        ),
        context_variables=ContextVariables(expertise="math")
    )
    ```
    """

    agent: Agent
    """The new agent to switch to"""

    message: Message | None = None
    """Optional message to add to the conversation"""

    context_variables: ContextVariables | None = None
    """Context variables to pass to the next agent"""


class ToolCallFailureResult(ToolCallResult):
    """Result of a failed tool call.

    Captures errors that occur during tool execution
    for proper error handling and reporting.

    Example:
    ```python
    result = ToolCallFailureResult(
        tool_call=failed_call,
        error=ValueError("Invalid input")
    )
    ```
    """

    error: Exception
    """The exception that occurred during tool execution"""


class CompletionResponse(BaseModel):
    """A response chunk from the language model.

    Represents a single chunk in a streaming response,
    including content updates and metadata.

    Example:
    ```python
    response = CompletionResponse(
        delta=Delta(content="Hello"),
        finish_reason=None,
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11
        ),
        response_cost=ResponseCost(
            prompt_tokens_cost=0.0001,
            completion_tokens_cost=0.0002
        )
    )
    ```
    """

    delta: Delta
    """The content update in this chunk"""

    finish_reason: str | None = None
    """Why the response ended (if it did)"""

    usage: Usage | None = None
    """Token usage statistics"""

    response_cost: ResponseCost | None = None
    """Cost information for this response"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class AgentResponse(BaseModel):
    """A processed response from an agent, including accumulated state.

    Combines the current update with accumulated content and
    tool calls from previous updates in the stream.

    Example:
    ```python
    response = AgentResponse(
        delta=Delta(content=" world"),
        content="Hello world",  # Accumulated
        tool_calls=[calc_call],  # Accumulated
        finish_reason=None,
        usage=Usage(...),
        response_cost=ResponseCost(...)
    )
    ```
    """

    delta: Delta
    """The content update in this response"""

    finish_reason: str | None = None
    """Why the response ended (if it did)"""

    content: str | None = None
    """Accumulated content so far"""

    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)
    """Accumulated tool calls"""

    usage: Usage | None = None
    """Token usage statistics"""

    response_cost: ResponseCost | None = None
    """Cost information for this response"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )


class ConversationState(BaseModel):
    """Complete state of a conversation.

    Captures the entire state of a conversation, including:
    - Final content and active agent
    - Message histories (full and agent-specific)
    - Usage statistics and costs

    Example:
    ```python
    state = ConversationState(
        content="Final response",
        agent=current_agent,
        agent_messages=[...],  # Current agent's context
        agent_queue=[backup_agent],  # Queued agents
        messages=[...],  # Full conversation history
        usage=Usage(...),
        response_cost=ResponseCost(...)
    )
    ```
    """

    content: str | None = None
    """Final content of the conversation"""

    agent: Agent | None = None
    """Currently active agent"""

    agent_messages: list[Message] = Field(default_factory=list)
    """Messages for the current agent"""

    agent_queue: list[Agent] = Field(default_factory=list)
    """Queue of agents waiting to be activated"""

    messages: list[Message] = Field(default_factory=list)
    """Complete conversation history"""

    usage: Usage | None = None
    """Total token usage statistics"""

    response_cost: ResponseCost | None = None
    """Total cost information"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
