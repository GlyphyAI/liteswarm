# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, field_serializer

from liteswarm.types.context import ContextVariables
from liteswarm.types.swarm import (
    Agent,
    AgentExecutionResult,
    AgentResponse,
    AgentResponseChunk,
    CompletionResponseChunk,
    Message,
    ToolCallResult,
)
from liteswarm.types.swarm_team import Artifact, Plan, Task, TaskResult


class SwarmEventBase(BaseModel):
    """Base class for all Swarm events in the system.

    Used for pattern matching and routing of events throughout the system.
    All event types inherit from this class and implement specific event data.
    """

    type: str
    """Discriminator field used to identify the specific event type."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentExecutionStartEvent(SwarmEventBase):
    """Event emitted when agent execution starts.

    Called at the very beginning of agent execution flow, before any
    message processing or tool calls. Used to initialize execution
    state and prepare for incoming events.
    """

    type: Literal["agent_execution_start"] = "agent_execution_start"
    """Discriminator field."""

    agent: Agent
    """Agent that starts the execution."""

    messages: list[Message]
    """Messages being sent to the agent."""

    context_variables: ContextVariables | None = None
    """Optional context variables for execution."""


class AgentExecutionCompleteEvent(SwarmEventBase):
    """Event emitted when agent execution completes.

    Called when an agent execution flow reaches completion, after all
    message processing and tool calls are done. Contains the final
    execution result with all responses and messages.
    """

    type: Literal["agent_execution_complete"] = "agent_execution_complete"
    """Discriminator field."""

    agent: Agent
    """Agent that completed execution."""

    execution_result: AgentExecutionResult
    """Result of the execution."""


class CompletionResponseChunkEvent(SwarmEventBase):
    """Event emitted for each streaming update from the language model.

    Called each time new content is received from the model, before any
    agent-specific processing occurs. Used for monitoring raw model output.
    """

    type: Literal["completion_response_chunk"] = "completion_response_chunk"
    """Discriminator field."""

    response_chunk: CompletionResponseChunk
    """Raw completion response chunk from the model."""


class AgentResponseChunkEvent(SwarmEventBase):
    """Event emitted for each streaming update from an agent.

    Called each time new content is received from an agent, including both
    text content and tool call updates. Used for real-time monitoring of
    agent responses.
    """

    type: Literal["agent_response_chunk"] = "agent_response_chunk"
    """Discriminator field."""

    agent: Agent
    """Agent that generated the response."""

    response_chunk: AgentResponseChunk
    """Processed agent response chunk."""


class AgentResponseEvent(SwarmEventBase):
    type: Literal["agent_response"] = "agent_response"
    """Discriminator field."""

    agent: Agent
    """Agent that generated the response."""

    response: AgentResponse
    """Response generated by the agent."""


class ToolCallResultEvent(SwarmEventBase):
    """Event emitted when a tool call execution completes.

    Called after a tool finishes execution, with either a result or error.
    Used for processing tool outputs and updating system state.
    """

    type: Literal["tool_call_result"] = "tool_call_result"
    """Discriminator field."""

    agent: Agent
    """Agent that made the tool call."""

    tool_call_result: ToolCallResult
    """Result of the tool execution."""


class AgentSwitchEvent(SwarmEventBase):
    """Event emitted when switching between agents.

    Called when the conversation transitions from one agent to another.
    The first agent in a conversation will have previous_agent as None.
    """

    type: Literal["agent_switch"] = "agent_switch"
    """Discriminator field."""

    prev_agent: Agent | None
    """Agent being switched from, None if first agent."""

    next_agent: Agent
    """Agent being switched to, never None."""


class AgentActivateEvent(SwarmEventBase):
    """Event emitted when an agent becomes active.

    Called when an agent is set as the active executor, either as the
    initial agent or after a switch. Used to track the currently
    active agent in the execution flow.
    """

    type: Literal["agent_activate"] = "agent_activate"
    """Discriminator field."""

    agent: Agent
    """Agent that was activated."""


class AgentStartEvent(SwarmEventBase):
    """Event emitted when an agent starts message processing.

    Called right before an agent begins processing its message queue,
    after instructions are resolved and context is prepared. Used to
    track the beginning of an agent's response generation phase.
    """

    type: Literal["agent_start"] = "agent_start"
    """Discriminator field."""

    agent: Agent
    """Agent that starts processing."""

    agent_instructions: str
    """Instructions resolved for the agent."""

    messages: list[Message]
    """Messages to be processed by the agent."""


class AgentCompleteEvent(SwarmEventBase):
    """Event emitted when an agent becomes stale.

    Called when an agent finishes its current task and becomes stale.
    Occurs after generating a response or completing tool-related processing.
    Contains the agent's final response and messages from its active phase.
    """

    type: Literal["agent_complete"] = "agent_complete"
    """Discriminator field."""

    agent: Agent
    """Agent that completed processing."""

    agent_instructions: str
    """Instructions used during processing."""

    response: AgentResponse | None
    """Final response from the agent."""

    messages: list[Message]
    """Messages generated during processing."""


class ErrorEvent(SwarmEventBase):
    """Event emitted when an error occurs during execution.

    Called when an error occurs during any phase of operation, including
    content generation, tool calls, or response processing. The agent
    may be None if the error occurred outside agent context.
    """

    type: Literal["error"] = "error"
    """Discriminator field."""

    agent: Agent | None
    """Agent that encountered the error, None for system-level errors."""

    error: Exception
    """Exception that occurred."""

    @field_serializer("error")
    def serialize_error(self, error: Exception) -> str:
        """Serialize Exception object to string representation.

        This method is used by Pydantic to convert Exception objects into
        a serializable format for JSON encoding. It ensures that error
        information can be properly transmitted and logged.

        Args:
            error: Exception object to serialize.

        Returns:
            String representation of the error.
        """
        return str(error)


class PlanCreateEvent(SwarmEventBase):
    """Event emitted when a new plan is successfully created.

    Called after a planning agent successfully creates a structured plan
    with a unique ID. Used to analyze the plan or prepare resources
    before execution.
    """

    type: Literal["plan_create"] = "plan_create"
    """Discriminator field."""

    plan: Plan
    """Newly created execution plan."""


class PlanExecutionStartEvent(SwarmEventBase):
    """Event emitted when a plan starts execution.

    Called when a plan starts execution, after member assignment but
    before actual processing. Used to track task progress and prepare
    resources.
    """

    type: Literal["plan_execution_start"] = "plan_execution_start"
    """Discriminator field."""

    plan: Plan
    """Plan that started execution."""


class TaskStartEvent(SwarmEventBase):
    """Event emitted when a task starts execution.

    Called when a task starts execution, after member assignment but
    before actual processing. Used to track task progress and prepare
    resources.
    """

    type: Literal["task_start"] = "task_start"
    """Discriminator field."""

    task: Task
    """Task beginning execution."""

    task_instructions: str
    """Instructions for the task."""

    messages: list[Message]
    """Messages to be sent to the agent executing the task."""


class TaskCompleteEvent(SwarmEventBase):
    """Event emitted when a task finishes execution.

    Called when a task completes execution successfully. Used to process
    results and trigger dependent tasks.
    """

    type: Literal["task_complete"] = "task_complete"
    """Discriminator field."""

    task: Task
    """Task that completed execution."""

    task_result: TaskResult
    """Result of the completed task."""

    task_context_variables: ContextVariables | None
    """Context variables used during task execution."""


class PlanExecutionCompleteEvent(SwarmEventBase):
    """Event emitted when all tasks in a plan are completed.

    Called when all tasks in a plan have finished execution successfully.
    Used to perform cleanup or trigger follow-up actions.
    """

    type: Literal["plan_execution_complete"] = "plan_execution_complete"
    """Discriminator field."""

    plan: Plan
    """Plan that completed execution."""

    artifact: Artifact
    """Artifact containing the results of plan execution."""


SwarmEvent = Annotated[
    AgentExecutionStartEvent
    | AgentExecutionCompleteEvent
    | CompletionResponseChunkEvent
    | AgentResponseChunkEvent
    | AgentResponseEvent
    | ToolCallResultEvent
    | AgentSwitchEvent
    | AgentActivateEvent
    | AgentStartEvent
    | AgentCompleteEvent
    | ErrorEvent
    | PlanCreateEvent
    | PlanExecutionStartEvent
    | TaskStartEvent
    | TaskCompleteEvent
    | PlanExecutionCompleteEvent,
    Discriminator("type"),
]
"""Type alias for all Swarm events."""
