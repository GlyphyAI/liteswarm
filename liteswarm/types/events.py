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
    """Event emitted when an agent is activated.

    Called when an agent is activated, either for the first time or after a switch.
    """

    type: Literal["agent_activate"] = "agent_activate"
    """Discriminator field."""

    agent: Agent
    """Agent that was activated."""


class AgentBeginEvent(SwarmEventBase):
    """Event emitted when an agent begins execution.

    Called when an agent begins execution, either for the first time or after a switch.
    """

    type: Literal["agent_begin"] = "agent_begin"
    """Discriminator field."""

    agent: Agent
    """Agent that began execution."""

    messages: list[Message]
    """Messages to be sent to the agent."""


class AgentCompleteEvent(SwarmEventBase):
    """Event emitted when an agent completes execution.

    Called when an agent completes execution, either successfully or with an error.
    """

    type: Literal["agent_complete"] = "agent_complete"
    """Discriminator field."""

    agent: Agent
    """Agent that completed execution."""

    response: AgentResponse | None
    """Response generated by the agent."""

    messages: list[Message]
    """Messages generated by the agent."""


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


class CompleteEvent(SwarmEventBase):
    """Event emitted when execution reaches completion.

    Called when a conversation reaches its natural conclusion or is
    terminated.
    """

    type: Literal["complete"] = "complete"
    """Discriminator field."""


class PlanCreatedEvent(SwarmEventBase):
    """Event emitted when a new plan is successfully created.

    Called after a planning agent successfully creates a structured plan
    with a unique ID. Used to analyze the plan or prepare resources
    before execution.
    """

    type: Literal["plan_created"] = "plan_created"
    """Discriminator field."""

    plan: Plan
    """Newly created execution plan."""


class TaskStartedEvent(SwarmEventBase):
    """Event emitted when a task begins execution.

    Called when a task starts execution, after member assignment but
    before actual processing. Used to track task progress and prepare
    resources.
    """

    type: Literal["task_started"] = "task_started"
    """Discriminator field."""

    task: Task
    """Task beginning execution."""

    messages: list[Message]
    """Messages to be sent to the agent executing the task."""


class TaskCompletedEvent(SwarmEventBase):
    """Event emitted when a task finishes execution.

    Called when a task completes execution successfully. Used to process
    results and trigger dependent tasks.
    """

    type: Literal["task_completed"] = "task_completed"
    """Discriminator field."""

    task: Task
    """Task that completed execution."""

    task_result: TaskResult
    """Result of the completed task."""

    task_context_variables: ContextVariables | None
    """Context variables used during task execution."""


class PlanCompletedEvent(SwarmEventBase):
    """Event emitted when all tasks in a plan are completed.

    Called when all tasks in a plan have finished execution successfully.
    Used to perform cleanup or trigger follow-up actions.
    """

    type: Literal["plan_completed"] = "plan_completed"
    """Discriminator field."""

    plan: Plan
    """Plan that completed execution."""

    artifact: Artifact
    """Artifact containing the results of plan execution."""


SwarmEvent = Annotated[
    CompletionResponseChunkEvent
    | AgentResponseChunkEvent
    | AgentResponseEvent
    | ToolCallResultEvent
    | AgentSwitchEvent
    | AgentActivateEvent
    | AgentBeginEvent
    | AgentCompleteEvent
    | ErrorEvent
    | CompleteEvent
    | PlanCreatedEvent
    | TaskStartedEvent
    | TaskCompletedEvent
    | PlanCompletedEvent,
    Discriminator("type"),
]
"""Type alias for all Swarm events."""
