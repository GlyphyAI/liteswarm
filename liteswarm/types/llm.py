# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, Generic, Literal, Self, TypeAlias

from litellm import (
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
    ModelConfig,
)
from litellm.types.utils import ChatCompletionAudioResponse, ChatCompletionDeltaToolCall
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from typing_extensions import TypedDict

from liteswarm.types.typing import TypeVar

AgentTool: TypeAlias = Callable[..., Any]
"""Function that can be called by agents.

Tools are functions that agents can use to perform actions, returning
simple values, new agents, or complex Result objects.

Examples:
    Simple calculation tool:
        ```python
        def calculate_sum(numbers: list[float]) -> float:
            return sum(numbers)
        ```

    Agent switching tool:
        ```python
        def switch_to_expert(domain: str) -> Agent:
            return Agent(
                id=f"{domain}-expert",
                instructions=f"You are a {domain} expert.",
                llm=LLM(model="gpt-4o")
            )
        ```

    Agent switching with context updates:
        ```python
        def switch_to_expert(domain: str) -> ToolResult:
            return ToolResult.switch_agent(
                content="Switching to expert",
                agent=Agent(
                    id=f"{domain}-expert",
                    instructions=f"You are a {domain} expert.",
                    llm=LLM(model="gpt-4o")
                ),
                context_variables=ContextVariables(specialty=domain),
            )
        ```

    Error handling:
        ```python
        def fetch_weather(location: str) -> dict[str, Any]:
            if location == "":
                raise ValueError("Location is required")

            return f"Weather in {location}: 20C"
        ```
"""

MessageRole: TypeAlias = Literal["assistant", "user", "system", "tool", "function", "developer"]
"""Role of the message author in a conversation.

Supported roles:
- assistant: Model-generated responses to user messages.
- user: End-user messages containing prompts or context.
- system: System-level instructions that guide model behavior.
- tool: Results from tool/function executions.
- function: Legacy alias for tool responses [deprecated].
- developer: Developer-provided instructions for reasoning models.
"""

ToolCall: TypeAlias = ChatCompletionDeltaToolCall
"""Tool call in a streaming response from the model.

Represents a function or tool invocation request with:
- id: Unique identifier for tracking the call.
- function: Function name and arguments to execute.
- type: Optional tool type identifier (e.g., 'function').
- index: Position in the sequence of tool calls.

Used in both complete messages and streaming deltas.
"""

AudioResponse: TypeAlias = ChatCompletionAudioResponse
"""Audio response generated by the model.

Contains audio data and metadata:
- id: Unique identifier for the response.
- data: Base64 encoded audio bytes in requested format.
- expires_at: Unix timestamp when audio will be purged.
- transcript: Text transcription of the audio content.

Used in conversations with audio capabilities.
"""

FinishReason: TypeAlias = Literal["stop", "length", "tool_calls", "function_call", "content_filter"]
"""OpenAI supported finish reasons for streaming responses.

Indicates why the model stopped generating tokens during a streaming response.

Supported finish reasons:
- stop: Model reached a natural stopping point or hit a provided stop sequence.
- length: Maximum allowed tokens were reached.
- tool_calls: Model called a tool.
- function_call: Deprecated equivalent of tool_calls.
- content_filter: Content was omitted due to safety or moderation filters.
"""


class ToolChoiceFunctionObject(TypedDict):
    """Function specification for tool choice.

    Defines the specific function an agent should use when multiple
    tools are available.
    """

    name: str
    """Name of the function to use."""


class ToolChoiceFunction(TypedDict):
    """Complete tool choice specification.

    Combines the choice type with the specific function to use.
    """

    type: Literal["function"]
    """Type of tool choice (always "function")."""

    function: ToolChoiceFunctionObject
    """Function specification."""


ToolChoice: TypeAlias = Literal["auto", "none", "required"] | ToolChoiceFunction
"""Specification for how an agent should use tools.

Controls tool selection and usage behavior:
- "auto": Agent decides when to use tools.
- "none": Agent cannot use tools.
- "required": Agent must use a tool.
- ToolChoiceFunction: Agent must use specific tool.

Examples:
    Automatic tool selection:
        ```python
        llm = LLM(
            model="gpt-4o",
            tools=[search, calculate],
            tool_choice="auto"
        )
        ```

    Force specific tool:
        ```python
        llm = LLM(
            model="gpt-4o",
            tools=[search, calculate],
            tool_choice={
                "type": "function",
                "function": {"name": "calculate"}
            }
        )
        ```
"""


class ResponseSchema(BaseModel):
    """Schema for validating structured responses.

    Defines the expected structure and validation rules for model outputs.

    Examples:
        Define review output schema:
            ```python
            schema = ResponseSchema(
                name="review_output",
                description="Code review response format",
                json_schema={
                    "type": "object",
                    "required": ["approved", "comments"],
                    "properties": {
                        "approved": {"type": "boolean"},
                        "comments": {"type": "array", "items": {"type": "string"}},
                    },
                },
            )
            ```
    """

    name: str
    """Schema identifier."""

    description: str | None = None
    """Schema purpose and usage."""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    """JSON schema definition."""

    strict: bool = False
    """Whether to enforce strict validation."""


class ResponseFormatText(TypedDict):
    """Text response format specification."""

    type: Literal["text"]
    """Response format type."""


class ResponseFormatJsonObject(TypedDict):
    """JSON object response format specification."""

    type: Literal["json_object"]
    """Response format type."""


class ResponseFormatJsonSchema(TypedDict):
    """JSON schema response format specification."""

    type: Literal["json_schema"]
    """Response format type."""

    json_schema: ResponseSchema
    """Schema for validation."""


class StreamOptions(BaseModel):
    """Configuration for streaming responses.

    Controls what additional information is included in streaming
    response chunks.
    """

    include_usage: bool | None = None
    """Whether to include token usage stats."""


ResponseFormatPydantic = TypeVar(
    "ResponseFormatPydantic",
    bound=BaseModel,
    default=BaseModel,
    covariant=True,
)
"""Type variable for Pydantic model used as response output.

This type variable allows for type-safe validation of response formats
by ensuring that the response format is a subclass of BaseModel.
"""

ResponseFormat: TypeAlias = (
    ResponseFormatText
    | ResponseFormatJsonObject
    | ResponseFormatJsonSchema
    | type[ResponseFormatPydantic]
)
"""Specification for model output format and validation.

Controls response structure and validation:
- ResponseFormatBasic: Text or generic JSON.
- ResponseFormatJsonSchema: Validated against JSON schema.
- type[BaseModel]: Validated against Pydantic model.

Examples:
    Text response:
        ```python
        llm = LLM(
            model="gpt-4o",
            response_format={"type": "text"}
        )
        ```

    JSON schema validation:
        ```python
        llm = LLM(
            model="gpt-4o",
            response_format={
                "type": "json_schema",
                "json_schema": ResponseSchema(
                    name="review",
                    json_schema={
                        "type": "object",
                        "properties": {
                            "approved": {"type": "boolean"},
                            "comments": {"type": "array"}
                        }
                    }
                )
            }
        )
        ```

    Pydantic model validation:
        ```python
        class ReviewOutput(BaseModel):
            approved: bool
            comments: list[str]

        llm = LLM(
            model="gpt-4o",
            response_format=ReviewOutput
        )
        ```
"""


class LLM(BaseModel, Generic[ResponseFormatPydantic]):
    """Configuration for language model interactions.

    Provides comprehensive control over LLM behavior including model
    selection, response formatting, tool usage, and performance settings.

    Examples:
        Basic configuration:
            ```python
            llm = LLM(
                model="gpt-4o",
                max_tokens=1000,
                temperature=0.7,
            )
            ```

        Advanced configuration:
            ```python
            class ReviewOutput(BaseModel):
                approved: bool
                comments: list[str]


            llm = LLM(
                model="gpt-4o",
                tools=[search_docs, analyze_code, run_tests],
                tool_choice="auto",
                parallel_tool_calls=True,
                response_format=ReviewOutput,
                temperature=0.7,
                stream_options=StreamOptions(include_usage=True),
            )
            ```
    """

    model: str
    """Model identifier."""

    tools: list[AgentTool] | None = None
    """Available tool functions."""

    tool_choice: ToolChoice | None = None
    """Tool selection behavior."""

    parallel_tool_calls: bool | None = None
    """Allow concurrent tool calls."""

    response_format: ResponseFormat[ResponseFormatPydantic] | None = None
    """Output format specification."""

    logprobs: bool | None = None
    """Include token logprobs."""

    top_logprobs: int | None = None
    """Number of top logprobs."""

    deployment_id: str | None = None
    """Model deployment identifier."""

    seed: int | None = None
    """Random seed for reproducibility."""

    user: str | None = None
    """User identifier for tracking."""

    logit_bias: dict[int, float] | None = None
    """Token probability adjustments."""

    frequency_penalty: float | None = None
    """Repetition reduction factor."""

    presence_penalty: float | None = None
    """Topic diversity factor."""

    max_tokens: int | None = None
    """Maximum response length."""

    max_completion_tokens: int | None = None
    """Maximum completion length."""

    modalities: list[ChatCompletionModality] | None = None
    """Response modality settings."""

    prediction: ChatCompletionPredictionContentParam | None = None
    """Prediction output settings."""

    audio: ChatCompletionAudioParam | None = None
    """Audio generation settings."""

    stop: str | list[str] | None = None
    """Response termination sequences."""

    stream_options: StreamOptions | None = None
    """Streaming configuration."""

    stream: bool | None = None
    """Enable response streaming."""

    n: int | None = None
    """Number of completions."""

    top_p: float | None = None
    """Nucleus sampling threshold."""

    temperature: float | None = None
    """Response randomness."""

    timeout: int | float | None = None
    """Request timeout seconds."""

    litellm_kwargs: dict[str, Any] | None = None
    """Additional LiteLLM options."""

    base_url: str | None = None
    """Base URL for API requests."""

    api_version: str | None = None
    """API version string."""

    api_key: str | None = None
    """API authentication key."""

    model_list: list[ModelConfig] | None = None
    """List of available model configurations."""

    extra_headers: dict[str, Any] | None = None
    """Additional HTTP headers to include in API requests."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_serializer("tools")
    def serialize_tools(
        self,
        tools: list[AgentTool] | None,
    ) -> list[dict[str, str]] | None:
        """Serialize tool functions for storage or transmission.

        Captures essential metadata about each tool:
        - name: Function name.
        - doc: Function docstring.
        - module: Module where function is defined.

        Note that the actual function implementation cannot be serialized,
        so tools will need to be provided again when recreating the LLM.

        Returns:
            List of tool metadata or None if no tools.
        """
        if not tools:
            return None

        tool_info: list[dict[str, str]] = []
        for tool in tools:
            info = {
                "name": getattr(tool, "__name__", "<unknown>"),
                "doc": getattr(tool, "__doc__", "<no docstring>"),
                "module": getattr(tool, "__module__", "<unknown>"),
            }
            tool_info.append(info)

        return tool_info

    @field_serializer("response_format")
    def serialize_response_format(
        self,
        response_format: ResponseFormat[ResponseFormatPydantic] | None,
    ) -> dict[str, Any] | None:
        """Serialize response format for API requests.

        Handles different response format types:
        - ResponseFormatBasic: Pass through (e.g., {"type": "text"})
        - ResponseFormatJsonSchema: Pass through with schema
        - type[BaseModel]: Convert to JSON schema

        Args:
            response_format: Format to serialize.

        Returns:
            API-compatible format specification or None.
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_json_schema()
        elif isinstance(response_format, dict):
            return {**response_format}
        else:
            return response_format

    @model_validator(mode="after")
    def check_litellm_kwargs_keys(self) -> Self:
        """Validate litellm_kwargs against main configuration.

        Ensures additional kwargs don't conflict with explicitly set fields.

        Raises:
            ValueError: If litellm_kwargs contains conflicting keys.
        """
        if self.litellm_kwargs:
            field_names = set(self.model_fields.keys()) - {"litellm_kwargs"}
            overlapping_keys = field_names.intersection(self.litellm_kwargs.keys())
            if overlapping_keys:
                raise ValueError(
                    f"litellm_kwargs contains keys that are already defined in LLM: {', '.join(overlapping_keys)}"
                )

        return self
