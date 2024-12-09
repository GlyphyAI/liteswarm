# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Any, Literal, Self, TypeAlias

from litellm import (
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
)
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from typing_extensions import TypedDict

AgentTool: TypeAlias = Callable[..., Any]
"""Function that can be called by an agent.

Tools are functions that agents can use to perform actions, returning
simple values, new agents, or complex Result objects.

Examples:
    Simple calculation tool:
        ```python
        def calculate_sum(numbers: list[float]) -> float:
            \"\"\"Calculate sum of numbers.\"\"\"
            return sum(numbers)
        ```

    Agent switching tool:
        ```python
        def switch_to_expert(domain: str) -> Agent:
            \"\"\"Switch to domain expert.\"\"\"
            return Agent(
                id=f"{domain}-expert",
                instructions=f"You are a {domain} expert.",
                llm=LLM(model="gpt-4o")
            )
        ```

    Agent switching with context updates:
        ```python
        def switch_to_expert(domain: str) -> ToolResult:
            \"\"\"Switch to domain expert.\"\"\"
            return ToolResult(
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
            \"\"\"Fetch weather data.\"\"\"
            if location == "":
                raise ValueError("Location is required")

            return f"Weather in {location}: 20C"
        ```
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


class ResponseFormatBasic(TypedDict):
    """Basic response format specification."""

    type: Literal["text", "json_object"]
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


ResponseFormat: TypeAlias = ResponseFormatBasic | ResponseFormatJsonSchema | type[BaseModel]
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


class LLM(BaseModel):
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

    response_format: ResponseFormat | None = None
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

    model_config = ConfigDict(
        extra="forbid",
        use_attribute_docstrings=True,
        arbitrary_types_allowed=True,
    )

    @field_serializer("response_format")
    def serialize_response_format(
        self, response_format: ResponseFormat | None
    ) -> dict[str, Any] | None:
        """Serialize response format for API requests.

        Converts different format types into API-compatible representations:
        - Pydantic models -> JSON schema.
        - Pydantic instances -> JSON.
        - Dict formats -> Pass through.

        Args:
            response_format: Format to serialize.

        Returns:
            API-compatible format specification.

        Examples:
            Model class to schema:
                ```python
                class Output(BaseModel):
                    value: int
                    details: str


                llm = LLM(response_format=Output)
                ```

            Dict format:
                ```python
                llm = LLM(response_format={"type": "json_object"})
                ```
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_json_schema()
        if isinstance(response_format, BaseModel):
            return response_format.model_dump_json()
        if isinstance(response_format, dict):
            return {**response_format}

        return response_format

    @model_validator(mode="after")
    def check_litellm_kwargs_keys(self) -> Self:
        """Validate litellm_kwargs against main configuration.

        Ensures additional kwargs don't conflict with explicitly set fields.

        Raises:
            ValueError: If litellm_kwargs contains conflicting keys.

        Examples:
            Valid configuration:
                ```python
                llm = LLM(
                    model="gpt-4o",
                    temperature=0.7,
                    litellm_kwargs={"custom_option": "value"},
                )
                ```

            Invalid configuration:
                ```python
                llm = LLM(
                    model="gpt-4o",
                    temperature=0.7,
                    litellm_kwargs={
                        "temperature": 0.5  # Raises ValueError
                    },
                )
                ```
        """
        if self.litellm_kwargs:
            field_names = set(self.model_fields.keys()) - {"litellm_kwargs"}
            overlapping_keys = field_names.intersection(self.litellm_kwargs.keys())
            if overlapping_keys:
                raise ValueError(
                    f"litellm_kwargs contains keys that are already defined in LLM: {', '.join(overlapping_keys)}"
                )

        return self
