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
"""A tool that can be called by an agent.

Tools are functions that agents can use to perform actions. They can:
- Return simple values (str, int, dict, etc.)
- Return new agents for agent switching
- Return Result objects for complex responses

Example:
```python
def calculate_sum(numbers: list[float]) -> float:
    \"\"\"Add up a list of numbers.\"\"\"
    return sum(numbers)

def switch_to_expert(topic: str) -> Agent:
    \"\"\"Switch to an expert agent for a specific topic.\"\"\"
    return Agent(
        id=f"{topic}-expert",
        instructions=f"You are an expert in {topic}.",
        llm=LLMConfig(model="gpt-4o", tools=[calculate_sum]),
    )
```
"""


class ToolChoiceFunctionObject(TypedDict):
    """Function specification for tool choice.

    Defines which specific function an agent should use
    when multiple tools are available.
    """

    name: str
    """Name of the function to use"""


class ToolChoiceFunction(TypedDict):
    """Complete tool choice specification.

    Combines the type of choice with the specific
    function to use.
    """

    type: Literal["function"]
    """Type of the tool choice (always "function")"""

    function: ToolChoiceFunctionObject
    """The function specification"""


ToolChoice: TypeAlias = Literal["auto", "none", "required"] | ToolChoiceFunction
"""Tool choice specification for agent behavior.

Controls how the agent selects and uses tools:
- "auto": Agent decides when to use tools
- "none": Agent cannot use tools
- "required": Agent must use a tool
- ToolChoiceFunction: Agent must use specific tool

Example:
```python
# Automatic tool selection
config = LLMConfig(
    model="gpt-4o",
    tools=[search, calculate],
    tool_choice="auto"
)

# Force specific tool
config = LLMConfig(
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
    """Schema specification for structured responses.

    Defines the expected structure and validation rules
    for model responses.

    Example:
    ```python
    schema = ResponseSchema(
        name="review_output",
        description="Output format for code reviews",
        json_schema={
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "comments": {"type": "array", "items": {"type": "string"}}
            }
        }
    )
    ```
    """

    name: str
    """Name of the schema"""

    description: str | None = None
    """Description of what the schema represents"""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    """The actual JSON schema definition"""

    strict: bool = False
    """Whether to enforce strict schema validation"""


class ResponseFormatBasic(TypedDict):
    """Basic response format specification."""

    type: Literal["text", "json_object"]
    """Type of response format (text, json_object)"""


class ResponseFormatJsonSchema(TypedDict):
    """JSON schema response format specification."""

    type: Literal["json_schema"]
    """Type of response format (json_schema)"""

    json_schema: ResponseSchema
    """The schema to validate against"""


class StreamOptions(BaseModel):
    """Configuration for streaming responses.

    Controls what additional information is included
    in streaming responses.
    """

    include_usage: bool | None = None
    """Whether to include token usage information"""


ResponseFormat: TypeAlias = ResponseFormatBasic | ResponseFormatJsonSchema | type[BaseModel]
"""Response format specification for model outputs.

Controls the structure and validation of responses:
- ResponseFormatText: Plain text responses
- ResponseFormatJsonObject: Generic JSON objects
- ResponseFormatJsonSchema: Validated against JSON schema
- type[BaseModel]: Validated against Pydantic model

Example:
```python
# Plain text
config = LLMConfig(
    model="gpt-4o",
    response_format={"type": "text"}
)

# JSON object
config = LLMConfig(
    model="gpt-4o",
    response_format={"type": "json_object"}
)

# JSON schema
config = LLMConfig(
    model="gpt-4o",
    response_format={
        "type": "json_schema",
        "json_schema": ResponseSchema(
            name="review",
            json_schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "comments": {"type": "array", "items": {"type": "string"}}
                }
            }
        )
    }
)

# Pydantic model
class ReviewOutput(BaseModel):
    approved: bool
    comments: list[str]

config = LLMConfig(
    model="gpt-4o",
    response_format=ReviewOutput
)
```
"""


class LLM(BaseModel):
    """LLM configuration used for agent interactions.

    Comprehensive configuration for LLM API calls, including:
    - Model selection and deployment
    - Response formatting and validation
    - Tool usage and control
    - Performance and cost optimization

    Example:
    ```python
    config = LLMConfig(
        model="gpt-4o",
        tools=[search_docs, generate_code],
        tool_choice="auto",
        response_format=ReviewOutput,  # Pydantic model
        max_tokens=1000,
        temperature=0.7,
        stream_options=StreamOptions(include_usage=True)
    )
    ```
    """

    model: str
    """The language model to use"""

    tools: list[AgentTool] | None = None
    """List of functions the agent can call"""

    tool_choice: ToolChoice | None = None
    """How the agent should choose tools ("auto", "none", etc.)"""

    parallel_tool_calls: bool | None = None
    """Whether multiple tools can be called simultaneously"""

    response_format: ResponseFormat | None = None
    """The response format to use for the model"""

    logprobs: bool | None = None
    """Whether to return logprobs"""

    top_logprobs: int | None = None
    """The number of top logprobs to return"""

    deployment_id: str | None = None
    """The deployment ID to use for the model"""

    seed: int | None = None
    """The seed to use for the model"""

    user: str | None = None
    """The user to use for the model"""

    logit_bias: dict[int, float] | None = None
    """Logit bias to use for the model"""

    frequency_penalty: float | None = None
    """Frequency penalty to use for the model"""

    presence_penalty: float | None = None
    """Presence penalty to use for the model"""

    max_tokens: int | None = None
    """Maximum number of tokens in the response"""

    max_completion_tokens: int | None = None
    """Maximum number of completion tokens in the response"""

    modalities: list[ChatCompletionModality] | None = None
    """Modality settings for the response"""

    prediction: ChatCompletionPredictionContentParam | None = None
    """Prediction outputs settings for the response"""

    audio: ChatCompletionAudioParam | None = None
    """Audio settings for the response"""

    stop: str | list[str] | None = None
    """Stop sequences for the response"""

    stream_options: StreamOptions | None = None
    """Stream options for the response"""

    stream: bool | None = None
    """Whether to stream the response"""

    n: int | None = None
    """Number of responses to generate"""

    top_p: float | None = None
    """Top P value to use for the response"""

    temperature: float | None = None
    """Temperature to use for the response"""

    timeout: int | float | None = None
    """Timeout for the response"""

    litellm_kwargs: dict[str, Any] | None = None
    """Additional keyword arguments to pass to litellm"""

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

        Handles different response format types:
        - Pydantic models -> JSON schema
        - Pydantic instances -> JSON representation
        - Dict formats -> Pass through

        Args:
            response_format: The format specification to serialize

        Returns:
            Serialized format ready for API request

        Example:
        ```python
        class Output(BaseModel):
            value: int

        # Model class -> JSON schema
        config = LLMConfig(response_format=Output)

        # Model instance -> JSON
        config = LLMConfig(response_format=Output(value=42))

        # Dict format -> Pass through
        config = LLMConfig(response_format={"type": "json_object"})
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
        """Validate that litellm_kwargs don't conflict with main config.

        Ensures that any additional kwargs passed to LiteLLM don't
        override the explicitly configured fields in LLMConfig.

        Raises:
            ValueError: If litellm_kwargs contains keys that conflict
                      with LLMConfig fields

        Example:
        ```python
        # Valid: no conflicts
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
            litellm_kwargs={
                "custom_option": "value"
            }
        )

        # Invalid: conflicts with main config
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
            litellm_kwargs={
                "temperature": 0.5  # Raises ValueError
            }
        )
        ```
        """
        if self.litellm_kwargs:
            field_names = set(self.model_fields.keys()) - {"litellm_kwargs"}
            overlapping_keys = field_names.intersection(self.litellm_kwargs.keys())
            if overlapping_keys:
                raise ValueError(
                    f"litellm_kwargs contains keys that are already defined in LLMConfig: {', '.join(overlapping_keys)}"
                )

        return self
