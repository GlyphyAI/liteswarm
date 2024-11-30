# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict

from liteswarm.types import Agent, ContextVariables

T = TypeVar("T", bound=BaseModel)

StrategyId: TypeAlias = Literal[
    "anthropic_pydantic",
    "openai_pydantic",
    "llm_json_object",
    "llm_json_tags",
]


class Strategy(BaseModel, Generic[T]):
    """Protocol for structured output strategies."""

    agent: Agent
    """Agent for this strategy."""

    response_parser: Callable[[str, ContextVariables], T]
    """Response parser for this strategy."""

    model_config = ConfigDict(extra="forbid")


class StrategyBuilder(BaseModel, Generic[T]):
    """Builder for structured output strategies."""

    create_strategy: Callable[[str], Strategy[T]]
    """Create a structured output strategy."""

    description: str
    """Strategy description."""

    default_model: str
    """Default model for this strategy."""

    available_models: list[str]
    """List of available models for this strategy."""

    model_config = ConfigDict(extra="forbid")


class StrategyBuilderRegistry(BaseModel, Generic[T]):
    """Registry for structured output strategy builders."""

    strategy_builders: dict[StrategyId, StrategyBuilder[T]]
    """Strategy builders."""

    default_strategy_id: StrategyId
    """Default strategy identifier."""

    default_strategy_prompt: str
    """Default strategy prompt."""

    model_config = ConfigDict(extra="forbid")
