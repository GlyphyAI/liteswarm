# Copyright 2025 GlyphyAI
#
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
    """Agent configuration for the LLM."""

    response_parser: Callable[[str, ContextVariables], T]
    """Function to parse and validate responses."""

    model_config = ConfigDict(extra="forbid")


class StrategyBuilder(BaseModel, Generic[T]):
    """Builder for creating structured output strategies."""

    create_strategy: Callable[[str], Strategy[T]]
    """Factory function to create a strategy instance."""

    description: str
    """Human-readable description of the strategy."""

    default_model: str
    """Default LLM model for this strategy."""

    available_models: list[str]
    """List of available LLM models for this strategy."""

    model_config = ConfigDict(extra="forbid")


class StrategyBuilderRegistry(BaseModel, Generic[T]):
    """Registry for managing multiple strategy builders."""

    strategy_builders: dict[StrategyId, StrategyBuilder[T]]
    """Map of strategy IDs to their builders."""

    default_strategy_id: StrategyId
    """ID of the default strategy."""

    default_strategy_prompt: str
    """Default prompt for strategies."""

    model_config = ConfigDict(extra="forbid")
