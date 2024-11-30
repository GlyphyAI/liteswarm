# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import os

from liteswarm.utils import enable_logging

from .repl import start_repl
from .strategies import anthropic_pydantic, llm_json_object, llm_json_tags, openai_pydantic
from .strategy import StrategyBuilder, StrategyBuilderRegistry, StrategyId
from .types import InnerMonologue
from .utils import generate_structured_response_typed

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"

STRATEGY_REGISTRY = StrategyBuilderRegistry[InnerMonologue](
    strategy_builders={
        "anthropic_pydantic": StrategyBuilder(
            create_strategy=anthropic_pydantic.create_strategy,
            description="Anthropic with Pydantic schema validation",
            default_model="claude-3-5-sonnet-20241022",
            available_models=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
        ),
        "openai_pydantic": StrategyBuilder(
            create_strategy=openai_pydantic.create_strategy,
            description="OpenAI with Pydantic schema validation",
            default_model="gpt-4o",
            available_models=["gpt-4o", "gpt-4o-mini"],
        ),
        "llm_json_object": StrategyBuilder(
            create_strategy=llm_json_object.create_strategy,
            description="Provider-agnostic JSON object parsing",
            default_model="gpt-4o",
            available_models=[
                "gpt-4o",
                "gpt-4o-mini",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
        ),
        "llm_json_tags": StrategyBuilder(
            create_strategy=llm_json_tags.create_strategy,
            description="Provider-agnostic JSON tags parsing",
            default_model="gpt-4o",
            available_models=[
                "gpt-4o",
                "gpt-4o-mini",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
        ),
    },
    default_strategy_id="openai_pydantic",
    default_strategy_prompt="Create a method to calculate the area of a circle in Python and test it.",
)


async def run_example(
    strategy_id: StrategyId,
    model: str | None = None,
    prompt: str | None = None,
) -> None:
    """Run the structured outputs example with the specified strategy."""
    strategy_builder = STRATEGY_REGISTRY.strategy_builders.get(strategy_id)
    if not strategy_builder:
        raise ValueError(f"Invalid strategy: {strategy_id}")

    model = model or strategy_builder.default_model
    prompt = prompt or STRATEGY_REGISTRY.default_strategy_prompt
    strategy = strategy_builder.create_strategy(model)

    print(f"\nUsing strategy: {strategy_id} ({strategy_builder.description})")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}\n")

    response = await generate_structured_response_typed(
        user_prompt=prompt,
        agent=strategy.agent,
        response_format=strategy.response_parser,
    )

    print(f"\nResponse:\n{response.model_dump_json(indent=2)}")


if __name__ == "__main__":
    enable_logging()
    asyncio.run(start_repl(STRATEGY_REGISTRY, run_example))
