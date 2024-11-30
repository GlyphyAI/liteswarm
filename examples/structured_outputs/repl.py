# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from collections.abc import Awaitable, Callable
from typing import NoReturn, TypeVar

from pydantic import BaseModel

from .strategy import StrategyBuilderRegistry, StrategyId

T = TypeVar("T", bound=BaseModel)


def get_model_input(available_models: list[str], default_model: str) -> str:
    """Get model input with selection or custom entry.

    Args:
        available_models: List of available models to choose from.
        default_model: Default model if no selection is made.

    Returns:
        Selected or custom model name.
    """
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    print(f"{len(available_models) + 1}. Enter custom model")

    while True:
        try:
            choice = input(
                f"\nSelect model (1-{len(available_models) + 1}) or press Enter for default: "
            ).strip()
            if not choice:
                return default_model

            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                return available_models[idx]
            elif idx == len(available_models):
                custom = input("Enter custom model name: ").strip()
                if custom:
                    return custom
            print(f"Invalid choice. Please select 1-{len(available_models) + 1}.")
        except ValueError:
            print(f"Invalid input. Please enter a number 1-{len(available_models) + 1}.")


def get_user_input(
    registry: StrategyBuilderRegistry[T],
) -> tuple[StrategyId, str, str]:
    """Get user input for strategy, model, and prompt."""
    print("\nAvailable strategies:")
    items = list(registry.strategy_builders.items())
    for i, (id, strategy_builder) in enumerate(items, 1):
        print(f"{i}. {id} - {strategy_builder.description}")
        print(f"   Default model: {strategy_builder.default_model}")

    while True:
        try:
            choice = input(f"\nSelect strategy (1-{len(items)}) or press Enter for default: ")
            choice = choice.strip()
            strategy_id: StrategyId | None = None

            if not choice:
                strategy_id = registry.default_strategy_id
                break

            idx = int(choice) - 1
            if 0 <= idx < len(items):
                strategy_id = items[idx][0]
                break

            print(f"Invalid choice. Please select 1-{len(items)}.")

        except ValueError:
            print(f"Invalid input. Please enter a number 1-{len(items)}.")

    strategy_builder = registry.strategy_builders[strategy_id]
    model = get_model_input(strategy_builder.available_models, strategy_builder.default_model)

    default_prompt = registry.default_strategy_prompt
    prompt = input(f"\nEnter prompt or press Enter for default:\n{default_prompt}\n> ").strip()
    prompt = prompt.strip() or default_prompt

    return strategy_id, model, prompt


async def start_repl(
    registry: StrategyBuilderRegistry[T],
    run_example: Callable[[StrategyId, str | None, str | None], Awaitable[None]],
) -> NoReturn:
    """Start a REPL loop featuring structured outputs strategies."""
    print("\nStructured Outputs Example")
    print("=" * 25)

    while True:
        strategy_id, model, prompt = get_user_input(registry)

        try:
            await run_example(strategy_id, model, prompt)
        except Exception as e:
            print(f"\nError: {e}")

        again = input("\nTry another? (y/N): ").lower()
        if again != "y":
            sys.exit()
