# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from collections.abc import Awaitable, Callable
from typing import NoReturn, TypeVar

from pydantic import BaseModel

from liteswarm.utils.misc import prompt

from .strategy import StrategyBuilderRegistry, StrategyId

T = TypeVar("T", bound=BaseModel)


async def get_model_input(available_models: list[str], default_model: str) -> str:
    """Get model selection from user input.

    Presents available models and allows user to select from list or enter
    a custom model name.

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
        count = len(available_models) + 1
        try:
            choice = await prompt(f"\nSelect model (1-{count}) or press Enter for default: ")
            if not choice:
                return default_model

            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                return available_models[idx]
            elif idx == len(available_models):
                custom = await prompt("Enter custom model name: ")
                if custom:
                    return custom
            print(f"Invalid choice. Please select 1-{count}.")
        except ValueError:
            print(f"Invalid input. Please enter a number 1-{count}.")


async def get_user_input(
    registry: StrategyBuilderRegistry[T],
) -> tuple[StrategyId, str, str]:
    """Get strategy, model and prompt selections from user.

    Presents available strategies and prompts user for selections.

    Args:
        registry: Registry containing available strategies.

    Returns:
        Tuple containing:
            - Selected strategy ID
            - Selected model name
            - User prompt or default prompt
    """
    print("\nAvailable strategies:")
    items = list(registry.strategy_builders.items())
    for i, (id, strategy_builder) in enumerate(items, 1):
        print(f"{i}. {id} - {strategy_builder.description}")
        print(f"   Default model: {strategy_builder.default_model}")

    strategy_id: StrategyId | None = None

    while True:
        try:
            choice = await prompt(
                f"\nSelect strategy (1-{len(items)}) or press Enter for default: "
            )

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
    model = await get_model_input(strategy_builder.available_models, strategy_builder.default_model)

    default_prompt = registry.default_strategy_prompt
    user_prompt = input(f"\nEnter prompt or press Enter for default:\n{default_prompt}\n> ").strip()
    user_prompt = user_prompt.strip() or default_prompt

    return strategy_id, model, user_prompt


async def start_repl(
    registry: StrategyBuilderRegistry[T],
    run_example: Callable[[StrategyId, str | None, str | None], Awaitable[None]],
) -> NoReturn:
    """Start interactive REPL for testing structured outputs.

    Continuously prompts user for strategy/model/prompt selections and
    runs examples until user exits.

    Args:
        registry: Registry containing available strategies.
        run_example: Function to run example with selected parameters.

    Returns:
        Never returns - loops until user exits.

    Raises:
        SystemExit: When user chooses to exit.
    """
    print("\nStructured Outputs Example")
    print("=" * 25)

    while True:
        strategy_id, model, user_prompt = await get_user_input(registry)

        try:
            await run_example(strategy_id, model, user_prompt)
        except Exception as e:
            print(f"\nError: {e}")

        again = (await prompt("\nTry another? (y/N): ")).lower()
        if again != "y":
            sys.exit()
