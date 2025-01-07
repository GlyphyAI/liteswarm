# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio

from liteswarm.utils import enable_logging
from liteswarm.utils.misc import prompt

from .example_chat import run as chat_example
from .example_team_chat import run as team_chat_example

enable_logging(default_level="DEBUG")


async def main() -> None:
    print("Select an example to run:")
    print("1. Chat example")
    print("2. Team chat example")
    choice = await prompt("Enter the number of the example to run: ")
    match choice:
        case "1":
            await chat_example()
        case "2":
            await team_chat_example()
        case _:
            print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
