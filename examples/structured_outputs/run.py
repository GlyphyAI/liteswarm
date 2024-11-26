# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json
import os

from pydantic import BaseModel

from liteswarm.types import LLM
from liteswarm.utils import enable_logging

from .types import CodingTask, Plan, Response
from .utils import generate_structured_response

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"

PLANNER_PROMPT = """
You are a planning assistant AI designed to help users break down their queries into manageable tasks and create execution plans. Your goal is to analyze the user's query, identify the necessary steps to complete it, and generate a structured plan using the provided JSON schema.

When generating your response, you must strictly adhere to the provided schema. Some properties in the schema may contain a union type with a `___UNKNOWN___` constant. For these properties, you must use the "___UNKNOWN___" placeholder. However, for all other properties without this type in their union, you should populate them with appropriate values based on the user's query.

To complete this task, follow these steps:

1. Carefully read and analyze the user's query.
2. Break down the query into a set of distinct, manageable tasks.
3. Generate an execution plan that outlines how to complete these tasks.
4. Ensure that your response strictly follows the provided JSON schema.

Use an inner monologue to think through the process before providing your final answer. This will help you organize your thoughts and ensure a comprehensive response.

Here is a JSON schema for your response format:

<json_schema>
{RESPONSE_FORMAT}
</json_schema>

Here's an example of how to structure your response:

<example_response>
{RESPONSE_EXAMPLE}
</example_response>
""".strip()


async def main() -> None:
    response_example = Response(
        inner_monologue="First, I'll analyze the user's query to understand the main goal and identify the necessary steps to achieve it. Then, I'll break these steps down into distinct tasks and create an execution plan. Finally, I'll format this information according to the provided JSON schema, being careful to use the '___UNKNOWN___' placeholder where required.",
        response=Plan(
            tasks=[
                CodingTask(
                    task_type="coding",
                    title="Write a hello world program in Python",
                    filepath="main.py",
                    code="print('Hello, world!')",
                )
            ],
        ),
    )

    def agent_instructions_builder(
        response_format: type[BaseModel],
        response_example: BaseModel | None,
    ) -> str:
        return PLANNER_PROMPT.format(
            RESPONSE_FORMAT=json.dumps(response_format.model_json_schema()),
            RESPONSE_EXAMPLE=response_example.model_dump_json() if response_example else "",
        )

    response = await generate_structured_response(
        user_prompt="Write a fibonacci calculation function in Python",
        agent_instructions=agent_instructions_builder,
        response_format=Response,
        response_example=response_example,
        llm=LLM(model="gpt-4o"),
    )

    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    enable_logging()
    asyncio.run(main())
