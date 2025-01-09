# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json
from typing import Literal

from pydantic import BaseModel

from liteswarm.experimental import LiteTeamChat
from liteswarm.types import LLM, Agent, ContextVariables, Task, TaskDefinition, TeamMember
from liteswarm.utils import dump_messages

ENGINEER_INSTRUCTIONS = """You are a world-class software engineer with expertise in multiple programming languages and best practices.

Your responsibilities:
1. Write clean, maintainable, and well-documented code
2. Follow language-specific conventions and best practices
3. Include appropriate error handling and edge cases
4. Add helpful comments explaining complex logic
5. Structure code for readability and maintainability

Focus on delivering production-quality code that is both efficient and easy to understand.
""".strip()


TASK_INSTRUCTIONS = """Implement the following user story:

{user_story}

Implementation requirements:
1. Language: {language}
2. Include appropriate error handling
3. Add docstrings and comments
4. Follow language-specific style guides
5. Consider edge cases and validation

Please structure your response as follows:
1. Brief explanation of your implementation approach
2. Code implementation with comments
3. Any important notes or considerations
""".strip()


class SoftwareTask(Task):
    type: Literal["software_task"]
    user_story: str
    language: str


class SoftwareTaskOutput(BaseModel):
    thoughts: list[str]
    filepath: str
    code: str


def create_team_members() -> list[TeamMember]:
    engineer_agent = Agent(
        id="engineer",
        instructions=ENGINEER_INSTRUCTIONS,
        llm=LLM(
            model="gpt-4o",
            response_format=SoftwareTaskOutput,
        ),
    )

    engineer_member = TeamMember.from_agent(
        engineer_agent,
        task_types=[SoftwareTask],
    )

    return [engineer_member]


def create_task_definitions() -> list[TaskDefinition]:
    def task_prompt_builder(task: SoftwareTask, context_variables: ContextVariables) -> str:
        return TASK_INSTRUCTIONS.format(
            user_story=task.user_story,
            language=task.language,
        )

    task_def = TaskDefinition(
        task_type=SoftwareTask,
        instructions=task_prompt_builder,
        response_format=SoftwareTaskOutput,
    )

    return [task_def]


async def run() -> None:
    team_members = create_team_members()
    task_definitions = create_task_definitions()
    chat = LiteTeamChat(
        members=team_members,
        task_definitions=task_definitions,
    )

    async for event in chat.send_message("Write a merge sort algorithm in Python."):
        if event.type == "agent_response_chunk":
            completion = event.response_chunk.completion
            if completion.delta.content:
                print(completion.delta.content, end="", flush=True)
            if completion.finish_reason == "stop":
                print()

    chat_messages = await chat.get_messages()
    messages = dump_messages(chat_messages, exclude_none=True)
    print(json.dumps(messages, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
