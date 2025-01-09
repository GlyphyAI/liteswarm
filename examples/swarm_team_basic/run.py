# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import json
from typing import Literal

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.experimental import SwarmTeam
from liteswarm.types import LLM, Agent, ContextVariables, Message, Task, TaskDefinition, TeamMember
from liteswarm.types.events import SwarmEvent

ENGINEER_INSTRUCTIONS = """
You are a world-class software engineer with expertise in multiple programming languages and best practices.
Focus on delivering production-quality code that is both efficient and easy to understand.
""".strip()


TASK_INSTRUCTIONS = """
Implement user story:
<user_story>
{user_story}
</user_story>

Requirements:
- Language: {language}
- Include error handling
- Add docstrings and comments
- Follow language-specific style guides
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


def handle_event(event: SwarmEvent) -> None:
    if event.type == "agent_response_chunk":
        completion = event.response_chunk.completion
        if completion.delta.content:
            print(completion.delta.content, end="", flush=True)
        if completion.finish_reason == "stop":
            print()


async def run() -> None:
    # Step 1: Create team members and task definitions
    team_members = create_team_members()
    task_definitions = create_task_definitions()

    # Step 2: Create swarm team
    swarm = Swarm()
    swarm_team = SwarmTeam(
        swarm=swarm,
        members=team_members,
        task_definitions=task_definitions,
    )

    # Step 3: Create plan
    print("Creating plan...")
    print("=" * 80)

    messages = [Message(role="user", content="Write a merge sort algorithm in Python.")]
    create_plan_stream = swarm_team.create_plan(messages)
    async for event in create_plan_stream:
        handle_event(event)

    plan_result = await create_plan_stream.get_return_value()
    messages = plan_result.all_messages

    print("=" * 80)
    print("\nExecuting plan...")
    print("=" * 80)

    # Step 4: Execute plan
    execute_plan_stream = swarm_team.execute_plan(plan_result.plan, messages)
    async for event in execute_plan_stream:
        handle_event(event)

    artifact = await execute_plan_stream.get_return_value()
    messages = artifact.all_messages
    messages_dump = [message.model_dump(exclude_none=True) for message in messages]

    print("=" * 80)
    print("\nAll messages:\n")
    print(json.dumps(messages_dump, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
