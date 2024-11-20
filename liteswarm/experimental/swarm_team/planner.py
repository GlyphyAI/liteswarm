# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from liteswarm.core.swarm import Swarm
from liteswarm.types import Result
from liteswarm.types.swarm import Agent, ContextVariables
from liteswarm.types.swarm_team import Plan, Task, TaskDefinition
from liteswarm.utils.misc import create_plan_schema, extract_json


class PromptTemplate(Protocol):
    """Protocol for prompt templates."""

    @property
    def template(self) -> str:
        """Return the template string."""
        ...

    def format_context(self, prompt: str, context: ContextVariables) -> str:
        """Format the prompt with the given context."""
        ...


class PlanningAgent(Protocol):
    """Protocol for planning agents that create task plans."""

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables,
        feedback: str | None = None,
    ) -> Result[Plan[Task]]:
        """Create a plan from the given prompt and context."""
        ...


class AgentPlanner(PlanningAgent):
    """Default implementation of the planning strategy."""

    def __init__(
        self,
        swarm: Swarm,
        agent: Agent,
        template: PromptTemplate,
        task_definitions: list[TaskDefinition],
    ) -> None:
        self.swarm = swarm
        self.agent = agent
        self.template = template
        self.task_definitions = {td.task_type: td for td in task_definitions}

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables,
        feedback: str | None = None,
    ) -> Result[Plan[Task]]:
        """Create a plan using the configured agent."""
        if feedback:
            full_prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"
        else:
            full_prompt = prompt

        task_definitions = list(self.task_definitions.values())
        plan_schema = create_plan_schema(task_definitions)
        output_format = plan_schema.model_json_schema()

        context = ContextVariables(**context)
        context.set_reserved("output_format", output_format)
        formatted_prompt = self.template.format_context(full_prompt, context)

        result = await self.swarm.execute(
            agent=self.agent,
            prompt=formatted_prompt,
            context_variables=context,
        )

        if not result.content:
            return Result(error=ValueError("Failed to create plan"))

        try:
            json_plan = extract_json(result.content)
            if not isinstance(json_plan, dict):
                raise TypeError("Invalid plan format")

            plan = plan_schema.model_validate(json_plan)

            for task in plan.tasks:
                if task.task_type not in self.task_definitions:
                    return Result(error=ValueError(f"Unknown task type: {task.task_type}"))

            if errors := plan.validate_dependencies():
                return Result(error=ValueError("\n".join(errors)))

            return Result(value=plan)

        except Exception as e:
            return Result(error=e)
