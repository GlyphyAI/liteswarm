# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from typing import Protocol, TypeAlias

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.types import Result
from liteswarm.types.llm import LLM
from liteswarm.types.swarm import Agent, ContextVariables
from liteswarm.types.swarm_team import Plan, TaskDefinition
from liteswarm.utils.misc import extract_json
from liteswarm.utils.pydantic import (
    change_field_type,
)
from liteswarm.utils.typing import is_callable, is_subtype, union

AGENT_PLANNER_INSTRUCTIONS = """
You are a task planning specialist.

Your role is to:
1. Break down complex requests into clear, actionable tasks
2. Ensure tasks have appropriate dependencies
3. Create tasks that match the provided task types
4. Consider team capabilities when planning

Each task must include:
- A clear title and description
- The appropriate task type
- Any dependencies on other tasks

Follow the output format specified in the prompt to create your plan.
""".strip()

PromptTemplate: TypeAlias = str | Callable[[str, ContextVariables], str]
"""Template for formatting prompts with context.

Can be either a static template string or a function that generates prompts
dynamically based on context.

Example:
    ```python
    # Static template
    template: PromptTemplate = "Process {prompt} with {context}"

    # Dynamic template
    def generate_prompt(prompt: str, context: ContextVariables) -> str:
        return f"Process {prompt} using {context.get('tools')}"
    ```
"""

PlanResponseFormat: TypeAlias = type[Plan] | Callable[[ContextVariables], type[Plan]]
"""Format specification for plan responses.

Can be either a Plan subclass or a function that generates a Plan subclass
based on context.
"""


class AgentPlanner(Protocol):
    """Protocol for agents that create task plans.

    Planning agents break down work into tasks, set dependencies, and validate
    against task definitions.

    Example:
        ```python
        class CustomPlanner(AgentPlanner):
            async def create_plan(
                self,
                prompt: str,
                feedback: str | None = None,
                context: ContextVariables | None = None,
            ) -> Result[Plan]:
                # Create and validate plan
                return Result(value=plan)
        ```
    """

    async def create_plan(
        self,
        prompt: str,
        feedback: str | None = None,
        context: ContextVariables | None = None,
    ) -> Result[Plan]:
        """Create a plan from the given prompt and context.

        Args:
            prompt: Description of work to be done
            feedback: Optional feedback on previous attempts
            context: Additional context variables

        Returns:
            Result containing either a valid Plan or an error
        """
        ...


class LiteAgentPlanner(AgentPlanner):
    """LLM-based implementation of the planning protocol.

    Creates plans by using an LLM to analyze requirements, generate structured plans,
    and validate them against task definitions.

    Example:
        ```python
        planner = LiteAgentPlanner(
            swarm=swarm,
            agent=Agent(id="planner", llm=LLM(model="gpt-4o")),
            task_definitions=[review_def, test_def]
        )
        ```
    """

    def __init__(
        self,
        swarm: Swarm,
        agent: Agent | None = None,
        prompt_template: PromptTemplate | None = None,
        task_definitions: list[TaskDefinition] | None = None,
        response_format: PlanResponseFormat | None = None,
    ) -> None:
        """Initialize a new planner instance.

        Args:
            swarm: Swarm instance for agent interactions
            agent: Optional custom planning agent
            prompt_template: Optional custom prompt template
            task_definitions: Available task types
            response_format: Optional plan response format
        """
        # Public properties
        self.swarm = swarm
        self.agent = agent or self._default_planning_agent()
        self.prompt_template = prompt_template or self._default_planning_prompt_template()
        self.response_format = response_format or self._default_planning_response_format()

        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)

    def _default_planning_agent(self) -> Agent:
        """Create the default planning agent.

        Returns:
            Agent configured with GPT-4o and planning instructions
        """
        return Agent(
            id="agent-planner",
            instructions=AGENT_PLANNER_INSTRUCTIONS,
            llm=LLM(model="gpt-4o"),
        )

    def _default_planning_prompt_template(self) -> PromptTemplate:
        """Create the default prompt template.

        Returns:
            Simple template that uses raw prompt
        """
        return lambda prompt, _: prompt

    def _default_planning_response_format(self) -> PlanResponseFormat:
        """Create the default plan response format.

        Returns:
            Plan schema with task types from registry
        """
        task_definitions = self._task_registry.get_task_definitions()
        task_schemas = union([td.task_schema for td in task_definitions])
        plan_schema_name = Plan.__name__

        plan_schema = change_field_type(
            model_type=Plan,
            field_name="tasks",
            new_type=list[task_schemas],  # type: ignore [valid-type]
            new_model_name=plan_schema_name,
        )

        return plan_schema

    def _unwrap_response_format(
        self,
        response_format: PlanResponseFormat,
        context: ContextVariables | None = None,
    ) -> type[Plan]:
        """Unwrap the response format into a Plan type.

        Args:
            response_format: Format to unwrap
            context: Optional context for dynamic formats

        Returns:
            Concrete Plan type

        Raises:
            ValueError: If format is invalid
        """
        if is_subtype(response_format, Plan):
            return response_format

        if is_callable(response_format):
            return response_format(context or ContextVariables())

        raise ValueError("Invalid response format")

    async def create_plan(
        self,
        prompt: str,
        feedback: str | None = None,
        context: ContextVariables | None = None,
    ) -> Result[Plan]:
        """Create a plan using the configured agent.

        Args:
            prompt: Description of work to be done
            feedback: Optional feedback on previous attempts
            context: Additional context variables

        Returns:
            Result containing either a valid Plan or an error

        Example:
            ```python
            result = await planner.create_plan(
                prompt="Review and test PR #123",
                feedback="Add security tests",
                context=ContextVariables(pr_url="github.com/org/repo/123"),
            )
            ```
        """
        context = context or ContextVariables()

        if feedback:
            prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"

        if is_callable(self.prompt_template):
            prompt = self.prompt_template(prompt, context)

        result = await self.swarm.execute(
            agent=self.agent,
            prompt=prompt,
            context_variables=context,
        )

        if not result.content:
            return Result(error=ValueError("Failed to create plan"))

        try:
            json_plan = extract_json(result.content)
            if not isinstance(json_plan, dict):
                raise TypeError("Unable to extract plan from response")

            plan_schema = self._unwrap_response_format(self.response_format, context)
            plan = plan_schema.model_validate(json_plan)

            for task in plan.tasks:
                if not self._task_registry.contains_task_type(task.type):
                    return Result(error=ValueError(f"Unknown task type: {task.type}"))

            if errors := plan.validate_dependencies():
                return Result(error=ValueError("\n".join(errors)))

            return Result(value=plan)

        except Exception as e:
            return Result(error=e)
