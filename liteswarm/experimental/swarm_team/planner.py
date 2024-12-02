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
from liteswarm.types.swarm_team import Plan, PlanResponseFormat, PromptTemplate, TaskDefinition
from liteswarm.utils.tasks import create_plan_with_tasks
from liteswarm.utils.typing import is_callable, is_subtype

AGENT_PLANNER_INSTRUCTIONS = """
You are a task planning specialist.

Your role is to:
1. Break down complex requests into clear, actionable tasks.
2. Ensure tasks have appropriate dependencies.
3. Create tasks that match the provided task types.
4. Consider team capabilities when planning.

Each task must include:
- A clear title and description.
- The appropriate task type.
- Any dependencies on other tasks.

Follow the output format specified in the prompt to create your plan.
""".strip()


class AgentPlanner(Protocol):
    """Protocol for agents that create task plans.

    Defines the interface for planning agents that can analyze prompts and create
    structured plans with tasks and dependencies.

    Examples:
        Create a custom planner:
            ```python
            class CustomPlanner(AgentPlanner):
                async def create_plan(
                    self,
                    prompt: str,
                    context: ContextVariables,
                    feedback: str | None = None
                ) -> Result[Plan]:
                    # Analyze prompt and create tasks
                    tasks = [
                        Task(id="task-1", title="First step"),
                        Task(id="task-2", title="Second step", dependencies=["task-1"])
                    ]
                    return Result(value=Plan(tasks=tasks))
            ```
    """

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables | None = None,
        feedback: str | None = None,
    ) -> Result[Plan]:
        """Create a plan from the given prompt and context.

        Args:
            prompt: Description of work to be done.
            context: Optional additional context variables.
            feedback: Optional feedback on previous attempts.

        Returns:
            Result containing either a valid Plan or an error.
        """
        ...


class LiteAgentPlanner(AgentPlanner):
    """LLM-based implementation of the planning protocol.

    Uses an LLM agent to analyze requirements and generate structured plans,
    validating them against task definitions.

    Examples:
        Create and use a planner:
            ```python
            # Define task types
            class ReviewTask(Task):
                pr_url: str
                review_type: str

            # Create task definitions
            review_def = TaskDefinition(
                task_schema=ReviewTask,
                task_instructions="Review {task.pr_url}"
            )

            # Initialize planner
            planner = LiteAgentPlanner(
                swarm=swarm,
                agent=Agent(id="planner", llm=LLM(model="gpt-4o")),
                task_definitions=[review_def]
            )

            # Create plan
            result = await planner.create_plan(
                prompt="Review PR #123",
                context=ContextVariables(pr_url="github.com/org/repo/123")
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
            swarm: Swarm client for agent interactions.
            agent: Optional custom planning agent.
            prompt_template: Optional custom prompt template.
            task_definitions: Available task types.
            response_format: Optional plan response format.
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
            Agent configured with GPT-4o and planning instructions.
        """
        return Agent(
            id="agent-planner",
            instructions=AGENT_PLANNER_INSTRUCTIONS,
            llm=LLM(model="gpt-4o"),
        )

    def _default_planning_prompt_template(self) -> PromptTemplate:
        """Create the default prompt template.

        Returns:
            Simple template that uses raw prompt.
        """
        return lambda prompt, _: prompt

    def _default_planning_response_format(self) -> PlanResponseFormat:
        """Create the default plan response format.

        Returns:
            Plan schema with task types from registered task definitions.
        """
        task_definitions = self._task_registry.get_task_definitions()
        task_types = [td.task_schema for td in task_definitions]
        return create_plan_with_tasks(task_types)

    def _parse_response(
        self,
        response: str,
        response_format: PlanResponseFormat,
        context: ContextVariables,
    ) -> Plan:
        """Parse agent response using schema.

        Args:
            response: Raw content to parse.
            response_format: Schema or parser function.
            context: Context for parsing.

        Returns:
            Parsed Plan model.

        Raises:
            TypeError: If response doesn't match schema.
            ValidationError: If response is invalid.

        Examples:
            Using model:
                ```python
                plan = self._parse_response(
                    response='{"tasks": [{"task_type": "coding", "id": "1", "title": "Write a hello world program in Python"}]}',
                    response_format=Plan,
                    context=ContextVariables()
                )
                ```

            Using parser:
                ```python
                def parse_plan_response(response: str, context: ContextVariables) -> Plan:
                    return Plan.model_validate_json(response)

                plan = self._parse_response(
                    response='{"tasks": [{"task_type": "coding", "id": "1", "title": "Write a hello world program in Python"}]}',
                    response_format=parse_plan_response,
                    context=ContextVariables()
                )
                ```
        """
        if is_subtype(response_format, Plan):
            return response_format.model_validate_json(response)

        if is_callable(response_format):
            return response_format(response, context)

        raise ValueError("Invalid response format")

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables | None = None,
        feedback: str | None = None,
    ) -> Result[Plan]:
        """Create a plan using the configured agent.

        Args:
            prompt: Description of work to be done.
            context: Optional context variables for planning.
            feedback: Optional feedback on previous attempts.

        Returns:
            Result containing either:
                - Valid Plan with tasks and dependencies.
                - Error if plan creation or validation fails.

        Examples:
            Create a plan:
                ```python
                result = await planner.create_plan(
                    prompt="Review and test PR #123",
                    context=ContextVariables(
                        pr_url="github.com/org/repo/123",
                        focus_areas=["security", "performance"]
                    )
                )

                if result.value:
                    plan = result.value
                    print(f"Created plan with {len(plan.tasks)} tasks")
                    for task in plan.tasks:
                        print(f"- {task.title}")
                ```
        """
        context = ContextVariables(context or ContextVariables())

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
            plan = self._parse_response(result.content, self.response_format, context)

            for task in plan.tasks:
                if not self._task_registry.contains_task_type(task.type):
                    return Result(error=ValueError(f"Unknown task type: {task.type}"))

            if errors := plan.validate_dependencies():
                return Result(error=ValueError("\n".join(errors)))

            return Result(value=plan)

        except Exception as e:
            return Result(error=e)
