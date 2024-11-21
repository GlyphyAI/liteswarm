# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import operator
from collections.abc import Callable
from functools import reduce
from typing import Protocol, TypeAlias

from liteswarm.core.swarm import Swarm
from liteswarm.types import Result
from liteswarm.types.swarm import Agent, ContextVariables
from liteswarm.types.swarm_team import Plan, TaskDefinition
from liteswarm.utils.misc import change_field_type, extract_json

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

Can be either:
- A static template string with placeholders
- A function that dynamically generates prompts

Example:
```python
# Static template
template: PromptTemplate = \"\"\"
Process this request: {prompt}
Using these tools: {context}
\"\"\"

# Dynamic template
def generate_prompt(prompt: str, context: ContextVariables) -> str:
    return dedent_prompt(f\"\"\"
    Process this request: {prompt}

    Available tools:
    {format_tools(context.get("tools", []))}

    Team context:
    - Project: {context.get("project_name")}
    - Priority: {context.get("priority")}
    \"\"\")

# Usage with static template
formatted = template.format(
    prompt="Analyze data",
    context=ContextVariables(available_tools=["tool1", "tool2"])
)

# Or with dynamic template
formatted = generate_prompt(
    prompt="Analyze data",
    context=ContextVariables(
        tools=["tool1", "tool2"],
        project_name="Analytics",
        priority="high"
    )
)
```
"""


class PlanningAgent(Protocol):
    """Protocol for planning agents that create task plans.

    Planning agents analyze prompts and create structured plans by:
    1. Breaking down work into tasks
    2. Setting appropriate dependencies
    3. Validating against task definitions
    4. Considering team capabilities

    Example:
    ```python
    class CustomPlanner(PlanningAgent):
        async def create_plan(
            self,
            prompt: str,
            context: ContextVariables,
            feedback: str | None = None
        ) -> Result[Plan]:
            # Build LLM instructions based on prompt and context
            # Retrieve & parse plan response
            # Validate plan & dependencies
            return Result(value=plan)
    ```
    """

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables,
        feedback: str | None = None,
    ) -> Result[Plan]:
        """Create a plan from the given prompt and context.

        Args:
            prompt: Description of what needs to be done
            context: Variables providing additional context
            feedback: Optional feedback on previous plan attempts

        Returns:
            Result containing either:
            - Successful Plan instance
            - Error if plan creation fails
        """
        ...


class AgentPlanner(PlanningAgent):
    """Default implementation of the planning strategy using an LLM agent.

    Creates plans by:
    1. Using an LLM agent to analyze requirements
    2. Generating JSON-structured plans
    3. Validating against task definitions
    4. Ensuring valid task dependencies

    Example:
    ```python
    # Create task definitions
    review_def = TaskDefinition.create(
        task_type="code_review",
        task_schema=CodeReviewTask,
        task_instructions="Review {task.pr_url}..."
    )

    test_def = TaskDefinition.create(
        task_type="testing",
        task_schema=TestingTask,
        task_instructions="Test {task.test_path}..."
    )

    # Create planner with custom template
    planner = AgentPlanner(
        swarm=swarm,
        agent=Agent.create(
            id="planner",
            model="gpt-4",
            instructions="You are a technical project planner..."
        ),
        template=CustomTemplate(),
        task_definitions=[review_def, test_def]
    )

    # Create plan
    result = await planner.create_plan(
        prompt="Review and test PR #123",
        context=ContextVariables(
            pr_url="https://github.com/org/repo/pull/123"
        )
    )
    ```
    """

    def __init__(
        self,
        swarm: Swarm,
        agent: Agent | None = None,
        template: PromptTemplate | None = None,
        task_definitions: list[TaskDefinition] | None = None,
    ) -> None:
        """Initialize a new AgentPlanner instance.

        Args:
            swarm: Swarm instance for agent interactions
            agent: Optional custom planning agent
            template: Optional custom prompt template
            task_definitions: List of available task types
        """
        self.swarm = swarm
        self.agent = agent or self._default_planning_agent()
        self.template = template or self._default_planning_prompt_template()
        self.task_definitions = {td.task_type: td for td in task_definitions or []}

    def _default_planning_agent(self) -> Agent:
        """Create a default agent if none is provided.

        Returns:
            Agent configured for planning tasks

        The default agent:
        - Uses gpt-4o for complex planning
        - Has specialized planning instructions
        - Focuses on task breakdown and dependencies
        """
        return Agent.create(
            id="agent-planner",
            model="gpt-4o",
            instructions=AGENT_PLANNER_INSTRUCTIONS,
        )

    def _default_planning_prompt_template(self) -> PromptTemplate:
        """Create a default prompt template.

        Returns:
            Simple template that uses raw prompt
        """
        return "{prompt}"

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables,
        feedback: str | None = None,
    ) -> Result[Plan]:
        """Create a plan using the configured agent.

        Args:
            prompt: Description of what needs to be done
            context: Context variables for planning
            feedback: Optional feedback on previous attempts

        Returns:
            Result containing either:
            - Successful Plan instance
            - Error if validation fails

        Example:
        ```python
        result = await planner.create_plan(
            prompt='''
            Review PR #123 which updates authentication:
            1. Security review of auth changes
            2. Test the new auth flow
            ''',
            context=ContextVariables(
                pr_url="https://github.com/org/repo/pull/123",
                team_capabilities={
                    "security_review": ["security-bot"],
                    "testing": ["test-bot"]
                }
            ),
            feedback="Please add input validation tests"
        )
        ```
        """
        if feedback:
            full_prompt = f"{prompt}\n\nPrevious feedback:\n{feedback}"
        else:
            full_prompt = prompt

        task_definitions = list(self.task_definitions.values())
        task_schemas = reduce(operator.or_, [td.task_schema for td in task_definitions])
        plan_schema_name = Plan.__name__
        plan_schema = change_field_type(
            model_type=Plan,
            field_name="tasks",
            new_type=list[task_schemas],  # type: ignore [valid-type]
            new_model_name=plan_schema_name,
        )

        output_format = plan_schema.model_json_schema()
        context = ContextVariables(**context)
        context.set_reserved("output_format", output_format)

        if callable(self.template):
            formatted_prompt = self.template(full_prompt, context)
        else:
            formatted_prompt = self.template.format(prompt=full_prompt, context=context)

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
