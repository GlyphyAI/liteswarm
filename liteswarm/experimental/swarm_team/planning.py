# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

import json_repair
from pydantic import ValidationError

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.response_repair import (
    LiteResponseRepairAgent,
    ResponseRepairAgent,
)
from liteswarm.types.exceptions import PlanValidationError, ResponseParsingError
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


class PlanningAgent(Protocol):
    """Protocol for agents that create task plans.

    Defines the interface for planning agents that can analyze prompts and create
    structured plans with tasks and dependencies.

    Examples:
        Create a custom planner:
            ```python
            class CustomPlanningAgent(PlanningAgent):
                async def create_plan(
                    self,
                    prompt: str,
                    context: ContextVariables | None = None,
                ) -> Plan:
                    # Analyze prompt and create tasks
                    tasks = [
                        Task(id="task-1", title="First step"),
                        Task(id="task-2", title="Second step", dependencies=["task-1"]),
                    ]
                    return Plan(tasks=tasks)
            ```
    """

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables | None = None,
    ) -> Plan:
        """Create a plan from the given prompt and context.

        Args:
            prompt: Description of work to be done.
            context: Optional additional context variables.

        Returns:
            A valid Plan object.

        Raises:
            PlanValidationError: If the plan fails validation or has invalid dependencies.
            ResponseParsingError: If the response cannot be parsed into a valid plan.
        """
        ...


class LitePlanningAgent(PlanningAgent):
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
                task_instructions="Review {task.pr_url}",
            )

            # Initialize planner
            planner = LitePlanningAgent(
                swarm=swarm,
                agent=Agent(id="planner", llm=LLM(model="gpt-4o")),
                task_definitions=[review_def],
            )

            # Create plan
            plan = await planner.create_plan(
                prompt="Review PR #123",
                context=ContextVariables(pr_url="github.com/org/repo/123"),
            )
            ```
    """

    def __init__(  # noqa: PLR0913
        self,
        swarm: Swarm,
        agent: Agent | None = None,
        prompt_template: PromptTemplate | None = None,
        task_definitions: list[TaskDefinition] | None = None,
        response_format: PlanResponseFormat | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
    ) -> None:
        """Initialize a new planner instance.

        Args:
            swarm: Swarm client for agent interactions.
            agent: Optional custom planning agent.
            prompt_template: Optional custom prompt template.
            task_definitions: Available task types.
            response_format: Optional plan response format.
            response_repair_agent: Optional custom response repair agent.
        """
        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)

        # Public properties
        self.swarm = swarm
        self.agent = agent or self._default_planning_agent()
        self.prompt_template = prompt_template or self._default_planning_prompt_template()
        self.response_format = response_format or self._default_planning_response_format()
        self.response_repair_agent = response_repair_agent or self._default_response_repair_agent()

    def _default_response_repair_agent(self) -> ResponseRepairAgent:
        """Create the default response repair agent for handling invalid planning responses.

        Creates and configures a LiteResponseRepairAgent instance using the current swarm.
        The repair agent helps recover from validation errors in planning responses by
        attempting to fix common issues like JSON formatting and schema violations.

        Returns:
            A configured repair agent instance using LiteResponseRepairAgent
                implementation with the current swarm.

        Examples:
            Basic usage:
                ```python
                planner = LitePlanningAgent(swarm=swarm)
                repair_agent = planner._default_response_repair_agent()
                assert isinstance(repair_agent, LiteResponseRepairAgent)
                ```

            Custom repair agent:
                ```python
                class CustomRepairAgent(ResponseRepairAgent):
                    async def repair_response(self, ...) -> Plan:
                        # Custom repair logic
                        pass

                planner = LitePlanningAgent(
                    swarm=swarm,
                    response_repair_agent=CustomRepairAgent(),
                )
                # Will use custom agent instead of default
                ```
        """
        return LiteResponseRepairAgent(swarm=self.swarm)

    def _default_planning_agent(self) -> Agent:
        """Create the default planning agent.

        Returns:
            Agent configured with GPT-4o and planning instructions.

        Examples:
            Create default agent:
                ```python
                agent = planner._default_planning_agent()
                # Returns Agent with:
                # - id: "agent-planner"
                # - model: "gpt-4o"
                # - planning-specific instructions
                ```

            Use in planner:
                ```python
                planner = LitePlanningAgent(swarm=swarm)
                # Automatically creates default agent if none provided
                assert planner.agent.id == "agent-planner"
                assert planner.agent.llm.model == "gpt-4o"
                ```
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

        Examples:
            Default template:
                ```python
                template = planner._default_planning_prompt_template()
                prompt = template("Create a plan", context={})
                assert prompt == "Create a plan"  # Returns raw prompt
                ```

            Custom template:
                ```python
                def custom_template(prompt: str, context: ContextVariables) -> str:
                    return f"{prompt} for {context.get('project_name')}"


                planner = LitePlanningAgent(swarm=swarm, prompt_template=custom_template)
                # Will format prompts with project name
                ```
        """
        return lambda prompt, _: prompt

    def _default_planning_response_format(self) -> PlanResponseFormat:
        """Create the default plan response format.

        Returns:
            Plan schema with task types from registered task definitions.

        Examples:
            Default format:
                ```python
                # With review and test task types
                planner = LitePlanningAgent(swarm=swarm, task_definitions=[review_def, test_def])
                format = planner._default_planning_response_format()
                # Returns Plan schema that accepts:
                # - ReviewTask
                # - TestTask
                ```

            Custom format:
                ```python
                def parse_plan(content: str, context: ContextVariables) -> Plan:
                    # Custom parsing logic
                    return Plan(tasks=[...])


                planner = LitePlanningAgent(swarm=swarm, response_format=parse_plan)
                # Will use custom parser instead of schema
                ```
        """
        task_definitions = self._task_registry.get_task_definitions()
        task_types = [td.task_schema for td in task_definitions]
        return create_plan_with_tasks(task_types)

    def _validate_plan(self, plan: Plan) -> Plan:
        """Validate a plan against task registry and dependency rules.

        Checks that all tasks are registered and that the dependency graph is
        a valid DAG without cycles.

        Args:
            plan: Plan to validate.

        Returns:
            The validated Plan if all checks pass.

        Raises:
            PlanValidationError: If the plan contains unknown task types or has invalid
                dependencies.

        Examples:
            Valid plan:
                ```python
                plan = Plan(
                    tasks=[
                        Task(id="1", type="review", title="Review code"),
                        Task(id="2", type="test", title="Run tests", dependencies=["1"]),
                    ]
                )
                validated_plan = planner._validate_plan(plan)  # Returns plan if valid
                ```

            Unknown task type:
                ```python
                plan = Plan(tasks=[Task(id="1", type="unknown", title="Invalid task")])
                # Raises PlanValidationError: Unknown task type
                planner._validate_plan(plan)
                ```

            Invalid dependencies:
                ```python
                plan = Plan(
                    tasks=[
                        Task(id="1", type="review", title="Task 1", dependencies=["2"]),
                        Task(id="2", type="test", title="Task 2", dependencies=["1"]),
                    ]
                )
                # Raises PlanValidationError: Invalid task dependencies
                planner._validate_plan(plan)
                ```
        """
        for task in plan.tasks:
            if not self._task_registry.contains_task_type(task.type):
                raise PlanValidationError(f"Unknown task type: {task.type}")

        if errors := plan.validate_dependencies():
            raise PlanValidationError("Invalid task dependencies", validation_errors=errors)

        return plan

    async def _parse_response(
        self,
        response: str,
        response_format: PlanResponseFormat,
        context: ContextVariables,
    ) -> Plan:
        """Parse agent response into a Plan object.

        Handles both direct Plan schemas and callable parsers.
        Uses json_repair to attempt basic JSON repair before validation.

        Args:
            response: Raw response string from agent.
            response_format: Schema or parser function.
            context: Context for parsing.

        Returns:
            Parsed Plan object.

        Raises:
            ValueError: If response format is invalid.
            ValidationError: If response cannot be parsed into Plan.
            ResponseParsingError: If there are other errors during parsing.

        Examples:
            Parse with schema:
                ```python
                response = '''
                {
                    "tasks": [
                        {
                            "id": "1",
                            "type": "review",
                            "title": "Review PR",
                            "dependencies": []
                        }
                    ]
                }
                '''
                plan = await planner._parse_response(
                    response=response,
                    response_format=Plan,
                    context=context,
                )
                # Returns Plan instance
                ```

            Parse with custom function:
                ```python
                def parse_plan(content: str, context: ContextVariables) -> Plan:
                    # Custom parsing logic
                    data = json.loads(content)
                    return Plan(tasks=[...])


                plan = await planner._parse_response(
                    response=response,
                    response_format=parse_plan,
                    context=context,
                )
                # Returns Plan via custom parser
                ```

            With json_repair:
                ```python
                response = '''
                {
                    'tasks': [  # Single quotes
                        {
                            id: "1",  # Missing quotes
                            type: "review",
                            title: "Review PR",
                            dependencies: []
                        }
                    ]
                }
                '''
                plan = await planner._parse_response(
                    response=response,
                    response_format=Plan,
                    context=context,
                )
                # Still returns valid Plan
                ```
        """
        if is_callable(response_format):
            return response_format(response, context)

        if not is_subtype(response_format, Plan):
            raise ValueError("Invalid response format")

        decoded_object = json_repair.repair_json(response, return_objects=True)
        if isinstance(decoded_object, tuple):
            decoded_object = decoded_object[0]

        return response_format.model_validate(decoded_object)

    async def _process_planning_result(
        self,
        agent: Agent,
        response: str,
        context: ContextVariables,
    ) -> Plan:
        """Process and validate a planning response.

        Attempts to parse and validate the response according to the response
        format. If validation fails, tries to recover using the response repair
        agent.

        Args:
            agent: Agent that produced the response.
            response: Raw response string.
            context: Context for parsing and repair.

        Returns:
            A valid Plan object.

        Raises:
            PlanValidationError: If the plan fails validation even after repair.
            ResponseParsingError: If the response cannot be parsed into a valid plan.

        Examples:
            Successful processing:
                ```python
                response = '''
                {
                    "tasks": [
                        {
                            "id": "1",
                            "type": "review",
                            "title": "Review PR",
                            "dependencies": []
                        }
                    ]
                }
                '''
                plan = await planner._process_planning_result(
                    agent=agent,
                    response=response,
                    context=context,
                )
                # Returns valid Plan
                ```

            With response repair:
                ```python
                response = '''
                {
                    tasks: [  # Invalid JSON
                        {
                            id: "1",
                            type: "review",
                            title: "Review PR"
                        }
                    ]
                }
                '''
                plan = await planner._process_planning_result(
                    agent=agent,
                    response=response,
                    context=context,
                )
                # Returns repaired and validated Plan
                ```

            Invalid task type:
                ```python
                response = '''
                {
                    "tasks": [
                        {
                            "id": "1",
                            "type": "unknown",  # Unknown type
                            "title": "Invalid task"
                        }
                    ]
                }
                '''
                # Raises PlanValidationError: Unknown task type
                plan = await planner._process_planning_result(
                    agent=agent,
                    response=response,
                    context=context,
                )
                ```
        """
        try:
            plan = await self._parse_response(
                response=response,
                response_format=self.response_format,
                context=context,
            )

            return self._validate_plan(plan)

        except ValidationError as validation_error:
            repaired_response = await self.response_repair_agent.repair_response(
                agent=agent,
                response=response,
                response_format=self.response_format,
                validation_error=validation_error,
                context=context,
            )

            return self._validate_plan(repaired_response)

        except Exception as e:
            raise ResponseParsingError(
                f"Error processing plan response: {e}",
                response=response,
                original_error=e,
            ) from e

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables | None = None,
    ) -> Plan:
        """Create a plan from the given prompt and context.

        Args:
            prompt: Description of work to be done.
            context: Optional additional context variables.

        Returns:
            A valid Plan object.

        Raises:
            PlanValidationError: If the generated plan fails validation or has invalid
                dependencies.
            ResponseParsingError: If the agent response cannot be parsed into a valid plan.

        Examples:
            Basic planning:
                ```python
                plan = await planner.create_plan(
                    prompt="Review and test the authentication changes in PR #123"
                )
                print(f"Created plan with {len(plan.tasks)} tasks")
                for task in plan.tasks:
                    print(f"- {task.title} ({task.type})")
                ```

            With context:
                ```python
                plan = await planner.create_plan(
                    prompt="Review the security changes",
                    context=ContextVariables(
                        pr_url="github.com/org/repo/123",
                        focus_areas=["authentication", "authorization"],
                        security_checklist=["SQL injection", "XSS", "CSRF"],
                    ),
                )
                # Plan tasks will incorporate context information
                ```

            Complex workflow:
                ```python
                plan = await planner.create_plan(
                    prompt=\"\"\"
                    Review and deploy the new payment integration:
                    1. Review code changes
                    2. Run security tests
                    3. Test payment flows
                    4. Deploy to staging
                    5. Monitor for issues
                    \"\"\",
                    context=ContextVariables(
                        pr_url="github.com/org/repo/456",
                        deployment_env="staging",
                        test_cases=["visa", "mastercard", "paypal"],
                        monitoring_metrics=["latency", "error_rate"]
                    )
                )
                # Plan will have tasks for each step with proper dependencies
                # - Code review task
                # - Security testing task (depends on review)
                # - Payment testing tasks (depend on security)
                # - Deployment task (depends on tests)
                # - Monitoring task (depends on deployment)
                ```

            Error handling:
                ```python
                try:
                    plan = await planner.create_plan(prompt="Review the changes")
                except PlanValidationError as e:
                    if "Unknown task type" in str(e):
                        print("Plan contains unsupported task types")
                    elif "Invalid task dependencies" in str(e):
                        print("Plan has invalid task dependencies")
                    else:
                        print(f"Plan validation failed: {e}")
                except ResponseParsingError as e:
                    print(f"Failed to parse planning response: {e}")
                else:
                    # Use the plan
                    pass
                ```

            Custom template:
                ```python
                # With custom prompt template
                planner = LitePlanningAgent(
                    swarm=swarm,
                    prompt_template=lambda p, c: f"{p} for {c.get('project')}",
                    task_definitions=[review_def, test_def],
                )
                plan = await planner.create_plan(
                    prompt="Review the changes",
                    context=ContextVariables(project="Payment API"),
                )
                # Template will format prompt as "Review the changes for Payment API"
                ```
        """
        context = ContextVariables(context or {})

        if is_callable(self.prompt_template):
            prompt = self.prompt_template(prompt, context)

        result = await self.swarm.execute(
            agent=self.agent,
            prompt=prompt,
            context_variables=context,
        )

        return await self._process_planning_result(
            agent=self.agent,
            response=result.content or "",
            context=context,
        )
