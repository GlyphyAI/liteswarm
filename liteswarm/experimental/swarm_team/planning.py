# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

import json_repair
import orjson
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

PLANNING_AGENT_SYSTEM_PROMPT = """
You are an expert task planning specialist with deep experience in breaking down complex work into efficient, minimal workflows.

Your role is to:
1. Analyze requests and identify the core essential tasks (aim for ~1-3 tasks when possible).
2. Combine related work into larger, meaningful tasks.
3. Create precise, well-scoped tasks that match the available task types.
4. Consider team capabilities and workload when planning.
5. Ensure each task has clear success criteria.

Each task must include:
1. A descriptive title that clearly states the goal.
2. A comprehensive description explaining:
   - What needs to be done (including subtasks if needed).
   - How to verify completion.
   - Any important constraints or requirements.
3. The correct task type from available options.
4. Dependencies that enable parallel execution while maintaining correctness.
5. Fields that are specified in the response format for the task type.

Best practices to follow:
- Aim for around 1-3 well-defined tasks when possible.
- Make each task larger and more comprehensive.
- Combine related work into single tasks.
- Include subtasks in descriptions rather than creating separate tasks.
- Enable parallel execution where possible.
- Include clear verification steps.
- Use consistent naming conventions.
- Keep descriptions concise (ideally under 50 words per task).

IMPORTANT: The most effective plans typically have around 1-3 tasks. While more tasks are allowed when necessary, strive to keep plans concise and focused.
""".strip()

PLANNING_AGENT_USER_PROMPT = """
You need to create an efficient and concise execution plan for the following request:

<user_request>
{PROMPT}
</user_request>

Important context to consider:

<context>
{CONTEXT}
</context>

Analysis steps:
1. Understand the core requirements and aim for around 1-3 essential tasks.
2. Combine related work into larger, comprehensive tasks.
3. Include subtasks within task descriptions rather than creating separate tasks.
4. Organize tasks for parallel execution where possible.
5. Ensure each task has clear success criteria.

Your response MUST be a JSON object exactly matching this schema:

<response_format>
{RESPONSE_FORMAT}
</response_format>

When responding, you MUST:
1. Return ONLY the JSON object, no other text.
2. Strictly follow the <response_format> schema.
3. Start your response with opening curly brace ('{{') and end with closing curly brace ('}}').
4. Use valid JSON syntax without wrapping in code blocks or other formatting.
5. Include all required fields for each task.

Now proceed to create the plan.
""".strip()


class PlanningAgent(Protocol):
    """Protocol for agents that create task plans.

    Defines the interface for planning agents that can analyze prompts and create
    structured plans with tasks and dependencies. The plans must be OpenAI-compatible
    for structured outputs.

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
                        Task(
                            # Base Task required fields
                            type="review",  # Must match Literal in task definition
                            id="task-1",
                            title="First step",
                            description="Review code changes",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=[],
                            metadata=None,
                        ),
                        Task(
                            # Base Task required fields
                            type="test",
                            id="task-2",
                            title="Second step",
                            description="Run test suite",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=["task-1"],  # Depends on review
                            metadata=None,
                        ),
                    ]
                    # Create plan with all required fields
                    return Plan(tasks=tasks, metadata=None)
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
    validating them against task definitions. The framework supports two approaches
    to structured outputs that can be used independently or together:

    1. Framework-level Parsing:
       - Response is parsed using response_format (Pydantic model or parser function)
       - Works with any LLM provider
       - Can be combined with LLM-level formats for guaranteed schema validation

    Example:
           ```python
           def parse_plan(content: str, context: ContextVariables) -> Plan:
               # Custom parsing logic for any response format
               return Plan(...)


           agent = LitePlanningAgent(
               swarm=swarm,
               response_format=parse_plan,
           )
           ```

    2. LLM-level Schema:
       - Uses provider's native structured output support (e.g., OpenAI)
       - Requires OpenAI-compatible schemas (no defaults, simple types)
       - Can be combined with framework-level parsing for additional validation

    Example:
           ```python
           class ReviewPlan(Plan):  # OpenAI-compatible
               tasks: list[ReviewTask]
               metadata: dict[str, Any] | None


           agent = LitePlanningAgent(
               swarm=swarm,
               agent=Agent(
                   llm=LLM(
                       model="gpt-4o",
                       response_format=ReviewPlan,  # LLM enforces schema
                   ),
               ),
               response_format=ReviewPlan,  # Framework validates result
           )
           ```

    Note:
        The two approaches can be used together for robust schema handling:
        - LLM-level ensures valid JSON output format
        - Framework-level provides additional validation and parsing
        - Base Task and Plan classes support both approaches
        - Custom schemas must maintain OpenAI compatibility if using both:
          - No default values
          - Simple JSON-serializable types
          - Discriminated unions (no oneOf)
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
                planning_agent = LitePlanningAgent(swarm=swarm)
                repair_agent = planning_agent._default_response_repair_agent()
                assert isinstance(repair_agent, LiteResponseRepairAgent)
                ```

            Custom repair agent:
                ```python
                class CustomRepairAgent(ResponseRepairAgent):
                    async def repair_response(self, ...) -> Plan:
                        # Custom repair logic
                        pass

                planning_agent = LitePlanningAgent(
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
                agent = planning_agent._default_planning_agent()
                # Returns Agent with:
                # - id: "agent-planner"
                # - model: "gpt-4o"
                # - planning-specific instructions
                ```

            Use in planner:
                ```python
                planning_agent = LitePlanningAgent(swarm=swarm)
                # Automatically creates default agent if none provided
                assert planning_agent.agent.id == "agent-planner"
                assert planning_agent.agent.llm.model == "gpt-4o"
                ```
        """
        return Agent(
            id="agent-planner",
            instructions=PLANNING_AGENT_SYSTEM_PROMPT,
            llm=LLM(model="gpt-4o"),
        )

    def _default_planning_prompt_template(self) -> PromptTemplate:
        """Create the default prompt template.

        Creates a template that guides the agent to create efficient plans with
        comprehensive tasks. The template includes the request, context, and
        response format requirements.

        Returns:
            Template function that formats prompts with context.

        Example:
            ```python
            template = planning_agent._default_planning_prompt_template()
            prompt = template(
                "Review and deploy changes",
                context=ContextVariables(pr_url="github.com/org/repo/123"),
            )
            ```
        """

        def default_template_builder(prompt: str, context: ContextVariables) -> str:
            if is_subtype(self.response_format, Plan):
                response_format = orjson.dumps(self.response_format.model_json_schema()).decode()
            else:
                response_format = None

            return PLANNING_AGENT_USER_PROMPT.format(
                PROMPT=prompt,
                CONTEXT=context,
                RESPONSE_FORMAT=response_format,
            )

        return default_template_builder

    def _default_planning_response_format(self) -> PlanResponseFormat:
        """Create the default plan response format.

        Returns:
            Plan schema with task types from registered task definitions.

        Examples:
            Default format:
                ```python
                # With review and test task types
                planning_agent = LitePlanningAgent(swarm=swarm, task_definitions=[review_def, test_def])
                format = planning_agent._default_planning_response_format()
                # Returns Plan schema that accepts:
                # - ReviewTask
                # - TestTask
                ```

            Custom format:
                ```python
                def parse_plan(content: str, context: ContextVariables) -> Plan:
                    # Custom parsing logic
                    return Plan(tasks=[...])


                planning_agent = LitePlanningAgent(swarm=swarm, response_format=parse_plan)
                # Will use custom parser instead of schema
                ```
        """
        task_definitions = self._task_registry.get_task_definitions()
        task_types = [td.task_type for td in task_definitions]
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
                        Task(
                            # Base Task required fields
                            type="review",
                            id="1",
                            title="Review code",
                            description="Review PR changes",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=[],
                            metadata=None,
                        ),
                        Task(
                            # Base Task required fields
                            type="test",
                            id="2",
                            title="Run tests",
                            description="Run test suite",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=["1"],
                            metadata=None,
                        ),
                    ],
                    metadata=None,
                )
                validated_plan = planner._validate_plan(plan)  # Returns plan if valid
                ```

            Unknown task type:
                ```python
                plan = Plan(
                    tasks=[
                        Task(
                            type="unknown",  # Unknown type
                            id="1",
                            title="Invalid task",
                            description="This will fail",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=[],
                            metadata=None,
                        )
                    ],
                    metadata=None,
                )
                planning_agent._validate_plan(plan)
                # Raises PlanValidationError: Unknown task type
                ```

            Invalid dependencies:
                ```python
                plan = Plan(
                    tasks=[
                        Task(
                            type="review",
                            id="1",
                            title="Task 1",
                            description="First task",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=["2"],  # Circular dependency
                            metadata=None,
                        ),
                        Task(
                            type="test",
                            id="2",
                            title="Task 2",
                            description="Second task",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=["1"],  # Circular dependency
                            metadata=None,
                        ),
                    ],
                    metadata=None,
                )
                planning_agent._validate_plan(plan)
                # Raises PlanValidationError: Invalid task dependencies
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
                            "type": "review",
                            "id": "1",
                            "title": "Review PR",
                            "description": "Review code changes",
                            "status": "pending",
                            "assignee": null,
                            "dependencies": [],
                            "metadata": null
                        }
                    ],
                    "metadata": null
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
                plan = await planning_agent._parse_response(
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
                            "type": "review",
                            "id": "1",
                            "title": "Review PR",
                            "description": "Review code changes",
                            "status": "pending",
                            "assignee": null,
                            "dependencies": [],
                            "metadata": null
                        }
                    ],
                    "metadata": null
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
                plan = await planning_agent._process_planning_result(
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
                plan = await planning_agent.create_plan(
                    prompt="Review and test the authentication changes in PR #123"
                )
                print(f"Created plan with {len(plan.tasks)} tasks")
                for task in plan.tasks:
                    print(f"- {task.title} ({task.type})")
                ```

            With context:
                ```python
                plan = await planning_agent.create_plan(
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
                plan = await planning_agent.create_plan(
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
                    plan = await planning_agent.create_plan(prompt="Review the changes")
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
                planning_agent = LitePlanningAgent(
                    swarm=swarm,
                    prompt_template=lambda p, c: f"{p} for {c.get('project')}",
                    task_definitions=[review_def, test_def],
                )
                plan = await planning_agent.create_plan(
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
