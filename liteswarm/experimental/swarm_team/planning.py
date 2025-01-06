# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

import json_repair
from pydantic import ValidationError
from typing_extensions import override

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.response_repair import (
    LiteResponseRepairAgent,
    ResponseRepairAgent,
)
from liteswarm.types.collections import (
    AsyncStream,
    ReturnableAsyncGenerator,
    ReturnItem,
    YieldItem,
    returnable,
)
from liteswarm.types.events import SwarmEvent
from liteswarm.types.exceptions import PlanValidationError, ResponseParsingError, SwarmTeamError
from liteswarm.types.llm import LLM
from liteswarm.types.swarm import Agent, ContextVariables, Message
from liteswarm.types.swarm_team import Plan, PlanResponseFormat, PlanResult, TaskDefinition
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
    """Protocol for task planning agents built on top of Swarm.

    Defines an interface for planning agents that analyze conversation messages
    and create structured plans with tasks and dependencies. Following Swarm's
    stateless design, planning agents maintain no internal state between operations.

    Examples:
        Create a custom planner:
            ```python
            class CustomPlanningAgent(PlanningAgent):
                @returnable
                async def create_plan(
                    self,
                    messages: list[Message],
                    context_variables: ContextVariables | None = None,
                ) -> AsyncStream[SwarmEvent, PlanResult]:
                    # Create tasks based on messages
                    tasks = [
                        Task(
                            type="review",  # Must match task definition
                            id="task-1",
                            title="First step",
                            description="Review code changes",
                            status=TaskStatus.PENDING,
                            assignee=None,
                            dependencies=[],
                            metadata=None,
                        ),
                        Task(
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

                    # Create and validate plan
                    plan = Plan(id="plan-1", tasks=tasks, metadata=None)
                    yield ReturnItem(PlanResult(plan=plan))
            ```

    Notes:
        - Each method returns an event stream for progress tracking
        - Plans must be validated before being returned
        - Task types must match registered definitions
    """

    def create_plan(
        self,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> ReturnableAsyncGenerator[SwarmEvent, PlanResult]:
        """Create a plan from conversation messages.

        Args:
            messages: List of conversation messages for planning.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning plan result.

        Raises:
            PlanValidationError: If plan fails validation or has invalid dependencies.
            ResponseParsingError: If response cannot be parsed into a valid plan.
        """
        ...


class LitePlanningAgent(PlanningAgent):
    """LLM-based planning agent built on top of Swarm.

    Uses an LLM agent to analyze conversation messages and generate structured plans,
    validating them against task definitions. Following Swarm's stateless design,
    the agent maintains no internal state between operations.

    The agent supports two approaches to structured outputs:

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
       - Uses provider's native structured output support
       - Response format must follow provider-specific rules
       - Can be combined with framework-level parsing for additional validation

    Example:
           ```python
           class ReviewPlan(Plan):
               tasks: list[ReviewTask]
               metadata: dict[str, Any] | None


           agent = LitePlanningAgent(
               swarm=swarm,
               agent=Agent(
                   llm=LLM(
                       model="gpt-4o",
                       response_format=ReviewPlan,
                   ),
               ),
               response_format=ReviewPlan,
           )
           ```

    Notes:
        - Each method returns an event stream for progress tracking
        - Plans are validated before being returned
        - Response format must follow LLM provider's schema requirements
    """

    def __init__(
        self,
        swarm: Swarm,
        agent: Agent | None = None,
        task_definitions: list[TaskDefinition] | None = None,
        response_format: PlanResponseFormat | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
    ) -> None:
        """Initialize a new planner instance.

        Args:
            swarm: Swarm client for agent interactions.
            agent: Optional custom planning agent.
            task_definitions: Available task types.
            response_format: Optional plan response format.
            response_repair_agent: Optional custom response repair agent.
        """
        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)

        # Public properties
        self.swarm = swarm
        self.agent = agent or self._default_planning_agent(response_format)
        self.response_format = response_format or self._default_planning_response_format()
        self.response_repair_agent = response_repair_agent or self._default_response_repair_agent()

    def _default_planning_agent(self, response_format: PlanResponseFormat | None = None) -> Agent:
        """Create a default planning agent.

        Creates and configures an agent with GPT-4o and specialized planning
        instructions. The agent is designed to break down complex tasks into
        minimal, efficient workflows.

        Args:
            response_format: Optional plan response format.

        Returns:
            Agent configured with planning instructions and GPT-4o model.
        """
        response_format = response_format or self._default_planning_response_format()

        return Agent(
            id="agent-planner",
            instructions=PLANNING_AGENT_SYSTEM_PROMPT,
            llm=LLM(
                model="gpt-4o",
                response_format=response_format,
            ),
        )

    def _default_response_repair_agent(self) -> ResponseRepairAgent:
        """Create a default response repair agent.

        Creates and configures a LiteResponseRepairAgent instance using the current
        swarm. The repair agent helps recover from validation errors in planning
        responses by fixing common issues like JSON formatting and schema violations.

        Returns:
            LiteResponseRepairAgent configured with the current swarm.
        """
        return LiteResponseRepairAgent(swarm=self.swarm)

    def _default_planning_response_format(self) -> PlanResponseFormat:
        """Create a default plan response format.

        Creates a plan schema that includes all registered task types. The schema
        is used for both framework-level parsing and LLM-level validation when
        supported.

        Returns:
            Plan schema configured with registered task types.
        """
        task_definitions = self._task_registry.get_task_definitions()
        task_types = [td.task_type for td in task_definitions]
        return create_plan_with_tasks(task_types)

    def _validate_plan(self, plan: Plan) -> Plan:
        """Validate a plan against task registry and dependency rules.

        Performs two-phase validation:
        1. Verifies all task types are registered and valid
        2. Checks that dependencies form a valid DAG without cycles

        Args:
            plan: Plan to validate.

        Returns:
            The validated plan if all checks pass.

        Raises:
            PlanValidationError: If plan contains unknown task types or has invalid
                dependencies.
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
        context_variables: ContextVariables | None = None,
    ) -> Plan:
        """Parse agent response into a plan object.

        Converts raw agent responses into structured plan objects using either
        direct schema validation or custom parser functions. Uses json_repair for
        basic error recovery before validation.

        Args:
            response: Raw response string from agent.
            response_format: Schema or parser function for validation.
            context_variables: Optional context for parsing.

        Returns:
            Validated plan object matching the response format.

        Raises:
            ValueError: If response format is invalid.
            ValidationError: If response cannot be parsed into plan.
            ResponseParsingError: If there are other errors during parsing.
        """
        if is_callable(response_format):
            return response_format(response, context_variables)

        if not is_subtype(response_format, Plan):
            raise ValueError("Invalid response format")

        # TODO: Use RepairAgent to fix JSON errors
        decoded_object = json_repair.repair_json(response, return_objects=True)
        if isinstance(decoded_object, tuple):
            decoded_object = decoded_object[0]

        return response_format.model_validate(decoded_object)

    async def _process_planning_result(
        self,
        agent: Agent,
        response: str,
        context_variables: ContextVariables | None = None,
    ) -> Plan:
        """Process and validate a planning response.

        Manages the complete lifecycle of response processing by parsing the raw
        response, attempting repair for invalid responses, and validating the
        resulting plan against task registry and dependency rules.

        Args:
            agent: Agent that produced the response.
            response: Raw response string from agent.
            context_variables: Optional context for processing.

        Returns:
            Complete validated plan object.

        Raises:
            PlanValidationError: If plan fails validation even after repair.
            ResponseParsingError: If response cannot be parsed into valid plan.
        """
        try:
            plan = await self._parse_response(
                response=response,
                response_format=self.response_format,
                context_variables=context_variables,
            )

            return self._validate_plan(plan)

        except ValidationError as validation_error:
            repaired_response = await self.response_repair_agent.repair_response(
                agent=agent,
                response=response,
                response_format=self.response_format,
                validation_error=validation_error,
                context_variables=context_variables,
            )

            return self._validate_plan(repaired_response)

        except Exception as e:
            raise ResponseParsingError(
                f"Error processing plan response: {e}",
                response=response,
                original_error=e,
            ) from e

    @override
    @returnable
    async def create_plan(
        self,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEvent, PlanResult]:
        """Create a plan from conversation messages.

        Uses an LLM agent to analyze messages and generate a structured plan with
        ordered tasks. The plan is validated to ensure task types are registered
        and dependencies form a valid DAG. Following Swarm's stateless design,
        the method maintains no internal state between operations.

        Args:
            messages: List of conversation messages for planning.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning plan result.

        Raises:
            PlanValidationError: If plan fails validation or has invalid dependencies.
            ResponseParsingError: If response cannot be parsed into valid plan.
            SwarmTeamError: If planning agent fails to return content.

        Examples:
            Basic planning:
                ```python
                stream = planning_agent.create_plan(
                    messages=[
                        Message(
                            role="user",
                            content="Review and test the authentication changes in PR #123",
                        ),
                    ],
                )

                # Process events during planning
                async for event in stream:
                    if event.type == "plan_created":
                        print(f"Created plan with {len(event.plan.tasks)} tasks")

                # Get final result after completion
                plan_result = await stream.get_return_value()
                for task in plan_result.plan.tasks:
                    print(f"- {task.title} ({task.type})")
                ```

            With context:
                ```python
                stream = planning_agent.create_plan(
                    messages=[
                        Message(
                            role="user",
                            content="Review the security changes",
                        ),
                    ],
                    context_variables=ContextVariables(
                        pr_url="github.com/org/repo/123",
                        focus_areas=["authentication", "authorization"],
                        security_checklist=["SQL injection", "XSS", "CSRF"],
                    ),
                )
                plan_result = await stream.get_return_value()
                # Plan tasks will incorporate context information
                ```

            Complex workflow:
                ```python
                stream = planning_agent.create_plan(
                    messages=[
                        Message(
                            role="user",
                            content=\"\"\"
                            Review and deploy the new payment integration:
                            1. Review code changes
                            2. Run security tests
                            3. Test payment flows
                            4. Deploy to staging
                            5. Monitor for issues
                            \"\"\",
                        ),
                    ],
                    context_variables=ContextVariables(
                        pr_url="github.com/org/repo/456",
                        deployment_env="staging",
                        test_cases=["visa", "mastercard", "paypal"],
                        monitoring_metrics=["latency", "error_rate"],
                    ),
                )
                plan_result = await stream.get_return_value()
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
                    stream = planning_agent.create_plan(
                        messages=[
                            Message(role="user", content="Review the changes"),
                        ],
                    )
                    plan_result = await stream.get_return_value()
                except PlanValidationError as e:
                    if "Unknown task type" in str(e):
                        print("Plan contains unsupported task types")
                    elif "Invalid task dependencies" in str(e):
                        print("Plan has invalid task dependencies")
                    else:
                        print(f"Plan validation failed: {e}")
                except ResponseParsingError as e:
                    print(f"Failed to parse planning response: {e}")
                except SwarmTeamError as e:
                    print(f"Planning failed: {e}")
                ```

        Notes:
            - Each call returns an event stream for progress tracking
            - Plans are validated before being returned
            - Task types must match registered definitions
            - Dependencies must form a valid DAG
        """
        stream = self.swarm.stream(
            agent=self.agent,
            messages=messages,
            context_variables=context_variables,
        )

        async for event in stream:
            yield YieldItem(event)

        result = await stream.get_return_value()
        response = result.agent_responses[-1] if result.agent_responses else None
        content = response.content if response else None
        if not content:
            raise SwarmTeamError("The planning agent did not return any content")

        plan = await self._process_planning_result(
            agent=self.agent,
            response=content,
            context_variables=context_variables,
        )

        yield ReturnItem(
            PlanResult(
                plan=plan,
                new_messages=result.new_messages,
                all_messages=result.all_messages,
            )
        )
