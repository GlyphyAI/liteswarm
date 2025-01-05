# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections import defaultdict
from uuid import uuid4

import json_repair
from pydantic import BaseModel, ValidationError

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planning import LitePlanningAgent, PlanningAgent, PlanResult
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.response_repair import (
    LiteResponseRepairAgent,
    ResponseRepairAgent,
)
from liteswarm.types.collections import AsyncStream, ReturnItem, YieldItem, returnable
from liteswarm.types.events import (
    PlanCompletedEvent,
    PlanCreatedEvent,
    SwarmEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
)
from liteswarm.types.exceptions import TaskExecutionError
from liteswarm.types.swarm import AgentExecutionResult, ContextVariables, Message
from liteswarm.types.swarm_team import (
    Artifact,
    ArtifactStatus,
    Plan,
    Task,
    TaskDefinition,
    TaskResponseFormat,
    TaskResult,
    TaskStatus,
    TeamMember,
)
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.typing import is_callable, is_subtype


class SwarmTeam:
    """Experimental orchestrator for specialized AI agent teams built on top of Swarm.

    SwarmTeam provides a high-level interface for executing complex tasks using teams
    of specialized AI agents. Following Swarm's stateless design, SwarmTeam maintains
    no internal state between operations - you need to implement your own abstractions
    for history management and other stateful features.

    The orchestrator uses a two-phase approach:

    1. Planning Phase:
        - Analyzes conversation messages to create structured plans
        - Uses planning agent to break down work into tasks
        - Validates task types and dependencies
        - Supports event streaming for progress tracking

    2. Execution Phase:
        - Assigns tasks to capable team members
        - Executes tasks in parallel when dependencies allow
        - Validates responses against schemas
        - Produces execution artifacts with results

    The framework supports structured outputs through a two-layer parsing system:
        - Framework-level parsing using Pydantic models
        - LLM-level schema validation (if supported)
        - Response repair for validation errors

    Examples:
        Basic workflow execution:
            ```python
            # 1. Define task types and outputs
            class ReviewTask(Task):
                type: Literal["code_review"]
                pr_url: str
                review_type: str


            class ReviewOutput(BaseModel):
                issues: list[str]
                approved: bool


            # 2. Create task definitions
            review_def = TaskDefinition(
                task_type=ReviewTask,
                instructions="Review {task.pr_url} focusing on {task.review_type}",
                response_format=ReviewOutput,
            )

            # 3. Create team members
            reviewer = TeamMember(
                id="senior-reviewer",
                agent=Agent(
                    id="reviewer",
                    instructions="You are a code reviewer.",
                    llm=LLM(model="gpt-4o"),
                ),
                task_types=[ReviewTask],
            )

            # 4. Create team
            swarm = Swarm()
            team = SwarmTeam(
                swarm=swarm,
                members=[reviewer],
                task_definitions=[review_def],
            )

            # 5. Create execution plan
            stream = team.create_plan(
                messages=[
                    Message(
                        role="user",
                        content="Review PR #123 for security issues",
                    ),
                ],
                context_variables=ContextVariables(
                    pr_url="github.com/org/repo/123",
                    review_type="security",
                ),
            )

            # Process planning events
            async for event in stream:
                if event.type == "plan_created":
                    print(f"Created plan with {len(event.plan.tasks)} tasks")

            # Get plan result
            plan_result = await stream.get_return_value()

            # 6. Execute plan
            stream = team.execute_plan(
                plan=plan_result.plan,
                messages=[Message(role="user", content="Review PR #123")],
                context_variables=ContextVariables(
                    pr_url="github.com/org/repo/123",
                ),
            )

            # Process execution events
            async for event in stream:
                if event.type == "task_started":
                    print(f"Started task: {event.task.title}")
                elif event.type == "task_completed":
                    print(f"Completed task: {event.task.title}")

            # Get execution result
            artifact = await stream.get_return_value()
            if artifact.status == ArtifactStatus.COMPLETED:
                for result in artifact.task_results:
                    output = result.output  # ReviewOutput instance
                    print(f"Review by: {result.assignee.id}")
                    print(f"Issues: {output.issues}")
                    print(f"Approved: {output.approved}")
            ```

        Direct task execution:
            ```python
            # Execute single task
            stream = team.execute_task(
                task=ReviewTask(
                    type="code_review",
                    id="review-1",
                    title="Security review",
                    description="Review PR for vulnerabilities",
                    status=TaskStatus.PENDING,
                    assignee=None,
                    dependencies=[],
                    metadata=None,
                    pr_url="github.com/org/repo/123",
                    review_type="security",
                ),
                messages=[Message(role="user", content="Review PR #123")],
                context_variables=ContextVariables(
                    pr_url="github.com/org/repo/123",
                ),
            )

            # Process execution events
            async for event in stream:
                if event.type == "task_completed":
                    print(f"Task completed: {event.task.title}")

            # Get task result
            result = await stream.get_return_value()
            print(f"Review by: {result.assignee.id}")
            print(f"Status: {result.task.status}")
            ```

    Notes:
        - Each method returns an event stream for progress tracking
        - Tasks are executed in parallel when dependencies allow
        - Responses are validated against task schemas
        - Custom agents can be provided for planning and repair
    """

    def __init__(
        self,
        swarm: Swarm,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        planning_agent: PlanningAgent | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
    ) -> None:
        """Initialize a new team instance.

        Creates a new team instance with specified members and task capabilities.
        The team uses a swarm client for agent interactions and maintains a registry
        of task definitions that team members can execute.

        Each team member is configured with:
        - An agent for task execution
        - A list of task types they can handle
        - Optional specialization metadata

        Task definitions specify:
        - Task type and required fields
        - Execution instructions
        - Optional response formats for validation

        Args:
            swarm: Swarm client for agent interactions.
            members: Team members with their capabilities.
            task_definitions: Task types the team can handle.
            planning_agent: Optional custom planning agent for task breakdown.
            response_repair_agent: Optional custom repair agent for validation errors.

        Notes:
            - Team maintains a registry of task definitions
            - Members are mapped to task types they can handle
            - Custom agents can be provided for planning and repair
        """
        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)
        self._team_capabilities = self._get_team_capabilities(members)

        # Public properties
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.planning_agent = planning_agent or self._default_planning_agent(task_definitions)
        self.response_repair_agent = response_repair_agent or self._default_response_repair_agent()

    # ================================================
    # MARK: Internal Helpers
    # ================================================

    def _default_planning_agent(self, task_definitions: list[TaskDefinition]) -> LitePlanningAgent:
        """Create a default planning agent.

        Creates and configures a LitePlanningAgent instance using the current swarm
        and provided task definitions. The planning agent is responsible for analyzing
        prompts and creating structured plans with tasks.

        Args:
            task_definitions: Task definitions to use for planning.

        Returns:
            Default planning agent.
        """
        return LitePlanningAgent(swarm=self.swarm, task_definitions=task_definitions)

    def _default_response_repair_agent(self) -> LiteResponseRepairAgent:
        """Create a default response repair agent.

        Creates and configures a LiteResponseRepairAgent instance using the current swarm.
        The repair agent helps recover from validation errors in task responses by
        attempting to fix common issues like JSON formatting and schema violations.

        Returns:
            Default response repair agent.
        """
        return LiteResponseRepairAgent(swarm=self.swarm)

    def _get_team_capabilities(self, members: list[TeamMember]) -> dict[str, list[str]]:
        """Map task types to capable team members.

        Creates a mapping of task types to team member IDs who can handle them.
        This mapping is used during task assignment to find capable members for
        each task type.

        Args:
            members: Team members to analyze for capabilities.

        Returns:
            Dict mapping task types to member IDs.
        """
        capabilities: dict[str, list[str]] = defaultdict(list[str])
        for member in members:
            for task_type in member.task_types:
                capabilities[task_type.get_task_type()].append(member.agent.id)

        return capabilities

    def _prepare_instructions(
        self,
        task: Task,
        task_definition: TaskDefinition,
        task_context_variables: ContextVariables | None = None,
    ) -> str:
        """Prepare task instructions for execution.

        Prepares a properly formatted instruction string by resolving either static
        templates or dynamic instruction functions. Dynamic instructions are resolved
        using context variables if provided.

        Args:
            task: Task requiring instruction preparation.
            task_definition: Definition containing instruction template or function.
            task_context_variables: Optional variables for dynamic resolution.

        Returns:
            Resolved instruction string ready for agent execution.
        """
        instructions = task_definition.instructions
        task_context_variables = task_context_variables or ContextVariables()
        if callable(instructions):
            return instructions(task, task_context_variables)

        return instructions

    def _parse_response(
        self,
        response: str,
        response_format: TaskResponseFormat,
        task_context_variables: ContextVariables | None = None,
    ) -> BaseModel:
        """Parse agent response using schema with error recovery.

        Converts raw agent responses into structured output models using either
        direct schema validation or custom parser functions. Uses json_repair for
        basic error recovery before validation.

        Args:
            response: Raw response string from agent.
            response_format: Schema or parser function for validation.
            task_context_variables: Optional context for parsing.

        Returns:
            Validated output model matching the response format.

        Raises:
            TypeError: If output doesn't match schema.
            ValidationError: If content is invalid.
            ValueError: If response format is invalid.
        """
        if is_callable(response_format):
            return response_format(response, task_context_variables)

        if not is_subtype(response_format, BaseModel):
            raise ValueError("Invalid response format")

        decoded_object = json_repair.repair_json(response, return_objects=True)
        if isinstance(decoded_object, tuple):
            decoded_object = decoded_object[0]

        return response_format.model_validate(decoded_object)

    async def _process_response(
        self,
        response: str,
        assignee: TeamMember,
        task: Task,
        task_execution_result: AgentExecutionResult,
        task_context_variables: ContextVariables | None = None,
    ) -> TaskResult:
        """Process agent response into task result.

        Manages the complete lifecycle of response processing by validating against
        schemas, attempting repair for invalid responses, and creating task results
        with proper metadata. The method supports both structured and unstructured
        responses based on task definition.

        Args:
            response: Raw response string from agent.
            assignee: Team member who executed the task.
            task: Task that was executed.
            task_execution_result: Raw execution result from swarm.
            task_context_variables: Optional context for processing.

        Returns:
            Complete task result with validated output and metadata.

        Raises:
            ValidationError: If response cannot be parsed and repair fails.
            ResponseRepairError: If response repair attempts fail.
        """
        task_definition = self._task_registry.get_task_definition(task.type)
        response_format = task_definition.response_format

        if not response_format:
            task_result = TaskResult(
                task=task,
                content=response,
                new_messages=task_execution_result.new_messages,
                all_messages=task_execution_result.all_messages,
                context_variables=task_context_variables,
                assignee=assignee,
            )

            return task_result

        try:
            output = self._parse_response(
                response=response,
                response_format=response_format,
                task_context_variables=task_context_variables,
            )

            task_result = TaskResult(
                task=task,
                content=response,
                output=output,
                new_messages=task_execution_result.new_messages,
                all_messages=task_execution_result.all_messages,
                context_variables=task_context_variables,
                assignee=assignee,
            )

            return task_result

        except ValidationError as validation_error:
            repaired_response = await self.response_repair_agent.repair_response(
                agent=assignee.agent,
                response=response,
                response_format=response_format,
                validation_error=validation_error,
                context_variables=task_context_variables,
            )

            task_result = TaskResult(
                task=task,
                content=response,
                output=repaired_response,
                new_messages=task_execution_result.new_messages,
                all_messages=task_execution_result.all_messages,
                context_variables=task_context_variables,
                assignee=assignee,
            )

            return task_result

    def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select best team member for task.

        Finds the most suitable team member for a task using a multi-step process:
        1. Uses pre-assigned member if specified in task
        2. Finds members capable of handling the task type
        3. Selects best match using basic selection logic

        Args:
            task: Task requiring assignment to a team member.

        Returns:
            Selected team member or None if no suitable match found.
        """
        if task.assignee and task.assignee in self.members:
            return self.members[task.assignee]

        eligible_member_ids = self._team_capabilities[task.type]
        eligible_members = [self.members[member_id] for member_id in eligible_member_ids]

        if not eligible_members:
            return None

        # TODO: Implement more sophisticated selection logic
        # Could consider:
        # - Member workload
        # - Task type specialization scores
        # - Previous task performance
        # - Agent polling/voting

        return eligible_members[0]

    # ================================================
    # MARK: Public API
    # ================================================

    @returnable
    async def create_plan(
        self,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEvent, PlanResult]:
        """Create a task execution plan from conversation messages.

        Uses the planning agent to analyze messages and generate a structured plan
        with ordered tasks that can be executed by team members. The plan is validated
        to ensure task types are registered and dependencies form a valid DAG.

        Args:
            messages: List of conversation messages for planning.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning plan result.

        Raises:
            PlanValidationError: If plan fails validation or has invalid dependencies.
            ResponseParsingError: If planning response cannot be parsed.

        Examples:
            Basic usage:
                ```python
                try:
                    context_variables = ContextVariables()
                    stream = team.create_plan(
                        messages=[Message(role="user", content="Review PR #123")],
                        context_variables=context_variables,
                    )

                    # Process events during planning
                    async for event in stream:
                        if event.type == "plan_created":
                            print(f"Created plan with {len(event.plan.tasks)} tasks")

                    # Get final result after completion
                    plan_result = await stream.get_return_value()
                    print(f"Plan ID: {plan_result.plan.id}")
                except PlanValidationError as e:
                    print(f"Invalid plan: {e}")
                ```

            With additional context:
                ```python
                try:
                    context_variables = ContextVariables(
                        pr_url="github.com/org/repo/123",
                        focus_areas=["security", "performance"],
                    )
                    stream = team.create_plan(
                        messages=[
                            Message(
                                role="user",
                                content="Review authentication changes in PR #123",
                            ),
                        ],
                        context_variables=context_variables,
                    )
                    plan_result = await stream.get_return_value()
                except (PlanValidationError, ResponseParsingError) as e:
                    print(f"Planning failed: {e}")
                ```

        Notes:
            - Plan creation can be customized with context variables
            - Plans are validated before being returned
            - Events provide visibility into planning progress
        """
        stream = self.planning_agent.create_plan(
            messages=messages,
            context_variables=context_variables,
        )

        async for event in stream:
            yield YieldItem(event)

        plan_result = await stream.get_return_value()
        yield YieldItem(PlanCreatedEvent(plan=plan_result.plan))
        yield ReturnItem(plan_result)

    @returnable
    async def execute_plan(
        self,
        plan: Plan,
        /,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEvent, Artifact]:
        """Execute a plan by running tasks in dependency order.

        Manages the complete lifecycle of plan execution by assigning tasks to
        appropriate team members, executing them in parallel when dependencies allow,
        and collecting results into an execution artifact.

        Args:
            plan: Plan with tasks to execute.
            messages: Optional conversation messages for context.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning execution artifact.

        Raises:
            SwarmTeamError: If team execution encounters internal errors.
            TaskExecutionError: If individual task execution fails.

        Examples:
            Execute plan with events:
                ```python
                context_variables = ContextVariables(pr_url="github.com/org/repo/123")
                stream = team.execute_plan(
                    plan=plan,
                    messages=[Message(role="user", content="Review PR #123")],
                    context_variables=context_variables,
                )

                # Process events during execution
                async for event in stream:
                    if event.type == "task_started":
                        print(f"Started task: {event.task.title}")
                    elif event.type == "task_completed":
                        print(f"Completed task: {event.task.title}")

                # Get final result after completion
                artifact = await stream.get_return_value()
                if artifact.status == ArtifactStatus.COMPLETED:
                    print(f"Successfully completed {len(artifact.task_results)} tasks")
                else:
                    print(f"Execution failed: {artifact.error}")
                ```

        Notes:
            - Tasks are executed in parallel when dependencies allow
            - Execution state is tracked in artifacts
            - Events provide visibility into execution progress
        """
        artifact_id = str(uuid4())
        artifact = Artifact(id=artifact_id, plan=plan, status=ArtifactStatus.EXECUTING)

        log_verbose(f"Executing plan: {plan.tasks}", level="DEBUG")
        current_messages = messages.copy() if messages else []
        current_context_variables = context_variables or ContextVariables()
        execution_messages: list[Message] = []
        task_results: list[TaskResult] = []

        try:
            while next_tasks := plan.get_next_tasks():
                log_verbose(f"Executing tasks: {next_tasks}", level="DEBUG")
                for task in next_tasks:
                    try:
                        task_definition = self._task_registry.get_task_definition(task.type)
                        task_instructions = self._prepare_instructions(
                            task=task,
                            task_definition=task_definition,
                            task_context_variables=current_context_variables,
                        )

                        task_message = Message(role="user", content=task_instructions)
                        task_stream = self.execute_task(
                            task,
                            messages=[*current_messages, task_message],
                            context_variables=current_context_variables,
                        )

                        async for event in task_stream:
                            yield YieldItem(event)

                        task_result = await task_stream.get_return_value()
                        task_results.append(task_result)

                        if task_result.new_messages:
                            new_messages = [task_message, *task_result.new_messages]
                            current_messages.extend(new_messages)
                            execution_messages.extend(new_messages)

                        if task_result.context_variables:
                            current_context_variables = ContextVariables(current_context_variables)
                            current_context_variables.update(task_result.context_variables)

                    except Exception as e:
                        artifact.status = ArtifactStatus.FAILED
                        artifact.error = e
                        artifact.task_results = task_results
                        yield ReturnItem(artifact)

            artifact.status = ArtifactStatus.COMPLETED
            artifact.task_results = task_results
            yield YieldItem(PlanCompletedEvent(plan=plan, artifact=artifact))
            yield ReturnItem(artifact)

        except Exception as error:
            artifact.status = ArtifactStatus.FAILED
            artifact.error = error
            artifact.task_results = task_results
            yield ReturnItem(artifact)

    @returnable
    async def execute_task(
        self,
        task: Task,
        /,
        messages: list[Message],
        context_variables: ContextVariables | None = None,
    ) -> AsyncStream[SwarmEvent, TaskResult]:
        """Execute a single task using an appropriate team member.

        Manages the complete lifecycle of task execution by finding a capable team
        member, preparing instructions, executing the task through swarm, and
        processing the response into a validated task result.

        Args:
            task: Task to execute, must match a registered task type.
            messages: Optional conversation messages for context.
            context_variables: Optional variables for dynamic resolution.

        Returns:
            ReturnableAsyncGenerator yielding events and returning task result.

        Raises:
            TaskExecutionError: If execution fails or no capable member found.

        Examples:
            Execute task with events:
                ```python
                context_variables = ContextVariables(pr_url="github.com/org/repo/123")
                stream = team.execute_task(
                    task=ReviewTask(
                        type="code_review",
                        id="review-1",
                        title="Security review of auth changes",
                        description="Review PR for security vulnerabilities",
                        status=TaskStatus.PENDING,
                        assignee=None,
                        dependencies=[],
                        metadata=None,
                        pr_url="github.com/org/repo/123",
                        review_type="security",
                    ),
                    messages=[Message(role="user", content="Review PR #123")],
                    context_variables=context_variables,
                )

                # Process events during execution
                async for event in stream:
                    if event.type == "task_started":
                        print(f"Started task: {event.task.title}")
                    elif event.type == "task_completed":
                        print(f"Completed task: {event.task.title}")

                # Get final result after completion
                task_result = await stream.get_return_value()
                print(f"Reviewer: {task_result.assignee.id}")
                print(f"Status: {task_result.task.status}")
                if task_result.output:
                    print(f"Findings: {task_result.output.issues}")
                ```

        Notes:
            - Task execution can be customized with context
            - Results are validated against task schemas
            - Events provide visibility into execution progress
        """
        assignee = self._select_matching_member(task)
        if not assignee:
            raise TaskExecutionError(
                f"No team member found for task type '{task.type}'",
                task=task,
            )

        try:
            task.status = TaskStatus.IN_PROGRESS
            task.assignee = assignee.agent.id
            yield YieldItem(TaskStartedEvent(task=task, messages=messages))

            task_stream = self.swarm.stream(
                agent=assignee.agent,
                messages=messages,
                context_variables=context_variables,
            )

            async for event in task_stream:
                yield YieldItem(event)

            task_execution_result = await task_stream.get_return_value()

            agent_responses = task_execution_result.agent_responses
            agent_response = agent_responses[-1] if agent_responses else None
            content = agent_response.content if agent_response else None
            if not content:
                raise ValueError("The agent did not return any content")

            task.status = TaskStatus.COMPLETED
            task_result = await self._process_response(
                response=content,
                assignee=assignee,
                task=task,
                task_context_variables=context_variables,
                task_execution_result=task_execution_result,
            )

            yield YieldItem(
                TaskCompletedEvent(
                    task=task,
                    task_result=task_result,
                    task_context_variables=context_variables,
                )
            )

            yield ReturnItem(task_result)

        except Exception as e:
            task.status = TaskStatus.FAILED
            raise TaskExecutionError(
                f"Failed to execute task: {task.title}",
                task=task,
                assignee=assignee,
                original_error=e,
            ) from e
