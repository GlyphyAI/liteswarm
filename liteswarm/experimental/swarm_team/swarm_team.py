# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planner import AgentPlanner, LiteAgentPlanner
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.stream_handler import SwarmTeamStreamHandler
from liteswarm.types.result import Result
from liteswarm.types.swarm import ContextVariables
from liteswarm.types.swarm_team import (
    ExecutionResult,
    Plan,
    PlanStatus,
    Task,
    TaskDefinition,
    TaskResponseFormat,
    TaskStatus,
    TeamMember,
)
from liteswarm.utils.typing import is_callable, is_subtype


class SwarmTeam:
    """Orchestrates a team of specialized agents for task execution.

    Manages agent teams that can execute different types of tasks, handling planning,
    execution, and result tracking.

    Example:
        ```python
        # Define task types and team members
        class ReviewTask(Task):
            type: Literal["review"]
            pr_url: str
            review_type: str

        class TestTask(Task):
            type: Literal["test"]
            pr_url: str
            test_type: str

        review_def = TaskDefinition(
            task_schema=ReviewTask,
            task_instructions="Review {task.pr_url}",
            task_response_format=ReviewOutput
        )

        reviewer = TeamMember(
            id="reviewer-1",
            agent=Agent(id="review-gpt", llm=LLM(model="gpt-4o")),
            task_types=[ReviewTask]
        )

        tester = TeamMember(
            id="tester-1",
            agent=Agent(id="tester-gpt", llm=LLM(model="gpt-4o")),
            task_types=[TestTask]
        )

        # Create and use team
        team = SwarmTeam(
            swarm=swarm,
            members=[reviewer, tester],
            task_definitions=[review_def]
        )

        plan = await team.create_plan("Review PR #123")
        results = await team.execute_plan(plan)
        ```
    """

    def __init__(
        self,
        swarm: Swarm,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        agent_planner: AgentPlanner | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
    ) -> None:
        """Initialize a new team.

        Args:
            swarm: Swarm client for agent interactions
            members: Team members with their capabilities
            task_definitions: Task types the team can handle
            agent_planner: Optional custom planning agent
            stream_handler: Optional event stream handler
        """
        # Public properties
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.stream_handler = stream_handler
        self.agent_planner = agent_planner or LiteAgentPlanner(
            swarm=self.swarm,
            task_definitions=task_definitions,
        )

        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)
        self._execution_history: list[ExecutionResult] = []
        self._team_capabilities = self._get_team_capabilities()
        self._context: ContextVariables = ContextVariables(
            team_capabilities=self._team_capabilities,
        )

    # ================================================
    # MARK: Internal Helpers
    # ================================================

    def _get_team_capabilities(self) -> dict[str, list[str]]:
        """Map task types to capable team members.

        Returns:
            Dict mapping task types to member IDs

        Example:
            ```python
            capabilities = team._get_team_capabilities()
            # {"review": ["reviewer-1"], "test": ["tester-1"]}
            ```
        """
        capabilities: dict[str, list[str]] = defaultdict(list[str])
        for member in self.members.values():
            for task_type in member.task_types:
                capabilities[task_type.get_task_type()].append(member.agent.id)

        return capabilities

    def _build_task_context(self, task: Task) -> ContextVariables:
        """Build context for task execution.

        Args:
            task: Task being executed

        Returns:
            Context with task details and history
        """
        context = ContextVariables(
            task=task.model_dump(),
            execution_history=self._execution_history,
            **self._context,
        )

        return context

    def _prepare_instructions(
        self,
        task: Task,
        task_definition: TaskDefinition,
        task_context: ContextVariables,
    ) -> str:
        """Prepare task instructions for execution.

        Handles both static templates and dynamic instruction generation.

        Args:
            task: Task being executed
            task_definition: Task type definition
            task_context: Context for instruction generation

        Returns:
            Final instructions for agent

        Example:
            ```python
            # Static template
            instructions = team._prepare_instructions(
                task=task,
                task_definition=TaskDefinition(
                    task_schema=Task,
                    task_instructions="Process {task.title}"
                ),
                task_context=context
            )

            # Dynamic generation
            def build_task_instructions(task: Task, context: ContextVariables) -> str:
                return f"Process {task.title} with {context.get('tool')}"

            instructions = team._prepare_instructions(
                task=task,
                task_definition=TaskDefinition(
                    task_schema=Task,
                    task_instructions=build_task_instructions,
                ),
                task_context=context
            )
            ```
        """
        instructions = task_definition.task_instructions
        return instructions(task, task_context) if callable(instructions) else instructions

    def _process_execution_result(
        self,
        task: Task,
        assignee: TeamMember,
        task_definition: TaskDefinition,
        content: str,
        task_context: ContextVariables,
    ) -> Result[ExecutionResult]:
        """Process agent response into execution result.

        Args:
            task: Executed task
            assignee: Team member who executed
            task_definition: Task type definition
            content: Raw agent response
            task_context: Execution context

        Returns:
            Result with execution details or error

        Example:
            ```python
            # Unstructured output
            result = team._process_execution_result(
                task=task,
                assignee=member,
                task_definition=TaskDefinition(
                    task_schema=Task,
                    task_instructions="Process data",
                ),
                content="Task completed",
                task_context=context
            )

            # Structured output
            result = team._process_execution_result(
                task=task,
                assignee=member,
                task_definition=TaskDefinition(
                    task_schema=Task,
                    task_instructions="Process data",
                    task_response_format=OutputSchema,
                ),
                content='{"status": "success"}',
                task_context=context
            )
            ```
        """
        response_format = task_definition.task_response_format

        if not response_format:
            execution_result = ExecutionResult(
                task=task,
                content=content,
                assignee=assignee,
                timestamp=datetime.now(),
            )

            return Result(value=execution_result)

        try:
            output = self._parse_response(
                content=content,
                response_format=response_format,
                task_context=task_context,
            )

            execution_result = ExecutionResult(
                task=task,
                content=content,
                output=output,
                assignee=assignee,
            )

            return Result(value=execution_result)
        except Exception as e:
            return Result(error=ValueError(f"Invalid task output: {e}"))

    def _parse_response(
        self,
        content: str,
        response_format: TaskResponseFormat,
        task_context: ContextVariables,
    ) -> BaseModel:
        """Parse agent response using schema.

        Args:
            content: Raw content to parse
            response_format: Schema or parser function
            task_context: Context for parsing

        Returns:
            Parsed output model

        Raises:
            TypeError: If output doesn't match schema
            ValidationError: If content is invalid

        Example:
            ```python
            # Using model
            output = team._parse_response(
                content='{"status": "success"}',
                response_format=OutputSchema,
                task_context=context
            )

            # Using parser
            def parse_output(content: str, context: ContextVariables) -> OutputSchema:
                return OutputSchema(status=content["result"])

            output = team._parse_response(
                content='{"result": "pass"}',
                response_format=parse_output,
                task_context=context
            )
            ```
        """
        if is_subtype(response_format, BaseModel):
            return response_format.model_validate_json(content)

        if is_callable(response_format):
            return response_format(content, task_context)

        raise ValueError("Invalid response format")

    def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select best team member for task.

        Tries to find a member by:
        1. Using assigned member if specified
        2. Finding members capable of task type
        3. Selecting best match (currently first available)

        Args:
            task: Task needing assignment

        Returns:
            Selected member or None if no match

        Example:
            ```python
            # With specific assignee
            member = team._select_matching_member(
                Task(type="review", assignee="reviewer-1")
            )

            # Based on task type
            member = team._select_matching_member(
                Task(type="review")
            )
            ```
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

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables | None = None,
    ) -> Result[Plan]:
        """Create a task execution plan from a natural language prompt.

        Uses a planning agent to analyze the prompt, break it down into tasks,
        and create a structured plan with appropriate dependencies.

        Args:
            prompt: Natural language description of work to be done
            context: Optional variables for plan customization (e.g., URLs, paths)

        Returns:
            Result with either:
            - Plan: Structured plan with ordered tasks
            - Error: If plan creation or validation fails

        Example:
            ```python
            # Basic usage
            result = await team.create_plan("Review and test PR #123")

            # With additional context
            result = await team.create_plan(
                prompt="Review authentication changes in PR #123",
                context=ContextVariables(
                    pr_url="github.com/org/repo/123",
                    focus_areas=["security", "performance"]
                )
            )
            ```
        """
        if context:
            self._context.update(context)

        result = await self.agent_planner.create_plan(
            prompt=prompt,
            context=self._context,
        )

        if not result.value:
            return result

        plan = result.value

        if self.stream_handler:
            await self.stream_handler.on_plan_created(plan)

        return Result(value=plan)

    async def execute_plan(self, plan: Plan) -> Result[list[ExecutionResult]]:
        """Execute a plan by running all its tasks in dependency order.

        Manages the complete execution lifecycle:
        1. Executes tasks when their dependencies are met
        2. Tracks execution results and updates plan status
        3. Handles failures and notifies via stream handler

        Args:
            plan: Plan with tasks to execute

        Returns:
            Result with either:
            - List[ExecutionResult]: Results from all tasks
            - Error: If any task fails or dependencies are invalid

        Example:
            ```python
            # Create and execute plan
            plan_result = await team.create_plan("Review PR #123")
            if plan_result.error:
                print(f"Planning failed: {plan_result.error}")

        Return:
            result = await team.execute_plan(plan_result.value)
            if result.value:
                for execution in result.value:
                    print(f"Task: {execution.task.title}")
                    print(f"Status: {execution.task.status}")
                    if execution.output:
                        print(f"Output: {execution.output.model_dump()}")
            ```
        """
        plan.status = PlanStatus.IN_PROGRESS
        results: list[ExecutionResult] = []

        try:
            while next_tasks := plan.get_next_tasks():
                for task in next_tasks:
                    result = await self.execute_task(task)
                    if result.error:
                        return Result(error=result.error)

                    if result.value:
                        results.append(result.value)
                    else:
                        return Result(error=ValueError(f"Failed to execute task {task.id}"))

            plan.status = PlanStatus.COMPLETED

            if self.stream_handler:
                await self.stream_handler.on_plan_completed(plan)

            return Result(value=results)

        except Exception as e:
            plan.status = PlanStatus.FAILED
            return Result(error=e)

    async def execute_task(self, task: Task) -> Result[ExecutionResult]:
        """Execute a single task using an appropriate team member.

        Handles the complete task lifecycle:
        1. Selects a capable team member
        2. Prepares execution context and instructions
        3. Executes task and processes response
        4. Updates task status and history

        Args:
            task: Task to execute, must match a registered task type

        Returns:
            Result with either:
            - ExecutionResult: Task execution details and output
            - Error: If execution fails or no capable member found

        Example:
            ```python
            # Execute a review task
            result = await team.execute_task(
                ReviewTask(
                    id="review-1",
                    title="Security review of auth changes",
                    pr_url="github.com/org/repo/123",
                    review_type="security"
                )
            )

            if result.value:
                execution = result.value
                print(f"Reviewer: {execution.assignee.id}")
                print(f"Status: {execution.task.status}")
                if execution.output:
                    print(f"Findings: {execution.output.issues}")
            ```
        """
        assignee = self._select_matching_member(task)
        if not assignee:
            return Result(error=ValueError(f"No team member found for task type '{task.type}'"))

        if self.stream_handler:
            await self.stream_handler.on_task_started(task)

        task.status = TaskStatus.IN_PROGRESS
        task.assignee = assignee.agent.id

        task_definition = self._task_registry.get_task_definition(task.type)
        if not task_definition:
            task.status = TaskStatus.FAILED
            return Result(error=ValueError(f"No TaskDefinition found for task type '{task.type}'"))

        task_context = self._build_task_context(task)
        task_instructions = self._prepare_instructions(
            task=task,
            task_definition=task_definition,
            task_context=task_context,
        )

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=task_instructions,
            context_variables=task_context,
        )

        if not result.content:
            task.status = TaskStatus.FAILED
            return Result(error=ValueError("The agent did not return any content"))

        execution_result = self._process_execution_result(
            task=task,
            assignee=assignee,
            task_definition=task_definition,
            content=result.content,
            task_context=task_context,
        )

        if execution_result.value:
            self._execution_history.append(execution_result.value)
            task.status = TaskStatus.COMPLETED
        elif execution_result.error:
            task.status = TaskStatus.FAILED

        if self.stream_handler:
            await self.stream_handler.on_task_completed(task)

        return execution_result
