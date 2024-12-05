# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections import defaultdict
from datetime import datetime

import json_repair
from pydantic import BaseModel, ValidationError

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planner import AgentPlanner, LiteAgentPlanner
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.response_repair import (
    LiteResponseRepairAgent,
    ResponseRepairAgent,
)
from liteswarm.experimental.swarm_team.stream_handler import SwarmTeamStreamHandler
from liteswarm.types.result import Result
from liteswarm.types.swarm import ContextVariables
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
from liteswarm.utils.typing import is_callable, is_subtype


class SwarmTeam:
    """Orchestrates a team of specialized agents for task execution.

    Manages agent teams that can execute different types of tasks, handling planning,
    execution, and result tracking.

    Examples:
        Define task types and team members:
            ```python
            # Define task and output schemas
            class ReviewTask(Task):
                pr_url: str
                review_type: str

            class ReviewOutput(BaseModel):
                issues: list[str]
                approved: bool

            # Create task definition
            review_def = TaskDefinition(
                task_schema=ReviewTask,
                task_instructions="Review {task.pr_url}",
                task_response_format=ReviewOutput
            )

            # Create team member
            reviewer = TeamMember(
                id="reviewer-1",
                agent=Agent(id="review-gpt", llm=LLM(model="gpt-4o")),
                task_types=[ReviewTask]
            )

            # Create and use team
            team = SwarmTeam(
                swarm=swarm,
                members=[reviewer],
                task_definitions=[review_def]
            )

            # Execute workflow
            plan = await team.create_plan("Review PR #123")
            results = await team.execute_plan(plan)
            ```
    """

    def __init__(  # noqa: PLR0913
        self,
        swarm: Swarm,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        agent_planner: AgentPlanner | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
    ) -> None:
        """Initialize a new team.

        Args:
            swarm: Swarm client for agent interactions.
            members: Team members with their capabilities.
            task_definitions: Task types the team can handle.
            agent_planner: Optional custom planning agent.
            response_repair_agent: Optional custom response repair agent.
            stream_handler: Optional event stream handler.
        """
        # Public properties
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.stream_handler = stream_handler
        self.agent_planner = agent_planner or LiteAgentPlanner(
            swarm=self.swarm,
            task_definitions=task_definitions,
        )
        self.response_repair_agent = response_repair_agent or LiteResponseRepairAgent(
            swarm=self.swarm,
        )

        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)
        self._artifacts: list[Artifact] = []
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
            Dict mapping task types to member IDs.

        Examples:
            Get team capabilities:
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
            task: Task being executed.

        Returns:
            Context with task details and history.

        Examples:
            Basic context:
                ```python
                task = Task(
                    id="review-1",
                    type="review",
                    title="Review PR",
                    pr_url="github.com/org/repo/123"
                )
                context = team._build_task_context(task)
                # Returns ContextVariables with:
                # - task details as dict
                # - execution history
                # - team capabilities
                ```

            Access context values:
                ```python
                context = team._build_task_context(task)
                task_data = context.get("task")  # Get task details
                artifacts = context.get("artifacts")  # Get previous results
                capabilities = context.get("team_capabilities")  # Get team info
                ```
        """
        context = ContextVariables(
            task=task.model_dump(),
            artifacts=self._artifacts,
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
            task: Task being executed.
            task_definition: Task type definition.
            task_context: Context for instruction generation.

        Returns:
            Final instructions for agent.

        Examples:
            Static template:
                ```python
                instructions = team._prepare_instructions(
                    task=task,
                    task_definition=TaskDefinition(
                        task_schema=Task,
                        task_instructions="Process {task.title}"
                    ),
                    task_context=context
                )
                ```

            Dynamic generation:
                ```python
                def generate_instructions(task: Task, task_context: ContextVariables) -> str:
                    return f"Process {task.title} with {task_context.get('tool')}"

                instructions = team._prepare_instructions(
                    task=task,
                    task_definition=TaskDefinition(
                        task_schema=Task,
                        task_instructions=generate_instructions,
                    ),
                    task_context=context
                )
                ```
        """
        instructions = task_definition.task_instructions
        return instructions(task, task_context) if callable(instructions) else instructions

    def _parse_response(
        self,
        response: str,
        response_format: TaskResponseFormat,
        task_context: ContextVariables,
    ) -> BaseModel:
        """Parse agent response using schema with error recovery.

        Args:
            response: Raw agent response to parse.
            response_format: Schema or parser function.
            task_context: Context for parsing.

        Returns:
            Parsed output model.

        Raises:
            TypeError: If output doesn't match schema.
            ValidationError: If content is invalid and cannot be repaired.

        Examples:
            Parse with model schema:
                ```python
                class ReviewOutput(BaseModel):
                    issues: list[str]
                    approved: bool


                response = '''
                {
                    "issues": ["Security risk in auth", "Missing tests"],
                    "approved": false
                }
                '''
                output = team._parse_response(
                    response=response,
                    response_format=ReviewOutput,
                    task_context=context,
                )
                # Returns ReviewOutput instance
                ```

            Parse with custom function:
                ```python
                def parse_review(content: str, context: ContextVariables) -> ReviewOutput:
                    # Custom parsing logic
                    data = json.loads(content)
                    return ReviewOutput(**data)


                output = team._parse_response(
                    response=response,
                    response_format=parse_review,
                    task_context=context,
                )
                # Returns ReviewOutput instance via custom parser
                ```

            With json_repair:
                ```python
                # Even with slightly invalid JSON
                response = '''
                {
                    'issues': ['Missing tests'],  # Single quotes
                    approved: false  # Missing quotes
                }
                '''
                output = team._parse_response(
                    response=response,
                    response_format=ReviewOutput,
                    task_context=context,
                )
                # Still returns valid ReviewOutput
                ```
        """
        if is_callable(response_format):
            return response_format(response, task_context)

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
        task_definition: TaskDefinition,
        task_context: ContextVariables,
    ) -> Result[TaskResult]:
        """Process agent response into task result.

        Attempts to parse and validate the response according to the task's
        expected format. If validation fails, tries to recover using the
        response repair agent.

        Args:
            response: Raw agent response after task execution.
            assignee: Team member who executed the task.
            task: Executed task.
            task_definition: Task type definition.
            task_context: Execution context.

        Returns:
            Result containing either:
                - Task result with validated output
                - Error if parsing fails and cannot be recovered

        Examples:
            Successful execution:
                ```python
                class ReviewOutput(BaseModel):
                    issues: list[str]
                    approved: bool


                task = Task(id="review-1", type="review", title="Review PR")
                assignee = TeamMember(
                    id="reviewer-1",
                    agent=Agent(id="review-gpt"),
                    task_types=[ReviewTask],
                )
                task_def = TaskDefinition(
                    task_schema=ReviewTask,
                    task_response_format=ReviewOutput,
                )

                response = '{"issues": [], "approved": true}'
                result = await team._process_response(
                    response=response,
                    assignee=assignee,
                    task=task,
                    task_definition=task_def,
                    task_context=context,
                )
                # Returns Result with TaskResult containing:
                # - task details
                # - assignee info
                # - parsed ReviewOutput
                ```

            With response repair:
                ```python
                # Invalid JSON that needs repair
                response = '''
                {
                    issues: ["Missing tests"]  # Missing quotes
                    'approved': false,  # Extra comma
                }
                '''
                result = await team._process_response(
                    response=response,
                    assignee=assignee,
                    task=task,
                    task_definition=task_def,
                    task_context=context,
                )
                # Returns repaired and validated TaskResult
                ```

            Without response format:
                ```python
                task_def = TaskDefinition(
                    task_schema=Task,
                    task_response_format=None,  # No format specified
                )
                response = "Task completed successfully"
                result = await team._process_response(
                    response=response,
                    assignee=assignee,
                    task=task,
                    task_definition=task_def,
                    task_context=context,
                )
                # Returns TaskResult with raw content
                ```
        """
        response_format = task_definition.task_response_format

        if not response_format:
            task_result = TaskResult(
                task=task,
                content=response,
                assignee=assignee,
                timestamp=datetime.now(),
            )

            return Result(value=task_result)

        try:
            output = self._parse_response(
                response=response,
                response_format=response_format,
                task_context=task_context,
            )

            task_result = TaskResult(
                task=task,
                content=response,
                output=output,
                assignee=assignee,
            )

            return Result(value=task_result)

        except ValidationError as validation_error:
            repair_result = await self.response_repair_agent.repair_response(
                agent=assignee.agent,
                response=response,
                response_format=task_definition.task_response_format,
                validation_error=validation_error,
                context=task_context,
            )

            if repair_result.error:
                return Result(error=repair_result.error)

            if not repair_result.value:
                return Result(error=ValueError("No content in repair response"))

            task_result = TaskResult(
                task=task,
                content=response,
                output=repair_result.value,
                assignee=assignee,
            )

            return Result(value=task_result)

        except Exception as e:
            return Result(error=ValueError(f"Invalid task output: {e}"))

    def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select best team member for task.

        Tries to find a member by:
            1. Using assigned member if specified.
            2. Finding members capable of task type.
            3. Selecting best match (currently first available).

        Args:
            task: Task needing assignment.

        Returns:
            Selected member or None if no match.

        Examples:
            With specific assignee:
                ```python
                member = team._select_matching_member(
                    Task(type="review", assignee="reviewer-1")
                )
                ```

            Based on task type:
                ```python
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
            prompt: Natural language description of work to be done.
            context: Optional variables for plan customization (e.g., URLs, paths).

        Returns:
            Result containing either:
                - A structured plan with ordered tasks.
                - Error if plan creation or validation fails.

        Examples:
            Basic usage:
                ```python
                result = await team.create_plan("Review and test PR #123")
                ```

            With additional context:
                ```python
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

        if self.stream_handler and result.value:
            await self.stream_handler.on_plan_created(result.value)

        return result

    async def execute_plan(self, plan: Plan) -> Result[Artifact]:
        """Execute a plan by running all its tasks in dependency order.

        Manages the complete execution lifecycle:
            1. Executes tasks when their dependencies are met.
            2. Tracks execution results and updates plan status.
            3. Handles failures and notifies via stream handler.
            4. Creates and returns an execution artifact.

        Args:
            plan: Plan with tasks to execute.

        Returns:
            Result containing either:
                - Execution artifact with plan results and task outputs.
                - Error if any task fails or dependencies are invalid.

        Examples:
            Create and execute a plan:
                ```python
                plan_result = await team.create_plan("Review PR #123")
                if plan_result.error:
                    print(f"Planning failed: {plan_result.error}")
                    raise plan_result.error

                result = await team.execute_plan(plan_result.value)
                if result.value:
                    artifact = result.value
                    print(f"Execution {artifact.id}:")
                    print(f"Status: {artifact.status}")
                    if artifact.error:
                        print(f"Failed: {artifact.error}")
                    else:
                        for task_result in artifact.task_results:
                            print(f"Task: {task_result.task.title}")
                ```
        """
        artifact_id = f"artifact_{len(self._artifacts) + 1}"
        artifact = Artifact(id=artifact_id, plan=plan, status=ArtifactStatus.EXECUTING)
        self._artifacts.append(artifact)

        plan.status = PlanStatus.IN_PROGRESS

        try:
            while next_tasks := plan.get_next_tasks():
                for task in next_tasks:
                    result = await self.execute_task(task)
                    if result.error:
                        artifact.status = ArtifactStatus.FAILED
                        artifact.error = result.error
                        return Result(error=result.error)

                    if result.value:
                        artifact.task_results.append(result.value)
                    else:
                        artifact.status = ArtifactStatus.FAILED
                        error = ValueError(f"Failed to execute task {task.id}")
                        artifact.error = error
                        return Result(error=error)

            plan.status = PlanStatus.COMPLETED
            artifact.status = ArtifactStatus.COMPLETED

            if self.stream_handler:
                await self.stream_handler.on_plan_completed(plan)

            return Result(value=artifact)

        except Exception as error:
            plan.status = PlanStatus.FAILED
            artifact.status = ArtifactStatus.FAILED
            artifact.error = error
            return Result(error=error)

    async def execute_task(self, task: Task) -> Result[TaskResult]:
        """Execute a single task using an appropriate team member.

        Handles the complete task lifecycle:
            1. Selects a capable team member.
            2. Prepares execution context and instructions.
            3. Executes task and processes response.
            4. Updates task status and history.

        Args:
            task: Task to execute, must match a registered task type.

        Returns:
            Result containing either:
                - Task execution result with outputs.
                - Error if execution fails or no capable member found.

        Examples:
            Execute a review task:
                ```python
                result = await team.execute_task(
                    ReviewTask(
                        id="review-1",
                        title="Security review of auth changes",
                        pr_url="github.com/org/repo/123",
                        review_type="security"
                    )
                )

                if result.value:
                    task_result = result.value
                    print(f"Reviewer: {task_result.assignee.id}")
                    print(f"Status: {task_result.task.status}")
                    if task_result.output:
                        print(f"Findings: {task_result.output.issues}")
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

        task_result = await self._process_agent_response(
            assignee=assignee,
            response=result.content,
            task=task,
            task_definition=task_definition,
            task_context=task_context,
        )

        if task_result.success():
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED

        if self.stream_handler:
            await self.stream_handler.on_task_completed(task)

        return task_result

    def get_artifacts(self) -> list[Artifact]:
        """Get all execution artifacts.

        Returns:
            List of all execution artifacts in chronological order.

        Examples:
            Analyze execution history:
                ```python
                artifacts = team.get_artifacts()
                for artifact in artifacts:
                    print(f"Execution {artifact.id}:")
                    print(f"Status: {artifact.status}")
                    if artifact.error:
                        print(f"Failed: {artifact.error}")
                    else:
                        print(f"Completed {len(artifact.task_results)} tasks")
                ```
        """
        return self._artifacts

    def get_latest_artifact(self) -> Artifact | None:
        """Get the most recent execution artifact.

        Returns:
            The most recent artifact or None if no artifacts exist.

        Examples:
            Check latest execution:
                ```python
                if artifact := team.get_latest_artifact():
                    print(f"Latest execution {artifact.id}:")
                    print(f"Status: {artifact.status}")
                    if artifact.error:
                        print(f"Failed: {artifact.error}")
                    else:
                        print(f"Completed {len(artifact.task_results)} tasks")
                else:
                    print("No executions yet")
                ```
        """
        return self._artifacts[-1] if self._artifacts else None
