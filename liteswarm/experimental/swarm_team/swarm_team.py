# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections import defaultdict

import json_repair
from pydantic import BaseModel, ValidationError

from liteswarm.core.event_handler import LiteSwarmEventHandler
from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planning import LitePlanningAgent, PlanningAgent
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.response_repair import (
    LiteResponseRepairAgent,
    ResponseRepairAgent,
)
from liteswarm.types.events import (
    SwarmTeamPlanCompletedEvent,
    SwarmTeamPlanCreatedEvent,
    SwarmTeamTaskCompletedEvent,
    SwarmTeamTaskStartedEvent,
)
from liteswarm.types.exceptions import TaskExecutionError
from liteswarm.types.swarm import ContextVariables
from liteswarm.types.swarm_team import (
    Artifact,
    ArtifactStatus,
    Plan,
    PlanFeedbackHandler,
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
    """Experimental framework for orchestrating complex agent workflows.

    SwarmTeam provides a high-level interface for executing tasks using a team of
    specialized AI agents. The framework uses a two-phase approach to task execution:

    1. Planning Phase: Analyzes prompts to create structured plans
        - Uses planning agent to break down work into tasks
        - Validates task types and dependencies
        - Supports interactive feedback loop
        - Uses Pydantic models for schema definitions

    2. Execution Phase: Executes tasks with specialized agents
        - Assigns tasks to capable team members
        - Handles structured inputs/outputs via framework-level parsing
        - Tracks execution state
        - Produces artifacts with results

    The framework supports structured outputs through a two-layer parsing system
    using Pydantic models that transform to JSON schemas. Users can choose to:
        - Use Pydantic models for direct LLM structured outputs (if supported)
        - Use custom formats with framework-level parsing
        - Combine both approaches for robust validation

    Examples:
        Create a team for code review (complete setup):
            ```python
            # 1. Define task type using Pydantic model
            class ReviewTask(Task):
                type: Literal["code_review"]  # Discriminator for task type
                pr_url: str
                review_type: str


            # 2. Create task definition with structured output
            class ReviewOutput(BaseModel):
                issues: list[str]
                approved: bool


            review_def = TaskDefinition(
                task_type=ReviewTask,
                instructions="Review {task.pr_url} focusing on {task.review_type}",
                response_format=ReviewOutput,  # Framework-level parsing
            )

            # 3. Create specialized agent
            review_agent = Agent(
                id="reviewer",
                instructions="You are a code reviewer.",
                llm=LLM(
                    model="gpt-4o",
                    response_format=ReviewOutput,  # Optional LLM-level format
                ),
            )

            # 4. Create team member
            members = [
                TeamMember(
                    id="senior-reviewer",
                    agent=review_agent,
                    task_types=[ReviewTask],
                ),
            ]

            # 5. Create team
            swarm = Swarm()
            team = SwarmTeam(
                swarm=swarm,
                members=members,
                task_definitions=[review_def],
            )

            # 6. Execute workflow
            artifact = await team.execute(
                prompt="Review PR #123 for security issues",
                context=ContextVariables(
                    pr_url="github.com/org/repo/123",
                    review_type="security",
                ),
            )

            # 7. Process structured results
            if artifact.status == ArtifactStatus.COMPLETED:
                for result in artifact.task_results:
                    output = result.output  # ReviewOutput instance
                    assert isinstance(output, ReviewOutput)
                    print(f"Review by: {result.assignee.id}")
                    print(f"Issues: {output.issues}")
                    print(f"Approved: {output.approved}")
            ```

        Interactive plan feedback:
            ```python
            class InteractiveFeedback(PlanFeedbackHandler):
                async def handle(
                    self,
                    plan: Plan,
                    prompt: str,
                    context: ContextVariables | None,
                ) -> tuple[str, ContextVariables | None] | None:
                    print("Proposed plan:")
                    for task in plan.tasks:
                        print(f"- {task.title}")

                    if input("Approve? [y/N]: ").lower() == "y":
                        return None

                    feedback = input("Enter feedback: ")
                    return f"Previous plan needs adjustments: {feedback}", context


            # Using async interface
            artifact = await team.execute(
                prompt="Create a Flutter TODO app",
                feedback_handler=InteractiveFeedback(),
            )

            # Or using sync interface
            artifact = team.execute_sync(
                prompt="Create a Flutter TODO app",
                feedback_handler=InteractiveFeedback(),
            )
            ```

    Notes:
        - Automated task planning and execution
        - Parallel task execution when possible
        - Task dependency management
        - Event-based progress tracking
        - Plan feedback and refinement
        - Structured output handling using Pydantic models
    """

    def __init__(
        self,
        swarm: Swarm,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        planning_agent: PlanningAgent | None = None,
        response_repair_agent: ResponseRepairAgent | None = None,
        event_handler: LiteSwarmEventHandler | None = None,
    ) -> None:
        """Initialize a new team.

        Args:
            swarm: Swarm client for agent interactions.
            members: Team members with their capabilities.
            task_definitions: Task types the team can handle.
            planning_agent: Optional custom planning agent.
            response_repair_agent: Optional custom response repair agent.
            event_handler: Optional event handler for team events.
        """
        # Internal state (private)
        self._task_registry = TaskRegistry(task_definitions)
        self._artifacts: list[Artifact] = []
        self._team_capabilities = self._get_team_capabilities(members)

        # Public properties
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.planning_agent = planning_agent or self._default_planning_agent(task_definitions)
        self.response_repair_agent = response_repair_agent or self._default_response_repair_agent()
        self.event_handler = event_handler or LiteSwarmEventHandler()

    # ================================================
    # MARK: Internal Helpers
    # ================================================

    def _default_planning_agent(self, task_definitions: list[TaskDefinition]) -> LitePlanningAgent:
        """Create a default planning agent.

        Args:
            task_definitions: Task definitions to use for planning.

        Returns:
            Default planning agent.
        """
        return LitePlanningAgent(swarm=self.swarm, task_definitions=task_definitions)

    def _default_response_repair_agent(self) -> LiteResponseRepairAgent:
        """Create a default response repair agent.

        Returns:
            Default response repair agent.
        """
        return LiteResponseRepairAgent(swarm=self.swarm)

    def _get_team_capabilities(self, members: list[TeamMember]) -> dict[str, list[str]]:
        """Map task types to capable team members.

        Args:
            members: Team members to consider when determining capabilities.

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
        for member in members:
            for task_type in member.task_types:
                capabilities[task_type.get_task_type()].append(member.agent.id)

        return capabilities

    def _build_task_context(
        self,
        task: Task,
        context: ContextVariables | None = None,
    ) -> ContextVariables:
        """Build context for task execution.

        Args:
            task: Task being executed.
            context: Optional context to extend.

        Returns:
            Context with task details and history.

        Examples:
            Basic context:
                ```python
                task = Task(
                    # Base Task required fields
                    type="review",
                    id="review-1",
                    title="Review PR",
                    description="Review the PR for security issues",
                    status=TaskStatus.PENDING,
                    assignee=None,
                    dependencies=[],
                    metadata=None,
                    # Additional fields from task definition
                    pr_url="github.com/org/repo/123",
                )
                context = ContextVariables(pr_url="github.com/org/repo/123")
                task_context = team._build_task_context(task, context)
                # Returns ContextVariables with:
                # - task details as dict
                # - execution history
                # - team capabilities
                # - provided context
                ```

            Access context values:
                ```python
                context = ContextVariables(tool="myTool")
                task_context = team._build_task_context(task, context)
                task_data = task_context.get("task")  # Get task details
                artifacts = task_context.get("artifacts")  # Get previous results
                capabilities = task_context.get("team_capabilities")  # Get team info
                tool = task_context.get("tool")  # Get provided context
                ```
        """
        task_context = ContextVariables(
            task=task,
            artifacts=self._artifacts,
            team_capabilities=self._team_capabilities,
        )

        if context:
            task_context.update(context)

        return task_context

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
                        task_type=Task,
                        instructions="Process {task.title}",
                    ),
                    task_context=context,
                )
                ```

            Dynamic generation:
                ```python
                def generate_instructions(task: Task, task_context: ContextVariables) -> str:
                    return f"Process {task.title} with {task_context.get('tool')}"


                instructions = team._prepare_instructions(
                    task=task,
                    task_definition=TaskDefinition(
                        task_type=Task,
                        instructions=generate_instructions,
                    ),
                    task_context=context,
                )
                ```
        """
        instructions = task_definition.instructions
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
            ValidationError: If content is invalid.
            ValueError: If response format is invalid.

        Examples:
            Parse with model schema:
                ```python
                class ReviewOutput(BaseModel):
                    issues: list[str]
                    approved: bool


                try:
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
                    print(output.model_dump())
                except ValidationError as e:
                    print(f"Invalid response format: {e}")
                ```

            Parse with custom function:
                ```python
                def parse_review(content: str, context: ContextVariables) -> ReviewOutput:
                    try:
                        # Custom parsing logic
                        data = json.loads(content)
                        return ReviewOutput(
                            issues=data["issues"],
                            approved=data["approved"],
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ValidationError(f"Failed to parse review: {e}")


                try:
                    output = team._parse_response(
                        response='{"issues": [], "approved": true}',
                        response_format=parse_review,
                        task_context=context,
                    )
                    print(output.model_dump())
                except ValidationError as e:
                    print(f"Failed to parse with custom function: {e}")
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
                try:
                    output = team._parse_response(
                        response=response,
                        response_format=ReviewOutput,
                        task_context=context,
                    )
                    print(output.model_dump())
                except ValidationError as e:
                    print(f"Failed to repair JSON: {e}")
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
    ) -> TaskResult:
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
            Task result with validated output.

        Raises:
            ValidationError: If response cannot be parsed and repair fails.
            ResponseRepairError: If response repair attempts fail.

        Examples:
            Successful execution:
                ```python
                class ReviewOutput(BaseModel):
                    issues: list[str]
                    approved: bool


                task = Task(
                    # Base Task required fields
                    type="code_review",
                    id="review-1",
                    title="Review PR",
                    description="Review code changes",
                    status=TaskStatus.PENDING,
                    assignee=None,
                    dependencies=[],
                    metadata=None,
                )
                assignee = TeamMember(
                    id="reviewer-1",
                    agent=Agent(id="review-gpt"),
                    task_types=[ReviewTask],
                )
                task_def = TaskDefinition(
                    task_type=ReviewTask,
                    instructions="Review {task.pr_url} focusing on {task.review_type}",
                    response_format=ReviewOutput,
                )

                try:
                    response = '{"issues": [], "approved": true}'
                    task_result = await team._process_response(
                        response=response,
                        assignee=assignee,
                        task=task,
                        task_definition=task_def,
                        task_context=context,
                    )
                    print(f"Task completed: {task_result.output}")
                except (ValidationError, ResponseRepairError) as e:
                    print(f"Failed to process response: {e}")
                ```

            Without response format:
                ```python
                task_def = TaskDefinition(
                    task_type=Task,
                    instructions="Process {task.title}",
                    response_format=None,  # No format specified
                )
                response = "Task completed successfully"
                task_result = await team._process_response(
                    response=response,
                    assignee=assignee,
                    task=task,
                    task_definition=task_def,
                    task_context=context,
                )
                print(task_result.content)  # Raw response content
                ```
        """
        response_format = task_definition.response_format

        if not response_format:
            task_result = TaskResult(
                task=task,
                content=response,
                context=task_context,
                assignee=assignee,
            )

            return task_result

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
                context=task_context,
                assignee=assignee,
            )

            return task_result

        except ValidationError as validation_error:
            repaired_response = await self.response_repair_agent.repair_response(
                agent=assignee.agent,
                response=response,
                response_format=response_format,
                validation_error=validation_error,
                context=task_context,
            )

            task_result = TaskResult(
                task=task,
                content=response,
                output=repaired_response,
                context=task_context,
                assignee=assignee,
            )

            return task_result

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
                member = team._select_matching_member(Task(type="review", assignee="reviewer-1"))
                ```

            Based on task type:
                ```python
                member = team._select_matching_member(Task(type="review"))
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

    async def execute(
        self,
        prompt: str,
        context: ContextVariables | None = None,
        *,
        feedback_handler: PlanFeedbackHandler | None = None,
    ) -> Artifact:
        """Execute a user request by creating and running a plan.

        Creates a plan from the prompt, optionally refines it through feedback,
        and executes all tasks in order.

        Args:
            prompt: Natural language description of work to be done.
            context: Optional context for plan customization and task execution.
            feedback_handler: Optional handler for reviewing and refining plans.

        Returns:
            Execution artifact containing plan, results, and any errors.

        Raises:
            SwarmTeamError: If team execution fails.
            PlanValidationError: If plan validation fails.
            TaskExecutionError: If task execution fails.

        Examples:
            Basic usage:
                ```python
                context = ContextVariables(pr_url="github.com/org/repo/123")
                artifact = await team.execute(
                    prompt="Review and test PR #123",
                    context=context,
                )
                if artifact.status == ArtifactStatus.FAILED:
                    print(f"Execution failed: {artifact.error}")
                else:
                    print(f"Completed {len(artifact.task_results)} tasks")
                ```

            With interactive feedback:
                ```python
                class InteractiveFeedback(PlanFeedbackHandler):
                    async def handle(
                        self,
                        plan: Plan,
                        prompt: str,
                        context: ContextVariables | None,
                    ) -> tuple[str, ContextVariables | None] | None:
                        print("Proposed plan:")
                        for task in plan.tasks:
                            print(f"- {task.title}")

                        if input("Approve? [y/N]: ").lower() == "y":
                            return None

                        feedback = input("Enter feedback: ")
                        return f"Previous plan needs adjustments: {feedback}", context


                artifact = await team.execute(
                    prompt="Create a Flutter TODO app",
                    feedback_handler=InteractiveFeedback(),
                )
                ```

        Notes:
            - Plan creation and execution can be customized with context
            - Feedback handlers can modify plans before execution
            - Execution artifacts are stored for history tracking
        """
        current_prompt = prompt
        current_context = context

        while True:
            try:
                plan = await self.create_plan(current_prompt, current_context)
            except Exception as e:
                artifact = Artifact(
                    id=f"artifact_{len(self._artifacts) + 1}",
                    status=ArtifactStatus.FAILED,
                    error=e,
                )
                self._artifacts.append(artifact)
                return artifact

            if feedback_handler:
                result = await feedback_handler.handle(plan, current_prompt, current_context)
                if result:
                    current_prompt, current_context = result
                    continue

            return await self.execute_plan(plan, current_context)

    async def create_plan(
        self,
        prompt: str,
        context: ContextVariables | None = None,
    ) -> Plan:
        """Create a task execution plan from a natural language prompt.

        Analyzes the prompt and generates a structured plan with ordered tasks
        that can be executed by team members.

        Args:
            prompt: Natural language description of work to be done.
            context: Optional context for plan customization.

        Returns:
            Structured plan with ordered tasks.

        Raises:
            PlanValidationError: If plan creation or validation fails.
            ResponseParsingError: If the planning response cannot be parsed.

        Examples:
            Basic usage:
                ```python
                try:
                    context = ContextVariables()
                    plan = await team.create_plan(
                        prompt="Review and test PR #123",
                        context=context,
                    )
                    print(f"Created plan with {len(plan.tasks)} tasks")
                except PlanValidationError as e:
                    print(f"Invalid plan: {e}")
                ```

            With additional context:
                ```python
                try:
                    context = ContextVariables(
                        pr_url="github.com/org/repo/123",
                        focus_areas=["security", "performance"],
                    )
                    plan = await team.create_plan(
                        prompt="Review authentication changes in PR #123",
                        context=context,
                    )
                except (PlanValidationError, ResponseParsingError) as e:
                    print(f"Planning failed: {e}")
                ```

        Notes:
            - Plan creation can be customized with context variables
            - Plans are validated before being returned
            - Events are emitted for plan creation tracking
        """
        result = await self.planning_agent.create_plan(
            prompt=prompt,
            context=context,
        )

        await self.event_handler.on_event(SwarmTeamPlanCreatedEvent(plan=result))
        return result

    async def execute_plan(
        self,
        plan: Plan,
        context: ContextVariables | None = None,
    ) -> Artifact:
        """Execute a plan by running all its tasks in dependency order.

        Assigns tasks to appropriate team members and executes them in parallel
        when possible, respecting dependencies.

        Args:
            plan: Plan with tasks to execute.
            context: Optional context for task execution.

        Returns:
            Execution artifact containing results and task outputs.

        Raises:
            SwarmTeamError: If team execution fails.
            TaskExecutionError: If task execution fails.

        Examples:
            Execute a plan:
                ```python
                context = ContextVariables(pr_url="github.com/org/repo/123")
                plan = await team.create_plan("Review PR #123", context)
                artifact = await team.execute_plan(plan, context)

                if artifact.status == ArtifactStatus.FAILED:
                    print(f"Execution failed: {artifact.error}")
                    print(f"Completed {len(artifact.task_results)} tasks before failure")
                else:
                    print(f"Successfully completed all {len(artifact.task_results)} tasks")

                for task_result in artifact.task_results:
                    print(f"Task: {task_result.task.title}")
                    print(f"Status: {task_result.task.status}")
                ```

        Notes:
            - Tasks are executed in parallel when dependencies allow
            - Execution artifacts are stored for history tracking
            - Events are emitted for execution progress tracking
        """
        artifact_id = f"artifact_{len(self._artifacts) + 1}"
        artifact = Artifact(id=artifact_id, plan=plan, status=ArtifactStatus.EXECUTING)
        self._artifacts.append(artifact)

        try:
            log_verbose(f"Executing plan: {plan.tasks}", level="DEBUG")
            while next_tasks := plan.get_next_tasks():
                log_verbose(f"Executing tasks: {next_tasks}", level="DEBUG")
                for task in next_tasks:
                    try:
                        task_result = await self.execute_task(task, context)
                    except Exception as e:
                        artifact.status = ArtifactStatus.FAILED
                        artifact.error = e
                        return artifact

                    artifact.task_results.append(task_result)

            artifact.status = ArtifactStatus.COMPLETED
            await self.event_handler.on_event(
                SwarmTeamPlanCompletedEvent(
                    plan=plan,
                    artifact=artifact,
                )
            )

            return artifact

        except Exception as error:
            artifact.status = ArtifactStatus.FAILED
            artifact.error = error
            return artifact

    async def execute_task(
        self,
        task: Task,
        context: ContextVariables | None = None,
    ) -> TaskResult:
        """Execute a single task using an appropriate team member.

        Finds a capable team member and assigns them to execute the task,
        monitoring progress and collecting results.

        Args:
            task: Task to execute, must match a registered task type.
            context: Optional context for task execution.

        Returns:
            Task execution result with outputs.

        Raises:
            TaskExecutionError: If execution fails or no capable member is found.

        Examples:
            Execute a task:
                ```python
                try:
                    context = ContextVariables(pr_url="github.com/org/repo/123")
                    task_result = await team.execute_task(
                        ReviewTask(
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
                        context=context,
                    )
                    print(f"Reviewer: {task_result.assignee.id}")
                    print(f"Status: {task_result.task.status}")
                    if task_result.output:
                        print(f"Findings: {task_result.output.issues}")
                except TaskExecutionError as e:
                    print(f"Task execution failed: {e}")
                ```

        Notes:
            - Task execution can be customized with context
            - Task results are validated against schemas
            - Events are emitted for task progress tracking
        """
        assignee = self._select_matching_member(task)
        if not assignee:
            raise TaskExecutionError(
                f"No team member found for task type '{task.type}'",
                task=task,
            )

        try:
            await self.event_handler.on_event(SwarmTeamTaskStartedEvent(task=task))
            task.status = TaskStatus.IN_PROGRESS
            task.assignee = assignee.agent.id

            task_definition = self._task_registry.get_task_definition(task.type)
            task_context = self._build_task_context(task, context)
            task_instructions = self._prepare_instructions(
                task=task,
                task_definition=task_definition,
                task_context=task_context,
            )

            result = await self.swarm.execute(
                agent=assignee.agent,
                prompt=task_instructions,
                context_variables=task_context,
                event_handler=self.event_handler,
            )

            if not result.content:
                raise ValueError("The agent did not return any content")

            task.status = TaskStatus.COMPLETED
            task_result = await self._process_response(
                response=result.content,
                assignee=assignee,
                task=task,
                task_definition=task_definition,
                task_context=task_context,
            )

            await self.event_handler.on_event(
                SwarmTeamTaskCompletedEvent(
                    task=task,
                    task_result=task_result,
                    task_context=task_context,
                )
            )

            return task_result

        except Exception as e:
            task.status = TaskStatus.FAILED
            raise TaskExecutionError(
                f"Failed to execute task: {task.title}",
                task=task,
                assignee=assignee,
                original_error=e,
            ) from e

    def get_artifacts(self) -> list[Artifact]:
        """Get all execution artifacts.

        Retrieves all execution artifacts in chronological order, providing
        a complete history of team executions.

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

        Notes:
            - Artifacts are returned in chronological order
            - Each artifact contains a complete execution record
            - Failed executions include error information
        """
        return self._artifacts

    def get_latest_artifact(self) -> Artifact | None:
        """Get the most recent execution artifact.

        Retrieves the most recent execution artifact, providing quick access
        to the latest team execution results.

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

        Notes:
            - Returns None if no executions have occurred
            - Provides quick access to latest results
            - Includes both successful and failed executions
        """
        return self._artifacts[-1] if self._artifacts else None
