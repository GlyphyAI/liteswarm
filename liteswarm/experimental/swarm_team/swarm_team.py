# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from datetime import datetime

from pydantic import BaseModel

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planner import AgentPlanner, PlanningAgent
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
    TaskOutput,
    TaskStatus,
    TeamMember,
)
from liteswarm.utils.unwrap import unwrap_task_output_type


class SwarmTeam:
    """Orchestrates a team of agents with planning and task execution capabilities.

    SwarmTeam manages a group of specialized agents that can execute different types
    of tasks. It handles:
    - Task planning and execution
    - Team member assignment
    - Context and history management
    - Structured outputs and validation

    Example:
    ```python
    # Define task types
    class CodeReviewTask(Task):
        pr_url: str
        review_type: Literal["security", "style"]

    class TestingTask(Task):
        test_path: str
        coverage_target: float

    # Create task definitions
    review_def = TaskDefinition.create(
        task_type="code_review",
        task_schema=CodeReviewTask,
        task_instructions="Review {task.pr_url}...",
        task_output=ReviewOutput
    )

    test_def = TaskDefinition.create(
        task_type="testing",
        task_schema=TestingTask,
        task_instructions="Test {task.test_path}...",
        task_output=TestResult
    )

    # Create team members
    reviewer = TeamMember(
        id="reviewer-1",
        agent=Agent.create(
            id="review-gpt",
            model="gpt-4o",
            instructions="You are a code reviewer..."
        ),
        task_types=["code_review"]
    )

    tester = TeamMember(
        id="tester-1",
        agent=Agent.create(
            id="test-gpt",
            model="gpt-4o",
            instructions="You are a testing expert..."
        ),
        task_types=["testing"]
    )

    # Initialize team
    team = SwarmTeam(
        swarm=swarm,
        members=[reviewer, tester],
        task_definitions=[review_def, test_def]
    )

    # Create and execute plan
    plan = await team.create_plan(
        "Review PR #123 and test the changes"
    )

    results = await team.execute_plan(plan)
    ```
    """

    def __init__(
        self,
        swarm: Swarm,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        planning_agent: PlanningAgent | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
    ) -> None:
        """Initialize a new SwarmTeam instance.

        Args:
            swarm: The Swarm client for agent interactions
            members: List of team members with their capabilities
            task_definitions: List of task types the team can handle
            planning_agent: Optional custom agent for plan creation
            stream_handler: Optional handler for streaming events
        """
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.task_definitions = task_definitions
        self.stream_handler = stream_handler
        self.planning_agent = planning_agent or AgentPlanner(
            swarm=self.swarm,
            task_definitions=task_definitions,
        )

        # Private state
        self._task_registry = TaskRegistry(task_definitions)
        self._execution_history: list[ExecutionResult] = []
        self._context: ContextVariables = ContextVariables(
            team_capabilities=self._get_team_capabilities(),
        )

    # ================================================
    # MARK: Task Execution Helpers
    # ================================================

    def _get_team_capabilities(self) -> dict[str, list[str]]:
        """Get a mapping of task types to team member capabilities.

        Creates a dictionary showing which team members can handle each task type.
        Used for planning and task assignment.

        Returns:
            Dict mapping task types to lists of member IDs

        Example:
            ```python
            capabilities = team._get_team_capabilities()
            # {
            #     "code_review": ["reviewer-1", "reviewer-2"],
            #     "testing": ["tester-1"]
            # }
            ```
        """
        capabilities: dict[str, list[str]] = {}
        for member in self.members.values():
            for task_type in member.task_types:
                if task_type not in capabilities:
                    capabilities[task_type] = []
                capabilities[task_type].append(member.agent.id)

        return capabilities

    def _build_task_context(
        self,
        task: Task,
        task_definition: TaskDefinition,
    ) -> ContextVariables:
        """Construct the context for task execution.

        Builds a context object containing:
        - Task details and metadata
        - Execution history
        - Team capabilities
        - Output format schema (if applicable)

        Args:
            task: The task being executed
            task_definition: Definition of the task type

        Returns:
            Context variables for task execution
        """
        context = ContextVariables(
            task=task.model_dump(),
            execution_history=self._execution_history,
            **self._context,
        )

        task_output = task_definition.task_output
        if task_output:
            task_output_type = unwrap_task_output_type(task_output)
            context.set_reserved("output_format", task_output_type.model_json_schema())

        return context

    def _prepare_instructions(
        self,
        task: Task,
        task_definition: TaskDefinition,
        task_context: ContextVariables,
    ) -> str:
        """Prepare task instructions for execution.

        Handles both static and dynamic instructions:
        - Static instructions are used as-is
        - Dynamic instructions are generated using task and context

        Args:
            task: The task being executed
            task_definition: Definition containing the instructions
            task_context: Context variables for instruction generation

        Returns:
            Final instructions string for the agent

        Example:
        ```python
        context = ContextVariables(
            tool_version="1.0.0",
            # ... other context variables
        )

        # Static instructions
        task_def = TaskDefinition.create(
            task_type="simple",
            task_schema=Task,
            task_instructions="Process {task.title}"
        )
        instructions = team._prepare_instructions(
            task=task,
            task_definition=task_def,
            task_context=context
        )

        # Dynamic instructions
        def generate_instructions(task: Task, context: ContextVariables) -> str:
            return f"Process {task.title} using {context.get('tool_version')}"

        task_def = TaskDefinition.create(
            task_type="dynamic",
            task_schema=Task,
            task_instructions=generate_instructions
        )
        instructions = team._prepare_instructions(
            task=task,
            task_definition=task_def,
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
        """Process the agent's response and create an ExecutionResult.

        Handles both unstructured and structured outputs:
        - For unstructured: Creates ExecutionResult with raw content
        - For structured: Parses content into specified output schema

        Args:
            task: The executed task
            assignee: Team member who executed the task
            task_definition: Definition of the task type
            content: Raw response from the agent
            task_context: Context used during execution

        Returns:
            Result containing either:
            - Successful ExecutionResult with parsed output
            - Error if output parsing fails

        Example:
        ```python
        # Unstructured output
        result = team._process_execution_result(
            task=task,
            assignee=member,
            task_definition=task_def,
            content='Task completed successfully',  # LLM response
            task_context=context
        )

        # Structured output with schema
        class OutputSchema(BaseModel):
            status: str
            items_processed: int

        def task_instructions(task: Task, context: ContextVariables) -> str:
            return f"Process {task.title}"

        result = team._process_execution_result(
            task=task,
            assignee=member,
            task_definition=TaskDefinition(
                task_type="simple_task",
                task_schema=Task,
                task_instructions=task_instructions,
                task_output=OutputSchema,
            ),
            content='{"status": "success", "items_processed": 42}',  # LLM response
            task_context=context
        )
        ```
        """
        output_schema = task_definition.task_output
        if not output_schema:
            execution_result = ExecutionResult(
                task=task,
                content=content,
                assignee=assignee,
                timestamp=datetime.now(),
            )
            return Result(value=execution_result)

        try:
            output = self._parse_output(
                output_schema=output_schema,
                content=content,
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

    def _parse_output(
        self,
        output_schema: TaskOutput,
        content: str,
        task_context: ContextVariables,
    ) -> BaseModel:
        """Parse the agent's output using the specified schema.

        Supports two types of output parsing:
        1. Direct schema validation using Pydantic model
        2. Custom parsing function that uses context

        Args:
            output_schema: Schema or parser function
            content: Raw content to parse
            task_context: Context variables for parsing

        Returns:
            Parsed output as a Pydantic model

        Raises:
            TypeError: If output doesn't match schema
            ValidationError: If content is invalid

        Example:
        ```python
        # Using Pydantic model
        class ReviewOutput(BaseModel):
            approved: bool
            comments: list[str]

        output = team._parse_output(
            output_schema=ReviewOutput,
            content='{"approved": true, "comments": ["Good work"]}',
            task_context=context
        )

        # Using custom parser
        def parse_review(content: str, context: ContextVariables) -> BaseModel:
            data = json.loads(content)
            return ReviewOutput(
                approved=data["result"] == "pass",
                comments=data["notes"]
            )

        output = team._parse_output(
            output_schema=parse_review,
            content='{"result": "pass", "notes": ["Good work"]}',
            task_context=context
        )
        ```
        """
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            output = output_schema.model_validate_json(content)
        else:
            output = output_schema(content, task_context)  # type: ignore

        return output

    def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select the best matching team member for a task.

        Selection process:
        1. Use assigned member if specified
        2. Find members capable of task type
        3. Select best match (currently first available)

        Future improvements could consider:
        - Member workload
        - Specialization scores
        - Past performance
        - Agent voting

        Args:
            task: Task needing assignment

        Returns:
            Selected team member or None if no match

        Example:
        ```python
        # With specific assignee
        task = Task(
            id="review-1",
            assignee="security-expert",
            task_type="security_review"
        )
        member = team._select_matching_member(task)  # Returns security-expert

        # Based on task type
        task = Task(
            id="review-2",
            task_type="code_review"
        )
        member = team._select_matching_member(task)  # Returns first available reviewer
        ```
        """
        if task.assignee and task.assignee in self.members:
            return self.members[task.assignee]

        eligible_members = [
            member for member in self.members.values() if task.task_type in member.task_types
        ]

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
        """Create a new plan from the prompt and context.

        Uses the planning agent to:
        1. Analyze the prompt and requirements
        2. Break down work into appropriate tasks
        3. Set up task dependencies
        4. Validate the resulting plan

        Args:
            prompt: Description of what needs to be done
            context: Optional additional context variables

        Returns:
            Result containing the created plan or error

        Example:
        ```python
        result = await team.create_plan(
            prompt='''
            Review PR #123 which updates authentication:
            1. Security review of auth changes
            2. Test the new auth flow
            ''',
            context=ContextVariables(
                pr_url="https://github.com/org/repo/pull/123",
                priority="high"
            )
        )

        if result.value:
            plan = result.value
            print(f"Created plan with {len(plan.tasks)} tasks")
        ```
        """
        if context:
            self._context.update(context)

        result = await self.planning_agent.create_plan(
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
        """Execute all tasks in the given plan.

        Handles the complete plan execution lifecycle:
        1. Updates plan status
        2. Executes tasks in dependency order
        3. Tracks execution results
        4. Manages failures and recovery

        Args:
            plan: The plan to execute

        Returns:
            Result containing list of execution results or error

        Example:
        ```python
        result = await team.execute_plan(plan)
        if result.value:
            for execution in result.value:
                print(f"Task {execution.task.id}:")
                print(f"- Status: {execution.task.status}")
                print(f"- Output: {execution.output}")
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
        """Execute a single task using the appropriate team member.

        Manages the complete task execution process:
        1. Selects the best team member
        2. Prepares task context
        3. Executes the task
        4. Processes and validates output

        Args:
            task: The task to execute

        Returns:
            Result containing execution details or error

        Example:
        ```python
        result = await team.execute_task(
            CodeReviewTask(
                id="review-1",
                title="Security review of auth changes",
                pr_url="https://github.com/org/repo/pull/123",
                review_type="security"
            )
        )

        if result.value:
            execution = result.value
            print(f"Review by: {execution.assignee.id}")
            print(f"Findings: {execution.output.issues}")
        ```
        """
        assignee = self._select_matching_member(task)
        if not assignee:
            return Result(
                error=ValueError(f"No team member found for task type '{task.task_type}'")
            )

        if self.stream_handler:
            await self.stream_handler.on_task_started(task)

        task.status = TaskStatus.IN_PROGRESS
        task.assignee = assignee.agent.id

        task_definition = self._task_registry.get_task_definition(task.task_type)
        if not task_definition:
            task.status = TaskStatus.FAILED
            return Result(
                error=ValueError(f"No TaskDefinition found for task type '{task.task_type}'")
            )

        task_context = self._build_task_context(task, task_definition)
        instructions = self._prepare_instructions(task, task_definition, task_context)

        result = await self.swarm.execute(
            agent=assignee.agent,
            prompt=instructions,
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