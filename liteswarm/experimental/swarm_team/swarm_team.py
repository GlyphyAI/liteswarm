# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from datetime import datetime

from pydantic import BaseModel

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.planner import AgentPlanner, PlanningAgent, PromptTemplate
from liteswarm.experimental.swarm_team.registry import TaskRegistry
from liteswarm.experimental.swarm_team.stream_handler import SwarmTeamStreamHandler
from liteswarm.types.result import Result
from liteswarm.types.swarm import Agent, ContextVariables
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
from liteswarm.utils.misc import dedent_prompt
from liteswarm.utils.unwrap import unwrap_task_output_type


class SwarmTeam:
    """Orchestrates a team of agents working on tasks according to a plan."""

    def __init__(
        self,
        swarm: Swarm,
        members: list[TeamMember],
        task_definitions: list[TaskDefinition],
        planning_agent: PlanningAgent | None = None,
        stream_handler: SwarmTeamStreamHandler | None = None,
    ) -> None:
        self.swarm = swarm
        self.members = {member.agent.id: member for member in members}
        self.task_definitions = task_definitions
        self.stream_handler = stream_handler
        self.planning_agent = planning_agent or AgentPlanner(
            swarm=self.swarm,
            agent=self._default_planning_agent(),
            template=self._default_planning_prompt_template(),
            task_definitions=task_definitions,
        )

        # Private state
        self._task_registry = TaskRegistry(task_definitions)
        self._execution_history: list[ExecutionResult] = []
        self._context: ContextVariables = ContextVariables(
            team_capabilities=self._get_team_capabilities(),
        )

    # ================================================
    # MARK: Private Helpers
    # ================================================

    def _default_planning_agent(self) -> Agent:
        """Create a default agent if none is provided."""
        return Agent.create(
            id="agent-planner",
            model="gpt-4o",
            instructions=dedent_prompt("""
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
            """),
        )

    def _default_planning_prompt_template(self) -> PromptTemplate:
        """Create a default prompt template."""

        class DefaultPromptTemplate:
            @property
            def template(self) -> str:
                return "{prompt}"

            def format_context(self, prompt: str, context: ContextVariables) -> str:
                return self.template.format(prompt=prompt)

        return DefaultPromptTemplate()

    def _get_team_capabilities(self) -> dict[str, list[str]]:
        """Get a mapping of task types to team member capabilities."""
        capabilities: dict[str, list[str]] = {}
        for member in self.members.values():
            for task_type in member.task_types:
                if task_type not in capabilities:
                    capabilities[task_type] = []
                capabilities[task_type].append(member.agent.id)

        return capabilities

    # ================================================
    # MARK: Task Execution Helpers
    # ================================================

    def _build_task_context(
        self,
        task: Task,
        task_definition: TaskDefinition,
    ) -> ContextVariables:
        """Construct the context for task execution."""
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
        """Prepare task instructions, handling callable instructions if necessary."""
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
        """Process the agent's response and create an ExecutionResult."""
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
        """Parse the agent's output based on the provided schema."""
        output: BaseModel
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            output = output_schema.model_validate_json(content)
        else:
            output = output_schema(content, task_context)  # type: ignore

        return output

    def _select_matching_member(self, task: Task) -> TeamMember | None:
        """Select the best matching team member for a task."""
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
        """Create a new plan from the prompt and context by delegating to AgentPlanner."""
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
        """Execute all tasks in the given plan."""
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
            return Result(error=e)

    async def execute_task(self, task: Task) -> Result[ExecutionResult]:
        """Execute a single task using the appropriate team member."""
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
