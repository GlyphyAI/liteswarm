# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pydantic import BaseModel

from liteswarm.types.swarm_team import PydanticResponseFormat, Task, TeamMember


class SwarmError(Exception):
    """Base exception class for all Swarm-related errors.

    Provides a common ancestor for all custom exceptions in the system,
    enabling unified error handling and logging of Swarm operations.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                )
            except SwarmError as e:
                logger.error(f"Swarm operation failed: {e}")
            ```

        Custom exception:
            ```python
            class ValidationError(SwarmError):
                \"\"\"Raised when agent input validation fails.\"\"\"
                def __init__(self, field: str, value: Any) -> None:
                    super().__init__(
                        f"Invalid value {value} for field {field}"
                    )
            ```
    """


class CompletionError(SwarmError):
    """Exception raised when LLM completion fails permanently.

    Indicates that the language model API call failed and exhausted
    all retry attempts. Preserves the original error for debugging
    and error reporting.

    Examples:
        Basic handling:
            ```python
            try:
                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                )
            except CompletionError as e:
                logger.error(
                    f"API call failed: {e}",
                    extra={
                        "error_type": type(e.original_error).__name__ if e.original_error else None,
                        "details": str(e.original_error) if e.original_error else None,
                    },
                )
            ```

        Fallback strategy:
            ```python
            try:
                result = await swarm.execute(
                    agent=primary_agent,
                    prompt="Hello",
                )
            except CompletionError:
                # Switch to backup model
                backup_agent = Agent(
                    id="backup",
                    instructions="You are a backup assistant.",
                    llm=LLM(model="gpt-4o"),
                )
                result = await swarm.execute(
                    agent=backup_agent,
                    prompt="Hello",
                )
            ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new CompletionError.

        Args:
            message: Human-readable error description.
            original_error: Optional underlying exception that caused the failure.
        """
        super().__init__(message)
        self.original_error = original_error


class ContextLengthError(SwarmError):
    """Exception raised when input exceeds model's context limit.

    Occurs when the combined length of conversation history and new
    input exceeds the model's maximum context window, even after
    attempting context reduction strategies.

    Examples:
        Basic handling:
            ```python
            try:
                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                )
            except ContextLengthError as e:
                logger.warning(
                    "Context length exceeded",
                    extra={
                        "current_length": e.current_length,
                        "error": str(e.original_error) if e.original_error else None,
                    },
                )
            ```

        Automatic model upgrade:
            ```python
            async def execute_with_fallback(
                swarm: Swarm,
                prompt: str,
                agent: Agent,
            ) -> AgentExecutionResult:
                try:
                    return await swarm.execute(
                        agent=agent,
                        prompt=prompt,
                    )
                except ContextLengthError:
                    # Switch to larger context model
                    large_agent = Agent(
                        id="large-context",
                        instructions=agent.instructions,
                        llm=LLM(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=200000,
                        ),
                    )
                    return await swarm.execute(
                        agent=large_agent,
                        prompt=prompt,
                    )
            ```
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        current_length: int | None = None,
        max_length: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new ContextLengthError.

        Args:
            message: Human-readable error description.
            model: Model that exceeded the context limit.
            current_length: Current context length that exceeded the limit.
            max_length: Maximum allowed context length.
            original_error: Optional underlying exception that caused the failure.
        """
        super().__init__(message)
        self.model = model
        self.current_length = current_length
        self.max_length = max_length
        self.original_error = original_error


class SwarmTeamError(SwarmError):
    """Base exception class for SwarmTeam-related errors.

    Provides a common ancestor for all SwarmTeam exceptions, enabling unified error
    handling for team operations like planning, task execution, and response processing.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await team.execute_plan(plan)
            except SwarmTeamError as e:
                logger.error(
                    f"Team execution failed: {e.message}",
                    extra={"error": str(e.original_error) if e.original_error else None},
                )
            ```

        Specific error types:
            ```python
            try:
                result = await team.execute_plan(plan)
            except PlanValidationError as e:
                logger.error("Plan validation failed", extra={"errors": e.validation_errors})
            except TaskExecutionError as e:
                logger.error(f"Task {e.task.id} failed", extra={"assignee": e.assignee.id})
            except SwarmTeamError as e:
                logger.error("Other team error occurred", extra={"error": str(e)})
            ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new SwarmTeamError.

        Args:
            message: Human-readable error description.
            original_error: Optional underlying exception that caused the failure.
        """
        super().__init__(message)
        self.original_error = original_error


class PlanValidationError(SwarmTeamError):
    """Exception raised when plan validation fails.

    Occurs when a plan fails validation checks. Common failures include using unknown
    task types that aren't registered with the team, having invalid dependencies between
    tasks such as circular references, missing required fields in task definitions, or
    providing invalid task configurations that don't match the schema.

    Examples:
        Handle validation failures:
            ```python
            try:
                plan = await team.create_plan("Review PR")
            except PlanValidationError as e:
                if e.validation_errors:
                    logger.error("Plan validation failed:")
                    for error in e.validation_errors:
                        logger.error(f"- {error}")
                else:
                    logger.error(f"Plan validation error: {e.message}")
            ```

        Custom validation handling:
            ```python
            try:
                plan = await team.create_plan(prompt)
            except PlanValidationError as e:
                if any("Unknown task type" in err for err in (e.validation_errors or [])):
                    logger.error("Plan contains unsupported task types")
                    # Register missing task types
                    team.register_task_definitions([new_task_def])
                    # Retry with updated task types
                    plan = await team.create_plan(prompt)
            ```
    """

    def __init__(
        self,
        message: str,
        validation_errors: list[str] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new PlanValidationError.

        Args:
            message: Human-readable error description.
            validation_errors: Optional list of specific validation failures.
            original_error: Optional underlying exception that caused validation to fail.
        """
        super().__init__(message, original_error)
        self.validation_errors = validation_errors


class TaskExecutionError(SwarmTeamError):
    """Exception raised when task execution fails.

    Occurs when a task fails to execute successfully. This can happen for several
    reasons: the team might not have any members capable of handling the task type,
    the execution might time out, the agent might encounter errors during execution,
    the response might fail parsing or validation, or tool execution might fail
    with errors.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await team.execute_task(task)
            except TaskExecutionError as e:
                logger.error(
                    f"Task execution failed: {e.message}",
                    extra={
                        "task": e.task.id,
                        "assignee": e.assignee.id if e.assignee else None,
                        "error": str(e.original_error) if e.original_error else None,
                    },
                )
            ```

        Retry with different agent:
            ```python
            try:
                result = await team.execute_task(task)
            except TaskExecutionError as e:
                if e.assignee:
                    # Try with different agent
                    backup = team.get_backup_member(e.assignee)
                    result = await team.execute_task(task, assignee=backup)
            ```
    """

    def __init__(
        self,
        message: str,
        task: Task,
        assignee: TeamMember | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new TaskExecutionError.

        Args:
            message: Human-readable error description.
            task: Task that failed to execute.
            assignee: Optional team member that failed to execute the task.
            original_error: Optional underlying exception that caused the failure.
        """
        super().__init__(message, original_error)
        self.task = task
        self.assignee = assignee


class ResponseParsingError(SwarmTeamError):
    """Exception raised when response parsing fails.

    Occurs when an agent's response cannot be parsed into the expected format.
    This typically happens when the response JSON is malformed, missing required
    fields, or contains invalid values that don't match the schema.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await team.execute_task(task)
            except ResponseParsingError as e:
                logger.error(
                    f"Response parsing failed: {e.message}",
                    extra={
                        "response": e.response,
                        "format": e.response_format.model_json_schema() if e.response_format else None,
                        "error": str(e.original_error) if e.original_error else None,
                    },
                )
            ```

        Retry with repair:
            ```python
            try:
                result = await team.execute_task(task)
            except ResponseParsingError as e:
                if e.response:
                    # Try to repair and parse again
                    repaired = repair_json(e.response)
                    result = parse_response(repaired, e.response_format)
            ```
    """

    def __init__(
        self,
        message: str,
        response: str | None = None,
        response_format: PydanticResponseFormat[BaseModel] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new ResponseParsingError.

        Args:
            message: Human-readable error description.
            response: Optional raw response that failed to parse.
            response_format: Optional expected response format.
            original_error: Optional underlying exception that caused parsing to fail.
        """
        super().__init__(message, original_error)
        self.response = response
        self.response_format = response_format


class ResponseRepairError(SwarmTeamError):
    """Exception raised when response repair fails.

    Occurs when attempts to repair an invalid response fail. This typically happens
    when the response format is incompatible with the target schema, the maximum
    repair attempts are exhausted, or when the agent consistently generates invalid
    responses. May also occur when required fields cannot be reconstructed or the
    response structure is too corrupted.

    Examples:
        Basic error handling:
            ```python
            try:
                repaired = await repair_agent.repair_response(
                    response=invalid_response,
                    response_format=ReviewOutput,
                    validation_error=error,
                )
            except ResponseRepairError as e:
                logger.error(
                    f"Response repair failed: {e.message}",
                    extra={
                        "response": e.response,
                        "format": e.response_format.model_json_schema() if e.response_format else None,
                        "error": str(e.original_error) if e.original_error else None,
                    },
                )
                # Fall back to raw response
                return TaskResult(
                    task=task,
                    content=None,
                    status=TaskStatus.FAILED,
                )
            ```

        With specialized repair agent:
            ```python
            try:
                result = await repair_agent.repair_response(response, format)
            except ResponseRepairError as e:
                if "max attempts" in str(e):
                    # Try with specialized repair agent
                    repair_specialist = Agent(
                        id="repair-specialist",
                        instructions="You are an expert at repairing malformed responses.",
                        llm=LLM(
                            model="gpt-4o",
                            temperature=0.3,  # Lower temperature for more consistent repairs
                        ),
                    )
                    return await repair_specialist.repair_response(
                        response=e.response,
                        response_format=e.response_format,
                    )
                raise  # Re-raise if not a max attempts error
            ```
    """

    def __init__(
        self,
        message: str,
        response: str | None = None,
        response_format: PydanticResponseFormat[BaseModel] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new ResponseRepairError.

        Args:
            message: Human-readable error description.
            response: Optional response that could not be repaired.
            response_format: Optional target format specification.
            original_error: Optional underlying exception from the last repair attempt.
        """
        super().__init__(message, original_error)
        self.response = response
        self.response_format = response_format


class MaxAgentSwitchesError(SwarmError):
    """Exception raised when too many agent switches occur.

    Indicates potential infinite switching loops or excessive agent transitions.
    This error helps prevent scenarios where agents continuously pass control
    between each other without making progress.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                )
            except MaxAgentSwitchesError as e:
                logger.error(
                    f"Too many agent switches: {e.message}",
                    extra={
                        "switch_count": e.switch_count,
                        "max_switches": e.max_switches,
                        "switch_history": e.switch_history,  # List of agent IDs in order
                    },
                )
            ```

        Fallback to single agent:
            ```python
            try:
                result = await swarm.execute(
                    agent=agent,
                    prompt="Hello",
                )
            except MaxAgentSwitchesError as e:
                # Get the first agent from switch history
                first_agent_id = e.switch_history[0] if e.switch_history else None
                if first_agent_id:
                    # Create a new swarm with just the first agent
                    single_swarm = Swarm(max_agent_switches=0)
                    result = await single_swarm.execute(
                        agent=agent,
                        prompt="Hello",
                    )
            ```
    """

    def __init__(
        self,
        message: str,
        switch_count: int,
        max_switches: int,
        switch_history: list[str] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new MaxAgentSwitchesError.

        Args:
            message: Human-readable error description.
            switch_count: Number of switches that occurred.
            max_switches: Maximum allowed switches.
            switch_history: Optional list of agent IDs in switch order.
            original_error: Optional underlying exception.
        """
        super().__init__(message)
        self.switch_count = switch_count
        self.max_switches = max_switches
        self.switch_history = switch_history
        self.original_error = original_error


class MaxResponseContinuationsError(SwarmError):
    """Exception raised when response needs too many continuations.

    Occurs when an agent's response exceeds length limits and requires more
    continuations than allowed. This helps prevent scenarios where responses
    grow indefinitely without reaching a natural conclusion.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await swarm.execute(prompt)
            except MaxResponseContinuationsError as e:
                logger.error(
                    f"Response too long: {e.message}",
                    extra={
                        "continuation_count": e.continuation_count,
                        "max_continuations": e.max_continuations,
                        "total_tokens": e.total_tokens,
                    },
                )
            ```

        Split into smaller tasks:
            ```python
            try:
                result = await swarm.execute(prompt)
            except MaxResponseContinuationsError as e:
                # Break into smaller chunks
                subtasks = split_task(prompt)
                results = []
                for subtask in subtasks:
                    result = await swarm.execute(subtask)
                    results.append(result)
                result = combine_results(results)
            ```
    """

    def __init__(
        self,
        message: str,
        continuation_count: int,
        max_continuations: int,
        total_tokens: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new MaxResponseContinuationsError.

        Args:
            message: Human-readable error description.
            continuation_count: Number of continuations attempted.
            max_continuations: Maximum allowed continuations.
            total_tokens: Optional total tokens generated.
            original_error: Optional underlying exception.
        """
        super().__init__(message)
        self.continuation_count = continuation_count
        self.max_continuations = max_continuations
        self.total_tokens = total_tokens
        self.original_error = original_error


class RetryError(SwarmError):
    """Exception raised when retry mechanism fails.

    Indicates that all retry attempts have been exhausted without success.
    This error provides detailed information about the retry process,
    including attempt counts, timing, and the original error that
    triggered retries.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await swarm.execute(prompt)
            except RetryError as e:
                logger.error(
                    f"Retry mechanism failed: {e.message}",
                    extra={
                        "attempts": e.attempts,
                        "total_duration": e.total_duration,
                        "backoff_strategy": e.backoff_strategy,
                        "original_error": str(e.original_error),
                    },
                )
            ```

        Fallback strategy:
            ```python
            try:
                result = await swarm.execute(prompt)
            except RetryError as e:
                if e.attempts >= 3:
                    # Switch to more reliable model
                    fallback_agent = Agent(
                        id="fallback",
                        llm=LLM(model="gpt-4o"),
                    )
                    result = await swarm.execute(prompt, agent=fallback_agent)
            ```
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        total_duration: float,
        backoff_strategy: dict[str, float],
        original_error: Exception,
    ) -> None:
        """Initialize a new RetryError.

        Args:
            message: Human-readable error description.
            attempts: Number of retry attempts made.
            total_duration: Total time spent retrying in seconds.
            backoff_strategy: Dictionary with retry settings.
            original_error: The underlying exception that caused retries.
        """
        super().__init__(message)
        self.attempts = attempts
        self.total_duration = total_duration
        self.backoff_strategy = backoff_strategy
        self.original_error = original_error
