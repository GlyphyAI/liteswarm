# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from liteswarm.types.swarm_team import PydanticResponseFormat, Task, TeamMember


class SwarmError(Exception):
    """Base exception class for all Swarm-related errors.

    Provides a common ancestor for all custom exceptions in the system,
    enabling unified error handling and logging of Swarm operations.

    Examples:
        Basic error handling:
            ```python
            try:
                result = await swarm.execute(prompt)
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
                response = await agent.complete(prompt)
            except CompletionError as e:
                logger.error(
                    f"API call failed: {e}",
                    extra={"error_type": type(e.original_error).__name__, "details": str(e.original_error)},
                )
            ```

        Fallback strategy:
            ```python
            try:
                response = await primary_agent.complete(prompt)
            except CompletionError:
                # Switch to backup model
                backup_agent = Agent(id="backup", llm=LLM(model="gpt-3.5-turbo"))
                response = await backup_agent.complete(prompt)
            ```
    """

    def __init__(
        self,
        message: str,
        original_error: Exception,
    ) -> None:
        """Initialize a new CompletionError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused the failure.
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
                response = await agent.complete(prompt)
            except ContextLengthError as e:
                logger.warning(
                    "Context length exceeded",
                    extra={"current_length": e.current_length, "error": str(e.original_error)},
                )
            ```

        Automatic model upgrade:
            ```python
            async def complete_with_fallback(prompt: str, agent: Agent) -> str:
                try:
                    return await agent.complete(prompt)
                except ContextLengthError:
                    # Switch to larger context model
                    large_agent = Agent(
                        id="large-context",
                        instructions=agent.instructions,
                        llm=LLM(model="claude-3-opus", max_tokens=100000),
                    )
                    return await large_agent.complete(prompt)
            ```
    """

    def __init__(
        self,
        message: str,
        current_length: int,
        original_error: Exception,
    ) -> None:
        """Initialize a new ContextLengthError.

        Args:
            message: Human-readable error description.
            current_length: Current context length that exceeded the limit.
            original_error: The underlying exception that caused the failure.
        """
        super().__init__(message)

        self.current_length = current_length
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
                    f"Task {e.task.id} ({e.task.type}) failed:",
                    extra={
                        "title": e.task.title,
                        "assignee": e.assignee.id if e.assignee else None,
                        "status": e.task.status,
                        "error": str(e.original_error) if e.original_error else None,
                    },
                )
            ```

        Retry with different assignee:
            ```python
            try:
                result = await team.execute_task(task)
            except TaskExecutionError as e:
                if e.assignee:
                    logger.warning(f"Task failed with {e.assignee.id}, trying backup")
                    # Try again with specific assignee
                    task.assignee = "backup-agent"
                    result = await team.execute_task(task)
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
            task: The task that failed to execute.
            assignee: Optional team member that attempted the task.
            original_error: Optional underlying exception that caused the failure.
        """
        super().__init__(message, original_error)
        self.task = task
        self.assignee = assignee


class ResponseParsingError(SwarmTeamError):
    """Exception raised when task response parsing fails.

    Occurs when an agent's response cannot be parsed according to the expected format.
    This typically happens due to syntax issues in the JSON response, missing fields
    that are required by the schema, incorrect data types in the response, general
    schema validation failures, or errors in custom parser implementations.

    Examples:
        Handle parsing failures:
            ```python
            try:
                output = team._parse_response(response, response_format)
            except ResponseParsingError as e:
                logger.error(
                    f"Failed to parse {e.response_format.__name__} response:",
                    extra={
                        "response": e.response,
                        "error_type": type(e.original_error).__name__,
                        "error": str(e.original_error),
                    },
                )
                # Attempt repair if possible
                if isinstance(e.original_error, ValidationError):
                    output = await repair_agent.repair_response(
                        e.response,
                        e.response_format,
                        e.original_error,
                    )
            ```

        With format-specific handling:
            ```python
            try:
                result = await team.execute_task(task)
            except ResponseParsingError as e:
                if e.response_format == ReviewOutput:
                    # Handle review output parsing failures
                    logger.error("Failed to parse review results")
                    result = await fallback_review_handler(e.response)
                else:
                    # Handle other format failures
                    raise
            ```
    """

    def __init__(
        self,
        message: str,
        response: str | None = None,
        response_format: PydanticResponseFormat | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize a new ResponseParsingError.

        Args:
            message: Human-readable error description.
            response: Optional raw response that failed to parse.
            response_format: Optional expected format specification.
            original_error: Optional underlying exception that caused parsing to fail.
        """
        super().__init__(message, original_error)
        self.response = response
        self.response_format = response_format


class ResponseRepairError(SwarmTeamError):
    """Exception raised when response repair fails.

    Occurs when attempts to repair an invalid response fail. This can happen when
    the maximum number of repair attempts is exhausted, when the response is
    fundamentally invalid and cannot be fixed, when the agent consistently fails
    to generate a valid response, or when the target response format's requirements
    cannot be satisfied with the given input.

    Examples:
        Handle repair failures:
            ```python
            try:
                repaired = await repair_agent.repair_response(
                    response,
                    response_format,
                    validation_error,
                )
            except ResponseRepairError as e:
                logger.error(
                    "Response repair failed after multiple attempts:",
                    extra={
                        "response": e.response,
                        "format": e.response_format.__name__,
                        "error": str(e.original_error),
                    },
                )
                # Fall back to raw response
                return TaskResult(
                    task=task,
                    content=e.response,
                    status=TaskStatus.COMPLETED_WITH_ERRORS,
                )
            ```

        With retry logic:
            ```python
            async def repair_with_retry(response: str, format: Type[BaseModel]) -> BaseModel:
                try:
                    return await repair_agent.repair_response(response, format)
                except ResponseRepairError as e:
                    if "max attempts" in str(e):
                        # Try one more time with different agent
                        backup_agent = Agent(id="repair-specialist", llm=LLM(model="gpt-4"))
                        return await backup_agent.repair_response(e.response, e.response_format)
                    raise
            ```
    """

    def __init__(
        self,
        message: str,
        response: str | None = None,
        response_format: PydanticResponseFormat | None = None,
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
