# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from pydantic import ValidationError

from liteswarm.core.swarm import Swarm
from liteswarm.types.result import Result
from liteswarm.types.swarm import Agent, ContextVariables
from liteswarm.types.swarm_team import PydanticModel, PydanticResponseFormat
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.typing import is_callable


class ResponseRepairAgent(Protocol):
    """Protocol for agents that handle response validation and repair.

    This protocol defines the interface for agents that can repair invalid responses
    by regenerating them with proper validation. It allows for different repair
    strategies while maintaining a consistent interface.
    """

    async def repair_response(
        self,
        agent: Agent,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel] | None,
        validation_error: ValidationError,
        context: ContextVariables,
    ) -> Result[PydanticModel]:
        """Repair an invalid response to match the expected format.

        The repair process should attempt to fix the invalid response while maintaining
        its semantic meaning. The implementation can use various strategies such as
        regeneration, modification, or transformation. The process should be guided
        by the validation error to understand what needs to be fixed. The repaired
        response must conform to the provided response format if one is specified.

        Args:
            agent: The agent that produced the invalid response. Can be used to
                regenerate or modify the response.
            response: The original invalid response content that needs repair.
            response_format: Expected format for the response. Can be either a
                Pydantic model class or a callable that returns one. If None,
                no format validation is performed.
            validation_error: The error from attempting to validate the original
                response. Contains details about what made the response invalid.
            context: Execution context containing variables that may be needed
                for response generation or validation.

        Returns:
            Result containing either the repaired response that matches the format,
            or an error if repair was not possible. The success case will contain
            a properly validated instance of the response format model.

        Example:
            ```python
            class ReviewOutput(BaseModel):
                issues: list[str]
                approved: bool

            class SimpleRepairAgent(ResponseRepairAgent):
                async def repair_response(
                    self,
                    agent: Agent,
                    response: str,
                    response_format: PydanticResponseFormat[ReviewOutput],
                    validation_error: ValidationError,
                    context: ContextVariables,
                ) -> Result[ReviewOutput]:
                    try:
                        # Simple repair strategy: add quotes to values
                        fixed = response.replace("true", '"true"')
                        return Result(
                            value=response_format.model_validate_json(fixed)
                        )
                    except Exception as e:
                        return Result(error=e)
            ```
        """
        ...


class LiteResponseRepairAgent:
    """Agent that repairs invalid responses by regenerating them.

    This agent attempts to fix invalid responses by removing the failed response,
    retrieving the last user message, and asking the original agent to try again.
    It will make multiple attempts to generate a valid response before giving up.

    Example:
        ```python
        class ReviewOutput(BaseModel):
            issues: list[str]
            approved: bool

        swarm = Swarm()
        repair_agent = LiteResponseRepairAgent(swarm)

        # Invalid response missing quotes
        response = "{issues: [Missing tests], approved: false}"
        result = await repair_agent.repair_response(
            agent=review_agent,
            response=response,
            response_format=ReviewOutput,
            validation_error=error,
            context=context,
        )
        if result.success():
            print(result.value.model_dump())  # Fixed response
        ```
    """

    def __init__(
        self,
        swarm: Swarm,
        max_attempts: int = 5,
    ) -> None:
        """Initialize the response repair agent.

        Args:
            swarm: Swarm instance for agent interactions.
            max_attempts: Maximum number of repair attempts before giving up (default: 5).

        Example:
            ```python
            swarm = Swarm()
            repair_agent = LiteResponseRepairAgent(
                swarm=swarm,
                max_attempts=5,
            )
            ```
        """
        self.swarm = swarm
        self.max_attempts = max_attempts

    def _parse_response(
        self,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel] | None,
        context: ContextVariables,
    ) -> Result[PydanticModel]:
        """Parse and validate a response string against the expected format.

        Attempts to parse the response string using the provided format. If the
        format is a callable, it's called with the response and context. If it's
        a BaseModel, the response is validated against it.

        Args:
            response: Response string to parse.
            response_format: Expected response format.
            context: Context variables for dynamic resolution.

        Returns:
            Result containing either the parsed response or an error.

        Example:
            ```python
            # With BaseModel format
            format = ReviewOutput
            result = agent._parse_response(
                '{"issues": [], "approved": true}',
                format,
                context,
            )
            assert result.success()
            assert isinstance(result.value, ReviewOutput)

            # With callable format
            def parse(content: str, ctx: dict) -> ReviewOutput:
                data = json.loads(content)
                return ReviewOutput(
                    issues=data["issues"],
                    approved=data["approved"],
                )

            result = agent._parse_response(
                '{"issues": [], "approved": true}',
                parse,
                context,
            )
            assert result.success()
            assert isinstance(result.value, ReviewOutput)
            ```
        """
        if not response_format:
            return Result(value=response)

        try:
            if is_callable(response_format):
                return Result(value=response_format(response, context))
            return Result(value=response_format.model_validate_json(response))
        except Exception as e:
            log_verbose(f"Error parsing response: {e}", level="ERROR")
            return Result(error=e)

    async def _regenerate_last_user_message(
        self,
        agent: Agent,
        context: ContextVariables,
    ) -> Result[str]:
        """Regenerate a response for the last user message in history.

        Removes the failed response, gets the last user message, and tries again.
        If anything goes wrong, we put the original messages back to keep history intact.

        Args:
            agent: The agent to use for regeneration.
            context: Execution context.

        Returns:
            Result containing either the new response content or an error.

        Example:
            ```python
            # Initial conversation
            swarm.append_message(Message(role="user", content="Review this code"))
            swarm.append_message(Message(role="assistant", content="Invalid JSON"))

            # Regenerate response
            result = await agent._regenerate_last_user_message(
                review_agent,
                context,
            )
            if result.success():
                print(result.value)  # New valid response
            else:
                print(f"Failed: {result.error}")
            ```
        """
        last_assistant_message = self.swarm.pop_last_message()
        if not last_assistant_message:
            return Result(error=ValueError("No message to regenerate"))

        last_user_message = self.swarm.pop_last_message()
        if not last_user_message or last_user_message.role != "user":
            self.swarm.append_message(last_assistant_message)
            return Result(error=ValueError("No user message found to regenerate"))

        try:
            result = await self.swarm.execute(
                agent=agent,
                prompt=last_user_message.content or "",
                context_variables=context,
            )
            return Result(value=result.content)
        except Exception as e:
            self.swarm.append_message(last_user_message)
            self.swarm.append_message(last_assistant_message)
            return Result(error=e)

    async def repair_response(
        self,
        agent: Agent,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel] | None,
        validation_error: ValidationError,
        context: ContextVariables,
    ) -> Result[PydanticModel]:
        """Attempt to repair an invalid response by regenerating it.

        If a response is invalid, we remove it and ask the agent to try again
        with the same user message. This repeats until we get a valid response
        or run out of attempts.

        Args:
            agent: The agent that produced the response.
            response: The failed response content.
            response_format: Expected response schema.
            validation_error: Validation error from the original response.
            context: Execution context.

        Returns:
            Result containing either a repaired response or an error.

        Example:
            ```python
            class ReviewOutput(BaseModel):
                issues: list[str]
                approved: bool

            # Invalid response
            response = "{issues: [], approved: invalid}"
            try:
                ReviewOutput.model_validate_json(response)
            except ValidationError as e:
                result = await repair_agent.repair_response(
                    agent=review_agent,
                    response=response,
                    response_format=ReviewOutput,
                    validation_error=e,
                    context=context,
                )
                if result.success():
                    output = result.value
                    assert isinstance(output, ReviewOutput)
                    print(f"Fixed: {output.model_dump()}")
                else:
                    print(f"Failed to repair: {result.error}")
            ```
        """
        try:
            for attempt in range(1, self.max_attempts + 1):
                log_verbose(f"Repair attempt {attempt}/{self.max_attempts}")

                regeneration_result = await self._regenerate_last_user_message(
                    agent=agent,
                    context=context,
                )

                if regeneration_result.failure():
                    log_verbose(f"Regeneration failed: {regeneration_result.error}", level="ERROR")
                    continue

                regenerated_response = regeneration_result.value
                if not regenerated_response:
                    continue

                if response_format:
                    log_verbose(f"Parsing response: {regenerated_response}")
                    parsed_response = self._parse_response(
                        response=regenerated_response,
                        response_format=response_format,
                        context=context,
                    )

                    if parsed_response.failure():
                        log_verbose(f"Parsing failed: {parsed_response.error}", level="ERROR")
                        continue

                    log_verbose(f"Parsed response: {parsed_response.model_dump_json()}")
                    return parsed_response
                else:
                    return Result(value=regenerated_response)

        except Exception as e:
            log_verbose(f"Repair failed with error: {e}", level="ERROR")
            return Result(error=e)

        return Result(
            error=ValueError(f"Failed to get valid response after {self.max_attempts} attempts")
        )
