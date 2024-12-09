# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

from pydantic import ValidationError

from liteswarm.core.swarm import Swarm
from liteswarm.types.exceptions import ResponseRepairError
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
        response_format: PydanticResponseFormat[PydanticModel],
        validation_error: ValidationError,
        context: ContextVariables,
    ) -> PydanticModel:
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
            A properly validated instance of the response format model.

        Raises:
            ResponseRepairError: If the response cannot be repaired.

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
                ) -> ReviewOutput:
                    # Simple repair strategy: add quotes to values
                    fixed = response.replace("true", '"true"')
                    return response_format.model_validate_json(fixed)
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
        try:
            output = await repair_agent.repair_response(
                agent=review_agent,
                response=response,
                response_format=ReviewOutput,
                validation_error=error,
                context=context,
            )
            print(output.model_dump())  # Fixed response
        except ResponseRepairError as e:
            print(f"Failed to repair: {e}")
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
        response_format: PydanticResponseFormat[PydanticModel],
        context: ContextVariables,
    ) -> PydanticModel:
        """Parse and validate a response string against the expected format.

        Attempts to parse the response string using the provided format. If the
        format is a callable, it's called with the response and context. If it's
        a BaseModel, the response is validated against it.

        Args:
            response: Response string to parse.
            response_format: Expected response format.
            context: Context variables for dynamic resolution.

        Returns:
            The parsed and validated response object.

        Raises:
            ValidationError: If the response fails format validation.
            ValueError: If the response format is invalid.

        Example:
            ```python
            # With BaseModel format
            format = ReviewOutput
            try:
                output = agent._parse_response(
                    '{"issues": [], "approved": true}',
                    format,
                    context,
                )
                assert isinstance(output, ReviewOutput)
            except ValidationError:
                print("Invalid response format")


            # With callable format
            def parse(content: str, ctx: dict) -> ReviewOutput:
                data = json.loads(content)
                return ReviewOutput(
                    issues=data["issues"],
                    approved=data["approved"],
                )


            try:
                output = agent._parse_response(
                    '{"issues": [], "approved": true}',
                    parse,
                    context,
                )
                assert isinstance(output, ReviewOutput)
            except ValidationError:
                print("Failed to parse response")
            ```
        """
        if is_callable(response_format):
            return response_format(response, context)

        return response_format.model_validate_json(response)

    async def _regenerate_last_user_message(
        self,
        agent: Agent,
        context: ContextVariables,
    ) -> str:
        """Regenerate a response for the last user message in history.

        Removes the failed response, gets the last user message, and tries again.
        If anything goes wrong, we put the original messages back to keep history intact.

        Args:
            agent: The agent to use for regeneration.
            context: Execution context.

        Returns:
            The new response content.

        Raises:
            ResponseRepairError: If regeneration fails or no messages are found.

        Example:
            ```python
            # Initial conversation
            swarm.append_message(Message(role="user", content="Review this code"))
            swarm.append_message(Message(role="assistant", content="Invalid JSON"))

            try:
                # Regenerate response
                new_response = await agent._regenerate_last_user_message(
                    review_agent,
                    context,
                )
                print(new_response)  # New valid response
            except ResponseRepairError as e:
                print(f"Failed to regenerate: {e}")
            ```
        """
        last_assistant_message = self.swarm.pop_last_message()
        if not last_assistant_message:
            raise ResponseRepairError("No message to regenerate")

        last_user_message = self.swarm.pop_last_message()
        if not last_user_message or last_user_message.role != "user":
            self.swarm.append_message(last_assistant_message)
            raise ResponseRepairError("No user message found to regenerate")

        try:
            result = await self.swarm.execute(
                agent=agent,
                prompt=last_user_message.content or "",
                context_variables=context,
            )

            if not result.content:
                raise ValueError("No response content")

            return result.content

        except Exception as e:
            self.swarm.append_message(last_user_message)
            self.swarm.append_message(last_assistant_message)
            raise ResponseRepairError(
                f"Failed to regenerate response: {e}",
                response=last_user_message.content,
                original_error=e,
            ) from e

    async def repair_response(
        self,
        agent: Agent,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel],
        validation_error: ValidationError,
        context: ContextVariables,
    ) -> PydanticModel:
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
            A repaired and validated response object.

        Raises:
            ResponseRepairError: If repair fails after maximum attempts or other errors occur.

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
                try:
                    output = await repair_agent.repair_response(
                        agent=review_agent,
                        response=response,
                        response_format=ReviewOutput,
                        validation_error=e,
                        context=context,
                    )
                    assert isinstance(output, ReviewOutput)
                    print(f"Fixed: {output.model_dump()}")
                except ResponseRepairError as e:
                    print(f"Failed to repair: {e}")
            ```
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                log_verbose(f"Repair attempt {attempt}/{self.max_attempts}")
                regenerated_response = await self._regenerate_last_user_message(
                    agent=agent,
                    context=context,
                )
            except Exception as e:
                log_verbose(f"Failed to regenerate response: {e}", level="ERROR")
                continue

            try:
                log_verbose(f"Parsing response: {regenerated_response}")
                parsed_response = self._parse_response(
                    response=regenerated_response,
                    response_format=response_format,
                    context=context,
                )
            except ValidationError as e:
                log_verbose(f"Failed to parse response: {e}", level="ERROR")
                continue
            except Exception as e:
                log_verbose(f"Failed to parse response: {e}", level="ERROR")
                raise ResponseRepairError(
                    f"Failed to parse response: {e}",
                    response=regenerated_response,
                    original_error=e,
                ) from e

            log_verbose(f"Parsed response: {parsed_response.model_dump_json()}")
            return parsed_response

        raise ResponseRepairError(
            f"Failed to get valid response after {self.max_attempts} attempts",
            response=response,
            original_error=validation_error,
        )
