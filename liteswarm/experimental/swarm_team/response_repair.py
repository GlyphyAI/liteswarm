# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

import json_repair
from pydantic import ValidationError
from typing_extensions import override

from liteswarm.core.swarm import Swarm
from liteswarm.types.exceptions import ResponseRepairError
from liteswarm.types.llm import LLM
from liteswarm.types.swarm import Agent, ContextVariables, Message
from liteswarm.types.swarm_team import PydanticModel, PydanticResponseFormat
from liteswarm.utils.logging import log_verbose
from liteswarm.utils.typing import is_callable

REPAIR_SYSTEM_PROMPT = """You are a specialized JSON repair agent. Your task is to fix invalid JSON responses to match their schema exactly.

CRITICAL REQUIREMENTS:
1. Return ONLY the fixed JSON object - no explanations, no wrappers, no additional fields
2. If repair is impossible, return this exact JSON:
   {"error": "<unable_to_repair>"}
3. DO NOT create new data or invent values - you must preserve the original meaning
4. The response must match the schema structure EXACTLY:
   - Do not add any wrapper objects
   - Include only the fields specified in the schema
   - Match the exact field names from the schema
5. Ensure proper JSON syntax:
   - All strings must be in double quotes
   - Arrays must be properly closed with brackets
   - Objects must be properly closed with braces
   - Use commas to separate items
6. Match the schema's type requirements exactly:
   - Boolean values must be `true` or `false` (not strings)
   - Numbers must be without quotes
   - Arrays must contain items of the correct type

Example repairs:

1. Missing quotes and wrong boolean:
Invalid: {
  name: Alice Johnson,
  email: alice@example.com,
  is_active: yes,
  roles: [admin, user]
}
Valid: {
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "is_active": true,
  "roles": ["admin", "user"]
}

2. Numeric type fixes and nested objects:
Invalid: {
  product_id: "12345",
  price: "29.99",
  metadata: {
    weight: "2.5",
    dimensions: {width: "10", height: 15.0, depth: "8"}
  }
}
Valid: {
  "product_id": "12345",
  "price": 29.99,
  "metadata": {
    "weight": 2.5,
    "dimensions": {"width": 10, "height": 15.0, "depth": 8}
  }
}

3. Array formatting and mixed types:
Invalid: {
  tags: [development, testing, qa],
  stats: {daily_visits: "156", conversion_rate: 0.0345},
  config: {debug_mode: yes, retry_count: "3", endpoints: [/api/v1, /api/v2]}
}
Valid: {
  "tags": ["development", "testing", "qa"],
  "stats": {"daily_visits": 156, "conversion_rate": 0.0345},
  "config": {"debug_mode": true, "retry_count": 3, "endpoints": ["/api/v1", "/api/v2"]}
}

4. Unrepairable case (completely different structure):
Invalid: {
  "unrelated_field": "some value",
  "wrong_structure": {
    "nested": ["wrong", "data", "types"]
  }
}
Response: {"error": "<unable_to_repair>"}"""


REPAIR_USER_PROMPT = """Here is what needs to be fixed:

ORIGINAL RESPONSE:
{response}

VALIDATION ERROR:
{validation_error}

EXPECTED SCHEMA:
{schema}

Remember: If repair is impossible while preserving meaning, return {{"error": "<unable_to_repair>"}}. Do not create new data or invent values."""


class ResponseRepairAgent(Protocol):
    """Protocol for response repair agents built on top of Swarm.

    Defines an interface for agents that repair invalid responses by fixing common
    validation issues. Following Swarm's stateless design, repair agents maintain
    no internal state between operations.

    Examples:
        Create a custom repair agent:
            ```python
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

    Notes:
        - Each method returns a validated response object
        - Repairs must preserve semantic meaning
        - Multiple repair strategies can be combined
    """

    async def repair_response(
        self,
        agent: Agent,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel],
        validation_error: ValidationError,
        context_variables: ContextVariables | None = None,
    ) -> PydanticModel:
        """Repair an invalid response to match the expected format.

        The repair process should attempt to fix the invalid response while maintaining
        its semantic meaning. The implementation can use various strategies such as
        regeneration, modification, or transformation. The process should be guided
        by the validation error to understand what needs to be fixed. The repaired
        response must conform to the provided response format if one is specified.

        Args:
            agent: Agent that produced the invalid response.
            response: Original invalid response content.
            response_format: Expected response schema.
            validation_error: Error from validation attempt.
            context_variables: Optional context for repair.

        Returns:
            Repaired and validated response object.

        Raises:
            ResponseRepairError: If repair fails.
        """
        ...


class LiteResponseRepairAgent(ResponseRepairAgent):
    """Multi-strategy response repair agent built on top of Swarm.

    Uses a combination of JSON repair and LLM-based repair to fix invalid responses:
    1. First tries json_repair for common JSON issues
    2. If that fails, uses an LLM to repair with schema guidance
    3. Validates the repaired response against format

    Following Swarm's stateless design, the agent maintains no internal state
    between operations.

    Examples:
        Basic repair:
            ```python
            class ReviewOutput(BaseModel):
                issues: list[str]
                approved: bool


            swarm = Swarm()
            repair_agent = LiteResponseRepairAgent(swarm)

            # Invalid response missing quotes
            response = "{issues: [], approved: invalid}"
            try:
                output = await repair_agent.repair_response(
                    agent=review_agent,
                    response=response,
                    response_format=ReviewOutput,
                    validation_error=error,
                )
                print(output.model_dump())  # Fixed response
            except ResponseRepairError as e:
                print(f"Failed to repair: {e}")
            ```

    Notes:
        - Each repair attempt uses multiple strategies
        - Repairs must preserve semantic meaning
        - Response format must follow LLM provider's schema requirements
    """

    def __init__(
        self,
        swarm: Swarm,
        repair_llm: LLM | None = None,
        max_attempts: int = 5,
    ) -> None:
        """Initialize a response repair agent.

        Args:
            swarm: Swarm instance for agent interactions.
            repair_llm: Optional custom LLM for repair.
            max_attempts: Maximum repair attempts (default: 5).
        """
        self.swarm = swarm
        self.repair_llm = repair_llm or LLM(
            model="gpt-4o",
            response_format={"type": "json_object"},
        )
        self.max_attempts = max_attempts

    def _parse_response(
        self,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel],
        context_variables: ContextVariables | None = None,
    ) -> PydanticModel:
        """Parse and validate a response string.

        Args:
            response: Response string to parse.
            response_format: Expected schema.
            context_variables: Optional context.

        Returns:
            Validated response object.

        Raises:
            ValidationError: If validation fails.
            ValueError: If format is invalid.
        """
        if is_callable(response_format):
            return response_format(response, context_variables)

        decoded_object = json_repair.repair_json(response, return_objects=True)
        if isinstance(decoded_object, tuple):
            decoded_object = decoded_object[0]

        return response_format.model_validate(decoded_object)

    async def _repair_with_llm(
        self,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel],
        validation_error: ValidationError,
        context_variables: ContextVariables | None = None,
    ) -> PydanticModel:
        """Use an LLM to repair an invalid response.

        Args:
            response: Invalid response to repair.
            response_format: Expected schema.
            validation_error: Error from validation.
            context_variables: Optional context.

        Returns:
            Validated response object.

        Raises:
            ResponseRepairError: If repair fails.
        """
        log_verbose("Attempting repair with LLM", level="DEBUG")
        schema = response_format.model_json_schema() if not is_callable(response_format) else None
        repair_instructions = REPAIR_USER_PROMPT.format(
            response=response,
            validation_error=validation_error,
            schema=schema,
        )

        log_verbose(
            f"Repair instructions:\n{repair_instructions}",
            level="DEBUG",
        )

        repair_agent = Agent(
            id="repair-specialist",
            instructions=REPAIR_SYSTEM_PROMPT,
            llm=self.repair_llm,
        )

        try:
            result = await self.swarm.execute(
                agent=repair_agent,
                messages=[Message(role="user", content=repair_instructions)],
                context_variables=context_variables,
            )

            if not result.last_agent_response.content:
                log_verbose("No content received from repair agent", level="ERROR")
                raise ResponseRepairError(
                    "No repaired content received",
                    response=response,
                    response_format=response_format,
                )

            log_verbose(
                f"Repair agent response:\n{result.last_agent_response.content}",
                level="DEBUG",
            )

            parsed_response = self._parse_response(
                response=result.last_agent_response.content,
                response_format=response_format,
                context_variables=context_variables,
            )

            log_verbose(
                f"Parsed response: {parsed_response.model_dump_json()}",
                level="DEBUG",
            )

            return parsed_response

        except Exception as e:
            log_verbose(f"Repair agent failed: {e}", level="ERROR")
            raise ResponseRepairError(
                f"Failed to repair response with LLM: {e}",
                response=response,
                response_format=response_format,
                original_error=e,
            ) from e

    async def _repair_with_json_repair(
        self,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel],
        context_variables: ContextVariables | None = None,
    ) -> PydanticModel:
        """Use json_repair to fix common JSON issues.

        Args:
            response: Invalid response to repair.
            response_format: Expected schema.
            context_variables: Optional context.

        Returns:
            Validated response object.

        Raises:
            ValidationError: If validation fails.
            ValueError: If format is invalid.
        """
        log_verbose("Attempting repair with json_repair", level="DEBUG")
        parsed_response = self._parse_response(
            response=response,
            response_format=response_format,
            context_variables=context_variables,
        )

        log_verbose(f"Parsed response: {parsed_response.model_dump_json()}")
        return parsed_response

    @override
    async def repair_response(
        self,
        agent: Agent,
        response: str,
        response_format: PydanticResponseFormat[PydanticModel],
        validation_error: ValidationError,
        context_variables: ContextVariables | None = None,
    ) -> PydanticModel:
        """Repair an invalid response to match the expected format.

        Uses multiple repair strategies in sequence:
        1. First tries json_repair for common JSON issues
        2. If that fails, uses an LLM to repair with schema guidance
        3. Validates the repaired response against format

        Args:
            agent: Agent that produced the response.
            response: Invalid response content.
            response_format: Expected schema.
            validation_error: Error from validation.
            context_variables: Optional context.

        Returns:
            Repaired and validated response object.

        Raises:
            ResponseRepairError: If repair fails after max attempts.

        Example:
            ```python
            # Invalid response
            response = "{issues: [], approved: invalid}"
            try:
                output = await repair_agent.repair_response(
                    agent=review_agent,
                    response=response,
                    response_format=ReviewOutput,
                    validation_error=error,
                )
                print(output.model_dump())  # Fixed response
            except ResponseRepairError as e:
                print(f"Failed to repair: {e}")
            ```
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                log_verbose(f"Repair attempt {attempt}/{self.max_attempts}")

                try:
                    return await self._repair_with_json_repair(
                        response=response,
                        response_format=response_format,
                        context_variables=context_variables,
                    )
                except (ValidationError, ValueError) as e:
                    log_verbose(f"json_repair failed: {e}", level="DEBUG")

                return await self._repair_with_llm(
                    response=response,
                    response_format=response_format,
                    validation_error=validation_error,
                    context_variables=context_variables,
                )

            except Exception as e:
                log_verbose(f"Repair attempt {attempt} failed: {e}", level="ERROR")

        raise ResponseRepairError(
            f"Failed to get valid response after {self.max_attempts} attempts",
            response=response,
            response_format=response_format,
            original_error=validation_error,
        )
