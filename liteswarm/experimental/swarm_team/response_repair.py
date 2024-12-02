# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
from typing import Protocol, get_type_hints

from pydantic import BaseModel

from liteswarm.core.swarm import Swarm
from liteswarm.types.result import Result
from liteswarm.types.swarm import Agent, ContextVariables
from liteswarm.types.swarm_team import (
    PydanticResponseFormat,
)
from liteswarm.utils.typing import is_callable, is_subtype

REPAIR_PROMPT_TEMPLATE = """
Your previous response had invalid format. {urgency}fix the response to match this schema exactly:
{schema}

Your previous response was:
{original_content}

Provide a corrected JSON response that matches the schema exactly. Do not include any additional text or explanations.
""".strip()


class ResponseRepairAgent(Protocol):
    """Protocol for agents that handle response validation and repair.

    Provides an interface for agents that can attempt to fix invalid responses
    from team members, ensuring they match the expected schema. This allows
    for different repair strategies while maintaining a consistent interface.
    """

    async def repair_response(
        self,
        agent: Agent,
        original_content: str,
        response_format: PydanticResponseFormat | None,
        context: ContextVariables,
    ) -> Result[str]: ...


class LiteResponseRepairAgent:
    """Default implementation of ResponseRepairAgent.

    Provides basic response repair functionality by:
    1. Extracting schema information from task definition
    2. Prompting the original agent with schema and previous response
    3. Retrying with increasing urgency up to max attempts
    """

    def __init__(
        self,
        swarm: Swarm,
        max_attempts: int = 3,
    ) -> None:
        """Initialize the response repair agent.

        Args:
            swarm: Swarm instance for agent interactions.
            max_attempts: Maximum number of repair attempts before giving up.
        """
        self.swarm = swarm
        self.max_attempts = max_attempts
        self._current_attempt = 0

    def _get_json_schema(self, response_format: PydanticResponseFormat | None) -> dict:
        """Extract and validate JSON schema from response format.

        Handles multiple response format types:
        1. Direct BaseModel schemas - extracts schema directly
        2. Callable formats returning BaseModel - extracts schema from return type
        3. Other formats - raises ValueError

        The schema is used to guide response repair by showing the agent
        the exact structure required for a valid response.

        Args:
            response_format: Response format specification, can be:
                - A BaseModel class (for schema extraction)
                - A callable returning BaseModel (for return type schema)
                - None (raises error)

        Returns:
            JSON schema dictionary describing the expected response format.

        Raises:
            ValueError: If response format is:
                - Not provided (None)
                - Not a BaseModel or callable
                - A callable that doesn't return a BaseModel
                - Missing required schema information

        Examples:
            Direct schema:
                ```python
                class ReviewOutput(BaseModel):
                    issues: list[str]
                    approved: bool

                schema = agent._get_json_schema(ReviewOutput)
                # Returns ReviewOutput's JSON schema
                ```

            Callable format:
                ```python
                def parse_review(content: str, context: dict) -> ReviewOutput:
                    # Parse content into ReviewOutput
                    pass

                schema = agent._get_json_schema(parse_review)
                # Returns ReviewOutput's JSON schema from return type
                ```
        """
        if not response_format:
            raise ValueError("Response format is not defined")

        if is_subtype(response_format, BaseModel):
            return response_format.model_json_schema()

        if is_callable(response_format):
            type_hints = get_type_hints(response_format)
            return_type = type_hints.get("return")
            if not is_subtype(return_type, BaseModel):
                raise ValueError("Response format must return a BaseModel explicitly")

            return return_type.model_json_schema()

        raise ValueError("Response format must be BaseModel or callable returning BaseModel")

    def _build_repair_prompt(
        self,
        schema_example: dict,
        original_content: str,
        attempt: int,
    ) -> str:
        """Build prompt for response repair attempt.

        Creates increasingly urgent prompts for each retry attempt to
        encourage the agent to fix the response format.

        Args:
            schema_example: JSON schema for expected format.
            original_content: Previous invalid response.
            attempt: Current attempt number (1-based).

        Returns:
            Formatted prompt string.
        """
        urgency = "Please try again and " if attempt == 1 else "This is urgent. You must "

        return REPAIR_PROMPT_TEMPLATE.format(
            urgency=urgency,
            schema=json.dumps(schema_example),
            original_content=original_content,
        )

    async def repair_response(
        self,
        agent: Agent,
        original_content: str,
        response_format: PydanticResponseFormat | None,
        context: ContextVariables,
    ) -> Result[str]:
        """Attempt to repair an invalid response.

        Makes up to max_attempts tries to get a valid response from the agent,
        with increasingly urgent prompting on each retry.

        Args:
            agent: The agent that produced the response.
            original_content: The failed response content.
            response_format: Expected response schema.
            context: Execution context.

        Returns:
            Result containing either:
                - Repaired response string that matches schema
                - Error if repair failed after max attempts
        """
        self._current_attempt += 1
        if self._current_attempt > self.max_attempts:
            self._current_attempt = 0
            return Result(
                error=ValueError(f"Failed to get valid response after {self.max_attempts} attempts")
            )

        try:
            schema_example = self._get_json_schema(response_format)
            repair_prompt = self._build_repair_prompt(
                schema_example=schema_example,
                original_content=original_content,
                attempt=self._current_attempt,
            )

            result = await self.swarm.execute(
                agent=agent,
                prompt=repair_prompt,
                context_variables=context,
            )

            repair_result: Result[str]
            if not result.content:
                repair_result = Result(error=ValueError("No content in repair response"))
            else:
                repair_result = Result(value=result.content)

            return repair_result

        except Exception as e:
            return Result(error=e)

        finally:
            self._current_attempt = 0
