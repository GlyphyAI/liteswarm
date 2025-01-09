# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest
from pydantic import BaseModel, ValidationError

from liteswarm.core.swarm import Swarm
from liteswarm.experimental.swarm_team.response_repair import LiteResponseRepairAgent
from liteswarm.types.exceptions import ResponseRepairError
from liteswarm.types.llm import LLM
from liteswarm.types.swarm import Agent, ContextVariables


class ReviewOutput(BaseModel):
    """Test model for response validation."""

    issues: list[str]
    approved: bool


@pytest.fixture
def swarm() -> Swarm:
    """Create a swarm instance for testing."""
    return Swarm()


@pytest.fixture
def repair_agent(swarm: Swarm) -> LiteResponseRepairAgent:
    """Create a repair agent instance for testing."""
    return LiteResponseRepairAgent(swarm=swarm)


@pytest.fixture
def test_agent() -> Agent:
    """Create a test agent that produces invalid responses."""
    return Agent(
        id="test-agent",
        instructions="You are a test agent.",
        llm=LLM(model="gpt-4o"),
    )


@pytest.mark.asyncio
async def test_json_repair_path(
    repair_agent: LiteResponseRepairAgent,
    test_agent: Agent,
) -> None:
    """Test that simple JSON formatting issues can be fixed without LLM."""
    # This response has unquoted strings and boolean
    invalid_response = "{issues: [Missing tests], approved: false}"

    try:
        ReviewOutput.model_validate_json(invalid_response)
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        fixed = await repair_agent.repair_response(
            agent=test_agent,
            response=invalid_response,
            response_format=ReviewOutput,
            validation_error=e,
            context_variables=ContextVariables(),
        )

        assert isinstance(fixed, ReviewOutput)
        assert fixed.issues == ["Missing tests"]
        assert fixed.approved is False


@pytest.mark.asyncio
async def test_llm_repair_path(
    repair_agent: LiteResponseRepairAgent,
    test_agent: Agent,
) -> None:
    """Test that complex validation issues are fixed using LLM."""
    # This response has a type error (string instead of boolean) and missing quotes
    invalid_response = "{issues: [Missing tests], approved: yes}"

    try:
        ReviewOutput.model_validate_json(invalid_response)
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        fixed = await repair_agent.repair_response(
            agent=test_agent,
            response=invalid_response,
            response_format=ReviewOutput,
            validation_error=e,
            context_variables=ContextVariables(),
        )

        assert isinstance(fixed, ReviewOutput)
        assert fixed.issues == ["Missing tests"]
        assert isinstance(fixed.approved, bool)


@pytest.mark.asyncio
async def test_custom_repair_model(swarm: Swarm, test_agent: Agent) -> None:
    """Test that custom repair model can be used."""
    repair_agent = LiteResponseRepairAgent(
        swarm=swarm,
        repair_llm=LLM(
            model="claude-3-5-sonnet-20241022",
            # Claude handles JSON well without explicit JSON mode
        ),
    )

    # Missing quotes around array item and using string for boolean
    invalid_response = '{"issues": [Missing tests], "approved": "false"}'

    try:
        ReviewOutput.model_validate_json(invalid_response)
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        fixed = await repair_agent.repair_response(
            agent=test_agent,
            response=invalid_response,
            response_format=ReviewOutput,
            validation_error=e,
            context_variables=ContextVariables(),
        )

        assert isinstance(fixed, ReviewOutput)
        assert fixed.issues == ["Missing tests"]
        assert isinstance(fixed.approved, bool)
        assert fixed.approved is False


@pytest.mark.asyncio
async def test_repair_failure(swarm: Swarm, test_agent: Agent) -> None:
    """Test that repair fails after max attempts with invalid response."""
    repair_agent = LiteResponseRepairAgent(swarm=swarm, max_attempts=1)

    # Completely malformed JSON that doesn't match the schema at all
    invalid_response = '{"random": "data", "that": ["does", "not"], "match": "schema", }'

    try:
        ReviewOutput.model_validate_json(invalid_response)
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        with pytest.raises(ResponseRepairError):
            await repair_agent.repair_response(
                agent=test_agent,
                response=invalid_response,
                response_format=ReviewOutput,
                validation_error=e,
                context_variables=ContextVariables(),
            )
