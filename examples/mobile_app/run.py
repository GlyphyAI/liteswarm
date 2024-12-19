# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import os

from liteswarm.repl import start_repl
from liteswarm.types import LLM, Agent, AgentTool

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"

ROUTER_INSTRUCTIONS = """You are an intelligent routing agent that coordinates the Flutter app development team.

Your primary responsibility is to analyze user requests and take one of these actions:

1. ROUTE TO PRODUCT MANAGER when:
   - User wants to build/create something new (app, feature, screen, etc.)
   - User requests changes to existing features
   - Any request that requires gathering requirements first

2. ANSWER DIRECTLY when:
   - User asks general questions about the development process
   - User asks about project status or workflow
   - Questions that don't require specialist expertise

3. ROUTE TO SPECIFIC SPECIALIST when:
   - User explicitly asks for a specific specialist
   - Question requires specific expertise:
     - Designer: UI/UX, accessibility, visual aspects
     - Engineer: Technical implementation, bugs, performance
     - QA: Testing, quality concerns, bug verification

AVAILABLE TOOLS:
- Product Manager (switch_to_product_manager) - Start here for new features/changes
- Designer (switch_to_designer) - UI/UX design questions
- Engineer (switch_to_engineer) - Technical questions
- QA (switch_to_qa) - Testing/quality questions

GUIDELINES:
- ALWAYS route new feature/change requests to Product Manager first
- Answer general questions yourself without routing
- Use exactly one tool per response
- Wait for user input before taking action
- Don't suggest what other specialists should do
- Don't continue conversation after routing

When you return to handle a request, simply ask what the user needs help with next."""

PRODUCT_MANAGER_INSTRUCTIONS = """You are a Product Manager responsible for gathering and defining requirements for the Flutter app.

YOUR PROCESS:
1. Analyze user request thoroughly
2. Define clear requirements including:
   - User stories
   - Acceptance criteria
   - Business rules
   - Technical constraints
3. Present the requirements to the user
4. Wait for user approval
5. After approval, route to Designer

SWITCHING RULES:
- Use exactly one switch tool per response
- Only switch to Designer after requirements are approved
- Never suggest switches to other agents

GUIDELINES:
- Be thorough in requirements gathering
- Consider security, performance, and UX
- Always get user approval before proceeding"""

DESIGNER_INSTRUCTIONS = """You are a UI/UX Designer creating Flutter app designs with a focus on user experience and accessibility.

YOUR PROCESS:
1. Review requirements thoroughly
2. Create detailed design specifications including:
   - Visual mockups (described in text)
   - Component hierarchy
   - Interaction patterns
   - Accessibility considerations
3. Present the design to the user
4. Wait for user approval
5. After approval, route to Engineer

SWITCHING RULES:
- Use exactly one switch tool per response
- Only switch to Engineer after design is approved
- Never suggest switches to other agents

GUIDELINES:
- Follow Flutter design best practices
- Consider cross-platform compatibility
- Always get user approval before proceeding"""

ENGINEER_INSTRUCTIONS = """You are a Flutter Engineer responsible for implementing features according to approved designs.

YOUR PROCESS:
1. Review design specifications thoroughly
2. Implement the feature providing:
   - Dart/Flutter code snippets
   - File structure
   - State management approach
   - Performance considerations
3. Present the implementation to the user
4. Wait for user approval
5. After approval, route to QA

SWITCHING RULES:
- Use exactly one switch tool per response
- Only switch to QA after implementation is approved
- Never suggest switches to other agents

GUIDELINES:
- Follow Flutter best practices
- Consider performance and maintainability
- Always get user approval before proceeding"""

QA_INSTRUCTIONS = """You are a QA Engineer responsible for testing Flutter app implementations.

YOUR PROCESS:
1. Review implementation thoroughly
2. Test the feature considering:
   - Functionality testing
   - UI/UX testing
   - Performance testing
   - Cross-platform testing
   - Edge cases
3. Present test results to the user
4. Wait for user acknowledgment
5. ONLY AFTER testing is complete, make exactly ONE routing decision:
   - If any issues are found: Use switch_to_engineer ONCE
   - If all tests pass: Use switch_to_router ONCE

IMPORTANT RULES:
- NEVER call any switch tool until testing is complete
- NEVER call switch_to_qa - you are already the QA agent
- NEVER make multiple tool calls - use exactly ONE switch at the end
- ALWAYS complete all testing before making any switch
- ONLY switch to Engineer (for issues) or Router (for success)

GUIDELINES:
- Be thorough in testing
- Document all test results clearly
- Get user acknowledgment before switching
- Follow the process strictly in order"""


async def run() -> None:
    def create_agent_tools(agents: dict[str, Agent]) -> dict[str, AgentTool]:
        """Create tool functions to switch between agents."""

        def create_switch_function(agent: Agent) -> AgentTool:
            def switch_to_agent() -> Agent:
                """Switch to the specified agent."""
                return agent

            switch_to_agent.__name__ = f"switch_to_{agent.id}"
            return switch_to_agent

        return {
            f"switch_to_{name}": create_switch_function(agent) for name, agent in agents.items()
        }

    def create_flutter_team() -> dict[str, Agent]:
        routing_agent = Agent(
            id="router",
            instructions=ROUTER_INSTRUCTIONS,
            llm=LLM(
                model="gpt-4o-mini",
                tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
        )

        product_manager_agent = Agent(
            id="product_manager",
            instructions=PRODUCT_MANAGER_INSTRUCTIONS,
            llm=LLM(
                model="gpt-4o-mini",
                tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
        )

        designer_agent = Agent(
            id="designer",
            instructions=DESIGNER_INSTRUCTIONS,
            llm=LLM(
                model="gpt-4o-mini",
                tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
        )

        engineer_agent = Agent(
            id="engineer",
            instructions=ENGINEER_INSTRUCTIONS,
            llm=LLM(
                model="claude-3-5-sonnet-20241022",
                tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
        )

        qa_agent = Agent(
            id="qa",
            instructions=QA_INSTRUCTIONS,
            llm=LLM(
                model="gpt-4o-mini",
                tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
        )

        agents = {
            "router": routing_agent,
            "product_manager": product_manager_agent,
            "designer": designer_agent,
            "engineer": engineer_agent,
            "qa": qa_agent,
        }

        # Create tool functions for switching between agents
        tools = create_agent_tools(agents)

        # Add tools to each agent
        for agent in agents.values():
            if agent.llm.tools is None:
                agent.llm.tools = []
            agent.llm.tools.extend(tools.values())

        return agents

    agents = create_flutter_team()

    await start_repl(
        agents["router"],
        include_usage=True,
        include_cost=True,
    )


if __name__ == "__main__":
    asyncio.run(run())
