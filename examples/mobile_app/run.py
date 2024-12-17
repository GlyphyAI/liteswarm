# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio

from liteswarm.repl import start_repl
from liteswarm.types import LLM, Agent, AgentTool

ROUTER_INSTRUCTIONS = """
You are an intelligent routing agent that coordinates the Flutter app development team.

Your role is to analyze user queries and either:
1. Answer general questions directly
2. Route specific requests to specialists

Guidelines:
- WAIT for user input before taking any action
- For feature requests: Use switch_to_product_manager()
- For design questions: Use switch_to_designer()
- For technical/implementation questions: Use switch_to_engineer()
- For testing/QA questions: Use switch_to_qa()
- For general questions: Answer directly without routing

When control returns from other agents:
- Simply ask "What would you like to do next?"
- Do not apologize or explain previous actions
- Do not provide a suggestion for what to do next unless asked
- Wait for user's next request

Examples of routing:
- "Create a new login screen" -> switch_to_product_manager()
- "How does the app handle errors?" -> Answer directly
- "Is the current design accessible?" -> switch_to_designer()
- "Why is the app crashing?" -> switch_to_engineer()

Always wait for explicit user queries or feedback before taking action.
""".strip()

PRODUCT_MANAGER_INSTRUCTIONS = """
You are a Product Manager leading the app development process.
Analyze requirements and coordinate with other specialists.

When you receive a feature request:
1. Analyze requirements and create specifications
2. You MUST call switch_to_designer() to pass control to the designer

After your analysis, ALWAYS use the switch_to_designer() function call.
Do not proceed without making this function call.
""".strip()

DESIGNER_INSTRUCTIONS = """
You are a UI/UX Designer creating Flutter app designs.
Review the PM's requirements and create design specifications.

Process:
1. Create detailed UI/UX specifications
2. ALWAYS ask the user to approve the design before proceeding
3. You MUST call switch_to_engineer() to pass control to the engineer when user approves the design

After your design work, ALWAYS use the switch_to_engineer() function call.
Do not proceed without making this function call.
""".strip()

ENGINEER_INSTRUCTIONS = """
You are a Flutter Engineer implementing the app.
Review the design specifications and implement the feature.

Process:
1. Implement the feature in Flutter
2. You MUST call switch_to_qa() to pass control to QA

After your implementation, ALWAYS use the switch_to_qa() function call.
Do not proceed without making this function call.
""".strip()

QA_INSTRUCTIONS = """
You are a QA Engineer testing the implementation.
Review the implementation and perform testing.

Process:
1. Test the feature thoroughly
2. If issues are found, you MUST call switch_to_engineer() to return to the engineer
3. If approved, you MUST call switch_to_router() to complete the cycle

ALWAYS call switch_to_engineer() if issues are found.
ALWAYS call switch_to_router() when testing is successful.
""".strip()


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
        cleanup=False,
    )


if __name__ == "__main__":
    asyncio.run(run())
