import asyncio

from litellm.types.utils import ChatCompletionDeltaToolCall

from liteswarm.swarm import Swarm
from liteswarm.types import Agent, FunctionTool, Message, ToolCallResult, TypedDelta


class ConsoleStreamHandler:
    async def on_stream(
        self,
        chunk: TypedDelta,
        agent: Agent | None,
    ) -> None:
        if chunk.content:
            print(f"{chunk.content}", end="", flush=True)

    async def on_error(
        self,
        error: Exception,
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Error: {str(error)}")

    async def on_agent_switch(
        self,
        previous_agent: Agent | None,
        next_agent: Agent,
    ) -> None:
        print(f"[{next_agent.agent_id}] Switched to {next_agent.agent_id}")

    async def on_complete(
        self,
        messages: list[Message],
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Completed")

    async def on_tool_call(
        self,
        tool_call: ChatCompletionDeltaToolCall,
        agent: Agent | None,
    ) -> None:
        print(f"[{agent.agent_id if agent else 'unknown'}] Tool call: {tool_call}")

    async def on_tool_call_result(
        self,
        tool_call_result: ToolCallResult,
        agent: Agent | None,
    ) -> None:
        print(
            f"[{agent.agent_id if agent else 'unknown'}] Tool call result: {tool_call_result}"
        )


async def run() -> None:
    def create_agent_tools(agents: dict[str, Agent]) -> dict[str, FunctionTool]:
        """Create tool functions to switch between agents."""

        def create_switch_function(agent: Agent) -> FunctionTool:
            def switch_to_agent() -> Agent:
                """Switch to the specified agent."""
                return agent

            switch_to_agent.__name__ = f"switch_to_{agent.agent_id}"
            return switch_to_agent

        return {
            f"switch_to_{name}": create_switch_function(agent)
            for name, agent in agents.items()
        }

    def create_flutter_team() -> dict[str, Agent]:
        agents = {
            "product_manager": Agent.create(
                agent_id="product_manager",
                model="claude-3-5-haiku-20241022",
                instructions="""You are a Product Manager leading the app development process.
                Analyze requirements and coordinate with other specialists.

                When you receive a feature request:
                1. Analyze requirements and create specifications
                2. You MUST call switch_to_designer() to pass control to the designer

                After your analysis, ALWAYS use the switch_to_designer() function call.
                Do not proceed without making this function call.""",
                function_tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
            "designer": Agent.create(
                agent_id="designer",
                model="claude-3-5-haiku-20241022",
                instructions="""You are a UI/UX Designer creating Flutter app designs.
                Review the PM's requirements and create design specifications.

                Process:
                1. Create detailed UI/UX specifications
                2. You MUST call switch_to_engineer() to pass control to the engineer

                After your design work, ALWAYS use the switch_to_engineer() function call.
                Do not proceed without making this function call.""",
                function_tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
            "engineer": Agent.create(
                agent_id="engineer",
                model="claude-3-5-haiku-20241022",
                instructions="""You are a Flutter Engineer implementing the app.
                Review the design specifications and implement the feature.

                Process:
                1. Implement the feature in Flutter
                2. You MUST call switch_to_qa() to pass control to QA

                After your implementation, ALWAYS use the switch_to_qa() function call.
                Do not proceed without making this function call.""",
                function_tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
            "qa": Agent.create(
                agent_id="qa",
                model="claude-3-5-haiku-20241022",
                instructions="""You are a QA Engineer testing the implementation.
                Review the implementation and perform testing.

                Process:
                1. Test the feature thoroughly
                2. If issues are found, you MUST call switch_to_engineer() to return to the engineer
                3. If approved, complete your response without making any function calls

                ALWAYS call switch_to_engineer() if any issues are found.""",
                function_tools=[],
                tool_choice="auto",
                temperature=0.0,
            ),
        }

        # Create tool functions for switching between agents
        tools = create_agent_tools(agents)

        # Add tools to each agent
        for agent in agents.values():
            agent.function_tools.extend(tools.values())

        return agents

    # Create stream handler
    console_handler = ConsoleStreamHandler()
    client = Swarm(stream_handler=console_handler)

    # Create team of agents
    team = create_flutter_team()

    # Start with Product Manager
    result = await client.execute(
        agent=team["product_manager"],
        prompt="Create a simple TODO list app with a list of tasks and a form to add new tasks",
    )

    print("\nExecution completed!")
    print("Final conversation:")
    for msg in result.messages:
        if msg["role"] != "system":
            print(f"\n[{msg['role']}]: {msg['content']}")

    print(f"\n\nResult messages:\n\n{result.messages}")


if __name__ == "__main__":
    asyncio.run(run())
