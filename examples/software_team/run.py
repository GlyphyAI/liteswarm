# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio

from liteswarm import enable_logging
from liteswarm.core import Swarm
from liteswarm.experimental import LiteTeamChat
from liteswarm.types import ApprovePlan, ArtifactStatus, Plan, PlanFeedback, RejectPlan
from liteswarm.utils.misc import prompt as prompt_user

from .handlers import EventHandler
from .planning import build_planning_agent_user_prompt, create_planning_agent
from .tasks import create_task_definitions
from .team import create_team_members
from .types import FileContent, Project, TechStack
from .utils import create_context_from_project, extract_project_from_artifact, print_artifact

enable_logging(default_level="DEBUG")

DEFAULT_PROMPT = "Create a simple todo list app"


def get_project_summary(project: Project) -> str:
    """Get a brief summary of the current project state."""
    if not project.tech_stack:
        return "No tech stack configured."

    return (
        f"Current project:\n"
        f"• Frameworks: {project.tech_stack.frameworks}\n"
        f"• Files: {len(project.files)} file(s) in {len(project.directories)} directory(ies)\n"
        f"• Directories: {', '.join(project.directories)}"
    )


def create_project() -> Project:
    """Create a new project with initial Flutter setup."""
    return Project(
        tech_stack=TechStack(
            platform="mobile",
            languages=["Dart"],
            frameworks=["Flutter"],
        ),
        directories=["lib"],
        files=[
            FileContent(
                filepath="lib/main.dart",
                content="import 'package:flutter/material.dart';\n\nvoid main() {\n  runApp(const MyApp());\n}\n\nclass MyApp extends StatelessWidget {\n  const MyApp({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return MaterialApp(\n      title: 'Flutter Demo',\n      theme: ThemeData(\n        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),\n        useMaterial3: true,\n      ),\n      home: const MyHomePage(title: 'Flutter Demo Home Page'),\n    );\n  }\n}\n\nclass MyHomePage extends StatefulWidget {\n  const MyHomePage({super.key, required this.title});\n\n  final String title;\n\n  @override\n  State<MyHomePage> createState() => _MyHomePageState();\n}\n\nclass _MyHomePageState extends State<MyHomePage> {\n  int _counter = 0;\n\n  void _incrementCounter() {\n    setState(() {\n      _counter++;\n    });\n  }\n\n  @override\n  Widget build(BuildContext context) {\n    return Scaffold(\n      appBar: AppBar(\n        backgroundColor: Theme.of(context).colorScheme.inversePrimary,\n        title: Text(widget.title),\n      ),\n      body: Center(\n        child: Column(\n          mainAxisAlignment: MainAxisAlignment.center,\n          children: <Widget>[\n            const Text(\n              'You have pushed the button this many times:',\n            ),\n            Text(\n              '$_counter',\n              style: Theme.of(context).textTheme.headlineMedium,\n            ),\n          ],\n        ),\n      ),\n      floatingActionButton: FloatingActionButton(\n        onPressed: _incrementCounter,\n        tooltip: 'Increment',\n        child: const Icon(Icons.add),\n    );\n  }\n}",
            )
        ],
    )


async def handle_plan_feedback(plan: Plan) -> PlanFeedback:
    """Handle user feedback on the generated plan."""
    # Show the plan
    print("\nProposed Plan:")
    print("-" * 30)
    print(f"Plan ID: {plan.id}")
    for i, task in enumerate(plan.tasks, 1):
        print(f"{i}. {task.title}")
        if task.description:
            print(f"   {task.description}")
    print("-" * 30)

    while True:
        choice = await prompt_user("\n1. Approve and execute\n2. Provide feedback\n3. Exit\n\nYour choice (1-3): ")  # fmt: skip

        match choice:
            case "1":
                return ApprovePlan(type="approve")
            case "2":
                feedback = await prompt_user("\nEnter your feedback: ")
                return RejectPlan(type="reject", feedback=feedback)
            case "3":
                raise KeyboardInterrupt("User chose to exit")
            case _:
                print("Invalid choice. Please try again.")
                continue


async def create_chat() -> LiteTeamChat:
    """Create a new chat."""
    swarm = Swarm()
    members = create_team_members()
    task_definitions = create_task_definitions()
    planning_agent = create_planning_agent(swarm, task_definitions)

    chat = LiteTeamChat(
        swarm=swarm,
        members=members,
        task_definitions=task_definitions,
        planning_agent=planning_agent,
    )

    return chat


async def main() -> None:
    """Run the software team example."""
    print("\nWelcome to the Software Team!")
    print("You are working on a Flutter project. All changes will be accumulated in the current project.")  # fmt: skip
    print("You can start a new project at any time to reset the state and history.\n")

    # Create team chat
    chat = await create_chat()
    project = create_project()
    context_variables = create_context_from_project(project)

    # Create event handler
    event_handler = EventHandler()

    while True:
        try:
            # Show current project state
            print("\n" + "=" * 50)
            print(get_project_summary(project))
            print("=" * 50)

            # Get the next task from user
            print("\nWhat would you like to build or modify in this project? (Press Enter for a default prompt)")  # fmt: skip
            print(f"Default: {DEFAULT_PROMPT}")
            prompt = await prompt_user("\n🗣️  Enter your query: ")
            prompt = prompt.strip() or DEFAULT_PROMPT
            prompt = build_planning_agent_user_prompt(prompt, context_variables)

            # Execute with feedback
            stream = chat.send_message(
                prompt,
                context_variables=context_variables,
                feedback_callback=handle_plan_feedback,
            )

            async for event in stream:
                event_handler.on_event(event)

            print()  # New line after streaming
            artifact = await stream.get_return_value()
            print_artifact(artifact)

            # Update project state if execution was successful
            if artifact.status == ArtifactStatus.COMPLETED:
                project = extract_project_from_artifact(artifact, project)
                context_variables = create_context_from_project(project)
                print(f"Project updated: {project.model_dump_json()}")

            # Ask what to do next
            print("\nWhat would you like to do next?")
            print("1. Continue working on this project")
            print("2. Start a new project (resets everything)")
            print("3. Exit")

            choice = await prompt_user("\nYour choice (1-3): ")
            match choice:
                case "1":
                    print("\nContinuing with the current project...")
                    print("Your next task will build upon the existing project state.")
                    continue
                case "2":
                    print("This will create a completely new project and reset all history.")
                    chat = await create_chat()
                    project = create_project()
                    context_variables = create_context_from_project(project)
                    continue
                case "3":
                    print("\nExiting. Thanks for using the software team!")
                    break
                case _:
                    print("Invalid choice. Please try again.")
                    continue

        except (KeyboardInterrupt, EOFError):
            print("\nExecution cancelled by user. Goodbye!")
            break

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Would you like to try again? (y/N)")
            if input().lower() != "y":
                break


if __name__ == "__main__":
    asyncio.run(main())
