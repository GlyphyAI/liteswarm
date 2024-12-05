# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import os

from liteswarm.core import Swarm
from liteswarm.experimental import SwarmTeam
from liteswarm.types import ArtifactStatus
from liteswarm.utils import enable_logging

from .handlers import InteractivePlanFeedbackHandler, SwarmStreamHandler, SwarmTeamStreamHandler
from .planner import create_planning_agent
from .tasks import create_task_definitions
from .team import create_team_members
from .types import FileContent, Project
from .utils import create_context_from_project, extract_project_from_artifact, print_artifact

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


def get_project_summary(project: Project) -> str:
    """Get a brief summary of the current project state.

    Args:
        project: The current project.

    Returns:
        A formatted string describing the project state.
    """
    if not project.tech_stack:
        return "No tech stack configured."

    return (
        f"Current project:\n"
        f"• Frameworks: {project.tech_stack.frameworks}\n"
        f"• Files: {len(project.files)} file(s) in {len(project.directories)} directory(ies)\n"
        f"• Directories: {', '.join(project.directories)}"
    )


def get_user_prompt(default_prompt: str = "Create a simple todo list app") -> str:
    """Get the user's prompt with a default option.

    Returns:
        The user's prompt or the default prompt if none provided.
    """
    print("\nWhat would you like to build or modify in this project? (Press Enter for a default prompt)")  # fmt: skip
    print(f"Default: {default_prompt}")
    user_input = input("\n> ").strip()
    return user_input or default_prompt


def create_project() -> Project:
    """Create a new project with initial Flutter setup.

    Returns:
        A new Project instance with default Flutter configuration.
    """
    return Project(
        tech_stack={
            "platform": "mobile",
            "languages": ["Dart"],
            "frameworks": ["Flutter"],
        },
        directories=["lib"],
        files=[
            FileContent(
                filepath="lib/main.dart",
                content="import 'package:flutter/material.dart';\n\nvoid main() {\n  runApp(const MyApp());\n}\n\nclass MyApp extends StatelessWidget {\n  const MyApp({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return MaterialApp(\n      title: 'Flutter Demo',\n      theme: ThemeData(\n        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),\n        useMaterial3: true,\n      ),\n      home: const MyHomePage(title: 'Flutter Demo Home Page'),\n    );\n  }\n}\n\nclass MyHomePage extends StatefulWidget {\n  const MyHomePage({super.key, required this.title});\n\n  final String title;\n\n  @override\n  State<MyHomePage> createState() => _MyHomePageState();\n}\n\nclass _MyHomePageState extends State<MyHomePage> {\n  int _counter = 0;\n\n  void _incrementCounter() {\n    setState(() {\n      _counter++;\n    });\n  }\n\n  @override\n  Widget build(BuildContext context) {\n    return Scaffold(\n      appBar: AppBar(\n        backgroundColor: Theme.of(context).colorScheme.inversePrimary,\n        title: Text(widget.title),\n      ),\n      body: Center(\n        child: Column(\n          mainAxisAlignment: MainAxisAlignment.center,\n          children: <Widget>[\n            const Text(\n              'You have pushed the button this many times:',\n            ),\n            Text(\n              '$_counter',\n              style: Theme.of(context).textTheme.headlineMedium,\n            ),\n          ],\n        ),\n      ),\n      floatingActionButton: FloatingActionButton(\n        onPressed: _incrementCounter,\n        tooltip: 'Increment',\n        child: const Icon(Icons.add),\n    );\n  }\n}",
            )
        ],
    )


def create_team() -> SwarmTeam:
    """Create a new software team.

    Creates fresh instances of Swarm and SwarmTeam to ensure no lingering state or history.

    Returns:
        A new SwarmTeam instance.
    """
    swarm = Swarm(
        stream_handler=SwarmStreamHandler(),
        include_usage=True,
        include_cost=True,
    )

    task_definitions = create_task_definitions()
    team_members = create_team_members()
    planning_agent = create_planning_agent(swarm, task_definitions)

    team = SwarmTeam(
        swarm=swarm,
        members=team_members,
        task_definitions=task_definitions,
        planning_agent=planning_agent,
        stream_handler=SwarmTeamStreamHandler(),
    )

    return team


async def main() -> None:
    """Run the software team example."""
    print("\nWelcome to the Software Team!")
    print("You are working on a Flutter project. All changes will be accumulated in the current project.")  # fmt: skip
    print("You can start a new project at any time to reset the state and history.\n")

    team = create_team()
    project = create_project()
    context = create_context_from_project(project)

    while True:
        try:
            # Show current project state
            print("\n" + "=" * 50)
            print(get_project_summary(project))
            print("=" * 50)

            # Get the next task from user
            prompt = get_user_prompt()

            # Execute the task
            artifact = await team.execute(
                prompt=prompt,
                context=context,
                feedback_handler=InteractivePlanFeedbackHandler(),
            )
            print_artifact(artifact)

            # Update project state if execution was successful
            if artifact.status == ArtifactStatus.COMPLETED:
                project = extract_project_from_artifact(artifact, project)
                context = create_context_from_project(project)

            # Ask what to do next
            print("\nWhat would you like to do next?")
            print("1. Continue working on this project")
            print("2. Start a new project (resets everything)")
            print("3. Exit")

            choice = input("\nYour choice (1-3): ").strip()
            match choice:
                case "1":
                    print("\nContinuing with the current project...")
                    print("Your next task will build upon the existing project state.")
                    continue
                case "2":
                    print("This will create a completely new project and reset all history.")
                    team = create_team()
                    project = create_project()
                    context = create_context_from_project(project)
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
    enable_logging()
    asyncio.run(main())
