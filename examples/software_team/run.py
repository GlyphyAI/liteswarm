# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import os

from liteswarm.core import Swarm
from liteswarm.experimental import SwarmTeam
from liteswarm.types import Artifact, ContextVariables
from liteswarm.utils import dedent_prompt, enable_logging

from .planner import create_agent_planner
from .stream import SwarmStreamHandler, SwarmTeamStreamHandler
from .tasks import create_task_definitions
from .team import create_team_members

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


USER_PROMPT = """
Create a Flutter TODO list app with the following features:

1. Add/edit/delete tasks
2. Mark tasks as complete
3. Local storage for persistence
4. Clean, modern UI design
""".strip()


def print_artifact(artifact: Artifact) -> None:
    """Print a detailed, well-formatted view of an execution artifact."""
    print("\n" + "=" * 50)
    print(f"🏷  Artifact ID: {artifact.id}")
    print(f"📊 Status: {_get_status_emoji(artifact.status)} {artifact.status}")
    print("=" * 50 + "\n")

    if artifact.error:
        print("❌ Execution Failed")
        print(f"Error: {artifact.error}")
        return

    print(f"✅ Successfully completed {len(artifact.task_results)} tasks\n")

    for i, task_result in enumerate(artifact.task_results, 1):
        agent_id = task_result.assignee.agent.id if task_result.assignee else "unknown"
        print(f"Task {i}/{len(artifact.task_results)}")
        print(f"├─ ID: {task_result.task.id}")
        print(f"├─ Type: {task_result.task.type}")
        print(f"├─ Title: {task_result.task.title}")
        print(f"├─ Status: {_get_status_emoji(task_result.task.status)} {task_result.task.status}")
        print(f"└─ Executed by: 🤖 {agent_id}\n")

        if task_result.output:
            print("   Output:")
            for key, value in task_result.output.model_dump().items():
                print(f"   ├─ {key}: {value}")
            print()

    print("=" * 50)


def _get_status_emoji(status: str) -> str:
    """Get an appropriate emoji for a status."""
    return {
        "COMPLETED": "✅",
        "FAILED": "❌",
        "IN_PROGRESS": "⏳",
        "PENDING": "⏳",
        "EXECUTING": "🔄",
    }.get(str(status).upper(), "❓")


async def main() -> None:
    """Run the software team example."""
    swarm = Swarm(
        stream_handler=SwarmStreamHandler(),
        include_usage=True,
        include_cost=True,
    )

    task_definitions = create_task_definitions()
    team_members = create_team_members()
    agent_planner = create_agent_planner(swarm, task_definitions)

    team = SwarmTeam(
        swarm=swarm,
        members=team_members,
        task_definitions=task_definitions,
        agent_planner=agent_planner,
        stream_handler=SwarmTeamStreamHandler(),
    )

    context = ContextVariables(
        tech_stack={
            "platform": "mobile",
            "languages": ["Dart"],
            "frameworks": ["Flutter"],
        },
        project={
            "directories": [
                "lib/main.dart",
            ],
            "files": [
                {
                    "filepath": "lib/main.dart",
                    "content": "import 'package:flutter/material.dart';\n\nvoid main() {\n  runApp(const MyApp());\n}\n\nclass MyApp extends StatelessWidget {\n  const MyApp({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return MaterialApp(\n      title: 'Flutter Demo',\n      theme: ThemeData(\n        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),\n        useMaterial3: true,\n      ),\n      home: const MyHomePage(title: 'Flutter Demo Home Page'),\n    );\n  }\n}\n\nclass MyHomePage extends StatefulWidget {\n  const MyHomePage({super.key, required this.title});\n\n  final String title;\n\n  @override\n  State<MyHomePage> createState() => _MyHomePageState();\n}\n\nclass _MyHomePageState extends State<MyHomePage> {\n  int _counter = 0;\n\n  void _incrementCounter() {\n    setState(() {\n      _counter++;\n    });\n  }\n\n  @override\n  Widget build(BuildContext context) {\n    return Scaffold(\n      appBar: AppBar(\n        backgroundColor: Theme.of(context).colorScheme.inversePrimary,\n        title: Text(widget.title),\n      ),\n      body: Center(\n        child: Column(\n          mainAxisAlignment: MainAxisAlignment.center,\n          children: <Widget>[\n            const Text(\n              'You have pushed the button this many times:',\n            ),\n            Text(\n              '$_counter',\n              style: Theme.of(context).textTheme.headlineMedium,\n            ),\n          ],\n        ),\n      ),\n      floatingActionButton: FloatingActionButton(\n        onPressed: _incrementCounter,\n        tooltip: 'Increment',\n        child: const Icon(Icons.add),\n    );\n  }\n}",
                }
            ],
        },
    )

    prompt = USER_PROMPT

    while True:
        plan_result = await team.create_plan(prompt, context)
        if plan_result.error or not plan_result.value:
            print(f"Failed to create plan: {plan_result.error}")
            return

        plan = plan_result.value

        print("\nReview the plan and choose an option:")
        print("1. Approve and execute")
        print("2. Provide feedback")
        print("3. Reject and exit")

        choice = input("\nYour choice (1-3): ").strip()

        match choice:
            case "1":
                artifact_result = await team.execute_plan(plan)
                if artifact_result.error:
                    print(f"Failed to execute plan: {artifact_result.error}")
                    return

                print_artifact(artifact_result.unwrap("No artifact found"))
                break
            case "2":
                feedback = input("\nEnter your feedback: ")
                prompt = dedent_prompt(f"{prompt}\n\nFeedback: {feedback}")
            case "3":
                print("Plan rejected. Exiting.")
                return
            case _:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    enable_logging()
    asyncio.run(main())
