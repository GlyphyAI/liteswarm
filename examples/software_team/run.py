# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import asyncio
import os

from liteswarm.core import Swarm
from liteswarm.experimental import SwarmTeam
from liteswarm.types import ContextVariables
from liteswarm.utils import dedent_prompt, enable_logging

from .planner import create_agent_planner
from .stream import SwarmStreamHandler, SwarmTeamStreamHandler
from .tasks import create_task_definitions
from .team import create_team_members

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


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
        platform="mobile",
        framework="flutter",
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

    prompt = dedent_prompt("""
    Create a Flutter TODO list app with the following features:

    1. Add/edit/delete tasks
    2. Mark tasks as complete
    3. Local storage for persistence
    4. Clean, modern UI design
    """)

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
                plan_output = await team.execute_plan(plan)
                if plan_output.error:
                    print(f"Failed to execute plan: {plan_output.error}")
                    return

                results = plan_output.value or []
                for result in results:
                    agent_id = result.assignee.agent.id if result.assignee else "unknown"
                    print(f"Task {result.task.id} completed by {agent_id}")

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
