import asyncio
import os

from liteswarm.logging import enable_logging
from liteswarm.swarm import Swarm
from liteswarm.swarm_team import SwarmTeam

from .agents import (
    create_agent_planner,
    create_team_members,
)
from .stream import SoftwareTeamStreamHandler, SwarmStreamHandler
from .tasks import create_task_definitions

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


async def main() -> None:
    """Run the software team example."""
    swarm = Swarm(stream_handler=SwarmStreamHandler())
    task_definitions = create_task_definitions()
    team_members = create_team_members()
    agent_planner = create_agent_planner(swarm, task_definitions)

    team = SwarmTeam(
        swarm=swarm,
        members=team_members,
        task_definitions=task_definitions,
        planning_agent=agent_planner,
        stream_handler=SoftwareTeamStreamHandler(),
    )

    context = {
        "platform": "mobile",
        "framework": "flutter",
        "project": {
            "directories": [
                "lib/main.dart",
                "lib/models",
                "lib/models/todo.dart",
                "lib/screens",
                "lib/screens/todo_list.dart",
            ],
            "files": [
                {
                    "filepath": "lib/main.dart",
                    "content": """
                    import 'package:flutter/material.dart';

                    void main() {
                      runApp(MyApp());
                    }

                    class MyApp extends StatelessWidget {
                      @override
                      Widget build(BuildContext context) {
                        return MaterialApp(
                          title: 'Todo App',
                          home: Container(),
                        );
                      }
                    }
                    """,
                }
            ],
        },
    }

    prompt = """Create a Flutter TODO list app with the following features:
    1. Add/edit/delete tasks
    2. Mark tasks as complete
    3. Local storage for persistence
    4. Clean, modern UI design
    """

    while True:
        plan_result = await team.create_plan(prompt, context=context)
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
                prompt = f"{prompt}\n\nFeedback: {feedback}"
            case "3":
                print("Plan rejected. Exiting.")
                return
            case _:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    enable_logging()
    asyncio.run(main())
