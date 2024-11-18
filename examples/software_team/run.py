import asyncio
import os

from liteswarm.logging import enable_logging
from liteswarm.swarm import Swarm
from liteswarm.swarm_team import PlanStatus, SwarmTeam, TeamMember
from liteswarm.types import ContextVariables

from .stream import SoftwareTeamStreamHandler, SwarmStreamHandler
from .team import (
    create_debug_engineer,
    create_flutter_engineer,
    create_planning_strategy,
)

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


def create_task_instructions(context: ContextVariables) -> str:
    """Create dynamic task instructions."""
    task = context["task"]
    execution_history = context.get("execution_history", [])
    project = context.get("project", {})

    # Get relevant files for this task
    relevant_files = []
    if files := project.get("files"):
        for file in files:
            if any(keyword in file["filepath"] for keyword in ["todo", "task", "list"]):
                relevant_files.append(file)

    # Build context-aware instructions
    instructions = f"""
    Implement the following software development task:

    Task Details:
    - ID: {task["id"]}
    - Title: {task["title"]}
    - Description: {task["description"]}

    Project Context:
    - Framework: Flutter
    - Platform: Mobile
    - Project Structure: {', '.join(project.get('directories', []))}
    """

    if relevant_files:
        instructions += "\nRelevant Files:"
        for file in relevant_files:
            instructions += f"\n- {file['filepath']}"

    if execution_history:
        instructions += "\nExecution History:"
        for result in execution_history[-3:]:  # Show last 3 tasks
            instructions += f"\n- {result.task.title} (by {result.assignee.agent.id})"

    instructions += """
    Important:
    1. Follow the existing project structure
    2. Maintain consistency with previous implementations
    3. Use root XML tags to structure your response
    4. Include implementation thoughts before files
    5. Consider the impact on dependent tasks

    Your response must follow this exact format:

    <thoughts>
    Explain your implementation approach here. This section should include:
    - What changes you're making and why
    - Key implementation decisions
    - Any important considerations
    - How this fits with existing code
    The content can be free-form text, formatted for readability.
    </thoughts>

    <files>
    Provide complete file contents in JSON format:
    [
        {
            "filepath": "path/to/file.dart",
            "content": "// Complete file contents here"
        }
    ]
    </files>
    """

    return instructions


async def main() -> None:
    """Run the software team example."""
    swarm = Swarm(stream_handler=SwarmStreamHandler())
    planning_strategy = create_planning_strategy(swarm)
    team_members = [
        TeamMember(
            agent=create_flutter_engineer(),
            task_types=["flutter"],
            metadata={"specialty": "mobile"},
        ),
        TeamMember(
            agent=create_debug_engineer(),
            task_types=["debug"],
            metadata={"specialty": "troubleshooting"},
        ),
    ]

    team = SwarmTeam(
        swarm=swarm,
        members=team_members,
        planning_strategy=planning_strategy,
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
        plan = await team.create_plan(prompt, context=context)

        print("\nReview the plan and choose an option:")
        print("1. Approve and execute")
        print("2. Provide feedback")
        print("3. Reject and exit")

        choice = input("\nYour choice (1-3): ").strip()

        match choice:
            case "1":
                plan.status = PlanStatus.APPROVED
                await team.execute_plan()
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
