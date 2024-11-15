import asyncio
import os

from liteswarm.logging import enable_logging
from liteswarm.swarm import Swarm
from liteswarm.swarm_team import SwarmTeam, TeamMember

from .stream import SoftwareTeamStreamHandler
from .team import (
    create_debug_engineer,
    create_flutter_engineer,
    create_planner,
)
from .types import SoftwarePlan, SoftwareTask

os.environ["LITESWARM_LOG_LEVEL"] = "DEBUG"


def print_plan_created(plan: SoftwarePlan) -> None:
    """Print the created development plan."""
    print("\nDevelopment Plan Created:")
    print("-------------------------")
    for task in plan.tasks:
        print(f"\nTask: {task.title}")
        print(f"Engineer: {task.engineer_type}")
        print(f"Description: {task.description}")
        if task.dependencies:
            print(f"Dependencies: {', '.join(task.dependencies)}")
        if task.deliverables:
            print("Deliverables:")
            for deliverable in task.deliverables:
                print(f"- {deliverable}")


def print_task_started(task: SoftwareTask) -> None:
    """Print when a task is started."""
    print(f"\nStarting Task: {task.title}")
    print(f"Assigned to: {task.assignee}")


def print_task_completed(task: SoftwareTask) -> None:
    """Print when a task is completed."""
    print(f"\nCompleted Task: {task.title}")


def print_plan_completed(plan: SoftwarePlan) -> None:
    """Print when the plan is completed."""
    print("\nPlan Completed!")
    print("All tasks have been executed successfully.")


async def main() -> None:
    """Run the software team example."""
    swarm = Swarm(stream_handler=SoftwareTeamStreamHandler())
    planner = create_planner(swarm=swarm)
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
        planner=planner,
        members=team_members,
        swarm=swarm,
        on_plan_created=print_plan_created,
        on_task_started=print_task_started,
        on_task_completed=print_task_completed,
        on_plan_completed=print_plan_completed,
    )

    context = {
        "platform": "mobile",
        "framework": "flutter",
        "files": [
            "lib/main.dart",
            "lib/models/todo.dart",
            "lib/screens/todo_list.dart",
        ],
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
                plan.status = "approved"
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
