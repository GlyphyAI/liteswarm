# LiteSwarm

LiteSwarm is a lightweight, extensible framework for building AI agent systems. It provides a minimal yet powerful foundation for creating both simple chatbots and complex agent teams, with customization possible at every level.

The framework is LLM-agnostic and supports 100+ language models through [litellm](https://github.com/BerriAI/litellm), including:
- OpenAI
- Anthropic (Claude)
- Google (Gemini)
- Azure OpenAI
- AWS Bedrock
- And many more

## Quick Navigation
- [Installation](#installation)
- [Requirements](#requirements)
- [Key Features](#key-features)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Key Concepts](#key-concepts)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

Choose your preferred installation method:

Using pip:
```bash
pip install liteswarm
```

Using uv (recommended for faster installation):
```bash
uv pip install liteswarm
```

Using poetry:
```bash
poetry add liteswarm
```

Using pipx (for CLI tools):
```bash
pipx install liteswarm
```

## Requirements

- Python 3.11 or higher
- Async support (asyncio)
- A valid API key for your chosen LLM provider

### API Keys
You can provide your API key in two ways:
1. Through environment variables:
   ```bash
   # For OpenAI
   export OPENAI_API_KEY=sk-...
   # For Anthropic
   export ANTHROPIC_API_KEY=sk-ant-...
   # For Google
   export GOOGLE_API_KEY=...
   ```

   or using os.environ:
   ```python
   import os

   # For OpenAI
   os.environ["OPENAI_API_KEY"] = "sk-..."
   # For Anthropic
   os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
   # For Google
   os.environ["GOOGLE_API_KEY"] = "..."
   ```

2. Using a `.env` file:
   ```env
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   ```

3. Using the `LLM` class:
   ```python
   from liteswarm.types import LLM

   llm = LLM(
       model="gpt-4o",
       api_key="sk-...", # or api_base, api_version, etc.
   )
   ```

See [litellm's documentation](https://docs.litellm.ai/docs/providers) for a complete list of supported providers and their environment variables.

## Key Features

- **Lightweight Core**: Minimal base implementation that's easy to understand and extend
- **LLM Agnostic**: Support for 100+ language models through litellm
- **Flexible Agent System**: Create agents with custom instructions and capabilities
- **Tool Integration**: Easy integration of Python functions as agent tools
- **Structured Outputs**: Built-in support for validating and parsing agent responses
- **Multi-Agent Teams**: Coordinate multiple specialized agents for complex tasks
- **Streaming Support**: Real-time response streaming with customizable handlers
- **Context Management**: Smart handling of conversation history and context
- **Cost Tracking**: Optional tracking of token usage and API costs

## Basic Usage

### Simple Agent

```python
import asyncio

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent


async def main() -> None:
    # Create an agent
    agent = Agent(
        id="assistant",
        instructions="You are a helpful AI assistant.",
        llm=LLM(
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
        ),
    )

    # Create swarm and execute
    swarm = Swarm()
    result = await swarm.execute(
        agent=agent,
        prompt="Hello!",
    )

    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

### Agent with Tools

```python
import asyncio

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent


async def main() -> None:
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b

    # Create a math agent with tools
    agent = Agent(
        id="math_agent",
        instructions="Use tools for calculations. Never calculate yourself.",
        llm=LLM(
            model="claude-3-5-haiku-20241022",
            tools=[calculate_sum],
            tool_choice="auto",
        ),
    )

    # Create swarm and execute
    swarm = Swarm()
    result = await swarm.execute(
        agent=agent,
        prompt="What is 2 + 2?",
    )

    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Features

### Agent Switching

Agents can dynamically switch to other agents during execution:

```python
import asyncio
import json

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, ToolResult


async def main() -> None:
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # Create a math agent with tools
    math_agent = Agent(
        id="math",
        instructions="You are a math expert.",
        llm=LLM(
            model="gpt-4o",
            tools=[multiply],
            tool_choice="auto",
        ),
    )

    def switch_to_math() -> ToolResult:
        """Switch to math agent for calculations."""
        return ToolResult(
            content="Switching to math expert",
            agent=math_agent,
        )

    # Create the main agent with switch tool
    main_agent = Agent(
        id="assistant",
        instructions="Help users and switch to math agent for calculations.",
        llm=LLM(
            model="gpt-4o",
            tools=[switch_to_math],
            tool_choice="auto",
        ),
    )

    # Agent will automatically switch when needed
    swarm = Swarm()
    result = await swarm.execute(
        agent=main_agent,
        prompt="What is 234 * 567?",
    )

    # Print the full conversation history
    messages = [m.model_dump(exclude_none=True) for m in result.messages]
    print(json.dumps(messages, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

### Agent Teams

The SwarmTeam class (from `liteswarm.experimental`) provides an experimental framework for orchestrating complex agent workflows with automated planning. It follows a two-phase process:

1. **Planning Phase**: 
   - Analyzes the prompt to create a structured plan
   - Breaks down work into specific tasks with dependencies
   - Supports interactive feedback loop for plan refinement
   - Validates task types and team capabilities

2. **Execution Phase**:
   - Executes tasks in dependency order
   - Assigns tasks to capable team members
   - Tracks progress and maintains execution state
   - Produces an artifact with results and updates

Here's a complete example:

```python
import asyncio
import json
from typing import Literal

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.experimental import SwarmTeam
from liteswarm.types import (
    LLM,
    Agent,
    ArtifactStatus,
    ContextVariables,
    Plan,
    PlanFeedbackHandler,
    Task,
    TaskDefinition,
    TeamMember,
)


# 1. Define task types and outputs
class WriteDocTask(Task):
    type: Literal["write_documentation"]
    topic: str
    target_audience: Literal["beginner", "intermediate", "advanced"]


class ReviewDocTask(Task):
    type: Literal["review_documentation"]
    content: str
    criteria: list[str]


class Documentation(BaseModel):
    content: str
    examples: list[str]
    see_also: list[str]


class ReviewFeedback(BaseModel):
    approved: bool
    issues: list[str]
    suggestions: list[str]


# 2. (Optional) Create interactive feedback handler
class InteractiveFeedback(PlanFeedbackHandler):
    async def handle(
        self,
        plan: Plan,
        prompt: str,
        context: ContextVariables | None,
    ) -> tuple[str, ContextVariables | None] | None:
        print("\nProposed plan:")
        for task in plan.tasks:
            print(f"- {task.title}")

        if input("\nApprove? [y/N]: ").lower() != "y":
            return "Please revise the plan", context
        else:
            return None


async def main() -> None:
    # 3. Create task definitions
    def build_write_doc_instructions(
        task: WriteDocTask,
        context: ContextVariables,
    ) -> str:
        return f"""
        Write a {task.target_audience}-level documentation about {task.topic}.

        Style Guide from context:
        {context.style_guide}

        You must return a JSON object that matches the following schema:
        {json.dumps(Documentation.model_json_schema())}
        """

    write_doc = TaskDefinition(
        task_type=WriteDocTask,
        instructions=build_write_doc_instructions,
        response_format=Documentation,
    )

    def build_review_doc_instructions(
        task: ReviewDocTask,
        context: ContextVariables,
    ) -> str:
        return f"""
        Review the following documentation:
        {task.content}

        Review criteria:
        {task.criteria}

        Style Guide to check against:
        {context.style_guide}

        You must return a JSON object that matches the following schema:
        {json.dumps(ReviewFeedback.model_json_schema())}
        """

    review_doc = TaskDefinition(
        task_type=ReviewDocTask,
        instructions=build_review_doc_instructions,
        response_format=ReviewFeedback,
    )

    # 4. Create specialized agents
    writer = Agent(
        id="tech_writer",
        instructions="""You are an expert technical writer who creates clear,
        concise documentation with practical examples.""",
        llm=LLM(
            model="gpt-4o",
            temperature=0.7,
        ),
    )

    reviewer = Agent(
        id="doc_reviewer",
        instructions="""You are a documentation reviewer who ensures accuracy,
        clarity, and completeness of technical documentation.""",
        llm=LLM(
            model="gpt-4o",
            temperature=0.3,  # Lower temperature for more consistent reviews
        ),
    )

    # 5. Create team members
    writer_member = TeamMember(
        id="writer",
        agent=writer,
        task_types=[WriteDocTask],
    )

    reviewer_member = TeamMember(
        id="reviewer",
        agent=reviewer,
        task_types=[ReviewDocTask],
    )

    # 6. Create swarm team
    team = SwarmTeam(
        swarm=Swarm(),
        members=[writer_member, reviewer_member],
        task_definitions=[write_doc, review_doc],
    )

    # 7. Execute the user request
    artifact = await team.execute(
        prompt="Create beginner-friendly documentation about Python list comprehensions",
        context=ContextVariables(
            style_guide="""
            - Use simple language
            - Include practical examples
            - Link to related topics
            - Start with basic concepts
            - Show common patterns
            """
        ),
        feedback_handler=InteractiveFeedback(),
    )

    # 8. Inspect and print the results
    if artifact.status == ArtifactStatus.COMPLETED:
        print("\nDocumentation Team Results:")
        for result in artifact.task_results:
            print(f"\nTask: {result.task.type}")

            if not result.output:
                continue

            match result.output:
                case Documentation() as doc:
                    print("\nContent:")
                    print(doc.content)
                    print("\nExamples:")
                    for example in doc.examples:
                        print(f"• {example}")
                    print("\nSee Also:")
                    for ref in doc.see_also:
                        print(f"• {ref}")

                case ReviewFeedback() as review:
                    print("\nReview Feedback:")
                    print(f"Approved: {review.approved}")
                    if review.issues:
                        print("\nIssues:")
                        for issue in review.issues:
                            print(f"• {issue}")
                    if review.suggestions:
                        print("\nSuggestions:")
                        for suggestion in review.suggestions:
                            print(f"• {suggestion}")


if __name__ == "__main__":
    asyncio.run(main())
```

The SwarmTeam will:
1. Create a plan with appropriate tasks and dependencies
2. Allow plan review/modification through feedback handler
3. Execute tasks in correct order using capable team members
4. Produce an artifact containing all results and updates

See the [software_team example](examples/software_team/run.py) for a complete implementation of a development team workflow.

### Streaming Responses

LiteSwarm supports real-time streaming of responses. Here's a simple example:

```python
import asyncio

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent


async def main():
    # Create an agent
    agent = Agent(
        id="explainer",
        instructions="You are a helpful assistant that explains concepts clearly.",
        llm=LLM(
            model="gpt-4o",
            temperature=0.7,
        ),
    )

    # Create swarm and start streaming
    swarm = Swarm()
    async for response in swarm.stream(
        agent=agent,
        prompt="Explain Python generators in 3-4 bullet points.",
    ):
        content = response.delta.content
        if content is not None:
            print(content, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

### Context Variables

Context variables let you pass data between interactions. Here's a simple example:

```python
import asyncio
import json

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, ContextVariables, ToolResult

mock_database = {
    "alice": {
        "language": "Python",
        "experience": "intermediate",
        "interests": ["web", "data science"],
    }
}


async def main():
    def get_user_preferences(user_id: str) -> ToolResult:
        """Get user preferences from a simulated database."""
        user_preferences = mock_database.get(user_id, {})
        return ToolResult(
            content=f"Found preferences for {user_id}: {user_preferences}",
            context_variables=ContextVariables(
                user_preferences=user_preferences,
                learning_path=[],  # Initialize empty learning path
            ),
        )

    def update_learning_path(topic: str, completed: bool = False) -> ToolResult:
        """Update the user's learning path with a new topic or mark as completed."""
        return ToolResult(
            content=f"{'Completed' if completed else 'Added'} topic: {topic}",
            context_variables=ContextVariables(
                topic=topic,
                completed=completed,
            ),
        )

    # Create an agent with tools
    agent = Agent(
        id="tutor",
        instructions=lambda context_variables: f"""
        You are a programming tutor tracking a student's learning journey.

        Current Context:
        - User ID: {json.dumps(context_variables.get('user_id', 'unknown'))}
        - User Preferences: {json.dumps(context_variables.get('user_preferences', {}))}
        - Learning Path: {json.dumps(context_variables.get('learning_path', []))}
        - Last Topic: {json.dumps(context_variables.get('topic', None))}
        - Last Topic Completed: {json.dumps(context_variables.get('completed', False))}

        Track their progress and suggest next steps based on their preferences and current progress.
        """,
        llm=LLM(
            model="gpt-4o",
            tools=[get_user_preferences, update_learning_path],
            tool_choice="auto",
            temperature=0.3,
        ),
    )

    # Create swarm and execute with initial context
    swarm = Swarm()

    # First interaction - get user preferences
    result = await swarm.execute(
        agent=agent,
        prompt="Start Alice's learning journey",
        context_variables=ContextVariables(user_id="alice"),
    )
    print("\nInitial Setup:", result.content)

    # Second interaction - suggest first topic
    result = await swarm.execute(
        agent=agent,
        prompt="What should Alice learn first?",
        context_variables=result.context_variables,
    )
    print("\nFirst Topic Suggestion:", result.content)

    # Third interaction - mark progress and get next topic
    result = await swarm.execute(
        agent=agent,
        prompt="Alice completed the first topic. What's next?",
        context_variables=result.context_variables,
    )
    print("\nProgress Update:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

### Structured Outputs

LiteSwarm provides two layers of structured output handling:

1. **LLM-level Response Format**:
   - Set via `response_format` in `LLM` class
   - Provider-specific structured output support
   - For OpenAI/Anthropic: Direct JSON schema enforcement
   - For other providers: Manual prompt engineering

2. **Framework-level Response Format**:
   - Set in `TaskDefinition` and `PlanningAgent`
   - Provider-agnostic parsing and validation
   - Supports both Pydantic models and custom parsers
   - Handles response repair and validation

Using Swarm directly with LLM-level response format:

```python
import asyncio

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent

CODE_TO_REVIEW = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a - b
"""


class ReviewOutput(BaseModel):
    issues: list[str]
    approved: bool


async def main() -> None:
    agent = Agent(
        id="reviewer",
        instructions="Review code and provide structured feedback",
        llm=LLM(
            model="gpt-4o",
            response_format=ReviewOutput,  # Direct OpenAI JSON schema support
        ),
    )

    swarm = Swarm()
    result = await swarm.execute(
        agent=agent,
        prompt=f"Review the code and provide structured feedback:\n{CODE_TO_REVIEW}",
    )

    if not result.content:
        print("Agent failed to produce a response")
        return

    # Currently, the content is the raw JSON output from the LLM,
    # so we need to parse it manually using a response_format Pydantic model.
    output = ReviewOutput.model_validate_json(result.content)

    if output.issues:
        print("Issues:")
        for issue in output.issues:
            print(f"- {issue}")

    print(f"\nApproved: {output.approved}")


if __name__ == "__main__":
    asyncio.run(main())
```

Using SwarmTeam with both layers (recommended for complex workflows):

```python
import asyncio
from typing import Literal

from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.experimental import LitePlanningAgent, SwarmTeam
from liteswarm.types import (
    LLM,
    Agent,
    ArtifactStatus,
    ContextVariables,
    Plan,
    Task,
    TaskDefinition,
    TeamMember,
)

CODE_TO_REVIEW = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a - bs  # Bug: Typo in variable name and wrong operator
"""


# 1. Define data structures for the review process
class ReviewTask(Task):
    type: Literal["code-review"]
    code: str
    language: str
    review_type: Literal["general", "security", "performance"]


class CodeReviewOutput(BaseModel):
    issues: list[str]
    approved: bool
    suggested_fixes: list[str]


class CodeReviewPlan(Plan):
    tasks: list[ReviewTask]


# 2. Create prompt builders
def build_review_prompt(prompt: str, context: ContextVariables) -> str:
    return f"""
    You're given the following user request:
    <request>
    {prompt}
    </request>

    Here is the code to review:
    <code language="{context.get('language', '')}" review_type="{context.get('review_type', '')}">
    {context.get('code', '')}
    </code>

    Please create a review plan consisting of 1 task.
    """.strip()


async def main() -> None:
    # 3. Create task definitions
    review_def = TaskDefinition(
        task_type=ReviewTask,
        instructions=lambda task, _: f"""
        Review the provided code focusing on {task.review_type} aspects.
        <code language="{task.language}">{task.code}</code>
        """,
        response_format=CodeReviewOutput,
    )

    # 4. Create agents
    review_agent = Agent(
        id="code-reviewer",
        instructions="You are an expert code reviewer.",
        llm=LLM(model="gpt-4o", response_format=CodeReviewOutput),
    )

    planning_agent = Agent(
        id="planning-agent",
        instructions="You are a planning agent that creates plans for code review tasks.",
        llm=LLM(model="gpt-4o", response_format=CodeReviewPlan),
    )

    # 5. Create team members
    review_member = TeamMember(
        id="senior-reviewer",
        agent=review_agent,
        task_types=[ReviewTask],
    )

    # 6. Set up swarm team
    swarm = Swarm()
    team = SwarmTeam(
        swarm=swarm,
        members=[review_member],
        task_definitions=[review_def],
        planning_agent=LitePlanningAgent(
            swarm=swarm,
            agent=planning_agent,
            prompt_template=build_review_prompt,
            task_definitions=[review_def],
            response_format=CodeReviewPlan,
        ),
    )

    # 7. Execute review request
    artifact = await team.execute(
        prompt="Review this Python code",
        context=ContextVariables(
            code=CODE_TO_REVIEW,
            language="python",
            review_type="general",
        ),
    )

    # 8. Show results
    if artifact.status == ArtifactStatus.COMPLETED:
        for result in artifact.task_results:
            if isinstance(result.output, CodeReviewOutput):
                assert result.assignee is not None
                print(f"\nReview by: {result.assignee.id}")
                print("\nIssues found:")
                for issue in result.output.issues:
                    print(f"- {issue}")
                print("\nSuggested fixes:")
                for fix in result.output.suggested_fixes:
                    print(f"- {fix}")
                print(f"\nApproved: {result.output.approved}")


if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates:

1. **LLM-level Format** (Provider-specific):
   - `response_format=CodeReviewOutput` in review agent's LLM
   - `response_format=CodeReviewPlan` in planning agent's LLM
   - OpenAI will enforce JSON schema at generation time

2. **Framework-level Format** (Provider-agnostic):
   - `response_format=CodeReviewOutput` in task definition
   - `response_format=CodeReviewPlan` in planning agent
   - Framework handles parsing, validation, and repair

The two-layer approach ensures:
- Structured outputs work with any LLM provider
- Automatic parsing and validation
- Consistent interface across providers
- Fallback to prompt-based formatting
- Response repair capabilities

See [examples/structured_outputs/run.py](examples/structured_outputs/run.py) for more examples of different structured output strategies.

> **Note about OpenAI Structured Outputs**
> 
> OpenAI's JSON schema support has certain limitations:
> - No default values in Pydantic models
> - No `oneOf` in union types (must use discriminated unions)
> - Some advanced Pydantic features may not be supported
>
> While LiteSwarm's base `Task` and `Plan` types are designed to be OpenAI-compatible, this compatibility must be maintained by users when subclassing these types. For example:
>
> ```python
> # OpenAI-compatible task type
> class ReviewTask(Task):
>     type: Literal["code-review"]  # Discriminator field
>     code: str                     # Required field, no default
>     language: str                 # Required field, no default
>     
>     # Not OpenAI-compatible - has default value
>     review_type: str = "general"  # Will work with other providers
> ```
>
> We provide utilities to help maintain compatibility:
> - `liteswarm.utils.pydantic` module contains helpers for:
>   - Converting Pydantic schemas to OpenAI format
>   - Restoring objects from OpenAI responses
>   - Handling schema transformations
>
> See [examples/structured_outputs/strategies/openai_pydantic.py](examples/structured_outputs/strategies/openai_pydantic.py) for practical examples of using these utilities.
>
> Remember: Base `Task` and `Plan` are OpenAI-compatible, but maintaining compatibility in subclasses is the user's responsibility if OpenAI structured outputs are needed.

## Key Concepts

1. **Agent**: An AI entity with specific instructions and capabilities
2. **Tool**: A Python function that an agent can call
3. **Swarm**: Orchestrator for agent interactions and conversations
4. **SwarmTeam**: Coordinator for multiple specialized agents
5. **Context Variables**: Dynamic data passed to agents and tools
6. **Stream Handler**: Interface for real-time response processing

## Best Practices

1. Use `ToolResult` for wrapping tool return values:
   ```python
   def my_tool() -> ToolResult:
       return ToolResult(
           content="Result",
           context_variables=ContextVariables(...)
       )
   ```

2. Implement proper error handling:
   ```python
   try:
       result = await team.execute_task(task)
   except TaskExecutionError as e:
       logger.error(f"Task failed: {e}")
   ```

3. Use context variables for dynamic behavior:
   ```python
   def build_instructions(context: ContextVariables) -> str:
       return f"Help {context['user_name']} with {context['task']}"
   ```

4. Leverage streaming for real-time feedback:
   ```python
   class MyStreamHandler(SwarmStreamHandler):
       async def on_stream(self, delta: Delta, agent: Agent) -> None:
           print(delta.content, end="", flush=True)
   ```

## Examples

The framework includes several example applications in the [examples/](examples/) directory:

- **Basic REPL** ([examples/repl/run.py](examples/repl/run.py)): Simple interactive chat interface showing basic agent usage
- **Calculator** ([examples/calculator/run.py](examples/calculator/run.py)): Tool usage and agent switching with a math-focused agent
- **Mobile App Team** ([examples/mobile_app/run.py](examples/mobile_app/run.py)): Complex team of agents (PM, Designer, Engineer, QA) building a Flutter app
- **Parallel Research** ([examples/parallel_research/run.py](examples/parallel_research/run.py)): Parallel tool execution for efficient data gathering
- **Structured Outputs** ([examples/structured_outputs/run.py](examples/structured_outputs/run.py)): Different strategies for parsing structured agent responses
- **Software Team** ([examples/software_team/run.py](examples/software_team/run.py)): Complete development team with planning, review, and implementation capabilities

Each example demonstrates different aspects of the framework:
```bash
# Run the REPL example
python -m examples.repl.run

# Run the calculator example
python -m examples.calculator.run

# Try the mobile app team
python -m examples.mobile_app.run

# Run the parallel research example
python -m examples.parallel_research.run

# Experiment with structured outputs
python -m examples.structured_outputs.run

# Run the software team example
python -m examples.software_team.run
```

## Contributing

We welcome contributions to LiteSwarm! We're particularly interested in:

1. **Adding Tests**: We currently have minimal test coverage and welcome contributions to:
   - Add unit tests for core functionality
   - Add integration tests for agent interactions
   - Add example-based tests for common use cases
   - Set up testing infrastructure and CI

2. **Bug Reports**: Open an issue describing:
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Your environment details
   - Any relevant code snippets

3. **Feature Requests**: Open an issue describing:
   - The use case for the feature
   - Expected behavior
   - Example code showing how it might work

4. **Code Contributions**: 
   - Fork the repository
   - Create a new branch for your feature
   - Include tests for new functionality
   - Submit a pull request with a clear description
   - Ensure CI passes and code follows our style guide

### Development setup:

```bash
# Clone the repository
git clone https://github.com/your-org/liteswarm.git
cd liteswarm

# Create virtual environment (choose one)
python -m venv .venv
# or
poetry install
# or
uv venv

# Install development dependencies
uv pip install -e ".[dev]"
# or
poetry install --with dev

# Run existing tests (if any)
pytest

# Run type checking
mypy .

# Run linting
ruff check .
```

### Code Style
- We use ruff for linting and formatting
- Type hints are required for all functions
- Docstrings should follow Google style
- New features should include tests

### Testing Guidelines
We're building our test suite and welcome contributions that:
- Add pytest-based tests
- Include both unit and integration tests
- Cover core functionality
- Demonstrate real-world usage
- Help improve test coverage
- Set up testing infrastructure

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code changes that neither fix bugs nor add features

## Citation

If you use LiteSwarm in your research or project, please cite our work:

```bibtex
@software{mozharovskii_2024_liteswarm,
    title = {{LiteSwarm: A Lightweight Framework for Building AI Agent Systems}},
    author = {Mozharovskii, Evgenii and {GlyphyAI}},
    year = {2024},
    url = {https://github.com/glyphyai/liteswarm},
    license = {MIT},
    version = {0.1.1}
}
```

## License

MIT License - see LICENSE file for details.
