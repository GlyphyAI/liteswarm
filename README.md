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
from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent

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
```

### Agent with Tools

```python
from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


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
```

## Advanced Features

### Agent Switching

Agents can dynamically switch to other agents during execution:

```python
from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, ToolResult

# Create specialized agents
math_agent = Agent(
    id="math",
    instructions="You are a math expert.",
    llm=LLM(model="gpt-4o"),
)

def switch_to_math() -> ToolResult:
    """Switch to math agent for calculations."""
    return ToolResult(
        content="Switching to math expert",
        agent=math_agent,
    )

# Create main agent with switching capability
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


# 1. Define task types
class ReviewTask(Task):
    pr_url: str
    review_type: str  # "security", "performance", etc.


class ImplementTask(Task):
    feature_name: str
    requirements: list[str]


# 2. Create task definitions with instructions
review_def = TaskDefinition(
    task_schema=ReviewTask,
    task_instructions="Review {task.pr_url} focusing on {task.review_type} aspects.",
)

implement_def = TaskDefinition(
    task_schema=ImplementTask,
    task_instructions="Implement {task.feature_name} following requirements:\n{task.requirements}",
)

# 3. Create specialized agents
review_agent = Agent(
    id="reviewer",
    instructions="You are a code reviewer focusing on quality and security.",
    llm=LLM(model="gpt-4o"),
)

dev_agent = Agent(
    id="developer",
    instructions="You are a developer implementing new features.",
    llm=LLM(model="gpt-4o"),
)

# 4. Create team members with capabilities
team_members = [
    TeamMember(
        id="senior-reviewer",
        agent=review_agent,
        task_types=[ReviewTask],
    ),
    TeamMember(
        id="backend-dev",
        agent=dev_agent,
        task_types=[ImplementTask],
    ),
]

# 5. Create the team
swarm = Swarm(include_usage=True)
team = SwarmTeam(
    swarm=swarm,
    members=team_members,
    task_definitions=[review_def, implement_def],
)


# 6. Optional: Add plan feedback handler
class InteractiveFeedback(PlanFeedbackHandler):
    async def handle(
        self,
        plan: Plan,
        prompt: str,
        context: ContextVariables | None,
    ) -> tuple[str, ContextVariables | None] | None:
        """Allow user to review and modify the plan before execution."""
        print("\nProposed plan:")
        for task in plan.tasks:
            print(f"- {task.title}")

        if input("\nApprove? [y/N]: ").lower() != "y":
            return "Please revise the plan", context
        else:
            return None


# 7. Execute workflow with planning
artifact = await team.execute(
    prompt="Implement a login feature and review it for security",
    context=ContextVariables(
        pr_url="github.com/org/repo/123",
        security_checklist=["SQL injection", "XSS", "CSRF"],
    ),
    feedback_handler=InteractiveFeedback(),
)

# 8. Check results
if artifact.status == ArtifactStatus.COMPLETED:
    print("Tasks completed:")
    for result in artifact.task_results:
        print(f"- {result.task.title}: {result.task.status}")
```

The SwarmTeam will:
1. Create a plan with appropriate tasks and dependencies
2. Allow plan review/modification through feedback handler
3. Execute tasks in correct order using capable team members
4. Produce an artifact containing all results and updates

See `examples/software_team/run.py` for a complete implementation of a development team.

### Streaming Responses

```python
async for response in swarm.stream(
    agent=agent,
    prompt="Generate a long response...",
):
    print(response.content, end="", flush=True)
```

### Context Variables

```python
result = await swarm.execute(
    agent=agent,
    prompt="Greet the user",
    context_variables=ContextVariables(
        user_name="Alice",
        language="en",
    ),
)
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
from pydantic import BaseModel

from liteswarm.core.swarm import Swarm
from liteswarm.types import LLM, Agent


class ReviewOutput(BaseModel):
    issues: list[str]
    approved: bool


agent = Agent(
    id="reviewer",
    instructions="Review code and provide structured feedback",
    llm=LLM(
        model="gpt-4o",
        response_format=ReviewOutput,  # Direct OpenAI JSON schema support
    ),
)

code = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a - b
"""

swarm = Swarm()
result = await swarm.execute(
    agent=agent,
    prompt=f"Review the code and provide structured feedback:\n{code}",
)

# Currently, the content is the raw JSON output from the LLM,
# so we need to parse it manually using a response_format Pydantic model.
output = ReviewOutput.model_validate_json(result.content)

if output.issues:
    print("Issues:")
    for issue in output.issues:
        print(f"- {issue}")

print(f"\nApproved: {output.approved}")
```

Using SwarmTeam with both layers (recommended for complex workflows):

```python
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


# Define output schema for code reviews
class CodeReviewOutput(BaseModel):
    issues: list[str]
    approved: bool
    suggested_fixes: list[str]


# Define task type with literal constraints
class ReviewTask(Task):
    type: Literal["code-review"]
    code: str
    language: str
    review_type: Literal["general", "security", "performance"]


# Define plan schema for planning agent
class CodeReviewPlan(Plan):
    tasks: list[ReviewTask]


# Create dynamic task instructions
def build_review_task_instructions(task: ReviewTask, context: ContextVariables) -> str:
    prompt = (
        "Review the provided code focusing on {task.review_type} aspects.\n"
        "Code to review:\n{task.code}"
    )
    return prompt.format(task=task)


# Create task definition with response format
review_def = TaskDefinition(
    task_schema=ReviewTask,
    task_instructions=build_review_task_instructions,
    # Framework-level: Used to parse and validate responses
    task_response_format=CodeReviewOutput,
)

# Create review agent with LLM-level response format
review_agent = Agent(
    id="code-reviewer",
    instructions="You are an expert code reviewer.",
    llm=LLM(
        model="gpt-4o",
        # LLM-level: Direct OpenAI JSON schema support
        response_format=CodeReviewOutput,
    ),
)

# Create planning agent with LLM-level response format
planning_agent = Agent(
    id="planning-agent",
    instructions="You are a planning agent that creates plans for code review tasks.",
    llm=LLM(
        model="gpt-4o",
        # LLM-level: Direct OpenAI JSON schema support
        response_format=CodeReviewPlan,
    ),
)


# Create dynamic planning prompt
PLANNING_PROMPT_TEMPLATE = """
User Request:
<request>{PROMPT}</request>

Code Context:
<code language="{LANGUAGE}" review_type="{REVIEW_TYPE}">
{CODE}
</code>

Please create a review plan consisting of 1 task.
""".strip()


def build_planning_prompt_template(prompt: str, context: ContextVariables) -> str:
    code = context.get("code", "")
    language = context.get("language", "")
    review_type = context.get("review_type", "")

    return PLANNING_PROMPT_TEMPLATE.format(
        PROMPT=prompt,
        CODE=code,
        LANGUAGE=language,
        REVIEW_TYPE=review_type,
    )


# Create team with both layers of structured outputs
swarm = Swarm()
team = SwarmTeam(
    swarm=swarm,
    members=[
        TeamMember(
            id="senior-reviewer",
            agent=review_agent,
            task_types=[ReviewTask],
        ),
    ],
    task_definitions=[review_def],
    planning_agent=LitePlanningAgent(
        swarm=swarm,
        agent=planning_agent,
        prompt_template=build_planning_prompt_template,
        task_definitions=[review_def],
        # Framework-level: Used to parse planning responses
        response_format=CodeReviewPlan,
    ),
)

# Execute review
code = """
def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a - bs
"""

artifact = await team.execute(
    prompt="Review this Python code",
    context=ContextVariables(
        code=code,
        language="python",
        review_type="general",
    ),
)

# Access structured output
if artifact.status == ArtifactStatus.COMPLETED:
    for result in artifact.task_results:
        # Output is automatically parsed into CodeReviewOutput
        output = result.output
        if not isinstance(output, CodeReviewOutput):
            raise TypeError(f"Unexpected output type: {type(output)}")

        print(f"\nReview by: {result.assignee.id}")
        print("\nIssues found:")
        for issue in output.issues:
            print(f"- {issue}")
        print("\nSuggested fixes:")
        for fix in output.suggested_fixes:
            print(f"- {fix}")
        print(f"\nApproved: {output.approved}")
```

This example demonstrates:

1. **LLM-level Format** (Provider-specific):
   - `response_format=CodeReviewOutput` in review agent's LLM
   - `response_format=CodeReviewPlan` in planning agent's LLM
   - OpenAI will enforce JSON schema at generation time

2. **Framework-level Format** (Provider-agnostic):
   - `task_response_format=CodeReviewOutput` in task definition
   - `response_format=CodeReviewPlan` in planning agent
   - Framework handles parsing, validation, and repair

The two-layer approach ensures:
- Structured outputs work with any LLM provider
- Automatic parsing and validation
- Consistent interface across providers
- Fallback to prompt-based formatting
- Response repair capabilities

See `examples/structured_outputs/run.py` for more examples of different structured output strategies.

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
> See `examples/structured_outputs/strategies/openai_pydantic.py` for practical examples of using these utilities.
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
   def get_instructions(context: ContextVariables) -> str:
       return f"Help {context['user_name']} with {context['task']}"
   ```

4. Leverage streaming for real-time feedback:
   ```python
   class MyStreamHandler(SwarmStreamHandler):
       async def on_stream(self, delta: Delta, agent: Agent) -> None:
           print(delta.content, end="")
   ```

## Examples

The framework includes several example applications in the `examples/` directory:

- **Basic REPL** (`examples/repl/run.py`): Simple interactive chat interface showing basic agent usage
- **Calculator** (`examples/calculator/run.py`): Tool usage and agent switching with a math-focused agent
- **Mobile App Team** (`examples/mobile_app/run.py`): Complex team of agents (PM, Designer, Engineer, QA) building a Flutter app
- **Parallel Research** (`examples/parallel_research/run.py`): Parallel tool execution for efficient data gathering
- **Structured Outputs** (`examples/structured_outputs/run.py`): Different strategies for parsing structured agent responses
- **Software Team** (`examples/software_team/run.py`): Complete development team with planning, review, and implementation capabilities

Each example demonstrates different aspects of the framework:
```bash
# Run the REPL example
python -m examples.repl.run

# Try the mobile app team
python -m examples.mobile_app.run

# Experiment with structured outputs
python -m examples.structured_outputs.run
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
@software{liteswarm2024,
  title = {LiteSwarm: A Lightweight Framework for Building AI Agent Systems},
  author = {Evgenii Mozharovskii and {GlyphyAI}},
  year = {2024},
  url = {https://github.com/glyphyai/liteswarm},
  note = {An extensible framework for building AI agent systems with a focus on 
          structured interactions, LLM-agnostic design, and composable architecture. 
          Features include multi-agent orchestration, structured outputs, and 
          provider-agnostic response parsing.}
}
```

## License

MIT License - see LICENSE file for details.