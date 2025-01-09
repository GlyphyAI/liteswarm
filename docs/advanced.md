# Advanced Features

Welcome to LiteSwarm's advanced features documentation. This guide explores the powerful capabilities that enable you to build sophisticated AI applications. Whether you're orchestrating complex agent workflows, handling structured outputs, or building stateful chat applications, you'll find detailed explanations and practical examples here.

## Table of Contents

1. [SwarmTeam](#swarmteam)
   - [Core Concepts](#core-concepts)
   - [Execution Flow](#execution-flow)
   - [Stateful Chat Integration](#stateful-chat-integration)

2. [Structured Outputs](#structured-outputs)
   - [Basic Usage](#basic-usage)
   - [Streaming and Parsing](#streaming-and-parsing-behavior)
   - [Agent Switching and Type Safety](#agent-switching-and-type-safety)
   - [Provider Support](#provider-support)
   - [Output Strategies](#output-strategies)

3. [Error Handling](#error-handling)
   - [Framework-Handled Errors](#framework-handled-errors)
   - [Developer-Handled Errors](#developer-handled-errors)
   - [Error Documentation](#error-documentation)

4. [Context Variables](#context-variables)
   - [Basic Usage](#basic-usage-1)
   - [Components Using Context](#components-using-context)
   - [Context Persistence](#context-persistence-and-passing)
   - [Best Practices](#best-practices)

5. [Tool Use and Agent Switching](#tool-use-and-agent-switching)
   - [Basic Tool Use](#basic-tool-use)
   - [Context Updates via Tools](#context-updates-via-tools)
   - [Agent Switching](#agent-switching)
   - [Building Agent Networks](#building-agent-networks)

6. [Chat API](#chat-api)
   - [Core Components](#core-components)
   - [Modular Architecture](#modular-architecture)
   - [Team Chat Support](#team-chat-support)

7. [Building with LiteSwarm](#building-with-liteswarm)

## SwarmTeam

SwarmTeam is an experimental orchestration framework for building collaborative agent workflows. It provides a structured way to:
- Define tasks and their dependencies
- Assign specialized agents to specific tasks
- Create execution plans as DAG graphs
- Share context and results between agents

This approach enhances LLMs' ability to solve complex problems by breaking them down into manageable tasks and leveraging specialized expertise.

### Core Concepts

1. **Tasks**: Define structured work units with clear inputs and outputs:
```python
from typing import Literal
from pydantic import BaseModel
from liteswarm.types import Task

class SoftwareTask(Task):
    type: Literal["software_task"]
    user_story: str
    language: str

class SoftwareTaskOutput(BaseModel):
    thoughts: list[str]
    filepath: str
    code: str
```

2. **Team Members**: Specialized agents that can handle specific task types:
```python
engineer_agent = Agent(
    id="engineer",
    instructions="You are a software engineer...",
    llm=LLM(model="gpt-4o", response_format=SoftwareTaskOutput),
)

engineer_member = TeamMember.from_agent(
    engineer_agent,
    task_types=[SoftwareTask],  # Tasks this member can handle
)
```

3. **Task Definitions**: Execution rules and output validation:
```python
task_def = TaskDefinition(
    task_type=SoftwareTask,
    instructions=task_prompt_builder,  # Function or template string
    response_format=SoftwareTaskOutput,
)
```

4. **Team Creation**: Assemble team with members and task definitions:
```python
swarm_team = SwarmTeam(
    swarm=swarm,
    members=[engineer_member],
    task_definitions=[task_def],
)
```

### Execution Flow

1. **Plan Creation**: Generate a DAG of tasks to solve the problem:
```python
# Create execution plan
create_plan_stream = swarm_team.create_plan(messages)
plan_result = await create_plan_stream.get_return_value()

# Plan defines task dependencies and execution order
print(plan_result.plan.tasks)  # List of tasks in execution order
```

2. **Plan Execution**: Execute tasks while sharing context:
```python
# Execute plan with context sharing
execute_plan_stream = swarm_team.execute_plan(plan_result.plan, messages)

# Stream execution progress
async for event in execute_plan_stream:
    if event.type == "task_started":
        print(f"Starting task: {event.task.type}")
    elif event.type == "task_completed":
        print(f"Completed task: {event.task.type}")
        # Access task results for context in next tasks
        print(event.result.parsed)

# Get final execution result
artifact = await execute_plan_stream.get_return_value()
```

### Stateful Chat Integration

While SwarmTeam itself is stateless, we provide `LiteTeamChat` for maintaining conversation state:

```python
from liteswarm.experimental import LiteTeamChat

# Create stateful team chat
team_chat = LiteTeamChat(swarm_team=swarm_team)

# Send message and get response with state management
async for event in team_chat.send_message("Create a Python web app"):
    if event.type == "task_started":
        print(f"Team member {event.member.id} working on {event.task.type}")
```

For a complete example showing how to build a software development team, see [swarm_team_basic example](../examples/swarm_team_basic/run.py).

## Structured Outputs

LiteSwarm provides robust support for structured outputs through Pydantic models. The framework handles both streaming partial JSON parsing and final validation to ensure type safety.

### Basic Usage

To get structured outputs, pass a Pydantic model to the LLM's `response_format`:

```python
from pydantic import BaseModel

from liteswarm.core import Swarm
from liteswarm.types import LLM, Agent, Message


class MathResult(BaseModel):
    thoughts: str
    result: int


agent = Agent(
    id="math_expert",
    instructions="You are a math expert.",
    llm=LLM(
        model="gpt-4o",
        response_format=MathResult,  # LLM will generate JSON matching this schema
    ),
)

swarm = Swarm()
stream = swarm.stream(
    agent,
    messages=[Message(role="user", content="What is 2 + 2 * 2?")],
    response_format=MathResult,  # Ensures type-safe execution result
)
```

### Streaming and Parsing Behavior

LiteSwarm handles JSON parsing in two stages:

1. **Streaming Chunks**: During streaming, each chunk contains:
   - Raw response snapshot
   - Partially parsed JSON objects (may not fully satisfy schema)
   ```python
   async for event in stream:
       if event.type == "agent_response_chunk":
           # Access partial parsed JSON object
           if event.response_chunk.parsed:
               partial = event.response_chunk.parsed  # Valid JSON object
   ```

2. **Final Validation**: After streaming completes:
   - Full response is collected and validated
   - Guaranteed to match provided Pydantic model
   ```python
   result = await stream.get_return_value()
   if result.agent_response.parsed:
       math_result = result.agent_response.parsed
       print(f"Result: {math_result.result}")  # MathResult instance
   ```

### Agent Switching and Type Safety

When using tools that can switch agents, LiteSwarm maintains type safety:

```python
# Execution result contains multiple agent responses
result = await swarm.execute(
    agent,
    messages,
    response_format=MathResult,  # Applied to final agent response
)

# Access final response with guaranteed type
if result.agent_response.parsed:
    math_result = result.agent_response.parsed

# Access all responses if needed
for response in result.responses:
    if response.parsed:
        # Type depends on agent's LLM response_format
        print(response.parsed)
```

### Chat API Integration

The Chat API provides the same type-safe structured outputs:

```python
from liteswarm.chat import LiteChat

chat = LiteChat()
stream = chat.send_message(
    "Calculate 2 + 2",
    agent=math_agent,
    response_format=MathResult,  # Type-safe chat responses
)

result = await stream.get_return_value()
if result.agent_response.parsed:
    math_result = result.agent_response.parsed
```

### Provider Support

Not all LLM providers support structured outputs. When using providers that do (like OpenAI):

1. The LLM will generate responses matching your schema
2. Streaming partial parsing is handled automatically
3. Final validation ensures type safety

For providers without schema support, you may need to:
- Include schema in prompts
- Handle parsing errors gracefully
- Use more robust validation

For OpenAI compatibility, see [Provider Limitations](#provider-limitations) below.

### Complete Examples

For complete working examples of structured outputs:

- Core API: [examples/structured_outputs/core](../examples/structured_outputs/core/run.py)
- Chat API: [examples/structured_outputs/chat](../examples/structured_outputs/chat/run.py)

### Provider Limitations

Not all LLM providers support JSON schemas or Pydantic models. Some have specific requirements:

- OpenAI doesn't allow models with default values
- Some providers require specific JSON formatting rules
- Streaming support varies by provider

For OpenAI compatibility, LiteSwarm provides helper methods:
```python
from liteswarm.utils.pydantic import remove_default_values, restore_default_values

# Remove defaults before passing to OpenAI
schema_no_defaults = remove_default_values(YourModel)
# Restore defaults after receiving response
restored_output = restore_default_values(response, YourModel)
```

### Output Strategies

LiteSwarm's built-in support for structured outputs focuses on two key features:

1. **LLM Response Format**
   - Pass Pydantic models to `response_format` in LLM configuration
   - Automatic JSON schema generation and validation
   - Streaming-aware partial JSON parsing

2. **Execution Result Format**
   - Type-safe validation via `response_format` in Swarm methods
   - Guaranteed parsed type in final execution results
   - Consistent across Core and Chat APIs

While these cover most use cases, we provide examples of alternative strategies in our playground:

```python
# Example: Prompt engineering strategy
instructions = """
Generate a response in the following JSON format:
{
    "thoughts": "string",  # Your reasoning
    "result": "number"     # Final calculation
}

Rules:
1. Use valid JSON syntax
2. Include both fields
3. Ensure result is a number
"""

# Example: XML + JSON hybrid strategy
instructions = """
Structure your response as follows:

<thoughts>
Your step-by-step reasoning in natural language
</thoughts>

<result>
{
    "calculation": number,
    "confidence": number
}
</result>
"""
```

These examples demonstrate different approaches, but note that they are not part of LiteSwarm's core functionality. For production use, we recommend using the built-in `response_format` support when possible.

For practical examples of different strategies and handling provider limitations, see our [structured_outputs/playground](../examples/structured_outputs/playground) example.

## Error Handling

LiteSwarm provides comprehensive error handling through specialized error types. While some errors are handled automatically by the framework, others require developer attention.

### Framework-Handled Errors

Some errors are automatically handled by LiteSwarm components:

1. **Swarm**:
   - Handles response continuation when output reaches max tokens
   - Manages agent switching and tool execution errors
   ```python
   # Automatic handling of max tokens
   result = await swarm.execute(agent, long_prompt)
   # If response hits max tokens, Swarm will:
   # 1. Detect truncation
   # 2. Continue generating
   # 3. Combine responses
   ```

2. **SwarmTeam**:
   - Repairs malformed JSON responses using repair agents
   - Manages task dependencies and execution order
   ```python
   # Automatic JSON repair
   result = await team.execute_task(task)
   # If JSON is invalid, SwarmTeam will:
   # 1. Detect parsing error
   # 2. Deploy repair agent
   # 3. Fix and validate response
   ```

### Developer-Handled Errors

Other errors require developer intervention:

```python
from liteswarm.types import (
    SwarmError,           # Base error class
    CompletionError,      # API/provider errors
    ContextLengthError,   # Context window exceeded
    MaxAgentSwitchesError,# Too many agent switches
    RetryError,          # Retry mechanism failed
)

try:
    result = await swarm.execute(agent, prompt)
except ContextLengthError as e:
    # Handle context length exceeded
    print(f"Context too long: {e.current_length} > {e.max_length}")
    # Developer should optimize context:
    # 1. Reduce message history
    # 2. Summarize context
    # 3. Use a model with larger context
except CompletionError as e:
    # Handle API errors
    print(f"API error: {e}")
    if e.original_error:
        print(f"Original error: {e.original_error}")
    # Developer should:
    # 1. Check API keys/quotas
    # 2. Implement fallback strategy
    # 3. Handle rate limits
except MaxAgentSwitchesError as e:
    # Handle excessive agent switching
    print(f"Too many switches: {e.switch_count}")
    print(f"Switch history: {e.switch_history}")
    # Developer should:
    # 1. Review agent instructions
    # 2. Adjust max switches limit
    # 3. Implement circuit breaker
except SwarmError as e:
    # Handle other Swarm errors
    print(f"Other error: {e}")
```

### Error Documentation

All LiteSwarm methods include detailed docstrings specifying potential errors:

```python
async def execute(
    self,
    agent: Agent,
    prompt: str,
) -> AgentExecutionResult:
    """Execute prompt with agent.

    Raises:
        CompletionError: If API call fails
        ContextLengthError: If context exceeds limit
        MaxAgentSwitchesError: If too many switches occur
    """
```

For a complete list of error types and their usage, see the [exceptions module](../liteswarm/types/exceptions.py).

## Context Variables

`ContextVariables` is a special mapping type for passing execution-specific context throughout the LiteSwarm ecosystem. It allows you to share data with various components during execution.

### Basic Usage

All entry methods accept `context_variables`:

```python
from liteswarm.types import ContextVariables, Message

# Create context with execution-specific data
context = ContextVariables(
    api_key="sk-...",
    database_connection=db_pool,
    user_preferences={
        "language": "Python",
        "style": "functional",
    },
)

# Pass to Swarm execution
result = await swarm.execute(
    agent=agent,
    messages=[Message(role="user", content="Hello!")],
    context_variables=context,
)

# Pass to Chat API
chat = LiteChat()
await chat.send_message(
    "Hello!",
    agent=agent,
    context_variables=context,
)

# Pass to SwarmTeam
team = SwarmTeam(swarm=swarm, members=members)
await team.execute_plan(
    plan=plan,
    messages=messages,
    context_variables=context,
)
```

### Components Using Context

The context variables are available to several components:

1. **Tool Functions**:
   ```python
   def fetch_data(
       query: str,
       context_variables: ContextVariables,  # Automatically injected
   ) -> dict:
       # Access API key from context
       api_key = context_variables.get("api_key")
       # Use connection from context
       db = context_variables.get("database_connection")
       return db.execute(query)
   ```

2. **Prompt Builders**:
   ```python
   def build_instructions(
       context_variables: ContextVariables,
   ) -> str:
       # Dynamic instructions based on context
       language = context_variables.get("user_preferences", {}).get("language", "Python")
       return f"""You are an expert {language} developer.
       Follow these style guidelines:
       {context_variables.get('style_guide')}
       """

   agent = Agent(
       id="developer",
       instructions=build_instructions,  # Will receive context_variables
       llm=LLM(model="gpt-4o"),
   )
   ```

3. **SwarmTeam Response Parsers**:
   ```python
   def parse_task_result(
       result: str,
       context_variables: ContextVariables,
   ) -> TaskOutput:
       # Use context-specific parsing rules
       parser = context_variables.get("custom_parser")
       return parser.parse(result)

   task_def = TaskDefinition(
       task_type=Task,
       response_parser=parse_task_result,  # Will receive context_variables
   )
   ```

### Context Persistence and Passing

Important notes about context variables behavior:

1. **No Persistence in Core API**:
   - Context variables are not stored between executions
   - Each execution requires passing fresh context
   - Use Chat API or implement your own storage if persistence needed

2. **Context Passing**:
   - Context variables from entry method are passed to all receivers
   - Tools, prompt builders, and parsers receive the same context
   - Context can be overridden during execution via tool results:
     ```python
     def update_preferences(prefs: dict) -> ToolResult:
         return ToolResult.update_context(
             content="Updated preferences",
             context_variables=ContextVariables(preferences=prefs),
         )
     ```

### Best Practices

1. Use for execution-specific data:
   - API keys and credentials
   - Database connections
   - User preferences
   - Environment-specific configuration

2. Avoid storing large data:
   - Use references to data instead of raw content
   - Consider caching for frequently accessed data
   - Pass service instances rather than raw connections

3. Type safety with Pydantic:
   ```python
   from pydantic import BaseModel

   class AppContext(BaseModel):
       api_key: str
       environment: str
       features: dict[str, bool]

   context = ContextVariables(
       **AppContext(
           api_key="sk-...",
           environment="production",
           features={"beta": True},
       ).model_dump()
   )
   ```

For more examples of context usage, see our [context_variables example](../examples/context_variables/run.py).

## Tool Use and Agent Switching

LiteSwarm provides a powerful tool system that enables agents to perform actions and switch between specialized agents during execution.

### Basic Tool Use

Tools are plain Python functions that take inputs and return JSON-serializable outputs:

```python
from typing import Any
from liteswarm.types import ToolResult

def search_docs(query: str) -> dict[str, Any]:
    """Simple tool returning JSON-serializable dict."""
    return {
        "results": [
            {"title": "API Guide", "content": "..."},
            {"title": "Examples", "content": "..."},
        ]
    }

def calculate(a: float, b: float, operation: str) -> float:
    """Tool returning simple value (must be JSON serializable)."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    raise ValueError(f"Unknown operation: {operation}")

# Add tools to agent
agent = Agent(
    id="assistant",
    instructions="You can search docs and perform calculations.",
    llm=LLM(
        model="gpt-4o",
        tools=[search_docs, calculate],  # Make tools available
        tool_choice="auto",              # Let agent decide when to use tools
    ),
)
```

### Context Updates via Tools

Tools can update execution context using `ToolResult`:

```python
def update_user_preferences(
    preferences: dict,
    context_variables: ContextVariables,  # Automatically injected
) -> ToolResult:
    """Tool that updates context variables."""
    return ToolResult.update_context(
        content="Updated preferences",
        context_variables=ContextVariables(
            user_preferences=preferences,
            last_update=datetime.now().isoformat(),
        ),
    )

def fetch_api_key(service: str) -> ToolResult:
    """Tool that adds API key to context."""
    key = get_service_key(service)
    return ToolResult.update_context(
        content=f"Retrieved {service} API key",
        context_variables=ContextVariables(
            api_key=key,
        ),
    )
```

### Agent Switching

Swarm supports dynamic agent switching through tools. This is a one-way operation - like building a directed graph of agent transitions:

```python
def switch_to_expert(domain: str) -> Agent:
    """Simple tool returning new agent."""
    return Agent(
        id=f"{domain}-expert",
        instructions=f"You are an expert in {domain}.",
        llm=LLM(model="gpt-4o"),
    )

def switch_with_context(
    domain: str,
    context_variables: ContextVariables,
) -> ToolResult:
    """Advanced switching with context update."""
    # Create new agent
    expert = Agent(
        id=f"{domain}-expert",
        instructions=f"You are an expert in {domain}.",
        llm=LLM(model="gpt-4o"),
    )
    
    # Switch with custom context
    return ToolResult.switch_agent(
        agent=expert,
        content=f"Switching to {domain} expert",
        context_variables=ContextVariables(
            specialty=domain,
            expertise_level="expert",
        ),
    )
```

### Building Agent Networks

Since agent switching is one-way, you need to provide tools for routing between agents:

```python
# Define routing tools
def switch_to_ui_expert() -> Agent:
    return Agent(id="ui", instructions="UI expert...")

def switch_to_backend_expert() -> Agent:
    return Agent(id="backend", instructions="Backend expert...")

# Create agents with routing capabilities
ui_agent = Agent(
    id="ui",
    instructions="""
    You are a UI expert. If question requires backend knowledge,
    use switch_to_backend_expert tool.
    """,
    llm=LLM(
        model="gpt-4o",
        tools=[switch_to_backend_expert],
    ),
)

backend_agent = Agent(
    id="backend",
    instructions="""
    You are a backend expert. If question requires UI knowledge,
    use switch_to_ui_expert tool.
    """,
    llm=LLM(
        model="gpt-4o",
        tools=[switch_to_ui_expert],
    ),
)
```

For a complete example of agent routing in a mobile development team, see [mobile_app example](../examples/mobile_app/run.py).

### Important Notes

1. **One-Way Transitions**:
   - Agent switches are one-directional
   - No automatic return to previous agent
   - Must use tools to implement routing logic

2. **Context Handling**:
   - `ToolResult.switch_agent` allows context updates
   - Can provide new message history
   - Can reset or update context variables

3. **Tool Requirements**:
   - Return values must be JSON serializable
   - Can return direct values or `ToolResult`
   - Tools can access context via parameters

## Chat API

While Swarm is designed to be stateless, many applications require persistent conversation state. The Chat API provides a modular and extensible framework built on top of Swarm for building stateful chat applications.

### Core Components

The Chat API is built around protocols (interfaces) with default implementations:

```python
from liteswarm.chat import LiteChat, ChatMemory, ChatSearch, ChatOptimization

# Basic chat with in-memory storage
chat = LiteChat()

# Send message and stream response
async for event in chat.send_message("Hello!", agent=agent):
    if event.type == "agent_response_chunk":
        print(event.response_chunk.content, end="")

# Access conversation history
messages = await chat.get_messages()
```

### Modular Architecture

Each component is replaceable with custom implementations:

1. **ChatMemory**: Message storage and retrieval
   ```python
   from liteswarm.chat import ChatMemory

   class DatabaseChatMemory(ChatMemory):
       """Custom memory implementation using database."""
       async def save_messages(self, messages: list[Message]) -> None:
           await self.db.insert_messages(messages)

       async def get_messages(self) -> list[ChatMessage]:
           return await self.db.fetch_messages()

   # Use custom memory implementation
   chat = LiteChat(memory=DatabaseChatMemory())
   ```

2. **ChatSearch**: Vector search for relevant context
   ```python
   from liteswarm.chat import ChatSearch

   class ElasticSearchChat(ChatSearch):
       """Custom search using Elasticsearch."""
       async def search(
           self,
           query: str,
           max_results: int | None = None,
           score_threshold: float | None = None,
       ) -> list[tuple[ChatMessage, float]]:
           results = await self.es.search(query)
           return [(ChatMessage(**doc), score) for doc, score in results]

       async def index(self) -> None:
           """Update search index with latest messages."""
           messages = await self.memory.get_messages()
           await self.es.index_messages(messages)

   # Use custom search implementation
   chat = LiteChat(search=ElasticSearchChat(memory=memory))
   ```

3. **ChatOptimization**: Context window management
   ```python
   from liteswarm.chat import ChatOptimization

   class CustomOptimizer(ChatOptimization):
       """Custom context optimization strategy."""
       async def optimize_context(self, model: str) -> list[ChatMessage]:
           messages = await self.memory.get_messages()
           return await self.optimize_messages(messages, model)

   # Use custom optimizer
   chat = LiteChat(
       optimization=CustomOptimizer(
           memory=memory,
           search=search,
       )
   )
   ```

### Team Chat Support

The Chat API also supports SwarmTeam with `LiteTeamChat`:

```python
from liteswarm.chat import LiteChatMemory, LiteChatSearch, LiteChatOptimization
from liteswarm.experimental import LiteTeamChat

# Create team chat with optional components
team_chat = LiteTeamChat(
    swarm=swarm,
    members=members,
    task_definitions=task_defs,
    memory=LiteChatMemory(),
    search=LiteChatSearch(),
    optimization=LiteChatOptimization(),
)

# Optional plan feedback callback
def feedback_callback(plan: Plan) -> PlanFeedback:
    return ApprovePlan(type="approve")

# Send message with optional feedback
async for event in team_chat.send_message(
    "Create a Python web app",
    context_variables=ContextVariables(project="my_app"),
    feedback_callback=feedback_callback,
):
    if event.type == "task_started":
        print(f"Team member {event.member.id} working on {event.task.type}")
    elif event.type == "task_completed":
        print(f"Task completed: {event.task.type}")
```

### Complete Examples

For working examples of Chat API usage:

1. Basic Usage:
   - Single-user chat: [examples/chat_basic/lite_chat](../examples/chat_basic/lite_chat)
   - Team chat: [examples/chat_basic/lite_team_chat](../examples/chat_basic/lite_team_chat)

2. Advanced Features:
   - Multi-user support
   - Session management
   - Client-server architecture
   See [examples/chat_api](../examples/chat_api) for a complete example of building a client-server chat application.

## More Examples

For more examples and detailed documentation, see the [examples directory](../examples). 

## Building with LiteSwarm

LiteSwarm provides a powerful foundation for building AI applications through its core components:

1. **Core Execution Engine (Swarm)**:
   - Stateless execution with type safety
   - Tool use and agent switching
   - Structured outputs with streaming support

2. **Team Orchestration (SwarmTeam)**:
   - Task-based workflow management
   - Collaborative problem solving
   - Plan creation and execution

3. **Stateful Conversations (Chat API)**:
   - Modular architecture for persistence
   - Context optimization and search
   - Team chat integration

These components can be used independently or combined to create sophisticated applications. For practical examples and implementation patterns, explore our [examples directory](../examples). 