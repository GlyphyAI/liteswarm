# Examples

The framework includes several example applications demonstrating different features and use cases.

## Basic Examples

### REPL ([example](../examples/repl))
Simple interactive chat interface showing basic agent usage.

### Calculator ([example](../examples/calculator))
Math agent with tool usage and switching.

### Context Variables ([example](../examples/context_variables))
Demonstrates passing execution-specific context to tools and agents.

## Chat Applications

### Basic Chat ([example](../examples/chat_basic))
Examples demonstrating how to use LiteSwarm's stateful components:
- `lite_chat`: Using the stateful chat interface built on top of Swarm
- `lite_team_chat`: Using the stateful chat interface built on top of SwarmTeam

### Client-Server Chat ([example](../examples/chat_api))
Example of building a client-server chat application:

- **[Server](../examples/chat_api/server)**: FastAPI server demonstrating:
  - User and session management
  - Streaming API endpoints
  - State handling
  - Multi-user support

- **[Client](../examples/chat_api/client)**: CLI client showing how to:
  - Handle real-time event streaming
  - Manage user sessions
  - Handle server connections

## Team Examples

### SwarmTeam Basic ([example](../examples/swarm_team_basic))
Introduction to SwarmTeam features:
- Task definition and planning
- Team member specialization
- Plan execution and feedback

### Mobile App Team ([example](../examples/mobile_app))
Complex team of agents (PM, Designer, Engineer, QA) building a Flutter app:
- Agent routing and switching
- Specialized task handling
- Team collaboration

### Software Team ([example](../examples/software_team))
Complete example demonstrating SwarmTeam's capabilities:
- Advanced planning and task execution
- Code generation and review
- Project management

## Advanced Examples

### Parallel Research ([example](../examples/parallel_research))
Parallel tool execution for efficient data gathering:
- Concurrent API calls
- Result aggregation
- Error handling

### Structured Outputs

1. **Core** ([example](../examples/structured_outputs/core))
   - Basic structured outputs with Pydantic
   - Streaming and parsing
   - Type safety

2. **Chat** ([example](../examples/structured_outputs/chat))
   - Structured outputs in chat context
   - Stateful parsing
   - Chat-specific features

3. **Playground** ([example](../examples/structured_outputs/playground))
   - Different parsing strategies
   - Provider compatibility
   - Complex output handling

## Running Examples

Basic examples:
```bash
# Try the REPL
make run-repl-example

# Run the calculator
make run-calculator-example

# Test context variables
make run-context-variables-example
```

Chat applications:
```bash
# Basic chat examples
make run-chat-lite-chat-example
make run-chat-lite-team-chat-example

# Client-server chat examples
make run-chat-api-server-example
make run-chat-api-client-example
```

Team examples:
```bash
# Basic team features
make run-swarm-team-basic-example

# Complex team examples
make run-mobile-app-example
make run-software-team-example
```

Advanced examples:
```bash
# Parallel execution
make run-parallel-research-example

# Structured outputs
make run-structured-outputs-core-example
make run-structured-outputs-chat-example
make run-structured-outputs-playground-example
```

For more details about specific features, see [Advanced Features](advanced.md). 
