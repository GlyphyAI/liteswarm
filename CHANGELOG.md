# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.1 (2025-01-09)

### Added
- Added missing common types (e.g. LLM and ContextVariables) to global namespace

### Fixed
- Resolved mypy type variance issues with generic types

## 0.5.0 (2025-01-09)

### Added
- Chat components built on top of Swarm and SwarmTeam for building chat applications
- Type-safe structured outputs with Pydantic model support
- New events for managing agent flow (activate, begin, complete, response)
- TypeVar type checking trick to work around Pydantic bound generics
- New examples showcasing chat and team chat functionality

### Changed
- **BREAKING**: Reworked core Swarm API to be fully stateless
- **BREAKING**: Removed session management in chat components
- Reorganized SwarmTeam to be stateless with streaming API support
- Simplified field names and event model naming for consistency
- Updated all examples to use new stateless API
- Improved type safety around response formats
- Simplified swarm team types generics
- Renamed get_result to get_return_value for ReturnableAsyncGenerator

### Removed
- Obsolete MessageStore, ContextManager, EventHandler
- Redundant unwrap utils
- Ambiguous messages file
- Deprecated types and components

[0.5.0]: https://github.com/glyphyai/liteswarm/releases/tag/0.5.0

## 0.4.0 (2024-12-26)

### Added
- ReturnableAsyncGenerator for unified event streaming API
- RAG strategy configuration type for better context optimization control
- Tool call result field for improved result handling
- Serialization support for agent instructions and LLM tools

### Changed
- Refactored event streaming architecture with cleaner separation of concerns
- Embedded completion response into agent response model for better encapsulation
- Renamed event models to remove redundant "Swarm" prefix
- Removed SwarmStream in favor of ReturnableAsyncGenerator
- Updated event handler to use new streaming API
- Improved documentation with up-to-date examples for core components

### Removed
- Deprecated SwarmStream module
- Redundant tool call event in favor of result-only events

[0.4.0]: https://github.com/glyphyai/liteswarm/releases/tag/0.4.0

## 0.3.0 (2024-12-23)

### Added
- Async interface for MessageStore protocol and implementations
- Batch operations for message management (add_messages, remove_messages)
- Unified ContextManager interface with create_context, optimize_context, and find_context methods

### Changed
- Renamed response types to better reflect streaming nature (AgentResponseChunk, CompletionResponseChunk)
- Updated event types to match new response chunk naming
- Improved context management to use MessageStore for data access
- Enhanced usage and cost tracking in REPL with proper accumulation

### Fixed
- Usage statistics accumulation in streaming responses

[0.3.0]: https://github.com/glyphyai/liteswarm/releases/tag/0.3.0

## 0.2.0 (2024-12-20)

### Added
- New message management system with MessageStore, MessageIndex, and ContextManager
- Event system with ConsoleEventHandler and SwarmEventHandler
- SwarmStream wrapper for improved response handling and parsing
- Task context and plan ID support
- New REPL commands for optimization and search

### Changed
- Refactored stream handler to use event-based system
- Enhanced message and context management
- Improved type system with stricter checks
- Updated examples to use new event system

### Fixed
- Context update and retry mechanism improvements
- Tool call pair filtering enhancements
- Response format and cleanup handling
- Variable initialization and error logging

### Removed
- Legacy stream handler and summarizer
- Unused types and context keys

[0.2.0]: https://github.com/glyphyai/liteswarm/releases/tag/0.2.0

## 0.1.1 (2024-12-10)

### Added
- Add py.typed file to support type checking with mypy

[0.1.1]: https://github.com/glyphyai/liteswarm/releases/tag/0.1.1

## 0.1.0 (2024-12-09)

### Added
- Core agent system with LLM-agnostic design and 100+ model support
- Multi-agent orchestration with structured outputs
- Real-time streaming and context management
- Tool integration system with flexible agent switching
- Production-ready examples (REPL, Mobile App Team, Software Team)

[0.1.0]: https://github.com/glyphyai/liteswarm/releases/tag/0.1.0