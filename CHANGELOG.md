# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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