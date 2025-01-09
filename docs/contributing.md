# Contributing

We welcome contributions to LiteSwarm! Here's how you can help:

## Areas for Contribution

1. **Adding Tests**
   - Unit tests for core functionality
   - Integration tests for agent interactions
   - Example-based tests for common use cases
   - Testing infrastructure and CI setup

2. **Bug Reports**
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Relevant code snippets

3. **Feature Requests**
   - Use case description
   - Expected behavior
   - Example code

4. **Code Contributions**
   - Fork the repository
   - Create feature branch
   - Add tests
   - Submit pull request
   - Ensure CI passes (if applicable)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/GlyphyAI/liteswarm.git
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

# Run tests
pytest

# Run type checking
mypy .

# Run linting
ruff check .
```

## Code Style

- Use ruff for linting and formatting
- Type hints required for all functions
- Google style docstrings
- Include tests for new features

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding/updating tests
- `refactor:` Code changes (no features/fixes)
