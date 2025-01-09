# ================================================
# MARK: Dev commands
# ================================================

.PHONY: install
install:
	uv pip install -e .

.PHONY: install-dev
install-dev:
	uv pip install -e ".[dev]"

.PHONY: install-docs
install-docs:
	uv pip install -e ".[docs]"

.PHONY: install-all
install-all:
	uv pip install -e ".[dev,examples,docs]"

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check . --fix

.PHONY: lint
lint:
	uv run ruff check .
	uv run mypy .

.PHONY: mypy
mypy:
	uv run mypy .

.PHONY: test
test:
	uv run pytest -v -s

# ================================================
# MARK: Commitizen commands
# ================================================

.PHONY: commit
commit:
	uv run cz commit

.PHONY: bump
bump:
	uv run cz bump

.PHONY: changelog
changelog:
	uv run cz changelog

.PHONY: release
release:
	# Ensure everything is clean and tested
	make lint
	make test
	# Update version and changelog
	uv run cz bump --changelog
	# Push changes
	git push
	git push --tags

# ================================================
# MARK: Example runners
# ================================================

.PHONY: run-calculator-example
run-calculator-example:
	python -m examples.calculator.run

.PHONY: run-context-variables-example
run-context-variables-example:
	python -m examples.context_variables.run

.PHONY: run-mobile-app-example
run-mobile-app-example:
	python -m examples.mobile_app.run

.PHONY: run-parallel-research-example
run-parallel-research-example:
	python -m examples.parallel_research.run

.PHONY: run-repl-example
run-repl-example:
	python -m examples.repl.run

.PHONY: run-software-team-example
run-software-team-example:
	python -m examples.software_team.run

.PHONY: run-structured-outputs-core-example
run-structured-outputs-core-example:
	python -m examples.structured_outputs.core.run

.PHONY: run-structured-outputs-chat-example
run-structured-outputs-chat-example:
	python -m examples.structured_outputs.chat.run

.PHONY: run-structured-outputs-playground-example
run-structured-outputs-playground-example:
	python -m examples.structured_outputs.playground.run

.PHONY: run-chat-lite-chat-example
run-chat-lite-chat-example:
	python -m examples.chat_basic.lite_chat.run

.PHONY: run-chat-lite-team-chat-example
run-chat-lite-team-chat-example:
	python -m examples.chat_basic.lite_team_chat.run

.PHONY: run-chat-api-server-example
run-chat-api-server-example:
	python -m examples.chat_api.server.run

.PHONY: run-chat-api-client-example
run-chat-api-client-example:
	python -m examples.chat_api.client.run

.PHONY: run-swarm-team-basic-example
run-swarm-team-basic-example:
	python -m examples.swarm_team_basic.run

# ================================================
# MARK: PR utility commands
# ================================================

.PHONY: save-pr-diff
save-pr-diff:
	@[ -n "$(PR_NUMBER)" ] || (echo "Error: PR_NUMBER is not set"; exit 1)
	@echo "Saving diff for PR_NUMBER=$(PR_NUMBER)"
	gh pr diff $(PR_NUMBER) > pr$(PR_NUMBER)_diff.patch

.PHONY: save-pr-commit-messages
save-pr-commit-messages:
	@[ -n "$(PR_NUMBER)" ] || (echo "Error: PR_NUMBER is not set"; exit 1)
	@echo "Saving commit messages for PR_NUMBER=$(PR_NUMBER)"
	gh pr view $(PR_NUMBER) --json commits --jq '.commits[] | "\(.messageHeadline) \(.messageBody)"' > pr$(PR_NUMBER)_commit_messages.txt

# ================================================
# MARK: Documentation commands
# ================================================

.PHONY: docs-serve
docs-serve:
	@echo "Starting documentation server..."
	@pkill -f "mkdocs serve" || true
	@mkdocs serve

.PHONY: docs-build
docs-build:
	@echo "Building documentation..."
	@mkdocs build

.PHONY: docs-clean
docs-clean:
	@echo "Cleaning documentation..."
	@rm -rf site/
