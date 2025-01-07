# ================================================
# MARK: Dev commands
# ================================================

.PHONY: install
install:
	uv pip install -e .

.PHONY: install-dev
install-dev:
	uv pip install -e ".[dev]"

.PHONY: install-all
install-all:
	uv pip install -e ".[dev,examples]"

.PHONY: format
format:
	ruff format .
	ruff check . --fix

.PHONY: lint
lint:
	ruff check .
	mypy .

.PHONY: mypy
mypy:
	mypy .

.PHONY: test
test:
	pytest -v -s

# ================================================
# MARK: Commitizen commands
# ================================================

.PHONY: commit
commit:
	cz commit

.PHONY: bump
bump:
	cz bump

.PHONY: changelog
changelog:
	cz changelog

.PHONY: release
release:
	# Ensure everything is clean and tested
	make lint
	make test
	# Update version and changelog
	cz bump --changelog
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

.PHONY: run-structured-outputs-example
run-structured-outputs-example:
	python -m examples.structured_outputs.run

.PHONY: run-chat-example
run-chat-example:
	python -m examples.chat_example.run

.PHONY: run-chat-server-example
run-chat-server-example:
	python -m examples.chat_server.run

.PHONY: run-chat-client-example
run-chat-client-example:
	python -m examples.chat_client.run

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
