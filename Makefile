.PHONY: install install-dev format lint test commit bump changelog release

# ================================================
# MARK: Dev commands
# ================================================

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

format:
	ruff format .
	ruff check . --fix

lint:
	ruff check .
	mypy .

test:
	pytest

# ================================================
# MARK: Commitizen commands
# ================================================

commit:
	cz commit

bump:
	cz bump

changelog:
	cz changelog

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

run-calculator-example:
	python -m examples.calculator.run

run-context-variables-example:
	python -m examples.context_variables.run

run-mobile-app-example:
	python -m examples.mobile_app.run

run-parallel-research-example:
	python -m examples.parallel_research.run

run-repl-example:
	python -m examples.repl.run

run-software-team-example:
	python -m examples.software_team.run

run-structured-outputs-example:
	python -m examples.structured_outputs.run
