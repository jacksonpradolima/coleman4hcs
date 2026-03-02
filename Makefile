.DEFAULT_GOAL := help

.PHONY: help install pre-commit-install lint format typecheck test test-cov docs docs-serve check-precommit clean interrogate build

## —— Coleman4HCS Makefile ——————————————————————————————————

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ——— Setup ———————————————————————————————————————————————

install: ## Install all dependencies (including dev, docs, notebook extras)
	uv sync --extra dev --extra docs --extra notebook
	uv pip install -e .

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push

# ——— Quality ——————————————————————————————————————————————

lint: ## Run ruff linter
	uv run ruff check .

format: ## Run ruff formatter
	uv run ruff format .

typecheck: ## Run pyright and ty type checkers
	uv run pyright coleman4hcs/
	uv run ty check coleman4hcs/

interrogate: ## Check docstring coverage with interrogate
	uv run interrogate coleman4hcs/ -v

# ——— Testing ——————————————————————————————————————————————

test: ## Run tests with pytest
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=coleman4hcs --cov-branch --cov-report=term-missing

# ——— Documentation ————————————————————————————————————————

docs: ## Build documentation with MkDocs
	uv run mkdocs build --strict

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

# ——— Build ————————————————————————————————————————————————

build: ## Build the package
	uv build

# ——— Pre-commit ———————————————————————————————————————————

check-precommit: test typecheck interrogate ## Run tests, type checks, and docstring coverage (used by pre-commit)

# ——— Cleanup ——————————————————————————————————————————————

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ site/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
