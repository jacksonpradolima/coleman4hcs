SHELL := /bin/bash

# Resolve uv path: if not found in PATH, default to ~/.local/bin/uv (where installer places it)
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
PRE_COMMIT := .venv/bin/pre-commit

# Default link mode for uv to avoid hardlink warnings on some filesystems
export UV_LINK_MODE ?= copy

.DEFAULT_GOAL := help

.PHONY: help ensure-uv setup install pre-commit-install lint format typecheck test test-cov docs docs-serve check-precommit clean interrogate build

## —— Coleman4HCS Makefile ——————————————————————————————————

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ——— Setup ———————————————————————————————————————————————

ensure-uv: ## Ensure uv is installed; install if missing
	@if ! command -v $(UV) >/dev/null 2>&1; then \
	  echo "uv not found, installing..."; \
	  curl -Ls https://astral.sh/uv/install.sh | bash; \
	  echo "uv installed at $$HOME/.local/bin/uv"; \
	fi

setup: ensure-uv ## Setup Python 3.14 for local development
	$(UV) python install 3.14
	# Ensure project Python is pinned to a compatible version for uv (avoids reading an old .python-version)
	$(UV) python pin 3.14
	$(UV) venv .venv --python 3.14 --allow-existing

install: ensure-uv setup ## Install all dependencies from uv.lock (including dev, docs, notebook extras) into .venv
	$(UV) sync --frozen --extra dev --extra docs --extra notebook
	# Ensure project is installed in editable mode
	$(UV) run --python $(PYTHON) --no-project pip install -e .

pre-commit-install: ## Install pre-commit hooks
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push

# ——— Quality ——————————————————————————————————————————————

lint: ensure-uv ## Run ruff linter
	$(UV) run ruff check .

format: ensure-uv ## Run ruff formatter
	$(UV) run ruff check --fix .

typecheck: ensure-uv ## Run pyright and ty type checkers
	# Run both static type checkers
	$(UV) run --python $(PYTHON) --no-project pyright coleman4hcs
	$(UV) run --python $(PYTHON) --no-project ty check .

interrogate: ensure-uv ## Check docstring coverage with interrogate
	$(UV) run interrogate coleman4hcs/ -v

# ——— Testing ——————————————————————————————————————————————

test: ensure-uv install ## Run tests with pytest in parallel if xdist is available; otherwise, run serially
	@if $(UV) run --python $(PYTHON) --no-project python -c 'import xdist' >/dev/null 2>&1; then \
	  $(UV) run --python $(PYTHON) --no-project pytest -n auto; \
	else \
	  echo "pytest-xdist não encontrado; executando em série..."; \
	  $(UV) run --python $(PYTHON) --no-project pytest; \
	fi

test-cov: ensure-uv install ## Run tests with coverage report in parallel if xdist is available; otherwise, run serially
	@if $(UV) run --python $(PYTHON) --no-project python -c 'import xdist' >/dev/null 2>&1; then \
	  $(UV) run --python $(PYTHON) --no-project pytest -n auto \
	    --cov=coleman4hcs --cov-branch \
	    --cov-report=term-missing:skip-covered \
	    --cov-report=xml \
	    --cov-report=html; \
	else \
	  $(UV) run --python $(PYTHON) --no-project pytest \
	    --cov=coleman4hcs --cov-branch \
	    --cov-report=term-missing:skip-covered \
	    --cov-report=xml \
	    --cov-report=html; \
	fi

# ——— Documentation ————————————————————————————————————————

docs: ensure-uv ## Build documentation with MkDocs
	$(UV) run mkdocs build --strict

docs-serve: ensure-uv ## Serve documentation locally
	$(UV) run mkdocs serve

# ——— Build ————————————————————————————————————————————————

build: ensure-uv ## Build the package
	$(UV) build

# ——— Pre-commit ———————————————————————————————————————————

check-precommit: test typecheck interrogate ## Run tests, type checks, and docstring coverage (used by pre-commit)

# ——— Cleanup ——————————————————————————————————————————————

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ site/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
