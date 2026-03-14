#!/usr/bin/env bash
# =============================================================================
# DevContainer post-create setup for Coleman4HCS
# =============================================================================
# Runs once after the container is created.  Installs uv, Python 3.14,
# project dependencies (including telemetry + clickhouse extras), pre-commit
# hooks, and prepares the environment so that observability works out of the
# box — zero manual steps.
# =============================================================================
set -e

echo "── Setting up Coleman4HCS DevContainer ──"

# 1. Install uv (the project's package manager)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Install Python 3.14, create the venv, and install ALL dependencies
#    (dev + docs + notebook + telemetry + clickhouse extras)
make setup
uv sync --frozen --extra dev --extra docs --extra notebook --extra telemetry --extra clickhouse
uv run --python .venv/bin/python --no-project pip install -e .

# 3. Install pre-commit hooks (non-critical)
make pre-commit-install || echo "⚠  pre-commit install failed — you can run 'make pre-commit-install' manually."

# 4. Seed .env from the example if it does not exist yet
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

# 5. Done — the post-start hook will start the observability stack and
#    enable telemetry in config.toml automatically on each container start.

echo ""
echo "✅  DevContainer ready!"
echo ""
echo "The observability stack (Grafana + OTel Collector) starts automatically."
echo ""
echo "  Grafana          → http://localhost:3000"
echo "  Prometheus metrics → http://localhost:8889/metrics"
echo ""
echo "Useful commands:"
echo "  make test          – run tests"
echo "  make lint          – run ruff linter"
echo "  make format        – run ruff formatter"
echo "  make docs-serve    – serve docs locally"
echo "  uv run python main.py  – run experiments (results + telemetry)"
echo ""
