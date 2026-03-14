#!/usr/bin/env bash
# =============================================================================
# DevContainer post-create setup for Coleman4HCS
# =============================================================================
# Runs once after the container is created.  Installs the correct Python,
# project dependencies (including telemetry + clickhouse extras), pre-commit
# hooks, and prepares the environment so that observability works out of the
# box — zero manual steps.
# =============================================================================
set -euo pipefail

echo "── Setting up Coleman4HCS DevContainer ──"

# 1. Install Python 3.14, create the venv, and install dev dependencies
make install

# 2. Install optional extras so observability works without extra pip commands
uv pip install -e ".[telemetry,clickhouse]"

# 3. Install pre-commit hooks
make pre-commit-install

# 4. Seed .env from the example if it does not exist yet
if [ ! -f .env ]; then
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
