#!/usr/bin/env bash
# =============================================================================
# DevContainer post-create setup for Coleman4HCS
# =============================================================================
# Runs once after the container is created.  Installs the correct Python,
# project dependencies, pre-commit hooks, and prepares the .env file.
# =============================================================================
set -euo pipefail

echo "── Setting up Coleman4HCS DevContainer ──"

# 1. Install Python 3.14 and create the venv via Make
make install

# 2. Install pre-commit hooks
make pre-commit-install

# 3. Seed .env from the example if it does not exist yet
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

echo ""
echo "✅  DevContainer ready!"
echo ""
echo "Useful commands:"
echo "  make test          – run tests"
echo "  make lint          – run ruff linter"
echo "  make format        – run ruff formatter"
echo "  make docs-serve    – serve docs locally"
echo ""
echo "Optional observability stack:"
echo "  cd examples/observability && docker compose up -d"
echo "  Grafana → http://localhost:3000"
echo ""
