# Getting Started

## Installation

### Option A: DevContainer (recommended)

The fastest way to start is with a [DevContainer](https://containers.dev/).
Open the repo in **VS Code** (or any DevContainer-compatible editor) and
select **"Reopen in Container"** — everything is set up automatically.

The DevContainer includes:

| What | Why |
|------|-----|
| **Python 3 + uv** | Project package manager |
| **Docker-in-Docker** | Run the optional observability stack inside the container |
| **VS Code extensions** | Ruff, Pylance, Pyright, Copilot, TOML, Jupyter, and more |
| **Port forwarding** | Grafana (3000), OTel Collector (4317/4318), ClickHouse (8123/9000), Prometheus (8889) |
| **Post-create script** | Runs `make install`, `make pre-commit-install`, seeds `.env` |

After the container builds you're ready to go — skip to
[Configuration](#configuration).

### Option B: Local setup

#### Prerequisites

- Python 3.14+
- [UV](https://docs.astral.sh/uv/)

#### Install

```bash
git clone https://github.com/jacksonpradolima/coleman4hcs.git
cd coleman4hcs
uv sync
uv pip install -e .
```

#### Development Setup

```bash
# Install all development dependencies
make install

# Install pre-commit hooks
make pre-commit-install
```

## Configuration

1. Copy the example environment file (DevContainer does this automatically):

```bash
cp .env.example .env
```

2. Edit `.env` and set `CONFIG_FILE=./config.toml`.

3. Customise `config.toml` to select datasets, policies, and reward functions.

## Running Experiments

```bash
uv run python main.py
```

## Optional: Observability Stack

If you want metrics and tracing during development, bring up the observability
stack (works inside the DevContainer thanks to Docker-in-Docker):

```bash
cd examples/observability && docker compose up -d
```

Then enable telemetry in `config.toml`:

```toml
[telemetry]
enabled = true
```

Grafana is available at [http://localhost:3000](http://localhost:3000).

See the [Observability guide](observability.md) for metric names, cardinality
rules, and ClickHouse setup.

---

See the [README](https://github.com/jacksonpradolima/coleman4hcs#readme) for
detailed usage instructions covering HCS strategies, contextual bandits, and
dataset preparation.
