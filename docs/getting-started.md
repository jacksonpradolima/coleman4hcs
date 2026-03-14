# Getting Started

## Installation

### Option A: DevContainer (recommended)

The fastest way to start is with a [DevContainer](https://containers.dev/).
Open the repo in **VS Code** (or any DevContainer-compatible editor) and
select **"Reopen in Container"** — **everything works out of the box**,
including the full observability stack.

**What happens automatically:**

1. Python 3 + uv + all dependencies (including telemetry & ClickHouse extras) are installed
2. Pre-commit hooks are configured
3. `.env` is seeded from `.env.example`
4. The **observability stack** (OTel Collector + Grafana) starts via Docker-in-Docker
5. **Telemetry is enabled** in `config.toml` so metrics flow to Grafana immediately

After the container builds, just run your experiment:

```bash
uv run python main.py
# Open http://localhost:3000 → Grafana shows metrics in real-time
```

No manual steps required — skip to [Running Experiments](#running-experiments).

**What's included:**

| What | Why |
|------|-----|
| **Python 3 + uv** | Project package manager |
| **Docker-in-Docker** | Runs the observability stack automatically |
| **VS Code extensions** | Ruff, Pylance, Pyright, Copilot, TOML, Jupyter, and more |
| **Telemetry + ClickHouse extras** | Pre-installed — no extra `pip install` needed |
| **OTel Collector + Grafana** | Started automatically on container start |
| **Port forwarding** | All service ports mapped to your host browser |

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

### DevContainer: just run it

If you're in the DevContainer, the observability stack is already running and
telemetry is enabled.  Just run:

```bash
uv run python main.py
# Results    → ./runs/       (Parquet)
# Checkpoints → ./checkpoints/
# Metrics    → http://localhost:3000 (Grafana, live)
```

### Local setup: choose your level

**Basic (Parquet only, no services needed):**

```bash
uv run python main.py
```

Results appear in `./runs/` (Parquet).  Query them with DuckDB:

```bash
uv run python -c "
import duckdb
print(duckdb.sql(\"\"\"
    SELECT policy, AVG(fitness) AS avg_napfd
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
    GROUP BY policy ORDER BY avg_napfd DESC
\"\"\"))
"
```

**With telemetry (OTel Collector + Grafana):**

```bash
# 1. Start the stack
cd examples/observability && docker compose up -d

# 2. Install telemetry extras
uv pip install coleman4hcs[telemetry]

# 3. Enable in config.toml → [telemetry] enabled = true

# 4. Run
uv run python main.py
# Grafana → http://localhost:3000
```

**With ClickHouse (optional results sink):**

```bash
cd examples/observability && docker compose --profile clickhouse up -d
uv pip install coleman4hcs[clickhouse]
# config.toml → [results] sink = "clickhouse"
uv run python main.py
```

## Port Reference

| Port | Service | URL |
|------|---------|-----|
| **3000** | Grafana | [http://localhost:3000](http://localhost:3000) |
| **4317** | OTel Collector (gRPC) | — (used by the framework) |
| **4318** | OTel Collector (HTTP) | — (used by the framework) |
| **8889** | Prometheus metrics | [http://localhost:8889/metrics](http://localhost:8889/metrics) |
| **8123** | ClickHouse (HTTP) | [http://localhost:8123](http://localhost:8123) *(profile: clickhouse)* |
| **9000** | ClickHouse (native) | — *(profile: clickhouse)* |

!!! note "DevContainer vs. local"
    In the DevContainer, the OTel Collector + Grafana start automatically
    and telemetry is enabled.  Locally, you start them manually with
    `docker compose up -d` and set `[telemetry] enabled = true`.

## Quick Reference: `config.toml` Sections

```toml
# ── Results ─────────────────────────────────────────────
[results]
enabled = true            # false → NullSink (discard all)
sink = "parquet"          # "parquet" (default) | "clickhouse"
out_dir = "./runs"        # Parquet output directory
batch_size = 1000         # Rows buffered before flush
top_k_prioritization = 0  # 0 = store hash only; >0 = keep top-k

# ── Checkpoints ─────────────────────────────────────────
[checkpoint]
enabled = true
interval = 50000          # Steps between saves
base_dir = "checkpoints"

# ── Telemetry ───────────────────────────────────────────
[telemetry]
enabled = false                          # DevContainer auto-enables this
otlp_endpoint = "http://localhost:4318"  # OTel Collector HTTP endpoint
service_name = "coleman4hcs"
```

See the [Setup & Architecture](setup.md) guide for a deeper explanation
of each layer and the [Observability guide](observability.md) for metric
names and cardinality rules.

---

See the [README](https://github.com/jacksonpradolima/coleman4hcs#readme) for
detailed usage instructions covering HCS strategies, contextual bandits, and
dataset preparation.
