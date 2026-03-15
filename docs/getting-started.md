# Getting Started

## Installation

### Option A: DevContainer (recommended)

The fastest way to start is with a [DevContainer](https://containers.dev/).
Open the repo in **VS Code** (or any DevContainer-compatible editor) and
select **"Reopen in Container"** — **everything works out of the box**,
including the full observability stack.

**What happens automatically:**

1. Python 3.14 + uv + all dependencies (including telemetry & ClickHouse extras) are installed
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
| **Python 3.14 + uv** | Project package manager |
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

## Checking Final Results

Grafana is for live execution behavior: throughput, latency, CPU, memory, and
run separation while the experiment is still running.

The final experiment facts are stored separately:

* Parquet, by default: `./runs/`
* Checkpoint progress for resume/recovery: `./checkpoints/`
* ClickHouse, when enabled: table `coleman_results`

### What happens if you run experiments again?

By default, Coleman4HCS **appends** new final results instead of replacing the
old ones.

For the default Parquet sink this means:

* new files are created under `./runs/`
* previous files are preserved
* old and new executions coexist in the same dataset
* the new `execution_id` lets you separate one run from another analytically

For the ClickHouse sink this means:

* new rows are inserted into `coleman_results`
* previous rows are preserved unless you explicitly delete them

Checkpoint behavior is different: for the same run and experiment, the latest
checkpoint progress is updated so recovery can continue from the most recent
durable step.

If you want a completely fresh analysis space, choose one of these options
before running again:

```bash
# Option 1: start from a clean Parquet directory
rm -rf ./runs

# Option 2: keep old runs, but write new ones elsewhere
# config.toml -> [results] out_dir = "./runs-new"

# Option 3: clean checkpoints too, if you do not want recovery state reused
rm -rf ./checkpoints
```

### Default Parquet output

After `uv run python main.py`, inspect the partitioned Parquet dataset:

```bash
find ./runs -name '*.parquet' | head
```

Query it directly with DuckDB:

```bash
uv run python -c "
import duckdb
print(duckdb.sql(\"\"\"
    SELECT scenario,
           execution_id,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc,
           AVG(process_memory_rss_mib) AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
    GROUP BY scenario, execution_id, policy, reward_function
    ORDER BY avg_napfd DESC
\"\"\")
)"
```

### DuckDB, in practice

DuckDB is the easiest way to inspect final results deeply without moving data
out of the Parquet dataset.

#### Open an interactive DuckDB session

```bash
uv run python -c "import duckdb; duckdb.connect('analysis.duckdb').execute('SELECT 1'); print('ready')"
```

Or use it directly from Python scripts and notebooks.

#### Inspect available columns

```bash
uv run python -c "
import duckdb
print(duckdb.sql(\"\"\"
    DESCRIBE
    SELECT *
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
\"\"\").df())
"
```

#### See which executions are available

```bash
uv run python -c "
import duckdb
print(duckdb.sql(\"\"\"
    SELECT scenario,
           execution_id,
           COUNT(*) AS rows,
           MIN(step) AS first_step,
           MAX(step) AS last_step
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
    GROUP BY scenario, execution_id
    ORDER BY scenario, execution_id
\"\"\").df())
"
```

#### Compare policies by final quality and resource cost

```bash
uv run python -c "
import duckdb
print(duckdb.sql(\"\"\"
    SELECT scenario,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc,
           AVG(prioritization_time) AS avg_prioritization_time,
           AVG(process_memory_rss_mib) AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
           MAX(wall_time_seconds) AS wall_time_seconds
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
    GROUP BY scenario, policy, reward_function
    ORDER BY avg_napfd DESC, avg_apfdc DESC
\"\"\").df())
"
```

#### Slice one specific execution

```bash
uv run python -c "
import duckdb
execution_id = 'replace-with-real-execution-id'
print(duckdb.sql(f\"\"\"
    SELECT experiment,
           step,
           policy,
           reward_function,
           fitness,
           cost,
           process_memory_rss_mib,
           process_cpu_utilization_percent
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
    WHERE execution_id = '{execution_id}'
    ORDER BY experiment, step, policy
\"\"\").df())
"
```

#### Export a filtered report

```bash
uv run python -c "
import duckdb
duckdb.sql(\"\"\"
    COPY (
        SELECT scenario,
               execution_id,
               policy,
               reward_function,
               AVG(fitness) AS avg_napfd,
               AVG(cost) AS avg_apfdc
        FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
        GROUP BY scenario, execution_id, policy, reward_function
    ) TO './runs/analysis/final-summary.csv' (HEADER, DELIMITER ',')
\"\"\")
print('exported ./runs/analysis/final-summary.csv')
"
```

### ClickHouse, in practice

If you want a long-lived analytical store instead of Parquet files, switch the
 results sink to ClickHouse.

#### Enable it

```toml
[results]
enabled = true
sink = "clickhouse"
```

Run the service locally:

```bash
cd examples/observability
docker compose --profile clickhouse up -d
```

#### Query it from Python

```bash
uv run python -c "
import clickhouse_connect

client = clickhouse_connect.get_client(host='localhost', port=8123, database='default')
rows = client.query('''
    SELECT scenario,
           execution_id,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc,
           AVG(process_memory_rss_mib) AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct
    FROM coleman_results
    GROUP BY scenario, execution_id, policy, reward_function
    ORDER BY avg_napfd DESC
''')
print(rows.result_rows)
"
```

#### Inspect stored schema

```bash
uv run python -c "
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost', port=8123, database='default')
print(client.query('DESCRIBE TABLE coleman_results').result_rows)
"
```

#### Clean old data when needed

ClickHouse also accumulates results by default. If you need a fresh table:

```bash
uv run python -c "
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost', port=8123, database='default')
client.command('TRUNCATE TABLE coleman_results')
print('coleman_results truncated')
"
```

### Resume and recovery

If checkpoints are enabled, Coleman4HCS writes one directory per run under
`./checkpoints/`. Each experiment keeps a `progress_ex<N>.json` file pointing
to the last durable checkpoint.

To inspect recovery state:

```bash
find ./checkpoints -name 'progress_*.json' -maxdepth 3 -print
```

When `uv run python main.py` is started again with the same configuration,
the framework loads the last saved checkpoint and resumes from the next step
instead of replaying completed builds.

### Notebook workflow

For a guided end-to-end workflow covering configuration, observability,
checkpoints, export, and analysis, open the marimo notebook example in
[workflow.py](workflow.py).

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
