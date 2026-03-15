# Setup & Architecture

## Default run (no Docker required)

Coleman4HCS is framework-first: the default installation works without any
external services.

```bash
pip install coleman4hcs
python main.py
```

Results are written as **partitioned Parquet files** (zstd compressed) under
`./runs/` by default.  You can query them directly with DuckDB or Polars
without loading everything into RAM.

Repeated runs append new result files by default. They do not replace previous
results unless you explicitly remove `./runs/` or choose another `out_dir`.

## Architecture overview

Coleman4HCS separates three concerns:

| Layer | Purpose | Default | Optional |
|-------|---------|---------|----------|
| **Results** | Persist experiment facts (NAPFD, APFDc, …) | Partitioned Parquet | ClickHouse sink |
| **Checkpoints** | Crash-safe resume | Local filesystem (pickle + progress.json) | — |
| **Telemetry** | Observability (latency, throughput) | Disabled (NoOp) | OpenTelemetry + Collector |

### Null adapters

When a layer is disabled its module resolves to a **Null implementation**
with near-zero overhead:

* `NullSink` — discards result rows
* `NullCheckpointStore` — never saves/loads checkpoints
* `NoOpTelemetry` — all instrument calls are instant no-ops

## Configuration

All settings live in `config.toml`:

```toml
[results]
enabled = true
sink = "parquet"
out_dir = "./runs"
batch_size = 1000
top_k_prioritization = 0  # 0 = hash only

[checkpoint]
enabled = true
interval = 50000
base_dir = "checkpoints"

[telemetry]
enabled = false
otlp_endpoint = "http://localhost:4318"
service_name = "coleman4hcs"
```

## Optional extras

```bash
# Telemetry (OpenTelemetry SDK)
pip install coleman4hcs[telemetry]

# ClickHouse sink
pip install coleman4hcs[clickhouse]
```

## Optional observability stack

See the [Observability guide](observability.md) for a Docker Compose stack
with OTel Collector + Grafana (ClickHouse under a profile).

## How to query results

Results are written as Hive-partitioned Parquet files under `./runs/`.  You
can query them directly with DuckDB (already a project dependency):

```sql
-- Average NAPFD per policy
SELECT policy, AVG(fitness) AS avg_napfd
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY policy
ORDER BY avg_napfd DESC;

-- Cost distribution per reward function
SELECT reward_function,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cost) AS median_cost
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY reward_function;
```

For a guided end-to-end walkthrough, see the official [workflow notebook](workflow.py).
