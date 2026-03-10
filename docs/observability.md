# Observability

This page describes the **optional** local observability stack for
debugging and profiling Coleman4HCS experiments.

!!! note "Framework-first guarantee"
    `python main.py` works without Docker or any of these services.
    Install optional extras only when you need them.

## Quick start

```bash
# Base stack (OTel Collector + Grafana)
cd examples/observability
docker compose up -d

# Enable telemetry in config.toml:
#   [telemetry]
#   enabled = true

# Run your experiment
python main.py
```

## With ClickHouse

```bash
docker compose --profile clickhouse up -d
pip install coleman4hcs[clickhouse]
```

## Tear down

```bash
docker compose --profile clickhouse down -v
```

## Metric names

| Metric | Type | Description |
|--------|------|-------------|
| `coleman.cycles_total` | Counter | Total experiment cycles processed |
| `coleman.bandit_update_latency` | Histogram (s) | Bandit arm-update latency |
| `coleman.prioritization_latency` | Histogram (s) | Test-case prioritization latency |
| `coleman.evaluation_latency` | Histogram (s) | Evaluation step latency |
| `coleman.napfd` | Histogram | NAPFD score distribution |
| `coleman.apfdc` | Histogram | APFDc score distribution |

### Cardinality rules

* **No `step` label** in metrics (would create unbounded cardinality).
* `run_id` is a resource attribute, not a metric label.
* Per-step detail is available in **traces** (span attributes).

## Example DuckDB queries over Parquet results

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
