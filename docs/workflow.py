"""Coleman4HCS workflow notebook.

Official marimo notebook covering configuration, observability,
resume/recovery, result export, and final analysis.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    """Load notebook dependencies and plotting defaults."""
    import json
    import os
    import tomllib
    from pathlib import Path

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.style.use("ggplot")
    sns.set_style("whitegrid")
    path_class = Path
    return duckdb, json, mo, os, path_class, pd, plt, sns, tomllib


@app.cell
def _(mo):
    """Introduce the end-to-end workflow covered by the notebook."""
    mo.md(
        """
        # Coleman4HCS Workflow

        This official notebook covers the full operational loop:

        1. Read active settings from `config.toml`
        2. Inspect live behavior in Grafana
        3. Inspect checkpoint progress for resume and recovery
        4. Read final experiment results from Parquet
        5. Export compact summaries for reports or papers
        6. Compare policies by quality and resource cost
        """
    )
    return


@app.cell
def _(mo, os, path_class, pd, tomllib):
    """Read the active runtime configuration from config.toml."""
    config_path = path_class(os.getenv("CONFIG_FILE", "config.toml"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    results_cfg = config.get("results", {})
    checkpoint_cfg = config.get("checkpoint", {})
    telemetry_cfg = config.get("telemetry", {})
    experiment_cfg = config.get("experiment", {})

    settings_df = pd.DataFrame(
        [
            {"setting": "datasets", "value": ", ".join(experiment_cfg.get("datasets", []))},
            {
                "setting": "scheduled_time_ratio",
                "value": ", ".join(str(value) for value in experiment_cfg.get("scheduled_time_ratio", [])),
            },
            {"setting": "results.enabled", "value": str(results_cfg.get("enabled", False))},
            {"setting": "results.sink", "value": results_cfg.get("sink", "parquet")},
            {"setting": "results.out_dir", "value": results_cfg.get("out_dir", "./runs")},
            {"setting": "checkpoint.enabled", "value": str(checkpoint_cfg.get("enabled", False))},
            {"setting": "checkpoint.base_dir", "value": checkpoint_cfg.get("base_dir", "checkpoints")},
            {"setting": "telemetry.enabled", "value": str(telemetry_cfg.get("enabled", False))},
            {"setting": "telemetry.otlp_endpoint", "value": telemetry_cfg.get("otlp_endpoint", "")},
        ]
    )

    mo.md("## Active Configuration")
    return checkpoint_cfg, mo, results_cfg, settings_df, telemetry_cfg


@app.cell
def _(settings_df):
    """Display the active configuration table."""
    return settings_df


@app.cell
def _(mo, telemetry_cfg):
    """Explain where live observability data is exposed."""
    grafana_url = "http://localhost:3000"
    collector_url = telemetry_cfg.get("otlp_endpoint", "http://localhost:4318")

    mo.md(
        f"""
        ## Live Observability

        Grafana is where you inspect live execution behavior:

        * Grafana: {grafana_url}
        * OTel collector endpoint: {collector_url}

        Use it for throughput, latency, CPU, memory, worker isolation,
        dataset slicing, and execution separation while the run is active.
        """
    )
    return


@app.cell
def _(checkpoint_cfg, json, mo, path_class, pd):
    """Inspect checkpoint progress files used for resume and recovery."""
    checkpoint_root = path_class(checkpoint_cfg.get("base_dir", "checkpoints"))
    checkpoint_files = sorted(checkpoint_root.glob("**/progress_*.json"))

    rows = []
    for checkpoint_file in checkpoint_files:
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": checkpoint_file.parent.name,
                "progress_file": str(checkpoint_file),
                "experiment": payload.get("experiment"),
                "step_committed": payload.get("step_committed"),
                "checkpoint_path": payload.get("checkpoint_path"),
                "timestamp": payload.get("timestamp"),
            }
        )

    checkpoint_df = pd.DataFrame(rows)
    mo.md("## Resume / Recovery State")
    return checkpoint_df, mo


@app.cell
def _(checkpoint_df, mo):
    """Display current checkpoint progress or explain its absence."""
    checkpoint_df if not checkpoint_df.empty else mo.md("No checkpoint progress files found.")
    return


@app.cell
def _(mo, path_class, results_cfg):
    """Describe where final experiment results are stored and how reruns behave."""
    runs_root = path_class(results_cfg.get("out_dir", "./runs"))
    parquet_files = sorted(runs_root.glob("**/*.parquet"))

    mo.md(
        f"""
        ## Final Results Storage

        Final experiment facts are stored in the results sink, not in Grafana.

        * Parquet root: {runs_root}
        * Parquet files found: {len(parquet_files)}
        * ClickHouse sink: enabled only when `results.sink = "clickhouse"`

        Re-running experiments in the same `runs` directory appends new Parquet
        files by default. Existing result files are preserved.
        """
    )
    return parquet_files, runs_root


@app.cell
def _(duckdb, parquet_files, pd):
    """Load raw Parquet rows and build an aggregated final-results summary."""
    if not parquet_files:
        raw_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        return raw_df, summary_df

    raw_df = duckdb.sql(
        """
        SELECT *
        FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
        """
    ).df()

    summary_df = duckdb.sql(
        """
        SELECT scenario,
               execution_id,
               experiment,
               policy,
               reward_function,
               AVG(fitness) AS avg_napfd,
               AVG(cost) AS avg_apfdc,
               AVG(prioritization_time) AS avg_prioritization_time,
               AVG(process_memory_rss_mib) AS avg_rss_mib,
               AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
               MAX(wall_time_seconds) AS wall_time_seconds
        FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
        GROUP BY scenario, execution_id, experiment, policy, reward_function
        ORDER BY avg_napfd DESC, avg_apfdc DESC
        """
    ).df()
    return raw_df, summary_df


@app.cell
def _(mo, summary_df):
    """Preview the aggregated final-results summary."""
    mo.md("## Final Results Summary")
    summary_df.head(25) if not summary_df.empty else mo.md("No Parquet results found yet.")
    return


@app.cell
def _(mo):
    """Show practical SQL snippets for DuckDB and ClickHouse users."""
    duckdb_example = """
SELECT scenario, execution_id, policy, AVG(fitness) AS avg_napfd
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY scenario, execution_id, policy
ORDER BY avg_napfd DESC;
""".strip()

    clickhouse_example = """
SELECT scenario, execution_id, policy, AVG(fitness) AS avg_napfd
FROM coleman_results
GROUP BY scenario, execution_id, policy
ORDER BY avg_napfd DESC;
""".strip()

    mo.vstack(
        [
            mo.md("## Query Snippets"),
            mo.md("### DuckDB"),
            mo.md(f"```sql\n{duckdb_example}\n```"),
            mo.md("### ClickHouse"),
            mo.md(f"```sql\n{clickhouse_example}\n```"),
        ]
    )
    return


@app.cell
def _(mo, runs_root, summary_df):
    """Export the current summary as a CSV artifact for reports."""
    export_dir = runs_root / "analysis"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "summary.csv"

    if not summary_df.empty:
        summary_df.to_csv(export_path, index=False)
        mo.md(f"## Export\nSummary exported to `{export_path}`.")
    else:
        mo.md("## Export\nNo summary exported because there are no result rows yet.")
    return


@app.cell
def _(mo, plt, sns, summary_df):
    """Plot average NAPFD per policy from the persisted final results."""
    if summary_df.empty:
        mo.md("## Analysis Plot\nRun an experiment first to generate result plots.")
        return

    top_policies = (
        summary_df.groupby("policy", as_index=False)["avg_napfd"].mean().sort_values("avg_napfd", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_policies, x="policy", y="avg_napfd", ax=ax)
    ax.set_title("Average NAPFD by Policy")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average NAPFD")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


@app.cell
def _(mo):
    """Summarize the persistence semantics of results and checkpoints."""
    mo.md(
        """
        ## Result Persistence Semantics

        * Parquet appends new files under `./runs/`
        * ClickHouse appends new rows to `coleman_results`
        * `execution_id` is the safest way to isolate one run analytically
        * checkpoints update the latest durable state for the same run and experiment
        * if you want a fresh analytical space, clean `./runs/` and optionally `./checkpoints/`
        """
    )
    return


@app.cell
def _(mo):
    """Close with recommended next steps after running the notebook."""
    mo.md(
        """
        ## Suggested Next Steps

        * Run `uv run python main.py` to generate fresh experiment data
        * Open Grafana to inspect live execution behavior while the run is active
        * Use the Parquet summary above for final comparisons and report export
        * Inspect `./checkpoints/` to verify resume and recovery progress
        * Switch to ClickHouse when you want a persistent analytical store instead of Parquet files
        """
    )
    return


if __name__ == "__main__":
    app.run()
