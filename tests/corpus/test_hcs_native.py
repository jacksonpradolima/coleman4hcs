"""Real native WTS/VTS, independently counted budgets/denominators and existing checkpoint recovery."""

import json
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from coleman.application.hcs import run_hcs_replay
from coleman.budget import BudgetMode
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.environment import Environment
from coleman.evaluation import NAPFDVerdictMetric
from coleman.policy import RandomPolicy
from coleman.reward import RNFailReward
from coleman.runner import create_agents, get_scenario_provider
from tests.corpus.test_hcs_etl import export


def check_native_accounting(completion, artifacts, strategy):
    """Independently count tiny suites on their declared projected cost lattice; never call the metric being checked."""
    root = completion.parent
    rows = pl.read_parquet(str(Path(artifacts) / "results/**/*.parquet"), hive_partitioning=True)
    union = pl.read_parquet(root / "datasets/owned-hcs/features-engineered.parquet")
    variants = pl.read_parquet(root / "datasets/owned-hcs/data-variants.parquet")
    expected = []
    for row in rows.iter_rows(named=True):
        if strategy == "wts":
            build = sorted(union["BuildId"].unique().to_list())[row["step"] - 1]
            suite = union if row["variant"] is None else variants.filter(pl.col("Variant") == row["variant"])
        else:
            suite = variants.filter(pl.col("Variant") == row["scenario"])
            build = sorted(suite["BuildId"].unique().to_list())[row["step"] - 1]
        suite = suite.filter(pl.col("BuildId") == build)
        assert suite.height
        order = json.loads(row["prioritization_order_top_k"])
        cases = {r["Name"]: r for r in suite.iter_rows(named=True)}
        assert len(order) == len(set(order)) == len(cases)
        assert set(order) == cases.keys()
        total = sum(r["Duration"] for r in cases.values())
        budget = total * row["budget_value"]
        cumulative, selected = 0, []
        for name in order:
            cumulative += cases[name]["Duration"]
            if cumulative <= budget:
                selected.append(name)
        detected = sum(cases[name]["Verdict"] for name in selected)
        nonpass = sum(c["Verdict"] for c in cases.values())
        assert row["total_build_duration"] == total
        assert row["tests_ran"] == len(selected)
        assert row["tests_not_ran"] == len(cases) - len(selected)
        assert row["detected"] == detected
        assert row["missed"] == nonpass - detected
        assert row["detected"] + row["missed"] == nonpass
        expected.append((row["scenario"], row["policy"], row["budget_value"], row["step"], row["variant"]))
    assert len(expected) == len(set(expected))
    return rows


@pytest.mark.parametrize("strategy,expected_rows", [("wts", 24), ("vts", 16)])
def test_actual_native_strategies_per_variant_denominators_and_idempotence(tmp_path, strategy, expected_rows):
    receipt, cache, completion = export(tmp_path)
    first = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    second = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    assert first.run_id == second.run_id
    assert second.artifacts_dir is not None
    rows = check_native_accounting(completion, second.artifacts_dir, strategy)
    assert rows.height == expected_rows
    assert set(rows["budget_value"]) == {0.5, 1.0}
    assert len(set(rows["policy"])) == 2
    markers = list(Path(second.artifacts_dir).rglob("progress_ex1.json"))
    assert markers
    if strategy == "wts":
        assert all(json.loads(m.read_bytes())["step_committed"] == 2 for m in markers)
    manifest = json.loads((Path(second.artifacts_dir) / "manifest.json").read_bytes())
    assert sum(x["type"] == "corpus_etl_receipt" for x in manifest["artifacts"]) == 1
    assert receipt.scope == "posthoc-test-only-serial-HCS-simulation"


@pytest.mark.parametrize("strategy", ["wts", "vts"])
def test_interrupted_native_matrix_continues_existing_checkpoint_without_double_rows(tmp_path, monkeypatch, strategy):
    _, cache, completion = export(tmp_path)
    original, interrupted = LocalCheckpointStore.save, False

    def fail_after_commit(self, payload):
        nonlocal interrupted
        original(self, payload)
        if not interrupted:
            interrupted = True
            raise OSError("owned interruption after HCS native checkpoint")

    with monkeypatch.context() as fault:
        fault.setattr(LocalCheckpointStore, "save", fail_after_commit)
        with pytest.raises(OSError, match="after HCS native checkpoint"):
            run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    markers = list((tmp_path / "native").rglob("progress_ex1.json"))
    assert markers
    assert json.loads(markers[0].read_bytes())["step_committed"] == 1
    after = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    rows = check_native_accounting(completion, after.artifacts_dir, strategy)
    assert rows.height == (24 if strategy == "wts" else 16)


@pytest.mark.parametrize("strategy,expected", [("wts", 24), ("vts", 16)])
@pytest.mark.parametrize("failed_save", [1, 2])
def test_flushed_uncommitted_tail_is_reconciled_before_native_resume(
    tmp_path, monkeypatch, strategy, expected, failed_save
):
    """Exercise the real gap between result flush and checkpoint commit, with/without older committed progress."""
    _, cache, completion = export(tmp_path)
    original, saves = LocalCheckpointStore.save, 0

    def fail_before_commit(self, payload):
        nonlocal saves
        saves += 1
        if saves == failed_save:
            raise OSError("owned interruption before checkpoint commit")
        original(self, payload)

    with monkeypatch.context() as fault:
        fault.setattr(LocalCheckpointStore, "save", fail_before_commit)
        with pytest.raises(OSError, match="before checkpoint commit"):
            run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    assert list((tmp_path / "native").rglob("*.parquet"))
    assert not list((tmp_path / "native").rglob("manifest.json"))
    after = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    rows = check_native_accounting(completion, after.artifacts_dir, strategy)
    assert rows.height == expected
    again = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    assert again.run_id == after.run_id
    assert check_native_accounting(completion, again.artifacts_dir, strategy).equals(rows)


def test_progress_without_checkpoint_rejects_without_discarding_any_native_results(tmp_path):
    _, cache, completion = export(tmp_path)
    result = run_hcs_replay(completion, cache, tmp_path / "native", strategy="wts")
    files = list((tmp_path / "native").rglob("*.parquet"))
    before = {p: p.read_bytes() for p in files}
    marker = next((tmp_path / "native").rglob("progress_ex1.json"))
    Path(json.loads(marker.read_bytes())["checkpoint_path"]).unlink()
    with pytest.raises(ValueError, match="without loadable state"):
        run_hcs_replay(completion, cache, tmp_path / "native", strategy="wts")
    assert result.artifacts_dir is not None
    assert {p: p.read_bytes() for p in files} == before


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("custom", ["native", "per_round_only"])
def test_direct_native_environment_preserves_buffered_preceding_execution(tmp_path, enabled, custom):
    """The library Environment may share one sink across executions with intervals beyond the final round."""
    _, _, completion = export(tmp_path)
    provider = get_scenario_provider(
        datasets_dir=str(completion.parent / "datasets"),
        dataset="owned-hcs",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
    )

    class PerRoundBudgetProvider:
        """An existing custom provider can expose budgets on each virtual round only."""

        budget_mode = BudgetMode.RATIO
        budget_value = None

        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def __iter__(self):
            return iter(self.actual)

        def __str__(self):
            return str(self.actual)

    if custom != "native":
        provider = PerRoundBudgetProvider(provider)
    env = Environment(
        create_agents(RandomPolicy(), RNFailReward(), [], seed=42),
        provider,
        NAPFDVerdictMetric(),
        results_config={"enabled": True, "out_dir": str(tmp_path / "results")},
        checkpoint_config={"enabled": enabled, "base_dir": str(tmp_path / "checkpoints"), "interval": 100},
    )
    env.run(experiments=2, trials=2, forecast_next=False)
    env.store_experiment()
    rows = pl.read_parquet(str(tmp_path / "results/**/*.parquet"), hive_partitioning=True)
    assert rows.height == 4
    assert rows.group_by("experiment").len().sort("experiment")["len"].to_list() == [2, 2]
    if enabled and custom == "native":
        before = {p: p.read_bytes() for p in (tmp_path / "results").rglob("*.parquet")}
        env.run(experiments=2, trials=2, forecast_next=False)
        env.store_experiment()
        assert {p: p.read_bytes() for p in (tmp_path / "results").rglob("*.parquet")} == before
        markers = list((tmp_path / "checkpoints").rglob("progress_ex*.json"))
        assert len(markers) == 2
        assert all(json.loads(path.read_bytes())["step_committed"] == 2 for path in markers)


def test_new_native_checkpoint_root_preserves_results_from_another_run(tmp_path):
    """Actual native replay must reject a reused result tree paired with a different run's checkpoints."""
    _, _, completion = export(tmp_path)

    def environment(base, execution_id):
        provider = get_scenario_provider(str(completion.parent / "datasets"), "owned-hcs", 0.5, False, False, {}, {})
        return Environment(
            create_agents(RandomPolicy(), RNFailReward(), [], seed=42),
            provider,
            NAPFDVerdictMetric(),
            results_config={"enabled": True, "out_dir": str(tmp_path / "results")},
            checkpoint_config={"enabled": True, "base_dir": str(base), "interval": 1},
            runtime_metadata={"execution_id": execution_id},
        )

    first = environment(tmp_path / "first-checkpoints", "original-independent-run")
    first.run_single(1, trials=2, forecast_next=False)
    first.store_experiment()
    before = {p: p.read_bytes() for p in (tmp_path / "results").rglob("*.parquet")}
    second = environment(tmp_path / "new-checkpoints", "different-independent-run")
    with pytest.raises(ValueError, match="not paired"):
        second.run_single(1, trials=2, forecast_next=False)
    assert {p: p.read_bytes() for p in (tmp_path / "results").rglob("*.parquet")} == before
    assert not list((tmp_path / "new-checkpoints").rglob("progress_ex1.json"))


def test_invalid_strategy_and_empty_projection_reject_before_native_execution(tmp_path):
    def unavailable(rows):
        for row in rows:
            row["duration"] = ""
        return rows

    _, cache, completion = export(tmp_path, rows_transform=unavailable)
    with pytest.raises(ValueError, match="no compatible native"):
        run_hcs_replay(completion, cache, tmp_path / "native", strategy="wts")
    invalid_strategy = cast(Any, "future")
    with pytest.raises(ValueError, match="strategy must be"):
        run_hcs_replay(completion, cache, tmp_path / "native", strategy=invalid_strategy)


@pytest.mark.parametrize("strategy", ["wts", "vts"])
def test_decimal_cost_projection_aligns_loader_bandit_and_variant_full_budgets(tmp_path, strategy):
    def decimals(rows):
        for row in rows:
            if row["duration"]:
                row["duration"] = {"A": "0.01", "B": "14.47", "C": "0.07"}[row["test_name"]]
        return rows

    _, cache, completion = export(tmp_path, rows_transform=decimals)
    observations = pl.read_parquet(completion.parent / "observations.parquet")
    pennies = observations.filter(pl.col("duration_lexeme") == "0.01")
    assert pennies["duration_decimal"].unique().to_list() == ["0.01"]
    assert pennies["native_duration_seconds"].unique().to_list() == [0.009999999776482582]
    result = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    rows = check_native_accounting(completion, result.artifacts_dir, strategy)
    full = rows.filter(pl.col("budget_value") == 1.0)
    assert full.height
    assert full["tests_not_ran"].sum() == 0


def test_unsupported_projected_dynamic_range_rejects_instead_of_unstable_full_budget(tmp_path):
    def wide_range(rows):
        for row in rows:
            if row["duration"]:
                row["duration"] = {"A": "1000000000000", "B": "0.000000000001", "C": "1"}[row["test_name"]]
        return rows

    with pytest.raises(ValueError, match="stable native prefix sums"):
        export(tmp_path, rows_transform=wide_range)
    assert not list((tmp_path / "etl").rglob("complete.json"))


@pytest.mark.parametrize("strategy,expected", [("wts", 12), ("vts", 8)])
def test_fully_excluded_round_preserves_pipeline_mapping_and_processed_checkpoint_step(tmp_path, strategy, expected):
    def unavailable_first(rows):
        for row in rows:
            if row["pipeline_id"] == "101":
                row["duration"] = ""
        return rows

    _, cache, completion = export(tmp_path, rows_transform=unavailable_first)
    native = pl.read_parquet(completion.parent / "datasets/owned-hcs/features-engineered.parquet")
    assert native["BuildId"].unique().to_list() == [2]
    first = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    second = run_hcs_replay(completion, cache, tmp_path / "native", strategy=strategy)
    assert first.run_id == second.run_id
    rows = check_native_accounting(completion, second.artifacts_dir, strategy)
    assert rows.height == expected
    assert rows["step"].unique().to_list() == [1]
