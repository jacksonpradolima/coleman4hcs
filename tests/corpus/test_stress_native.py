"""Real native CI policies/sinks/checkpoints with independent source accounting."""

import json
from pathlib import Path

import pytest

from coleman.application.stress import run_stress_replay
from coleman.artifacts import ArtifactWriter
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.corpus.stress_artifacts import export_stress
from tests.corpus.stress_support import case, check_native


@pytest.mark.parametrize("feedback,budget", [("censored", "severe"), ("full", "full")])
def test_actual_native_rankings_exact_tiny_metrics_and_completed_checkpoint_retry(tmp_path, feedback, budget):
    value = case(variants="changing", duration="heavy_tail", feedback=feedback, budget=budget)
    completion = export_stress(value, ArtifactWriter(tmp_path / "export"))
    a = run_stress_replay(completion, tmp_path / "native")
    assert a.artifacts_dir is not None
    rows = check_native(a, value)
    assert rows.height == 8
    stable = {
        p: p.read_bytes() for p in Path(a.artifacts_dir).rglob("*") if p.is_file() and p.suffix in {".parquet", ".pkl"}
    }
    b = run_stress_replay(completion, tmp_path / "native")
    assert b.run_id == a.run_id
    assert check_native(b, value).equals(rows)
    assert stable == {p: p.read_bytes() for p in stable}
    markers = list(Path(a.artifacts_dir).rglob("progress_ex1.json"))
    assert markers
    assert all(json.loads(p.read_bytes())["step_committed"] == value.builds for p in markers)
    manifest = json.loads((Path(a.artifacts_dir) / "manifest.json").read_bytes())
    assert sum(item["type"] == "corpus_etl_receipt" for item in manifest["artifacts"]) == 1


@pytest.mark.parametrize("when", ["before", "after"])
def test_actual_checkpoint_interruption_recovers_without_duplicate_or_omitted_rows(tmp_path, monkeypatch, when):
    value = case(churn="collisions", variants="changing")
    completion = export_stress(value, ArtifactWriter(tmp_path / "export"))
    original = LocalCheckpointStore.save
    interrupted = False

    def stop(self, payload):
        nonlocal interrupted
        if interrupted:
            return original(self, payload)
        interrupted = True
        if when == "after":
            original(self, payload)
        raise OSError("owned stop at native checkpoint")

    with monkeypatch.context() as fault:
        fault.setattr(LocalCheckpointStore, "save", stop)
        with pytest.raises(OSError, match="owned stop"):
            run_stress_replay(completion, tmp_path / "native")
    assert not list((tmp_path / "native").rglob("manifest.json"))
    after = run_stress_replay(completion, tmp_path / "native")
    assert check_native(after, value).height == 8
    assert run_stress_replay(completion, tmp_path / "native").run_id == after.run_id


@pytest.mark.parametrize(
    "feedback,budget", [("delayed", "full"), ("missing", "full"), ("corrected", "full"), ("full", "severe")]
)
def test_ineligible_feedback_fails_before_native_output(tmp_path, feedback, budget):
    completion = export_stress(case(feedback=feedback, budget=budget), ArtifactWriter(tmp_path / "export"))
    with pytest.raises(ValueError, match="Native RNFail|Full-label"):
        run_stress_replay(completion, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("policies", [(), ("Random", "Random"), ("LinUCB",)])
def test_unsupported_candidates_reject_before_output(tmp_path, policies):
    completion = export_stress(case(), ArtifactWriter(tmp_path / "export"))
    with pytest.raises(ValueError, match="Random/UCB"):
        run_stress_replay(completion, tmp_path / "must-not-exist", policies=policies)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("churn", ["additions", "removals"])
def test_rare_churn_native_full_budget_detects_one_failure_in_every_active_build(tmp_path, churn):
    value = case(failures="rare", churn=churn, variants="changing", budget="full", feedback="full", builds=4)
    completion = export_stress(value, ArtifactWriter(tmp_path / "export"))
    result = run_stress_replay(completion, tmp_path / "native", policies=("Random",))
    rows = check_native(result, value)
    assert rows.height == 4
    assert rows["detected"].to_list() == [1] * 4
    assert rows["missed"].to_list() == [0] * 4


@pytest.mark.parametrize("churn", ["additions", "removals"])
def test_zero_work_cases_remain_valid_exports_but_reject_native_undefined_cost_before_output(tmp_path, churn):
    value = case(failures="rare", churn=churn, builds=8, feedback="full", budget="full")
    completion = export_stress(value, ArtifactWriter(tmp_path / "export"))
    with pytest.raises(ValueError, match="positive full-suite work"):
        run_stress_replay(completion, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
