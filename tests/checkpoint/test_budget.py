"""Native budget-scoped checkpoint namespace and explicit legacy ambiguity regressions."""

import json

import pytest

from coleman.budget import BudgetMode
from coleman.checkpoint.budget import budget_checkpoint_config
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload


@pytest.mark.parametrize(
    "scenario",
    [
        "",
        "owned",
        "team/project",
        "progress_ex1.json",
        "team/progress_ex1.json/project",
        "budget-v1/" + "a" * 64 + "/project",
    ],
)
def test_close_budgets_and_modes_have_distinct_stable_namespaces(tmp_path, scenario):
    config = dict(enabled=True, base_dir=str(tmp_path), interval=1)
    points = [(BudgetMode.RATIO, 0.50001), (BudgetMode.RATIO, 0.50002), (BudgetMode.FIXED_TIME, 0.50001)]
    roots = [budget_checkpoint_config(config, points, mode, value)["base_dir"] for mode, value in points]
    assert len(set(roots)) == 3
    assert all("budget-v1" in r for r in roots)
    assert not list(tmp_path.iterdir())
    for index, root in enumerate(roots, 1):
        store = LocalCheckpointStore(root)
        store.save(CheckpointPayload(run_id=scenario, experiment=1, step=index, agents=[], bandit=None))
    for index, (mode, value) in enumerate(points, 1):
        root = budget_checkpoint_config(config, points, mode, value)["base_dir"]
        payload = LocalCheckpointStore(root).load(scenario, 1)
        assert payload is not None
        assert payload.step == index
    assert config == dict(enabled=True, base_dir=str(tmp_path), interval=1)


@pytest.mark.parametrize("scenario", ["owned", "team/project"])
def test_single_budget_preserves_actual_legacy_checkpoint_and_disabled_is_noop(tmp_path, scenario):
    config = dict(enabled=True, base_dir=str(tmp_path), interval=1)
    store = LocalCheckpointStore(str(tmp_path))
    store.save(CheckpointPayload(run_id=scenario, experiment=1, step=7, agents=[], bandit=None))
    marker = tmp_path / scenario / "progress_ex1.json"
    before = marker.read_bytes()
    point = (BudgetMode.RATIO, 0.5)
    resolved = budget_checkpoint_config(config, [point], *point)
    assert resolved == config
    payload = LocalCheckpointStore(resolved["base_dir"]).load(scenario, 1)
    assert payload is not None
    assert payload.step == 7
    assert marker.read_bytes() == before
    disabled = {**config, "enabled": False}
    assert budget_checkpoint_config(disabled, [point, (BudgetMode.RATIO, 1.0)], *point) == disabled


def test_directory_with_marker_like_name_is_not_legacy_checkpoint_evidence(tmp_path):
    (tmp_path / "progress_ex1.json").mkdir()
    config = dict(enabled=True, base_dir=str(tmp_path))
    points = [(BudgetMode.RATIO, 0.5), (BudgetMode.RATIO, 1.0)]
    resolved = budget_checkpoint_config(config, points, *points[0])
    assert resolved["base_dir"] != str(tmp_path)
    assert not (tmp_path / "budget-v1").exists()


@pytest.mark.parametrize(
    "scenario",
    ["owned", "owned_context", "extension-owned", "team/project", "budget-v1", "budget-v1/not-a-scope/project"],
)
def test_ambiguous_old_multi_budget_marker_rejects_without_reading_pickle_or_rewriting(tmp_path, scenario):
    legacy = tmp_path / scenario
    legacy.mkdir(parents=True)
    marker = legacy / "progress_ex1.json"
    original = json.dumps(dict(checkpoint_path="do-not-read.pkl", step_committed=3)).encode()
    marker.write_bytes(original)
    config = dict(enabled=True, base_dir=str(tmp_path))
    points = [(BudgetMode.RATIO, 0.5), (BudgetMode.RATIO, 1.0)]
    with pytest.raises(ValueError, match="Ambiguous legacy multi-budget checkpoint"):
        budget_checkpoint_config(config, points, *points[0])
    assert marker.read_bytes() == original
    assert not list(tmp_path.glob("budget-v1/" + "?" * 64))


@pytest.mark.parametrize("scenario", ["", "team/project", "budget-v1/" + "a" * 64 + "/project"])
def test_actual_nested_legacy_store_cannot_be_mistaken_for_a_new_budget_scope(tmp_path, scenario):
    store = LocalCheckpointStore(str(tmp_path))
    store.save(CheckpointPayload(run_id=scenario, experiment=1, step=7, agents=[], bandit=None))
    config = dict(enabled=True, base_dir=str(tmp_path))
    points = [(BudgetMode.RATIO, 0.5), (BudgetMode.RATIO, 1.0)]
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Ambiguous legacy multi-budget checkpoint"):
        budget_checkpoint_config(config, points, *points[0])
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    "kind", ["missing_run_id", "wrong_run_id", "oversized", "malformed", "invalid_encoding", "non_object"]
)
def test_ambiguous_scoped_progress_metadata_rejects_without_unpickling_or_writing(tmp_path, kind):
    config = dict(enabled=True, base_dir=str(tmp_path))
    points = [(BudgetMode.RATIO, 0.5), (BudgetMode.RATIO, 1.0)]
    root = budget_checkpoint_config(config, points, *points[0])["base_dir"]
    LocalCheckpointStore(root).save(
        CheckpointPayload(run_id="team/project", experiment=1, step=7, agents=[], bandit=None)
    )
    marker = next(tmp_path.rglob("progress_ex1.json"))
    metadata = json.loads(marker.read_bytes())
    if kind == "missing_run_id":
        metadata.pop("run_id")
    else:
        metadata["run_id"] = "another/project"
    payload = json.dumps(metadata).encode()
    if kind == "oversized":
        payload = b" " * (1024**2 + 1)
    elif kind == "malformed":
        payload = b"{"
    elif kind == "invalid_encoding":
        payload = b"\xff"
    elif kind == "non_object":
        payload = b"[]"
    marker.write_bytes(payload)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Ambiguous legacy multi-budget checkpoint"):
        budget_checkpoint_config(config, points, *points[0])
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
