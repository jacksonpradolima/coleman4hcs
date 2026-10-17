"""Native budget-scoped checkpoint namespace and explicit legacy ambiguity regressions."""

import json

import pytest

from coleman.budget import BudgetMode
from coleman.checkpoint.budget import budget_checkpoint_config
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload


def test_close_budgets_and_modes_have_distinct_stable_namespaces(tmp_path):
    config = dict(enabled=True, base_dir=str(tmp_path), interval=1)
    points = [(BudgetMode.RATIO, 0.50001), (BudgetMode.RATIO, 0.50002), (BudgetMode.FIXED_TIME, 0.50001)]
    roots = [budget_checkpoint_config(config, points, mode, value)["base_dir"] for mode, value in points]
    assert len(set(roots)) == 3
    assert all("budget-v1" in r for r in roots)
    assert not list(tmp_path.iterdir())
    for index, root in enumerate(roots, 1):
        store = LocalCheckpointStore(root)
        store.save(CheckpointPayload(run_id="owned", experiment=1, step=index, agents=[], bandit=None))
    for index, (mode, value) in enumerate(points, 1):
        root = budget_checkpoint_config(config, points, mode, value)["base_dir"]
        payload = LocalCheckpointStore(root).load("owned", 1)
        assert payload is not None
        assert payload.step == index
    assert config == dict(enabled=True, base_dir=str(tmp_path), interval=1)


def test_single_budget_preserves_actual_legacy_checkpoint_and_disabled_is_noop(tmp_path):
    config = dict(enabled=True, base_dir=str(tmp_path), interval=1)
    store = LocalCheckpointStore(str(tmp_path))
    store.save(CheckpointPayload(run_id="owned", experiment=1, step=7, agents=[], bandit=None))
    marker = tmp_path / "owned/progress_ex1.json"
    before = marker.read_bytes()
    point = (BudgetMode.RATIO, 0.5)
    resolved = budget_checkpoint_config(config, [point], *point)
    assert resolved == config
    payload = LocalCheckpointStore(resolved["base_dir"]).load("owned", 1)
    assert payload is not None
    assert payload.step == 7
    assert marker.read_bytes() == before
    disabled = {**config, "enabled": False}
    assert budget_checkpoint_config(disabled, [point, (BudgetMode.RATIO, 1.0)], *point) == disabled


@pytest.mark.parametrize("scenario", ["owned", "owned_context", "extension-owned"])
def test_ambiguous_old_multi_budget_marker_rejects_without_reading_pickle_or_rewriting(tmp_path, scenario):
    legacy = tmp_path / scenario
    legacy.mkdir()
    marker = legacy / "progress_ex1.json"
    original = json.dumps(dict(checkpoint_path="do-not-read.pkl", step_committed=3)).encode()
    marker.write_bytes(original)
    config = dict(enabled=True, base_dir=str(tmp_path))
    points = [(BudgetMode.RATIO, 0.5), (BudgetMode.RATIO, 1.0)]
    with pytest.raises(ValueError, match="Ambiguous legacy multi-budget checkpoint"):
        budget_checkpoint_config(config, points, *points[0])
    assert marker.read_bytes() == original
    assert not (tmp_path / "budget-v1").exists()
