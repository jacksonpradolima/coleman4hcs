"""Real native extension, checkpoint continuation and point-in-time policy integration."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from coleman.observations import DecisionSignal, QuantityMeanModel
from coleman.results.domain import DomainRecord

ROOT = Path(__file__).parents[2]


@pytest.fixture
def example():
    name = "coleman_owned_observation_ci_fixture"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/point-in-time/ci.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.mark.parametrize("stage", ["signal", "domain_record", "checkpoint"])
def test_native_checkpoint_resumes_after_signal_publication_without_duplicates(example, tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    publish = example.publish_signal
    interrupted = False

    def fail_after_complete_signal(signal, writer):
        nonlocal interrupted
        path = publish(signal, writer)
        if not interrupted:
            interrupted = True
            raise OSError("injected after complete signal and before native collection/checkpoint")
        return path

    collect_record = example.MonitorCollector.collect_domain_record

    def fail_after_domain_record(self, record):
        nonlocal interrupted
        collect_record(self, record)
        if not interrupted:
            interrupted = True
            raise OSError("injected after complete domain record and before checkpoint")

    from coleman.checkpoint.checkpoint_store import LocalCheckpointStore

    save = LocalCheckpointStore.save

    def fail_after_checkpoint(self, payload):
        nonlocal interrupted
        save(self, payload)
        if payload.step == 2 and not interrupted:
            interrupted = True
            raise OSError("injected after committed checkpoint")

    with monkeypatch.context() as fault:
        if stage == "signal":
            fault.setattr(example, "publish_signal", fail_after_complete_signal)
        elif stage == "domain_record":
            fault.setattr(example.MonitorCollector, "collect_domain_record", fail_after_domain_record)
        else:
            fault.setattr(LocalCheckpointStore, "save", fail_after_checkpoint)
        with pytest.raises(OSError, match="injected after"):
            example.run_case("owned", 604800)
    markers = list((tmp_path / "owned/native").rglob("progress_ex1.json"))
    assert len(markers) == 1
    progress = markers[0]
    assert json.loads(progress.read_text())["step_committed"] == (2 if stage == "checkpoint" else 1)
    pending = list((tmp_path / "owned/native").rglob("decision-signals/*.json"))
    assert len(pending) == (1 if stage == "signal" else 3)
    assert DecisionSignal.from_json(pending[0].read_bytes()).request.decision_id == "build-2"
    assert (
        len(list((tmp_path / "owned/native").rglob("domain-records/*.json")))
        == {
            "signal": 0,
            "domain_record": 1,
            "checkpoint": 3,
        }[stage]
    )
    resumed = example.run_case("owned", 604800)
    assert json.loads(progress.read_text())["step_committed"] == 3
    root = Path(resumed.artifacts_dir)
    signals = [DecisionSignal.from_json(path.read_bytes()) for path in root.rglob("decision-signals/*.json")]
    records = [DomainRecord.from_json(path.read_bytes()) for path in root.rglob("domain-records/*.json")]
    assert len(signals) == len(records) == 6
    assert len({record.execution_id for record in records}) == 1
    assert len({signal.logical_id for signal in signals}) == 6
    assert {record.decision_id for record in records} == {signal.logical_id for signal in signals}
    assert all(signal.availability == "available" for signal in signals)
    native = pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True)
    assert native.height == 3
    for row in native.filter(pl.col("step") > 1).iter_rows(named=True):
        assert json.loads(row["prioritization_order_top_k"]) == ["t2", "t3", "t1"]
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert sum(item["type"] == "decision_signal" for item in manifest["artifacts"]) == 6
    assert sum(item["type"] == "observation_model" for item in manifest["artifacts"]) == 1
    again = example.run_case("owned", 604800)
    assert again.run_id == resumed.run_id
    assert len(list(root.rglob("domain-records/*.json"))) == 6
    assert pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True).height == 3
    marker = progress.read_bytes()
    with pytest.raises(ValueError, match="Conflicting immutable"):
        example.run_case("owned", 0)
    assert progress.read_bytes() == marker


def test_stale_native_signals_use_full_base_order(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = example.run_case("fallback", 0)
    root = Path(result.artifacts_dir)
    signals = [DecisionSignal.from_json(path.read_bytes()) for path in root.rglob("decision-signals/*.json")]
    assert len(signals) == 6
    assert all(signal.availability == "stale" and signal.measurement.value is None for signal in signals)
    native = pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True)
    for row in native.filter(pl.col("step") > 1).iter_rows(named=True):
        assert json.loads(row["prioritization_order_top_k"]) == ["t1", "t2", "t3"]
    models = list(root.rglob("observation-models/*.json"))
    assert len(models) == 1
    model = QuantityMeanModel.from_json(models[0].read_bytes())
    assert [estimate.quantity.value for estimate in model.estimates] == [4, 1, 2]


def test_policy_does_not_read_realized_durations_or_verdicts(example):
    configuration = example.DemoConfiguration(
        schema_version=1,
        model=example.frozen_model(example.owned_history()),
        maximum_event_age=example.Quantity(value=604800, semantics=example.EVENT_AGE_SECONDS),
    )
    policy = example.MeanOrder(configuration)
    for durations, verdicts in (([1, 2, 4], [0, 1, 1]), ([999, 0, 100], [1, 0, 0])):
        agent = SimpleNamespace(
            t=1, actions=pl.DataFrame({"Name": ["t1", "t2", "t3"], "Duration": durations, "Verdict": verdicts})
        )
        assert policy.choose_all(agent) == ["t2", "t3", "t1"]


@pytest.mark.parametrize("child", ["inputs", "native", "checkpoints", "example-inputs.json"])
def test_native_example_rejects_existing_symlink_children(example, tmp_path, monkeypatch, child):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root = workspace / "owned"
    root.mkdir(parents=True)
    outside.mkdir()
    (root / child).symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(workspace)
    with pytest.raises(ValueError, match="symlinks"):
        example.run_case("owned", 604800)
    assert not list(outside.iterdir())


def test_native_resume_rejects_changed_actual_runspec(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example.run_case("owned", 604800)
    path = tmp_path / "owned/native-spec.json"
    altered = json.loads(path.read_bytes())
    altered["execution"]["seed"] = 123
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Conflicting immutable"):
        example.run_case("owned", 604800)
