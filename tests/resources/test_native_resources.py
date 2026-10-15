"""Real native checkpoints and resource artifacts across publication/collection failures."""

import hashlib
import json
from types import SimpleNamespace

import polars as pl
import pytest

from coleman.resources import ResourceLedger, ResourceReplayRun
from coleman.results.domain import DomainRecord


@pytest.mark.parametrize("stage", ["run", "ledger", "domain_record", "checkpoint"])
def test_native_recovery_has_no_double_consumption_or_duplicate_records(example, tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    publish_run, publish_ledger = example.publish_resource_run, example.publish_ledger
    collect = example.MonitorCollector.collect_domain_record
    from coleman.checkpoint.checkpoint_store import LocalCheckpointStore

    save = LocalCheckpointStore.save
    interrupted = False

    def after_run(run, writer):
        nonlocal interrupted
        path = publish_run(run, writer)
        if not interrupted:
            interrupted = True
            raise OSError("injected after resource run")
        return path

    def after_ledger(ledger, writer):
        nonlocal interrupted
        path = publish_ledger(ledger, writer)
        if not interrupted:
            interrupted = True
            raise OSError("injected after ledger checkpoint")
        return path

    def after_record(self, record):
        nonlocal interrupted
        collect(self, record)
        if not interrupted:
            interrupted = True
            raise OSError("injected after domain record")

    def after_checkpoint(self, payload):
        nonlocal interrupted
        save(self, payload)
        if payload.step == 2 and not interrupted:
            interrupted = True
            raise OSError("injected after native checkpoint")

    with monkeypatch.context() as fault:
        if stage == "run":
            fault.setattr(example, "publish_resource_run", after_run)
        elif stage == "ledger":
            fault.setattr(example, "publish_ledger", after_ledger)
        elif stage == "domain_record":
            fault.setattr(example.MonitorCollector, "collect_domain_record", after_record)
        else:
            fault.setattr(LocalCheckpointStore, "save", after_checkpoint)
        with pytest.raises(OSError, match="injected after"):
            example.main("owned")
    root = tmp_path / "owned"
    initial = list(root.rglob("resource-runs/*.json"))
    assert len(initial) == 1
    recorded = initial[0].read_bytes()
    markers = list(root.rglob("progress_ex1.json"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_bytes())["step_committed"] == (2 if stage == "checkpoint" else 1)
    report = example.main("owned")
    assert report.status == "completed"
    assert len(report.runs) == 4
    assert initial[0].read_bytes() == recorded
    runs = [ResourceReplayRun.from_json(path.read_bytes()) for path in root.rglob("resource-runs/*.json")]
    ledgers = [
        ResourceLedger.from_json(path.read_bytes()) for path in root.rglob("resource-ledgers/*/checkpoints/*.json")
    ]
    records = [DomainRecord.from_json(path.read_bytes()) for path in root.rglob("domain-records/*.json")]
    assert len(runs) == len(ledgers) == len(records) == 4
    assert {run.ledger.to_json() for run in runs} == {ledger.to_json() for ledger in ledgers}
    assert len({record.logical_key for record in records}) == 4
    assert len({record.execution_id for record in records}) == 2
    native_roots = [path.parent for path in root.rglob("manifest.json") if (path.parent / "results").exists()]
    frames = [pl.read_parquet(str(path / "results/**/*.parquet"), hive_partitioning=True) for path in native_roots]
    assert len(frames) == 2
    assert all(frame.height == 3 for frame in frames)
    snapshots = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("resource-runs/*.json")}
    assert example.main("owned").to_json() == report.to_json()
    assert snapshots == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots}


def test_native_policy_uses_prior_estimates_only(example):
    protocol, _ = example.fixture()
    policy = example.EstimateOrder(protocol, "shortest-estimate")
    for durations, verdicts in (([1, 2, 4], [0, 1, 1]), ([999, 0, 888], [1, 0, 0])):
        agent = SimpleNamespace(
            actions=pl.DataFrame({"Name": ["t1", "t2", "t3"], "Duration": durations, "Verdict": verdicts})
        )
        assert policy.choose_all(agent) == ["t2", "t3", "t1"]


@pytest.mark.parametrize("child", ["inputs", "base-order", "shortest-estimate", "example-inputs.json"])
def test_native_example_rejects_existing_symlink_children(example, tmp_path, monkeypatch, child):
    root = tmp_path / "owned"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / child).symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        example.main("owned")
    assert not list(outside.iterdir())


def test_native_resume_binds_actual_runspec_before_checkpoint_read(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example.main("owned")
    spec = tmp_path / "owned/native-spec-base-order.json"
    changed = json.loads(spec.read_bytes())
    changed["execution"]["seed"] = 999
    spec.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Conflicting immutable"):
        example.main("owned")
