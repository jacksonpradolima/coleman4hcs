"""Actual native execution across source, journal, sidecar and checkpoint failures."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.results.domain import DomainRecord
from coleman.shadow import BuildOpened, ShadowState, load_shadow_state


@pytest.mark.parametrize("stage", ["batch", "frame", "domain_record", "checkpoint"])
def test_native_interruption_recovery_preserves_first_source_and_required_ci(example, tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    write = example.ArtifactWriter.write_immutable
    collect = example.MonitorCollector.collect_domain_record
    save = LocalCheckpointStore.save
    interrupted = False

    def after_write(self, path, payload):
        nonlocal interrupted
        result = write(self, path, payload)
        target = (stage == "batch" and path.parent.name == "shadow-source-batches") or (
            stage == "frame" and path.parent.name == "frames" and path.name == "0003.json"
        )
        if target and not interrupted:
            interrupted = True
            raise OSError("injected after source commit")
        return result

    def after_record(self, record):
        nonlocal interrupted
        collect(self, record)
        if not interrupted:
            interrupted = True
            raise OSError("injected after sidecar commit")

    def after_checkpoint(self, payload):
        nonlocal interrupted
        save(self, payload)
        if payload.step == 2 and not interrupted:
            interrupted = True
            raise OSError("injected after native checkpoint")

    with monkeypatch.context() as fault:
        if stage in {"batch", "frame"}:
            fault.setattr(example.ArtifactWriter, "write_immutable", after_write)
        elif stage == "domain_record":
            fault.setattr(example.MonitorCollector, "collect_domain_record", after_record)
        else:
            fault.setattr(LocalCheckpointStore, "save", after_checkpoint)
        with pytest.raises(OSError, match="injected after"):
            example.main("owned")
    root = tmp_path / "owned"
    batches = list(root.rglob("shadow-source-batches/*.json"))
    assert len(batches) == 1
    first = batches[0].read_bytes()
    progress = next(root.rglob("progress_ex1.json"))
    assert json.loads(progress.read_bytes())["step_committed"] == (2 if stage == "checkpoint" else 1)
    report, native = example.main("owned")
    assert report.evidence().received_logical_builds == 2
    assert report.evidence().scored_logical_builds == 2
    assert report.evidence().outcome == "inconclusive"
    assert report.evidence().forward_logical_builds == 0
    assert batches[0].read_bytes() == first
    assert len(list(root.rglob("shadow-source-batches/*.json"))) == 2
    records = [DomainRecord.from_json(path.read_bytes()) for path in root.rglob("domain-records/*.json")]
    assert len(records) == len({record.logical_key for record in records}) == 2
    builds = report.state.builds()
    assert len(builds) == 2
    assert all(build.actual_ledger.balance.consumed >= 7000 for build in builds)
    assert all(build.actual_ledger.balance.reserved == 0 for build in builds)
    assert all(build.resource_accounting == "settled" for build in builds)
    actual = pl.read_parquet(str(native.artifacts_dir) + "/results/**/*.parquet", hive_partitioning=True)
    assert actual.height == 3
    snapshots = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("frames/*.json")}
    again, replay_native = example.main("owned")
    assert again.to_json() == report.to_json()
    assert replay_native.run_id == native.run_id
    assert snapshots == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots}
    assert {row.event.build_id for row in report.state.journal if isinstance(row.event, BuildOpened)} == {
        "build-2",
        "build-3",
    }


def test_predecision_native_observer_ignores_future_duration_and_failure(example, tmp_path):
    config = example.configuration()
    rankings = []
    for index, (durations, verdicts) in enumerate((([1, 2, 4], [0, 1, 1]), ([999, 0, 888], [1, 0, 0]))):
        root = tmp_path / str(index)
        observer = example.RequiredSuiteObserver(config, root)
        agent = SimpleNamespace(
            t=1, actions=pl.DataFrame({"Name": example.BASE, "Duration": durations, "Verdict": verdicts})
        )
        assert observer.choose_all(agent) == list(example.BASE)
        state = load_shadow_state(config.protocol, example.ArtifactWriter(root))
        opening = state.source_event("build-2:opened")
        assert isinstance(opening, BuildOpened)
        rankings.append(tuple(decision.request.ranking for decision in opening.decisions))
        assert all(decision.signals == () or len(decision.signals) == 3 for decision in opening.decisions)
    assert rankings[0] == rankings[1]


@pytest.mark.parametrize("child", ["inputs", "native", "checkpoints", "example-inputs.json"])
def test_example_rejects_symlink_resume_without_writing_target(example, tmp_path, monkeypatch, child):
    root = tmp_path / "owned"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / child).symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        example.main("owned")
    assert not list(outside.iterdir())


def test_native_resume_checks_actual_specification_before_checkpoint_read(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example.main("owned")
    spec = tmp_path / "owned/native-spec.json"
    data = json.loads(spec.read_bytes())
    data["execution"]["seed"] = 999
    spec.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Conflicting immutable"):
        example.main("owned")


def test_native_artifact_manifest_lists_committed_shadow_contracts(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report, native = example.main("owned")
    manifest = json.loads((Path(native.artifacts_dir) / "manifest.json").read_bytes())
    kinds = {item["type"] for item in manifest["artifacts"]}
    assert {"shadow_protocol", "shadow_event", "shadow_checkpoint", "shadow_source_batch", "domain_records"} <= kinds
    from coleman.manifest import generate_manifest

    manifest = json.loads(generate_manifest(tmp_path / "owned", run_id="shadow-report").read_bytes())
    assert any(item["type"] == "shadow_report" for item in manifest["artifacts"])
    assert ShadowState.from_json(report.state.to_json()).fingerprint == report.state.fingerprint
