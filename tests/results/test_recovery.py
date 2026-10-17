"""Native checkpoint recovery retains committed evidence and retries interrupted atomic shard updates."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import polars as pl
import pyarrow.parquet as pq
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.budget import BudgetMode
from coleman.results.parquet_sink import ParquetSink
from coleman.results.recovery import _OWNER_METADATA_KEY, ResultCheckpoint, _CheckpointBinding, _CheckpointOwner
from tests.results.test_results import _make_row


def checkpoint(**changes):
    return replace(ResultCheckpoint("TestScenario", 1, BudgetMode.RATIO, 0.5, 1), **changes)


def recover(sink, progress):
    """Use the same persisted checkpoint-directory pairing for one owned test run."""
    sink.recover_checkpoint(progress, checkpoint_root=Path(sink.out_dir) / ".owned-test-checkpoints")


def prepared_sink(root, experiment=1):
    """Bind before the first row, matching actual native Environment orchestration."""
    sink = ParquetSink(str(root))
    recover(sink, checkpoint(experiment=experiment, step=0))
    return sink


def read(root):
    return pl.read_parquet(str(root / "**/*.parquet"), hive_partitioning=True).sort("execution_id")


def test_recovery_preserves_committed_rows_other_coordinates_schema_and_idempotence(tmp_path):
    sink = prepared_sink(tmp_path)
    rows = [
        _make_row(step=1, execution_id="committed"),
        _make_row(step=2, execution_id="uncommitted"),
        _make_row(step=2, variant="linux", execution_id="uncommitted-variant"),
        _make_row(step=2, experiment=2, execution_id="other-experiment"),
        _make_row(step=2, scenario="Other", execution_id="other-scenario"),
        _make_row(step=2, budget_value=0.5000000000000001, execution_id="close-budget"),
        _make_row(step=2, budget_mode="fixed_time", execution_id="other-mode"),
    ]
    for row in rows:
        sink.write_row(row)
    sink.flush()
    before = read(tmp_path)
    original_metadata = {p: pq.read_schema(p).metadata for p in tmp_path.rglob("*.parquet")}
    recover(sink, checkpoint())
    after = read(tmp_path)
    assert after.schema == before.schema
    expected = before.filter(~pl.col("execution_id").str.starts_with("uncommitted"))
    assert after.equals(expected)
    assert all(pq.read_schema(p).metadata == original_metadata[p] for p in tmp_path.rglob("*.parquet"))
    files = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    recover(sink, checkpoint())
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == files


def test_absent_checkpoint_removes_all_uncommitted_rows_only(tmp_path):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row())
    sink.flush()
    recover(sink, checkpoint(step=0))
    assert not list(tmp_path.rglob("*.parquet"))
    recover(sink, checkpoint(step=0))


def test_atomic_replacement_error_preserves_original_and_blocks_recovery(tmp_path, monkeypatch):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row(step=1))
    sink.write_row(_make_row(step=2))
    sink.flush()
    path = next(tmp_path.rglob("*.parquet"))
    original = path.read_bytes()

    def fail(*args):
        raise OSError("owned atomic replacement failure")

    with monkeypatch.context() as fault:
        fault.setattr("coleman.results.recovery.os.replace", fail)
        progress = checkpoint()
        with pytest.raises(OSError, match="atomic replacement failure"):
            recover(sink, progress)
    assert path.read_bytes() == original
    assert not list(tmp_path.rglob(".recovery-*.tmp"))
    recover(sink, checkpoint())
    assert read(tmp_path)["step"].to_list() == [1]


def test_recovery_requires_empty_buffer_and_never_discards_new_work(tmp_path):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row(step=2))
    progress = checkpoint()
    with pytest.raises(ValueError, match="empty result buffer"):
        recover(sink, progress)
    sink.flush()
    assert read(tmp_path)["step"].to_list() == [2]


def test_independent_recovery_removing_a_previously_enumerated_shard_is_safe(tmp_path, monkeypatch):
    first, other = prepared_sink(tmp_path), prepared_sink(tmp_path, experiment=2)
    first.write_row(_make_row(experiment=1))
    other.write_row(_make_row(experiment=2))
    first.flush()
    other.flush()
    original = pq.ParquetFile
    vanished = next(p for p in tmp_path.rglob("*.parquet") if pq.read_table(p)["experiment"].to_pylist() == [2])

    def concurrent_removal(path, *args, **kwargs):
        if path == vanished:
            path.unlink()
        return original(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr("coleman.results.recovery.pq.ParquetFile", concurrent_removal)
        recover(first, checkpoint(step=0))
    assert not list(tmp_path.rglob("*.parquet"))


def test_interrupted_reconciliation_converges_before_any_replayed_rows(tmp_path, monkeypatch):
    sink = prepared_sink(tmp_path)
    for policy in ("UCB", "Random"):
        for step in (1, 2):
            sink.write_row(_make_row(step=step, policy=policy, execution_id=f"{policy}-{step}"))
    sink.flush()
    expected = read(tmp_path).filter(pl.col("step") == 1)
    original, calls = os.replace, 0

    def interrupt_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("owned second recovery shard interruption")
        original(source, destination)

    progress = checkpoint()
    with monkeypatch.context() as fault:
        fault.setattr("coleman.results.recovery.os.replace", interrupt_second)
        with pytest.raises(OSError, match="second recovery shard"):
            recover(sink, progress)
    assert read(tmp_path).height == 3
    recover(sink, progress)
    assert read(tmp_path).equals(expected)


def test_incomplete_shard_never_publishes_and_failed_sink_requires_reopen(tmp_path, monkeypatch):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row())

    def incomplete(table, *, root_path, **options):
        (Path(root_path) / "partial.tmp").write_bytes(b"owned incomplete parquet")
        raise OSError("owned interrupted encoder")

    with monkeypatch.context() as fault:
        fault.setattr("coleman.results.parquet_sink.pq.write_to_dataset", incomplete)
        with pytest.raises(OSError, match="interrupted encoder"):
            sink.flush()
    assert not list(tmp_path.rglob("*.parquet"))
    row = _make_row()
    with pytest.raises(RuntimeError, match="reopen the sink"):
        sink.write_row(row)
    with pytest.raises(RuntimeError, match="reopen the sink"):
        sink.flush()
    reopened = prepared_sink(tmp_path)
    recover(reopened, checkpoint(step=0))
    reopened.write_row(row)
    reopened.flush()
    assert read(tmp_path).height == 1


def test_real_process_death_between_shard_publications_leaves_only_complete_results(tmp_path):
    script = """
import os,sys
from coleman.results.parquet_sink import ParquetSink
from coleman.results.recovery import ResultCheckpoint
from coleman.budget import BudgetMode
from pathlib import Path
original=os.replace
def die_after_first(source,destination):
    original(source,destination)
    os._exit(97)
sink=ParquetSink(sys.argv[1])
sink.recover_checkpoint(ResultCheckpoint('TestScenario',1,BudgetMode.RATIO,0.5,0),checkpoint_root=Path(sys.argv[1])/'.owned-test-checkpoints')
for policy in ('Random','UCB'):
    sink.write_row(dict(scenario='TestScenario',experiment=1,step=1,policy=policy,
        reward_function='RNFail',budget_mode='ratio',budget_value=0.5))
os.replace=die_after_first
sink.flush()
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, check=False)
    assert result.returncode == 97, result.stderr.decode()
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    assert list(tmp_path.rglob("*.tmp"))
    assert read(tmp_path).height == 1
    reopened = prepared_sink(tmp_path)
    recover(reopened, checkpoint(step=0))
    assert not list(tmp_path.rglob("*.parquet"))
    for policy in ("Random", "UCB"):
        reopened.write_row(_make_row(policy=policy))
    reopened.flush()
    assert read(tmp_path).height == 2


@pytest.mark.parametrize("committed", [0, 1])
def test_concurrent_independent_executions_recover_their_own_distinct_shards(tmp_path, monkeypatch, committed):
    original, creators = ArtifactWriter.write_immutable, Barrier(2)

    def aligned_owner_creation(self, path, content):
        if path.name == ".native-results-owner-v1.json":
            creators.wait(timeout=30)
        return original(self, path, content)

    monkeypatch.setattr(ArtifactWriter, "write_immutable", aligned_owner_creation)

    def write(experiment):
        sink = prepared_sink(tmp_path, experiment=experiment)
        for step in (1, 2):
            sink.write_row(_make_row(experiment=experiment, step=step))
        sink.flush()
        return sink

    with ThreadPoolExecutor(max_workers=2) as pool:
        sinks = list(pool.map(write, [1, 2]))
        futures = [
            pool.submit(recover, sink, checkpoint(experiment=i, step=committed)) for i, sink in enumerate(sinks, 1)
        ]
        for future in futures:
            future.result()
    if committed:
        rows = read(tmp_path)
        assert sorted(rows["experiment"].to_list()) == [1, 2]
        assert rows["step"].to_list() == [1, 1]
    else:
        assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("fresh", ["different_directory", "emptied_directory"])
def test_new_checkpoint_directory_cannot_erase_an_existing_independent_run(tmp_path, fresh):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row(execution_id="original-independent-run"))
    sink.flush()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    base = tmp_path / ".owned-test-checkpoints"
    if fresh == "emptied_directory":
        shutil.rmtree(base)
    else:
        base = tmp_path / "another-checkpoint-directory"
    progress = checkpoint(step=0)
    with pytest.raises(ValueError, match="not paired"):
        sink.recover_checkpoint(progress, checkpoint_root=base)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == before


@pytest.mark.parametrize("loaded,step", [(False, 0), (True, 1)])
def test_unbound_legacy_tail_rejects_without_deleting_or_guessing_ownership(tmp_path, loaded, step):
    sink = ParquetSink(str(tmp_path))
    sink.write_row(_make_row(step=2, execution_id="legacy-independent-run"))
    sink.flush()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    progress = checkpoint(step=step)
    with pytest.raises(ValueError, match="Unpaired legacy"):
        sink.recover_checkpoint(progress, checkpoint_root=tmp_path / "checkpoints", checkpoint_loaded=loaded)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == before
    assert not (tmp_path / ".native-checkpoint-bindings-v1").exists()


def test_complete_committed_legacy_coordinate_adopts_pairing_without_rewriting_rows(tmp_path):
    sink = ParquetSink(str(tmp_path))
    sink.write_row(_make_row())
    sink.flush()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    sink.recover_checkpoint(checkpoint(), checkpoint_root=tmp_path / "checkpoints", checkpoint_loaded=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == before
    assert len(list((tmp_path / ".native-checkpoint-bindings-v1").glob("*.json"))) == 1
    sink.write_row(_make_row(step=2))
    sink.flush()
    sink.recover_checkpoint(checkpoint(), checkpoint_root=tmp_path / "checkpoints", checkpoint_loaded=True)
    assert read(tmp_path)["step"].to_list() == [1]
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == before


def test_unmanaged_writer_tail_is_not_reclaimed_as_the_paired_runs_work(tmp_path):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row())
    sink.flush()
    unmanaged = ParquetSink(str(tmp_path))
    unmanaged.write_row(_make_row(step=2, execution_id="other-independent-writer"))
    unmanaged.flush()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    progress = checkpoint()
    with pytest.raises(ValueError, match="Unowned Parquet tail"):
        recover(sink, progress)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == before


def test_future_shard_ownership_version_rejects_without_removing_rows(tmp_path):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row(step=2))
    sink.flush()
    path = next(tmp_path.rglob("*.parquet"))
    with pq.ParquetFile(path) as shard:
        table = shard.read()
    metadata = dict(table.schema.metadata)
    record = json.loads(metadata[_OWNER_METADATA_KEY])
    record["schema_version"] = 2
    metadata[_OWNER_METADATA_KEY] = json.dumps(record).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)
    original = path.read_bytes()
    progress = checkpoint(step=0)
    with pytest.raises(ValueError):
        recover(sink, progress)
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["owner", "binding"])
@pytest.mark.parametrize("version", [None, 2, True, "1", "oversized"])
def test_unknown_or_invalid_ownership_metadata_fails_before_result_deletion(tmp_path, kind, version):
    sink = prepared_sink(tmp_path)
    sink.write_row(_make_row(step=2))
    sink.flush()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    metadata = (
        tmp_path / ".owned-test-checkpoints/.native-results-owner-v1.json"
        if kind == "owner"
        else next((tmp_path / ".native-checkpoint-bindings-v1").glob("*.json"))
    )
    payload = json.loads(metadata.read_bytes())
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version
    metadata.write_text(" " * 4097 if version == "oversized" else json.dumps(payload))
    progress = checkpoint(step=0)
    with pytest.raises(ValueError):
        recover(sink, progress)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == before


def test_scope_digest_is_stable_across_progress_but_separates_exact_budget_and_execution():
    first = checkpoint()
    assert first.scope_id == checkpoint(step=100).scope_id
    assert first.scope_id != checkpoint(value=0.5000000000000001).scope_id
    assert first.scope_id != checkpoint(experiment=2).scope_id


def test_pairing_version_one_golden_roundtrip():
    fixture = Path(__file__).parents[1] / "fixtures/results/native-pairing-v1.json"
    raw = json.loads(fixture.read_bytes())
    owner = _CheckpointOwner.from_json(json.dumps(raw["owner"]))
    binding = _CheckpointBinding.from_json(json.dumps(raw["binding"]))
    assert json.loads(owner.to_json()) == raw["owner"]
    assert json.loads(binding.to_json()) == raw["binding"]
    assert binding.scope_id == checkpoint().scope_id


@pytest.mark.parametrize(
    "changes",
    [
        dict(step=-1),
        dict(step=True),
        dict(experiment=-1),
        dict(scenario=""),
        dict(value=float("nan")),
        dict(value=float("inf")),
        dict(value=True),
        dict(mode="unknown"),
    ],
)
def test_invalid_checkpoint_scope_rejects(changes):
    with pytest.raises(ValueError, match="Result checkpoint"):
        checkpoint(**changes)
