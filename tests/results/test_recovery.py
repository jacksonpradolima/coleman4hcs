"""Native checkpoint recovery retains committed evidence and retries interrupted atomic shard updates."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from coleman.budget import BudgetMode
from coleman.results.parquet_sink import ParquetSink
from coleman.results.recovery import ResultCheckpoint
from tests.results.test_results import _make_row


def checkpoint(**changes):
    return replace(ResultCheckpoint("TestScenario", 1, BudgetMode.RATIO, 0.5, 1), **changes)


def read(root):
    return pl.read_parquet(str(root / "**/*.parquet"), hive_partitioning=True).sort("execution_id")


def test_recovery_preserves_committed_rows_other_coordinates_schema_and_idempotence(tmp_path):
    sink = ParquetSink(str(tmp_path))
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
    sink.recover_checkpoint(checkpoint())
    after = read(tmp_path)
    assert after.schema == before.schema
    expected = before.filter(~pl.col("execution_id").str.starts_with("uncommitted"))
    assert after.equals(expected)
    files = {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    sink.recover_checkpoint(checkpoint())
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.parquet")} == files


def test_absent_checkpoint_removes_all_uncommitted_rows_only(tmp_path):
    sink = ParquetSink(str(tmp_path))
    sink.write_row(_make_row())
    sink.flush()
    sink.recover_checkpoint(checkpoint(step=0))
    assert not list(tmp_path.rglob("*.parquet"))
    sink.recover_checkpoint(checkpoint(step=0))


def test_atomic_replacement_error_preserves_original_and_blocks_recovery(tmp_path, monkeypatch):
    sink = ParquetSink(str(tmp_path))
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
            sink.recover_checkpoint(progress)
    assert path.read_bytes() == original
    assert not list(tmp_path.rglob(".recovery-*.tmp"))
    sink.recover_checkpoint(checkpoint())
    assert read(tmp_path)["step"].to_list() == [1]


def test_recovery_requires_empty_buffer_and_never_discards_new_work(tmp_path):
    sink = ParquetSink(str(tmp_path))
    sink.write_row(_make_row(step=2))
    progress = checkpoint()
    with pytest.raises(ValueError, match="empty result buffer"):
        sink.recover_checkpoint(progress)
    sink.flush()
    assert read(tmp_path)["step"].to_list() == [2]


def test_independent_recovery_removing_a_previously_enumerated_shard_is_safe(tmp_path, monkeypatch):
    first, other = ParquetSink(str(tmp_path)), ParquetSink(str(tmp_path))
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
        first.recover_checkpoint(checkpoint(step=0))
    assert not list(tmp_path.rglob("*.parquet"))


def test_interrupted_reconciliation_converges_before_any_replayed_rows(tmp_path, monkeypatch):
    sink = ParquetSink(str(tmp_path))
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
            sink.recover_checkpoint(progress)
    assert read(tmp_path).height == 3
    sink.recover_checkpoint(progress)
    assert read(tmp_path).equals(expected)


def test_incomplete_shard_never_publishes_and_failed_sink_requires_reopen(tmp_path, monkeypatch):
    sink = ParquetSink(str(tmp_path))
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
    reopened = ParquetSink(str(tmp_path))
    reopened.recover_checkpoint(checkpoint(step=0))
    reopened.write_row(row)
    reopened.flush()
    assert read(tmp_path).height == 1


def test_real_process_death_between_shard_publications_leaves_only_complete_results(tmp_path):
    script = """
import os,sys
from coleman.results.parquet_sink import ParquetSink
original=os.replace
def die_after_first(source,destination):
    original(source,destination)
    os._exit(97)
sink=ParquetSink(sys.argv[1])
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
    reopened = ParquetSink(str(tmp_path))
    reopened.recover_checkpoint(checkpoint(step=0))
    assert not list(tmp_path.rglob("*.parquet"))
    for policy in ("Random", "UCB"):
        reopened.write_row(_make_row(policy=policy))
    reopened.flush()
    assert read(tmp_path).height == 2


@pytest.mark.parametrize("committed", [0, 1])
def test_concurrent_independent_executions_recover_their_own_distinct_shards(tmp_path, committed):
    def write(experiment):
        sink = ParquetSink(str(tmp_path))
        for step in (1, 2):
            sink.write_row(_make_row(experiment=experiment, step=step))
        sink.flush()
        return sink

    with ThreadPoolExecutor(max_workers=2) as pool:
        sinks = list(pool.map(write, [1, 2]))
        futures = [
            pool.submit(sink.recover_checkpoint, checkpoint(experiment=i, step=committed))
            for i, sink in enumerate(sinks, 1)
        ]
        for future in futures:
            future.result()
    if committed:
        rows = read(tmp_path)
        assert sorted(rows["experiment"].to_list()) == [1, 2]
        assert rows["step"].to_list() == [1, 1]
    else:
        assert not list(tmp_path.rglob("*.parquet"))


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
