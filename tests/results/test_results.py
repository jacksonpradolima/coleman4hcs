"""Tests for the results sink interfaces and implementations."""

import json
import threading

import pyarrow.parquet as pq

from coleman4hcs.results.parquet_sink import ParquetSink, _hash_order, _top_k
from coleman4hcs.results.sink_base import NullSink, ResultsSink
from coleman4hcs.results.writer import ResultsWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(**overrides):
    """Build a minimal result row dict with sensible defaults."""
    row = {
        "scenario": "TestScenario",
        "experiment": 1,
        "step": 1,
        "policy": "UCB",
        "reward_function": "RNFail",
        "sched_time": 0.5,
        "sched_time_duration": 50.0,
        "total_build_duration": 100.0,
        "prioritization_time": 1.5,
        "detected": 5,
        "missed": 2,
        "tests_ran": 10,
        "tests_not_ran": 3,
        "ttf": 2.0,
        "ttf_duration": 10.0,
        "time_reduction": 90.0,
        "fitness": 0.8,
        "cost": 0.6,
        "rewards": 0.9,
        "avg_precision": 0.75,
        "prioritization_order": ["tc1", "tc2", "tc3"],
    }
    row.update(overrides)
    return row


# ============================================================================
# NullSink
# ============================================================================


class TestNullSink:
    def test_is_results_sink(self):
        assert isinstance(NullSink(), ResultsSink)

    def test_write_row_is_noop(self):
        sink = NullSink()
        sink.write_row({"a": 1})  # should not raise

    def test_flush_is_noop(self):
        NullSink().flush()

    def test_close_is_noop(self):
        NullSink().close()


# ============================================================================
# _hash_order / _top_k helpers
# ============================================================================


class TestHashOrder:
    def test_deterministic(self):
        order = ["a", "b", "c"]
        assert _hash_order(order) == _hash_order(order)

    def test_different_orders_differ(self):
        assert _hash_order(["a", "b"]) != _hash_order(["b", "a"])

    def test_is_sha256_hex(self):
        h = _hash_order([1, 2, 3])
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_none_order(self):
        h = _hash_order(None)
        assert isinstance(h, str)
        assert len(h) == 64


class TestTopK:
    def test_none_k_returns_empty(self):
        assert _top_k(["a", "b", "c"], None) == ""

    def test_non_list_returns_empty(self):
        assert _top_k("not_a_list", 2) == ""

    def test_returns_json_list(self):
        result = _top_k(["a", "b", "c", "d"], 2)
        assert json.loads(result) == ["a", "b"]

    def test_k_larger_than_list(self):
        result = _top_k(["a"], 5)
        assert json.loads(result) == ["a"]


# ============================================================================
# ParquetSink
# ============================================================================


class TestParquetSink:
    def test_write_and_flush(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=5)
        for i in range(3):
            sink.write_row(_make_row(step=i))
        sink.flush()

        # Should produce at least one parquet file
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) >= 1

        table = pq.read_table(parquet_files[0])
        assert table.num_rows == 3

    def test_auto_flush_on_batch_size(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=2)
        sink.write_row(_make_row(step=1))
        sink.write_row(_make_row(step=2))  # triggers flush
        # Buffer should be empty after auto-flush
        assert len(sink._buffer) == 0

    def test_prioritization_order_hashed(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=10)
        sink.write_row(_make_row())
        sink.flush()

        parquet_files = list(tmp_path.rglob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        assert "prioritization_order_hash" in table.column_names
        assert "prioritization_order" not in table.column_names

    def test_top_k_stored_when_set(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=10, top_k=2)
        sink.write_row(_make_row(prioritization_order=["a", "b", "c", "d"]))
        sink.flush()

        parquet_files = list(tmp_path.rglob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        top_k_val = table.column("prioritization_order_top_k")[0].as_py()
        assert json.loads(top_k_val) == ["a", "b"]

    def test_close_flushes(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_thread_safety(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=50)
        errors = []

        def writer(start):
            try:
                for i in range(20):
                    sink.write_row(_make_row(step=start + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sink.flush()
        assert not errors
        parquet_files = list(tmp_path.rglob("*.parquet"))
        total = sum(pq.read_table(f).num_rows for f in parquet_files)
        assert total == 100


# ============================================================================
# ResultsWriter (thread-safe queue)
# ============================================================================


class TestResultsWriter:
    def test_enqueue_and_stop(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        writer = ResultsWriter(sink, max_queue_size=50)
        for i in range(10):
            writer.enqueue(_make_row(step=i))
        writer.stop()

        parquet_files = list(tmp_path.rglob("*.parquet"))
        total = sum(pq.read_table(f).num_rows for f in parquet_files)
        assert total == 10

    def test_auto_start(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        writer = ResultsWriter(sink, max_queue_size=50)
        assert not writer._started
        writer.enqueue(_make_row())
        assert writer._started
        writer.stop()

    def test_null_sink_writer(self):
        sink = NullSink()
        writer = ResultsWriter(sink, max_queue_size=50)
        for i in range(10):
            writer.enqueue(_make_row(step=i))
        writer.stop()  # should not raise
