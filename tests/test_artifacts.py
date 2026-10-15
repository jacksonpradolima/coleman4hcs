"""Tests for artifact writer utilities."""

from __future__ import annotations

import json
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.budget import BudgetMode


def test_artifact_writer_path_is_deterministic(tmp_path):
    writer = ArtifactWriter(tmp_path)

    path_a = writer.path_for(
        run_id="run123",
        dataset_id="my-dataset",
        sched_time_ratio=0.5,
        execution_id="exec-1",
        artifact_type="quality",
        ext="json",
    )
    path_b = writer.path_for(
        run_id="run123",
        dataset_id="my-dataset",
        sched_time_ratio=0.5,
        execution_id="exec-1",
        artifact_type="quality",
        ext="json",
    )

    assert path_a == path_b
    assert path_a.as_posix().endswith("run123/artifacts/quality/time_ratio_50/my-dataset/exec-1/artifact.json")


def test_artifact_writer_json_and_csv_are_atomic(tmp_path):
    writer = ArtifactWriter(tmp_path)

    json_path = Path(tmp_path) / "r1" / "artifacts" / "x" / "a.json"
    csv_path = Path(tmp_path) / "r1" / "artifacts" / "x" / "a.csv"

    writer.write_json_atomic(json_path, {"a": 1, "b": 2})
    writer.write_csv_atomic(csv_path, [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])

    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == {"a": 1, "b": 2}

    assert "x,y" in csv_path.read_text(encoding="utf-8")


def test_artifact_writer_path_uses_budget_segment_for_non_ratio(tmp_path):
    writer = ArtifactWriter(tmp_path)

    path = writer.path_for(
        run_id="run123",
        dataset_id="my-dataset",
        sched_time_ratio=None,
        budget_mode=BudgetMode.FIXED_TIME.value,
        budget_value=30.0,
        execution_id="exec-1",
        artifact_type="quality",
        ext="json",
    )

    assert "budget_fixed_time_30" in path.as_posix()


def test_immutable_binary_stream_limits_interruption_and_utf8_compatibility(tmp_path):
    from collections.abc import Iterable
    from typing import cast

    import pytest

    writer = ArtifactWriter(tmp_path)
    path = tmp_path / "complete.bin"
    writer.write_immutable_stream(path, (b"\x00\xff", b"end"), maximum_bytes=5)
    assert path.read_bytes() == b"\x00\xffend"
    writer.write_immutable_stream(path, (b"\x00", b"\xffend"), maximum_bytes=5)
    with pytest.raises(ValueError, match="Conflicting"):
        writer.write_immutable_stream(path, (b"other",), maximum_bytes=5)
    target = tmp_path / "partial.bin"

    def interrupted():
        yield b"begin"
        raise OSError("checksum validation interrupted")

    interrupted_chunks = interrupted()
    with pytest.raises(OSError, match="interrupted"):
        writer.write_immutable_stream(target, interrupted_chunks, maximum_bytes=10)
    with pytest.raises(ValueError, match="byte limit"):
        writer.write_immutable_stream(target, (b"too long",), maximum_bytes=1)
    invalid_chunks = cast(Iterable[bytes], ("not binary",))
    with pytest.raises(TypeError, match="must be bytes"):
        writer.write_immutable_stream(target, invalid_chunks, maximum_bytes=10)
    for bound in (True, -1, 1.5):
        invalid_bound = cast(int, bound)
        with pytest.raises(ValueError, match="integer"):
            writer.write_immutable_stream(target, (), maximum_bytes=invalid_bound)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    text = tmp_path / "old.json"
    writer.write_immutable(text, '"CI: ação"\n')
    assert text.read_bytes() == '"CI: ação"\n'.encode()
    writer.write_immutable(text, '"CI: ação"\n')
    empty = tmp_path / "empty"
    writer.write_immutable_stream(empty, (), maximum_bytes=0)
    assert empty.read_bytes() == b""
