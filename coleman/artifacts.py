"""Deterministic artifact writing helpers for hooks and extensions."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ArtifactWriter:
    """Path builder and atomic writer for hook artifacts."""

    def __init__(self, run_root: str | Path) -> None:
        """Initialize writer with the base runs root path."""
        self.run_root = Path(run_root)

    def path_for(
        self,
        *,
        run_id: str,
        dataset_id: str | None,
        sched_time_ratio: float | None = None,
        budget_mode: str | None = None,
        budget_value: float | None = None,
        execution_id: str | None,
        artifact_type: str,
        ext: str,
        stem: str = "artifact",
    ) -> Path:
        """Build a deterministic artifact path under a run folder."""
        safe_dataset = _safe_segment(dataset_id or "dataset_unknown")
        safe_execution = _safe_segment(execution_id or "execution_unknown")
        safe_type = _safe_segment(artifact_type)
        safe_stem = _safe_segment(stem)
        ext = ext.lstrip(".")

        effective_budget_mode = (budget_mode or "ratio").lower()
        effective_budget_value = budget_value
        if effective_budget_value is None and sched_time_ratio is not None:
            effective_budget_value = float(sched_time_ratio)

        ratio_segment = "time_ratio_na"
        if effective_budget_mode == "ratio" and effective_budget_value is not None:
            ratio_segment = f"time_ratio_{int(round(float(effective_budget_value) * 100)):02d}"
        elif effective_budget_value is not None:
            value_segment = str(
                int(effective_budget_value)
                if int(effective_budget_value) == effective_budget_value
                else effective_budget_value
            ).replace(".", "_")
            ratio_segment = f"budget_{_safe_segment(effective_budget_mode)}_{_safe_segment(value_segment)}"

        base = self.run_root / run_id / "artifacts" / safe_type / ratio_segment / safe_dataset / safe_execution
        return base / f"{safe_stem}.{ext}"

    def write_json_atomic(self, path: Path, payload: Any, indent: int = 2) -> None:
        """Write JSON payload atomically (best effort across platforms)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def write_immutable(self, path: Path, content: str) -> None:
        """Publish complete UTF-8 bytes once, with atomic create-if-absent.

        Identical replay succeeds; conflicting bytes raise ``ValueError``.
        A unique, fsynced temporary file is linked to the final name so concurrent
        processes cannot overwrite a winner or expose a partially written record.
        Temporary files are never completed artifacts. A crash after publication
        is recovered by replay. Requires a local filesystem supporting hard links
        and directory fsync; unsupported storage raises ``OSError``. No lock file
        or in-memory duplicate index needs recovery. Parent-directory durability
        across power loss is outside this process-interruption contract.
        """
        encoded = content.encode("utf-8")
        self.write_immutable_stream(path, (encoded,), maximum_bytes=len(encoded))

    def write_immutable_stream(self, path: Path, chunks: Iterable[bytes], *, maximum_bytes: int) -> None:
        """Publish a bounded binary stream using the same immutable local-file contract.

        The iterator is exhausted before publication, allowing its final checksum
        verification to fail without creating a completed artifact. Bytes above
        the nonnegative integral limit raise ValueError; non-byte chunks raise
        TypeError. Identical concurrent replay succeeds; conflicting contents
        raise ValueError. Iterator/I/O errors propagate and temporary files are
        removed during ordinary exception unwinding. SIGKILL can leave only an
        uncommitted temporary file. Caller-owned iterators must close their input.
        No full source or existing artifact is loaded into memory for comparison.
        """
        if type(maximum_bytes) is not int or maximum_bytes < 0:
            raise ValueError("Immutable stream requires a nonnegative integer byte limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                total = 0
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("Immutable stream chunks must be bytes")
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise ValueError("Immutable stream exceeds byte limit")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            _publish_immutable_file(path, Path(temporary))
        finally:
            Path(temporary).unlink(missing_ok=True)

    def write_csv_atomic(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """Write row-oriented CSV payload atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        fieldnames = sorted({key for row in rows for key in row})
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


def _publish_immutable_file(path: Path, temporary: Path) -> None:
    """Link complete fsynced bytes once and fsync the parent; existing contents must match exactly."""
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not _same_file_bytes(path, temporary):
            raise ValueError(f"Conflicting immutable artifact: {path}") from None
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _same_file_bytes(left: Path, right: Path) -> bool:
    """Compare complete binary contents in fixed-size chunks without metadata-based caching."""
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            chunk = first.read(65536)
            if chunk != second.read(65536):
                return False
            if not chunk:
                return True


def _safe_segment(value: str) -> str:
    """Sanitize one filesystem path segment using conservative ASCII-safe chars."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "artifact"
