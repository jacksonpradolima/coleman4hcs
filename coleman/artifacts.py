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
    """Build artifact paths and publish local files for extensions.

    Parameters
    ----------
    run_root : str or pathlib.Path
        Caller-owned local artifact root. Construction stores the path without
        creating directories or validating a security boundary.

    Notes
    -----
    The caller owns path selection and trusted parent directories. This helper
    is not a hostile-filesystem sandbox. Immutable publication supports
    cooperative concurrent writers; mutable JSON/CSV helpers have a narrower
    single-writer contract. Directory durability across power loss is outside
    the process-interruption guarantee.
    """

    def __init__(self, run_root: str | Path) -> None:
        """Store the caller-owned artifact root without filesystem writes.

        Parameters
        ----------
        run_root : str or pathlib.Path
            Local base directory used by ``path_for``. Relative paths retain the
            caller's working-directory interpretation.
        """
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
        """Build a deterministic legacy artifact path without creating files.

        Parameters
        ----------
        run_id : str
            Trusted run namespace, normally the configuration-derived run ID.
        dataset_id : str or None
            Dataset label; ``None`` uses a placeholder. Unsafe label characters
            are replaced for this segment.
        sched_time_ratio : float or None, optional
            Legacy dimensionless ratio, used when ``budget_value`` is absent.
        budget_mode : str or None, optional
            Budget label. ``None`` selects ``ratio``.
        budget_value : float or None, optional
            Ratio, seconds or test count according to the selected mode.
        execution_id : str or None
            Execution label; ``None`` uses a placeholder.
        artifact_type : str
            Artifact family label, sanitized as a path segment.
        ext : str
            Trusted filename extension, with a leading dot removed.
        stem : str, optional
            Filename stem, sanitized as a path segment.

        Returns
        -------
        pathlib.Path
            Path beneath the composed run/artifact/budget/dataset/execution labels.

        Notes
        -----
        Ratio labels round to integer percentages for compatibility and are not
        collision-free identities for arbitrary fractional budgets. ``run_id`` and
        ``ext`` are caller-trusted rather than containment-validated. Use typed
        contract fingerprints and immutable family namespaces when exact identity
        is required; this path builder does not validate budget semantics.
        """
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
        """Replace a local JSON file after writing and fsyncing its contents.

        Parameters
        ----------
        path : pathlib.Path
            Caller-selected destination. Missing parent directories are created.
        payload : object
            JSON-serializable Python value. This legacy helper does not enforce
            versioned record semantics or finite-number validation.
        indent : int, optional
            JSON indentation width; keys are sorted and UTF-8 is preserved.

        Raises
        ------
        TypeError
            If the payload cannot be encoded by the standard JSON encoder.
        OSError
            If writing, flushing or replacing the file fails.

        Notes
        -----
        Overwrites the destination. Its fixed ``.tmp`` sibling is for a single
        cooperative writer, not concurrent publication or a complete-last multi-file
        receipt. Use ``write_immutable`` for immutable contract artifacts.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def write_immutable(self, path: Path, content: str) -> None:
        """Publish complete UTF-8 bytes once with atomic create-if-absent.

        Parameters
        ----------
        path : pathlib.Path
            Caller-selected local destination; missing parents are created.
        content : str
            Exact text bytes after UTF-8 encoding. Existing identical bytes are
            accepted as an idempotent replay.

        Raises
        ------
        ValueError
            If an existing destination has different contents.
        OSError
            If the filesystem cannot perform the required local I/O or hard-link
            publication and directory fsync.

        Notes
        -----
        A unique fsynced temporary file is linked to the final name. Cooperative
        concurrent writers cannot overwrite a winner or expose partial contents.
        Temporary files are not completed artifacts; a crash after publication is
        recovered by identical replay. No lock file or in-memory duplicate index
        needs recovery. Trusted parents and a suitable local filesystem are required.
        Parent-directory durability across power loss is outside this contract.
        """
        encoded = content.encode("utf-8")
        self.write_immutable_stream(path, (encoded,), maximum_bytes=len(encoded))

    def write_immutable_stream(self, path: Path, chunks: Iterable[bytes], *, maximum_bytes: int) -> None:
        """Publish a bounded stream with immutable local-file semantics.

        Parameters
        ----------
        path : pathlib.Path
            Caller-selected destination; missing parents are created.
        chunks : collections.abc.Iterable[bytes]
            Caller-owned byte iterator. It is exhausted before publication so final
            checksum validation may fail without exposing a completed artifact.
        maximum_bytes : int
            Nonnegative integral upper bound on encoded bytes; booleans are rejected.

        Raises
        ------
        ValueError
            If the bound is invalid, contents exceed it, or existing bytes conflict.
        TypeError
            If a yielded chunk is not bytes.
        OSError
            If local writing, hard-link publication or directory fsync fails.

        Notes
        -----
        Iterator errors propagate; ordinary unwinding removes the unique temporary
        file. Abrupt termination can leave an uncommitted temporary file. Callers
        must close their input resources. Identical concurrent replay succeeds;
        comparison does not load a full source/artifact into memory. The trusted
        local-filesystem and power-loss limits of ``write_immutable`` apply.
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
        """Replace a local CSV file after writing and fsyncing its contents.

        Parameters
        ----------
        path : pathlib.Path
            Caller-selected destination; missing parents are created.
        rows : list[dict[str, Any]]
            Materialized rows. The sorted union of keys determines the header;
            omitted row fields are emitted as empty fields.

        Raises
        ------
        OSError
            If writing, flushing or replacing the destination fails.

        Notes
        -----
        Overwrites the destination through a fixed ``.tmp`` sibling. This is a
        single-writer legacy export, not concurrent immutable publication or a
        versioned unit/missingness contract. An interrupted write may leave its
        uncommitted temporary file.
        """
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
