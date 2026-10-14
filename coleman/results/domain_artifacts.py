"""Shared local-sink adapter over the existing immutable artifact writer.

Domain artifacts are additive JSON sidecars, never legacy native result columns.
Each commit is synchronous: a successful write is visible before a checkpoint
can advance, independently of a native sink's buffered ``flush`` lifecycle.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.results.domain import DomainRecord, fingerprint
from coleman.results.sink_base import ResultsSink


class DomainArtifacts:
    """Typed query/commit adapter shared by local DuckDB and Parquet sinks.

    One logical key has one canonical JSON artifact. No table rewrite, duplicate
    index or second runner is required. Iteration returns a snapshot of filenames,
    not a transaction across records. Domain records and legacy native rows are
    not jointly atomic; replay repairs domain writes ahead of a checkpoint, but
    does not deduplicate legacy rows. Use on local filesystems only.
    """

    def __init__(self, out_dir: str | Path) -> None:
        """Configure sidecar root without creating files or opening a database."""
        self.root = Path(out_dir) / "domain-records"
        self.writer = ArtifactWriter(out_dir)

    def write(self, record: DomainRecord) -> None:
        """Validate and commit synchronously; conflicting replay raises ValueError.

        Revalidation also rejects instances built through Pydantic's unchecked
        construction/copy APIs. Failed writes cannot publish partial JSON.
        """
        validated = DomainRecord.from_json(record.to_json())
        path = self.root / f"{validated.identity_digest.removeprefix('sha256:')}.json"
        self.writer.write_immutable(path, validated.to_json() + "\n")

    def read(self, run_id: str, execution_id: str, decision_id: str, *, namespace: str) -> DomainRecord | None:
        """Read a logical key; absent sidecars (including legacy runs) return None.

        Unknown versions or corrupt/miskeyed artifacts fail closed. Reading never
        migrates or rewrites a file, and never infers provenance for old outputs.
        """
        key = (namespace, run_id, execution_id, decision_id)
        path = self.root / f"{fingerprint(list(key)).removeprefix('sha256:')}.json"
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return None
        record = DomainRecord.from_json(content)
        if record.logical_key != key:
            raise ValueError("Domain artifact logical key does not match its filename")
        return record

    def records(self) -> Iterator[DomainRecord]:
        """Query committed records in identity order, ignoring interrupted .tmp files.

        Raises on any unknown version or inconsistent filename. Concurrent new
        commits may appear only in the next scan; already committed files are
        immutable. Empty legacy directories yield no domain records.
        """
        for path in sorted(self.root.glob("*.json")):
            record = DomainRecord.from_json(path.read_bytes())
            if path.stem != record.identity_digest.removeprefix("sha256:"):
                raise ValueError("Domain artifact logical key does not match its filename")
            yield record


class LocalResultsSink(ResultsSink, abc.ABC):
    """Optional local artifact capability, independent of native buffering/shards.

    Concrete sinks provide ``out_dir`` and the unchanged native row lifecycle.
    This adapter holds no open files, locks or duplicate state, so it is picklable.
    """

    out_dir: str

    def write_domain_record(self, record: DomainRecord) -> None:
        """Commit immediately; identical replay succeeds and conflicts raise ValueError.

        Legacy tables are untouched. Filesystem/validation errors propagate before
        the caller can publish a checkpoint; no asynchronous queue hides errors.
        """
        DomainArtifacts(self.out_dir).write(record)

    def read_domain_record(
        self, run_id: str, execution_id: str, decision_id: str, *, namespace: str
    ) -> DomainRecord | None:
        """Read a committed v1 sidecar; legacy runs without sidecars return None."""
        return DomainArtifacts(self.out_dir).read(run_id, execution_id, decision_id, namespace=namespace)

    def domain_records(self) -> Iterator[DomainRecord]:
        """Query immutable sidecars in identity order, independently of native rows."""
        return DomainArtifacts(self.out_dir).records()
