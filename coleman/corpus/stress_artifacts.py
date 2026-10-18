"""Deterministic offline stress exports using existing complete-last immutable publication."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import validate_source_storage
from coleman.corpus.stress_contracts import StressCase, StressFile, StressReceipt
from coleman.corpus.stress_world import StressObservation, generate_build
from coleman.results.domain import fingerprint


def stress_producer() -> str:
    """Bind owned generator/contracts and immutable storage; old producers are never silently retagged."""
    modules = {
        "coleman.corpus": (
            "stress_contracts.py",
            "stress_world.py",
            "stress_design.py",
            "stress_replay.py",
            "stress_feedback.py",
            "stress_artifacts.py",
            "materialization.py",
        ),
        "coleman": ("artifacts.py",),
        "coleman.results": ("domain.py", "versioned.py"),
        "coleman.selection": ("contracts.py",),
    }
    return fingerprint(
        [
            [package + "/" + name, hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()]
            for package, names in sorted(modules.items())
            for name in names
        ]
    )


def _writer_version() -> str:
    """Name both pinned tabular libraries participating in deterministic byte encoding."""
    return f"polars-{pl.__version__}-pyarrow-{pa.__version__}"


def _target(root: Path, relative: str) -> Path:
    """Contain fixed role paths in an external trusted-local tree; reject symlinks before writes."""
    path = root / relative
    validate_source_storage(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Stress artifact escapes its trusted output tree")
    return path


def _native(rows: tuple[StressObservation, ...]) -> pa.Table:
    """Project complete synthetic truth for post-hoc scoring; no context or hidden risk enters the policy."""
    return pl.DataFrame(
        {
            "BuildId": [row.build + 1 for row in rows],
            "Name": [row.execution_name for row in rows],
            "Duration": [row.duration_seconds for row in rows],
            "CalcPrio": [0] * len(rows),
            "LastRun": [""] * len(rows),
            "Verdict": [row.verdict for row in rows],
        },
        schema={
            "BuildId": pl.Int64,
            "Name": pl.String,
            "Duration": pl.Float64,
            "CalcPrio": pl.Int64,
            "LastRun": pl.String,
            "Verdict": pl.Int64,
        },
    ).to_arrow()


def _payloads(case: StressCase) -> Iterator[tuple[str, bytes, int]]:
    """Bound each generated build to 64 MiB and encoded native bytes to 512 MiB.

    Only one truth build/table and one Parquet byte buffer are retained. At most
    16 builds x 150,000 variant executions are encoded; the covering design is
    never expanded into executions. Failure leaves no completion publication.
    """
    buffer = io.BytesIO()
    writer = None
    total = 0
    try:
        for build in range(case.builds):
            rows = generate_build(case, build)
            truth = b"".join(row.to_bytes() for row in rows)
            if len(truth) > 64 * 1024**2:
                raise ValueError("Stress truth build exceeds 64 MiB")
            table = _native(rows)
            if writer is None:
                writer = pq.ParquetWriter(
                    buffer,
                    table.schema,
                    compression="NONE",
                    use_dictionary=False,
                    write_statistics=False,
                    data_page_size=65536,
                    version="2.6",
                )
            writer.write_table(table, row_group_size=8192)
            if buffer.tell() > 512 * 1024**2:
                raise ValueError("Stress native Parquet exceeds 512 MiB")
            total += len(rows)
            yield f"truth/build-{build:02}.jsonl", truth, len(rows)
    finally:
        if writer is not None:
            writer.close()
    if buffer.tell() > 512 * 1024**2:
        raise ValueError("Stress native Parquet footer exceeds 512 MiB")
    yield "datasets/stress/features-engineered.parquet", buffer.getvalue(), total


def _record(path: str, payload: bytes, rows: int) -> StressFile:
    """Declare exact bytes, logical count and named roles before immutable publication."""
    return StressFile(
        path=path, sha256="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload), rows=rows
    )


def export_stress(case: StressCase, writer: ArtifactWriter) -> Path:
    """Materialize one case outside Git, publishing complete.json only after every file.

    Identical concurrent attempts converge using ArtifactWriter hard-link
    publication. Interruptions may leave immutable individual files; retry
    verifies/reuses them and completes. Changed producer/writer selects a new
    namespace. No network, legacy rewrite, registry admission, candidate fit or
    native execution occurs. Path/version/resource conflicts raise ValueError;
    filesystem errors propagate. Trusted local parents are required.
    """
    case = StressCase.model_validate(case)
    validate_source_storage(writer.run_root)
    identity = fingerprint([case.fingerprint, stress_producer(), _writer_version()]).removeprefix("sha256:")
    root = _target(writer.run_root, "stress/" + identity)
    records = []
    for path, payload, rows in _payloads(case):
        record = _record(path, payload, rows)
        writer.write_immutable_stream(_target(root, path), (payload,), maximum_bytes=record.size_bytes)
        records.append(record)
    receipt = StressReceipt(
        schema_version=1, case=case, producer=stress_producer(), parquet_writer=_writer_version(), files=tuple(records)
    )
    completion = _target(root, "complete.json")
    writer.write_immutable(completion, receipt.to_json() + "\n")
    return completion


def validate_stress(receipt: StressReceipt, root: Path) -> None:
    """Regenerate all expected bytes offline, without trusting self-signed table digests.

    Unknown producers/writers require the original installed package; old
    artifacts remain unchanged and no implicit migration occurs. No repairs,
    publication, Parquet parsing, policy fit or execution happen. A missing,
    truncated, modified, oversized or unsafe file raises ValueError/OSError.
    """
    receipt = StressReceipt.model_validate(receipt)
    validate_source_storage(root)
    if (receipt.producer, receipt.parquet_writer) != (stress_producer(), _writer_version()):
        raise ValueError("Unsupported stress producer/writer; use the recorded package")
    for record, (path, payload, rows) in zip(receipt.files, _payloads(receipt.case), strict=True):
        expected = _record(path, payload, rows)
        if record != expected:
            raise ValueError("Stress receipt disagrees with regenerated truth")
        with _target(root, path).open("rb") as stream:
            actual = stream.read(record.size_bytes + 1)
        if actual != payload:
            raise ValueError("Stress artifact bytes disagree with regenerated truth")
