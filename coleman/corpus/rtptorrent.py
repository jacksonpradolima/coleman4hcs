"""Bounded offline conversion of pinned RTPTorrent 1.1 raw class-result CSV members."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_materialization
from coleman.corpus.rtptorrent_contracts import REPORTED_SECONDS, ClassExecution, RTPTorrentDataset, RTPTorrentSpec
from coleman.selection.contracts import Quantity

_HEADER = ("travisJobId", "testName", "index", "duration", "count", "failures", "errors", "skipped")


def _integer(text: str, *, optional: bool = False) -> int | None:
    """Parse canonical nonnegative ASCII integer counts/labels, preserving empty optional values."""
    if not text and optional:
        return None
    if re.fullmatch(r"0|[1-9]\d{0,18}", text, flags=re.ASCII) is None:
        raise ValueError("Expected a canonical nonnegative ASCII integer")
    return int(text)


def _decimal_literal(text: str) -> bool:
    """Check ASCII decimal syntax with separate, unambiguous mantissa and exponent matches."""
    coefficient, separator, exponent = text.lower().partition("e")
    mantissa = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", coefficient, flags=re.ASCII)
    return mantissa is not None and (not separator or re.fullmatch(r"[+-]?\d+", exponent, flags=re.ASCII) is not None)


def _duration(text: str) -> Quantity | None:
    """Parse finite reported seconds exactly; empty is unavailable, never observed zero.

    At most 64 ASCII characters reach Decimal construction. Malformed, nonfinite,
    negative and overprecision literals raise ValueError; no file is written.
    """
    if not text:
        return None
    if len(text) > 64 or not _decimal_literal(text):
        raise ValueError("Invalid or overlong duration")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal duration") from exc
    if not value.is_finite():
        raise ValueError("Nonfinite duration")
    return Quantity(value=value, semantics=REPORTED_SECONDS)


def _fields(raw: bytes) -> tuple[str, ...]:
    """Parse one bounded UTF-8 physical line; multiline records and oversized fields reject.

    The source format contains Java class labels, not multiline text. Rejecting
    quoted newlines is explicit. This does not mutate csv.field_size_limit or
    other process-global parser state, including during concurrent conversions.
    """
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        row = next(reader)
        if next(reader, None) is not None:
            raise ValueError("Multiple CSV records inside one physical input line")
    except (csv.Error, StopIteration) as exc:
        raise ValueError("Expected one complete physical CSV record") from exc
    if any(len(value) > 1024 for value in row):
        raise ValueError("RTPTorrent CSV field exceeds 1024 characters")
    return tuple(row)


def _execution(values: tuple[str, ...]) -> ClassExecution:
    """Convert the exact eight-field raw layout; reject filtered baseline-table substitution."""
    if len(values) != len(_HEADER):
        raise ValueError("RTPTorrent raw class-result rows require exactly eight columns")
    duration = _duration(values[3])
    return ClassExecution.model_validate(
        {
            "job_id": _integer(values[0]),
            "test_class": values[1],
            "position": _integer(values[2]),
            "duration": duration,
            "duration_status": "observed" if duration is not None else "unavailable",
            **{name: _integer(text, optional=True) for name, text in zip(_HEADER[4:], values[4:], strict=True)},
        }
    )


def convert_rtptorrent_source(spec: RTPTorrentSpec, cache: SourceCache) -> RTPTorrentDataset:
    """Verify pinned source/member bytes and convert a bounded raw CSV, retaining all jobs.

    Existing materialization enforces specific data terms, byte digests, archive
    relations and offline cache bounds. This operation reads only verified local
    cache bytes; no transport, implicit download, clock or output file is used.
    Unknown layouts, malformed records, conflicting duplicates, negative/nonfinite
    durations, inconsistent counts and byte/row bounds raise ValueError. I/O errors
    propagate. No partial conversion artifact or completed receipt is written.
    Rows with unavailable quantities are retained and cannot enter native replay.
    """
    spec = RTPTorrentSpec.model_validate(spec)
    validate_materialization(spec.materialization, cache)
    path = cache.member_path(spec.materialization, spec.member)
    member = next(item for item in spec.materialization.spec.members if item.name == spec.member)
    hasher = hashlib.sha256()
    total = 0
    records: dict[tuple[int, str, int], ClassExecution] = {}
    count = 0
    with path.open("rb") as stream:
        header = stream.readline(spec.maximum_line_bytes + 1)
        hasher.update(header)
        total += len(header)
        if len(header) > spec.maximum_line_bytes or _fields(header) != _HEADER:
            raise ValueError("Expected bounded raw RTPTorrent header; published baseline tables are filtered inputs")
        while raw := stream.readline(spec.maximum_line_bytes + 1):
            count += 1
            total += len(raw)
            hasher.update(raw)
            if len(raw) > spec.maximum_line_bytes or count > spec.maximum_rows or total > member.size_bytes:
                raise ValueError("RTPTorrent input exceeds declared line/row bounds")
            try:
                item = _execution(_fields(raw))
            except (ValueError, csv.Error, StopIteration) as exc:
                raise ValueError(f"Invalid RTPTorrent source row {count}: {exc}") from exc
            key = (item.job_id, item.test_class, item.position)
            previous = records.setdefault(key, item)
            if previous != item:
                raise ValueError(f"Conflicting RTPTorrent execution identity at source row {count}")
    if total != member.size_bytes or "sha256:" + hasher.hexdigest() != member.sha256:
        raise ValueError("Parsed RTPTorrent bytes disagree with pinned member")
    return RTPTorrentDataset(
        schema_version=1,
        spec=spec,
        input_rows=count,
        duplicate_rows_excluded=count - len(records),
        executions=tuple(records.values()),
    )


def validate_rtptorrent_dataset(dataset: RTPTorrentDataset, cache: SourceCache) -> None:
    """Recompute conversion against actual pinned cache bytes; metadata/row invariants alone are insufficient."""
    dataset = RTPTorrentDataset.model_validate(dataset)
    if convert_rtptorrent_source(dataset.spec, cache) != dataset:
        raise ValueError("Converted RTPTorrent records disagree with verified source bytes")


def publish_rtptorrent_dataset(dataset: RTPTorrentDataset, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Verify actual source correspondence before immutable deterministic publication.

    Reuses the existing artifact writer for interrupted writes, identical concurrent
    retries and conflicting writes. The external source cache and output family
    must remain physically separate. Output contains converted source rows: callers
    must honor the recipe's redistribution rule and never commit third-party data
    merely because a source was licensed for local use. Metadata reads do not imply
    source validation or scientific admission. No learning checkpoint is created.
    """
    dataset = RTPTorrentDataset.model_validate(dataset)
    validate_cache_separation(cache.root, writer.run_root)
    path = writer.run_root / "corpus-rtptorrent" / (dataset.fingerprint.removeprefix("sha256:") + ".json")
    if path.is_symlink() or not path.resolve().is_relative_to(writer.run_root.resolve()):
        raise ValueError("Converted artifacts must remain within their output tree")
    validate_rtptorrent_dataset(dataset, cache)
    writer.write_immutable(path, dataset.to_json() + "\n")
    return path
