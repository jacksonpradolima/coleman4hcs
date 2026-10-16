"""Bounded offline LRTS metadata/raw-report assessment using existing verified source caches."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, NamedTuple

from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_contracts import (
    MEANINGS,
    METADATA_COLUMNS,
    METADATA_MEASURES,
    LRTSCase,
    LRTSDataset,
    LRTSMetadataRow,
    LRTSSpec,
    LRTSSuite,
)
from coleman.corpus.lrts_raw import RawReport
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_materialization
from coleman.corpus.materialization_contracts import PackageMember
from coleman.results.domain import Measurement, decode_json

_COUNTS = (
    ("passcount", "reported_pass_methods"),
    ("failcount", "reported_fail_methods"),
    ("skipcount", "reported_skip_methods"),
    ("num_pass_class", "processed_pass_classes"),
    ("num_fail_class", "processed_fail_classes"),
    ("num_trans_class", "processed_transition_classes"),
    ("num_pass_method", "processed_pass_methods"),
    ("num_fail_method", "processed_fail_methods"),
)


class _MetadataSlice(NamedTuple):
    """Typed selected metadata and conserved physical-row counts, without a loose result dictionary."""

    rows: tuple[LRTSMetadataRow, ...]
    input_rows: int
    outside_build: int
    duplicates_excluded: int


def _scalar(text: str) -> Decimal | int:
    """Reuse exact JSON numeric syntax for bounded CSV scalars; no Unicode/underscore coercion."""
    if len(text) > 64 or not text or text != text.strip():
        raise ValueError("LRTS numeric literal must have 1..64 unpadded characters")
    try:
        value = decode_json(text, exact_decimals=True)
    except (ValueError, DecimalException) as exc:
        raise ValueError("Invalid LRTS numeric literal") from exc
    if type(value) is int or isinstance(value, Decimal):
        return value
    raise ValueError("LRTS numeric fields require actual finite numbers")


def _count(text: str) -> int | None:
    """Keep empty counts unavailable and canonical integer counts exact within signed-64 bounds."""
    if not text:
        return None
    value = _scalar(text)
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("LRTS counts require canonical nonnegative signed-64 integers")
    return value


def _build_id(text: str) -> int:
    """Require a nonzero source build label without float coercion or rounding."""
    value = _count(text)
    if value is None or value < 1:
        raise ValueError("LRTS build labels must be positive integers")
    return value


def _measurement(name: str, value: Decimal | int | None) -> Measurement:
    """Preserve canonical source binary64 quantities; reject rounding, underflow and nonfinite values.

    Reported numeric fields use seconds, not exact modeled runner costs. A decimal
    literal must round-trip through binary64's canonical spelling before becoming
    a measurement. Null stays explicitly unavailable; genuine zero is canonical.
    """
    observed = None
    if value is not None:
        if not 0 <= value <= 10**12:
            raise ValueError("LRTS reported quantity is outside [0,10^12] seconds")
        observed = float(value)
        if not math.isfinite(observed) or Decimal(str(observed)) != Decimal(value):
            raise ValueError("LRTS source quantity is not canonical finite binary64; rounding is prohibited")
        if observed == 0:
            observed = 0.0
    return Measurement(
        name=name,
        value=observed,
        unit="s",
        meaning=MEANINGS[name],
        direction="none",
        status="unavailable" if observed is None else "observed",
        minimum=0.0,
    )


def _fields(raw: bytes) -> tuple[str, ...]:
    """Decode one UTF-8 physical CSV record with bounded fields and no global parser mutation."""
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        row = next(reader)
        if next(reader, None) is not None:
            raise ValueError("Multiple LRTS CSV records inside one physical line")
    except (csv.Error, StopIteration) as exc:
        raise ValueError("Expected one complete LRTS CSV record") from exc
    if any(len(value) > 1024 for value in row):
        raise ValueError("LRTS metadata field exceeds 1024 characters")
    return tuple(row)


def _metadata_row(values: tuple[str, ...]) -> LRTSMetadataRow:
    """Validate a selected metadata row immediately; unselected numeric fields are not interpreted."""
    row: dict[str, str] = dict(zip(METADATA_COLUMNS, values, strict=True))
    return LRTSMetadataRow.model_validate(
        {
            "project": row["project"],
            "pull_request": row["pr_name"],
            "build_id": _build_id(row["build_id"]),
            "stage_id": row["stage_id"],
            "base_branch": row["pr_base_branch"] or None,
            "build_result": row["build_result"] or None,
            "build_head_sha": row["build_head_sha"] or None,
            "build_date": row["build_date"] or None,
            "trunk_sha": row["trunk_sha"] or None,
            "trunk_sha_source": row["trunk_sha_source"] or None,
            **{target: _count(row[source]) for source, target in _COUNTS},
            "measurements": tuple(
                _measurement(name, _scalar(row[column]) if row[column] else None) for column, name in METADATA_MEASURES
            ),
        }
    )


def _member(spec: LRTSSpec, cache: SourceCache, *, metadata: bool) -> tuple[Path, PackageMember]:
    """Resolve only explicitly pinned source-cache members; no URI or transport is executed."""
    receipt = spec.metadata_materialization if metadata else spec.report_materialization
    name = spec.metadata_member if metadata else spec.report_member
    return cache.member_path(receipt, name), next(item for item in receipt.spec.members if item.name == name)


def _selected_metadata(values: tuple[str, ...], spec: LRTSSpec) -> LRTSMetadataRow | None:
    """Check physical row shape/identity before interpreting only the selected build's quantities."""
    if len(values) != len(METADATA_COLUMNS):
        raise ValueError("LRTS metadata row has unexpected columns")
    if (values[0], values[1], _build_id(values[3])) != (spec.project, spec.pull_request, spec.build_id):
        return None
    return _metadata_row(values)


def _merge_metadata(rows: dict[str, LRTSMetadataRow], item: LRTSMetadataRow) -> int:
    """Count equivalent selected-stage duplicates and reject conflicting identities or excess stages."""
    previous = rows.setdefault(item.stage_id, item)
    if previous != item:
        raise ValueError("Conflicting LRTS metadata for one processed stage")
    if len(rows) > 64:
        raise ValueError("LRTS selected build exceeds 64 processed stages")
    return int(previous is not item)


def _metadata(spec: LRTSSpec, cache: SourceCache) -> _MetadataSlice:
    """Stream all pinned metadata bytes and select one build without losing exclusion accounting."""
    path, member = _member(spec, cache, metadata=True)
    hasher = hashlib.sha256()
    total = count = outside = duplicates = 0
    rows: dict[str, LRTSMetadataRow] = {}
    with path.open("rb") as stream:
        header = stream.readline(spec.maximum_line_bytes + 1)
        hasher.update(header)
        total += len(header)
        if len(header) > spec.maximum_line_bytes or _fields(header) != METADATA_COLUMNS:
            raise ValueError("Expected exact bounded LRTS paper metadata header, not a filtered outcome table")
        while raw := stream.readline(spec.maximum_line_bytes + 1):
            count += 1
            total += len(raw)
            hasher.update(raw)
            if len(raw) > spec.maximum_line_bytes or count > spec.maximum_metadata_rows or total > member.size_bytes:
                raise ValueError("LRTS metadata exceeds declared byte/line/row limits")
            item = _selected_metadata(_fields(raw), spec)
            if item is None:
                outside += 1
            else:
                duplicates += _merge_metadata(rows, item)
    if total != member.size_bytes or "sha256:" + hasher.hexdigest() != member.sha256:
        raise ValueError("Parsed LRTS metadata bytes disagree with pinned member")
    if not rows:
        raise ValueError("No LRTS metadata matches the selected project/PR/build")
    return _MetadataSlice(tuple(rows.values()), count, outside, duplicates)


def _report_shape(value: Any, spec: LRTSSpec) -> None:
    """Bound raw suite/case allocation before constructing typed DTOs; unknown layouts reject."""
    if not isinstance(value, dict) or not isinstance(value.get("suites"), list):
        raise ValueError("LRTS requires a Jenkins report object with suites")
    if len(value["suites"]) > spec.maximum_suites:
        raise ValueError("LRTS report exceeds declared suite limit")
    count = 0
    for suite in value["suites"]:
        if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list):
            raise ValueError("LRTS suites require explicit case arrays")
        count += len(suite["cases"])
        if count > spec.maximum_cases:
            raise ValueError("LRTS report exceeds declared method limit")


def _report(spec: LRTSSpec, cache: SourceCache) -> RawReport:
    """Read at most the pinned byte count plus one and verify the bytes actually decoded."""
    path, member = _member(spec, cache, metadata=False)
    with path.open("rb") as stream:
        payload = stream.read(member.size_bytes + 1)
    if len(payload) != member.size_bytes or "sha256:" + hashlib.sha256(payload).hexdigest() != member.sha256:
        raise ValueError("Parsed LRTS report bytes disagree with pinned member")
    try:
        decoded = decode_json(payload, exact_decimals=True)
        _report_shape(decoded, spec)
        return RawReport.model_validate(decoded)
    except (ValueError, DecimalException, RecursionError) as exc:
        raise ValueError("Invalid bounded LRTS Jenkins report") from exc


def _suites(report: RawReport) -> tuple[LRTSSuite, ...]:
    """Project typed raw suites, preserving positions, node links and every repeated method."""
    return tuple(
        LRTSSuite(
            position=index,
            name=suite.name,
            reported_id=suite.reported_id,
            node_id=suite.node_id,
            enclosing_blocks=suite.enclosing_blocks,
            enclosing_block_names=suite.enclosing_block_names,
            raw_timestamp=suite.timestamp,
            duration=_measurement("lrts_suite_duration", suite.duration),
            cases=tuple(
                LRTSCase(
                    position=position,
                    class_name=case.class_name,
                    method_name=case.method_name,
                    source_status=case.status,
                    skipped=case.skipped,
                    duration=_measurement("lrts_case_duration", case.duration),
                )
                for position, case in enumerate(suite.cases)
            ),
        )
        for index, suite in enumerate(report.suites)
    )


def convert_lrts_source(spec: LRTSSpec, cache: SourceCache) -> LRTSDataset:
    """Verify and join a bounded metadata/raw-report slice without granting temporal admission.

    Existing source materialization retains byte identity and specific data terms.
    This function performs offline reads only, with no download, diagnostic-feature
    extraction, stage-name normalization or test deduplication. Invalid schemas,
    inconsistent selected sources, nonfinite/rounded quantities, duplicate conflicts
    and resource limits raise ValueError; I/O errors propagate. No output is written.
    """
    spec = LRTSSpec.model_validate(spec)
    validate_materialization(spec.metadata_materialization, cache)
    validate_materialization(spec.report_materialization, cache)
    selected = _metadata(spec, cache)
    report = _report(spec, cache)
    return LRTSDataset(
        schema_version=1,
        spec=spec,
        metadata_input_rows=selected.input_rows,
        metadata_outside_build=selected.outside_build,
        metadata_duplicates_excluded=selected.duplicates_excluded,
        metadata=selected.rows,
        report_pass_methods=report.pass_methods,
        report_fail_methods=report.fail_methods,
        report_skip_methods=report.skip_methods,
        report_duration=_measurement("lrts_report_duration", report.duration),
        reported_empty=report.empty,
        suites=_suites(report),
    )


def validate_lrts_dataset(dataset: LRTSDataset, cache: SourceCache) -> None:
    """Reconvert actual verified cache bytes; artifact metadata alone never proves source correspondence."""
    dataset = LRTSDataset.model_validate(dataset)
    if convert_lrts_source(dataset.spec, cache) != dataset:
        raise ValueError("Converted LRTS records disagree with verified source bytes")


def publish_lrts_dataset(dataset: LRTSDataset, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Revalidate source correspondence before deterministic immutable publication.

    Existing artifact writes supply concurrent retry/idempotence and interrupted
    publication behavior. Source cache and output remain separate. Converted rows
    honor the source's redistribution rule; recipe-only rows must stay local.
    This function writes one source artifact, never a completed experiment or a
    learning checkpoint. I/O errors propagate; conflicts/unsafe outputs reject.
    """
    dataset = LRTSDataset.model_validate(dataset)
    validate_cache_separation(cache.root, writer.run_root)
    path = writer.run_root / "corpus-lrts" / (dataset.fingerprint.removeprefix("sha256:") + ".json")
    if path.is_symlink() or not path.resolve().is_relative_to(writer.run_root.resolve()):
        raise ValueError("Converted LRTS artifacts must remain within their output tree")
    validate_lrts_dataset(dataset, cache)
    writer.write_immutable(path, dataset.to_json() + "\n")
    return path
