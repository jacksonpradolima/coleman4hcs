"""Offline controlled-fault conversion through existing source integrity and immutable artifact services."""

from __future__ import annotations

import hashlib
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus.controlled_contracts import (
    ControlledDataset,
    ControlledObservation,
    ControlledReport,
    ControlledSpec,
)
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_materialization


def convert_controlled_source(spec: ControlledSpec, cache: SourceCache) -> ControlledDataset:
    """Verify pinned inputs/report and convert bounded observations without executing foreign code.

    Each input receipt is checked against actual offline cached bytes. Identical
    role/test/trial repetitions are excluded with counts; conflicting repetitions
    raise ValueError. Aborted/missing/unexpected outcomes remain evidence, not a
    substituted pass or fault detection. Source/input terms remain independent of
    the report's own terms. No source is fetched, extracted again or executed.
    Schema/size/digest/context conflicts raise ValueError; I/O errors propagate.
    """
    spec = ControlledSpec.model_validate(spec)
    for item in spec.context.inputs:
        validate_materialization(item.materialization, cache)
    receipt = spec.report_materialization
    validate_materialization(receipt, cache)
    member = next(item for item in receipt.spec.members if item.name == spec.report_member)
    path = cache.member_path(receipt, spec.report_member)
    with path.open("rb") as stream:
        payload = stream.read(member.size_bytes + 1)
    if len(payload) != member.size_bytes or "sha256:" + hashlib.sha256(payload).hexdigest() != member.sha256:
        raise ValueError("Parsed controlled report bytes disagree with pinned member")
    report = ControlledReport.from_json(payload)
    if report.context_fingerprint != spec.context.fingerprint:
        raise ValueError("Controlled report context fingerprint mismatch")
    unique: dict[tuple[str, str, str], ControlledObservation] = {}
    for observation in report.observations:
        prior = unique.setdefault(observation.logical_key, observation)
        if prior != observation:
            raise ValueError("Conflicting controlled execution duplicate")
    return ControlledDataset(
        schema_version=1,
        spec=spec,
        input_rows=len(report.observations),
        duplicates_excluded=len(report.observations) - len(unique),
        observations=tuple(unique.values()),
    )


def validate_controlled_dataset(dataset: ControlledDataset, cache: SourceCache) -> None:
    """Reconvert actual source bytes; embedded metadata alone never proves correspondence or execution."""
    dataset = ControlledDataset.model_validate(dataset)
    if convert_controlled_source(dataset.spec, cache) != dataset:
        raise ValueError("Controlled dataset disagrees with pinned source observations")


def publish_controlled_dataset(dataset: ControlledDataset, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Reverify before atomic immutable publication; interrupted work never becomes a completed experiment.

    Concurrent identical retries share one content-addressed artifact. Source cache
    and output must be separate. This writes no learning checkpoint and grants no
    evaluation admission. Recipe-only observations must remain local. A content
    conflict raises ValueError; filesystem/integrity errors propagate.
    """
    dataset = ControlledDataset.model_validate(dataset)
    validate_cache_separation(cache.root, writer.run_root)
    path = writer.run_root / "corpus-controlled" / (dataset.fingerprint.removeprefix("sha256:") + ".json")
    if path.is_symlink() or not path.resolve().is_relative_to(writer.run_root.resolve()):
        raise ValueError("Controlled artifacts must remain within their output tree")
    validate_controlled_dataset(dataset, cache)
    writer.write_immutable(path, dataset.to_json() + "\n")
    return path
