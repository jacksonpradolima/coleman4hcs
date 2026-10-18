"""Malformed source bindings, resource limits and forged metadata reject before completion."""

from dataclasses import replace
from typing import cast

import polars as pl
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission import (
    EvidenceKind,
    SourceEvidence,
    assess_source,
    audit_corpus,
    load_source_evidence,
    publish_corpus_admission,
    validate_admission_completion,
)
from coleman.corpus.admission_contracts import RealityFact
from coleman.corpus.admission_profiles import profile_table
from coleman.corpus.registry import load_registry
from coleman.corpus.stress_contracts import StressCase, StressFactors, StressFile, StressReceipt
from coleman.results.domain import fingerprint
from tests.corpus.admission_support import changed, lrts_source, review
from tests.corpus.test_admission_contracts import normalized


@pytest.mark.parametrize(
    "kind", ["column", "identity", "group", "verdict-type", "boolean-duration", "string-duration", "nonbinary"]
)
def test_normalized_profile_requires_complete_explicit_numeric_and_identity_schema(kind):
    table = normalized([("one", "group", 1, 0)])
    frame = table.frame
    if kind == "column":
        frame = frame.drop("duration")
    elif kind in {"identity", "group"}:
        name = "observation_id" if kind == "identity" else "group_id"
        frame = frame.with_columns(pl.lit(1).alias(name))
    elif kind == "verdict-type":
        frame = frame.with_columns(pl.col("verdict").cast(pl.Boolean))
    elif kind in {"boolean-duration", "string-duration"}:
        frame = frame.with_columns(pl.col("duration").cast(pl.Boolean if kind == "boolean-duration" else pl.String))
    else:
        frame = frame.with_columns(pl.lit(2, dtype=pl.Int64).alias("verdict"))
    with pytest.raises(ValueError):
        profile_table(table._replace(frame=frame))


def test_empty_and_unidentified_rows_remain_explicit_without_inventing_groups():
    empty = profile_table(normalized([]))
    assert empty.rows == empty.source_groups == 0 and empty.duration.minimum is None
    unidentified = profile_table(normalized([(None, None, None, None)]))
    assert unidentified.rows == unidentified.missing_identity_rows == 1
    assert unidentified.unique_observations == unidentified.source_groups == 0
    with pytest.raises(ValueError, match="complete source group"):
        profile_table(normalized([("one", None, 1, 0)]))


@pytest.mark.parametrize(
    "mutation",
    [
        {"rows": 3},
        {"duration": None},
        {"conflicting_identities": 1},
        {"source_groups": 3},
        {"source_groups": 0},
        {"execution_group_fields": ("g", "g")},
    ],
)
def test_profile_identity_and_group_conservation_reject_forged_counts(mutation):
    profile = profile_table(normalized([("one", "g", 1, 0), ("two", "g", 2, 1)]))
    with pytest.raises(ValueError):
        changed(profile, **mutation)


def test_measurement_names_and_required_counters_cannot_disappear_or_duplicate():
    profile = profile_table(normalized([("one", "g", 1, 0)]))
    for measurements in (profile.measurements + profile.measurements[:1], profile.measurements[:-1]):
        with pytest.raises(ValueError):
            changed(profile, measurements=measurements)
    observed = RealityFact(
        axis="language", status="observed", values=("Python",), locator="context", limitation="Owned metadata."
    )
    with pytest.raises(ValueError, match="Duplicate production"):
        changed(observed, values=("Python", "Python"))


def oversized_stress_receipt():
    """Unverified metadata only: a valid 2.4M-row receipt must fail audit preflight without opening any source."""
    case = StressCase(schema_version=1, builds=16, factors=StressFactors(size="large", variants="stable"))
    files = tuple(
        StressFile(path=f"truth/build-{i:02}.jsonl", sha256=fingerprint(["unverified", i]), size_bytes=1, rows=150000)
        for i in range(16)
    )
    native = StressFile(
        path="datasets/stress/features-engineered.parquet",
        sha256=fingerprint("unverified-native"),
        size_bytes=1,
        rows=2400000,
    )
    return StressReceipt(
        schema_version=1,
        case=case,
        producer=fingerprint("unverified"),
        parquet_writer="unverified",
        files=files + (native,),
    )


def test_oversized_source_and_excess_bindings_reject_before_any_source_io(tmp_path):
    payload = oversized_stress_receipt()
    binding = SourceEvidence(payload, review(payload), tmp_path / "must-not-open")
    with pytest.raises(ValueError, match="two-million-row"):
        assess_source(binding)
    with pytest.raises(ValueError, match="64-source"):
        audit_corpus(load_registry(), (binding,) * 65)
    assert not binding.root.exists()


def test_invalid_source_bindings_and_conflicting_reviews_never_publish(tmp_path):
    source, completion = lrts_source(tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        assess_source(replace(source, controlled_verifier=object()))
    with pytest.raises(ValueError, match="exact source payload"):
        assess_source(replace(source, review=changed(source.review, payload_fingerprint=fingerprint("wrong"))))
    weaker = replace(source, review=changed(source.review, status="rejected"))
    with pytest.raises(ValueError, match="Conflicting evidence"):
        audit_corpus(load_registry(), (source, weaker))
    for kind in ("unknown", "native_profile"):
        with pytest.raises(ValueError, match="family|Parquet"):
            load_source_evidence(cast(EvidenceKind, kind), completion, source.review)
    large = tmp_path / "oversized-contract.json"
    with large.open("wb") as stream:
        stream.truncate(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="16 MiB"):
        load_source_evidence("lrts", large, source.review, cache=source.cache)
    report = audit_corpus(load_registry(), (source,))
    other_card = changed(report.assessments[0].card, languages=("owned-conflicting-language",))
    registry = changed(
        report.registry,
        cards=tuple(
            other_card if c.manifest.benchmark_id == other_card.manifest.benchmark_id else c
            for c in report.registry.cards
        ),
    )
    with pytest.raises(ValueError, match="silently replace"):
        audit_corpus(registry, (source,))


def test_completion_requires_fixed_role_containment_and_bounded_metadata(tmp_path):
    source, _ = lrts_source(tmp_path)
    report = audit_corpus(load_registry(), (source,))
    completion = publish_corpus_admission(report, (source,), ArtifactWriter(tmp_path / "audit"))
    alias = completion.with_name("alias.json")
    alias.write_bytes(completion.read_bytes())
    with pytest.raises(ValueError, match="fixed complete.json"):
        validate_admission_completion(alias, (source,))
    original = completion.read_bytes()
    completion.write_bytes(b" " * (16 * 1024 + 1))
    with pytest.raises(ValueError, match="16 KiB"):
        validate_admission_completion(completion, (source,))
    completion.write_bytes(original)
    human = completion.with_name("report.md")
    original_human = human.read_bytes()
    human.unlink()
    outside = tmp_path / "outside-human.md"
    outside.write_bytes(original_human)
    human.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|escapes"):
        validate_admission_completion(completion, (source,))
