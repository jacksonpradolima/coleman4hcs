"""Versioned admission invariants, legacy readers, unavailable evidence and deterministic identity."""

import json
from itertools import permutations
from pathlib import Path

import polars as pl
import pytest

from coleman.corpus.admission import SourceEvidence, assess_source, audit_corpus, validate_corpus_sources
from coleman.corpus.admission_contracts import (
    AdmissionCompletion,
    AdmissionFile,
    CorpusAdmission,
    ProducerVerification,
    RealityFact,
    SourceAssessment,
    SourceReview,
)
from coleman.corpus.admission_profiles import ProfileTable, profile_table
from coleman.corpus.contracts import ObservationIdentity
from coleman.corpus.profiles import NativeSourceProfile, profile_native_source
from coleman.corpus.registry import load_registry
from coleman.results.domain import fingerprint
from coleman.selection.contracts import UnitSemantics
from tests.corpus.admission_support import FIXTURES, changed, controlled_source, lrts_source, review
from tests.corpus.test_profiles import local_card


def normalized(rows):
    """Create source-independent owned analytical rows, including deliberately invalid numeric observations."""
    return ProfileTable(
        pl.DataFrame(
            rows,
            schema={"observation_id": pl.String, "group_id": pl.String, "duration": pl.Float64, "verdict": pl.Int64},
            orient="row",
        ),
        ObservationIdentity(observation_fields=("execution_id",), group_fields=("source_group",)),
        UnitSemantics(unit="s", meaning="owned-reported-elapsed-seconds-v1"),
    )


def producer_metadata():
    """Valid owned proof schema metadata; no actual original SDK or raw-wire verification is claimed."""
    return ProducerVerification.model_validate(
        {
            "schema_version": 1,
            "payload_fingerprint": fingerprint("owned-dataset"),
            "producer_sha256": fingerprint("owned-producer"),
            "package": {
                "uri": "urn:coleman:owned-wheel-metadata",
                "version": "owned-v1",
                "citation": "Owned schema only",
                "sha256": fingerprint("owned-wheel"),
                "size_bytes": 1,
                "availability": "verified_local",
            },
            "capture_receipt_fingerprint": fingerprint("owned-receipt"),
            "capture_plan_fingerprint": fingerprint("owned-plan"),
            "raw_packets": (fingerprint("owned-buggy"), fingerprint("owned-fixed")),
            "verifier_sha256": fingerprint("owned-verifier"),
            "python_version": "owned-contract",
            "dependencies": ({"name": "coleman", "version": "owned-v1"},),
        }
    )


def valid_contract(model):
    """Start version mutations from fully valid records rather than trivially incomplete JSON."""
    if model is ProducerVerification:
        return producer_metadata()
    if model is CorpusAdmission:
        return model.from_json((FIXTURES / "corpus-admission-owned-v1.json").read_bytes())
    if model is SourceAssessment or model is SourceReview:
        assessment = SourceAssessment.from_json((FIXTURES / "source-assessment-native-v1.json").read_bytes())
        return assessment.review if model is SourceReview else assessment
    roles = tuple(
        AdmissionFile(path=name, sha256=fingerprint(name), size_bytes=1) for name in ("admission.json", "report.md")
    )
    return AdmissionCompletion(schema_version=1, report_fingerprint=fingerprint("owned-report"), files=roles)


def test_versioned_owned_goldens_reuse_an_actual_legacy_profile_without_granting_admission():
    payload = NativeSourceProfile.from_json((FIXTURES / "native-source-profile-v1.json").read_bytes())
    source = Path(__file__).parents[2] / "examples/fakedata/features-engineered.parquet"
    evidence = SourceEvidence(payload, review(payload, status="unverified"), source)
    assessment = assess_source(evidence)
    golden = FIXTURES / "source-assessment-native-v1.json"
    assert assessment.to_json() + "\n" == golden.read_text()
    assert SourceAssessment.from_json(golden.read_bytes()) == assessment
    actual = audit_corpus(load_registry(), (evidence,))
    metadata_only = changed(
        actual, producer=fingerprint("owned-admission-schema-golden-producer-not-runtime-verification")
    )
    whole = FIXTURES / "corpus-admission-owned-v1.json"
    assert metadata_only.to_json() + "\n" == whole.read_text()
    assert CorpusAdmission.from_json(whole.read_bytes()) == metadata_only
    assert metadata_only.covered_regimes == ()
    assert len(metadata_only.uncovered_regimes) == 7
    with pytest.raises(ValueError, match="recorded installed package"):
        validate_corpus_sources(metadata_only, (evidence,))
    # The actual legacy profile/card keep their original reader and bytes.
    assert NativeSourceProfile.from_json(payload.to_json()) == payload


def test_explicit_zero_negative_null_nonfinite_and_unknown_outcomes():
    profile = profile_table(
        normalized(
            [
                ("a", "g", 0, 0),
                ("b", "g", -2, 1),
                ("c", "g", None, None),
                ("d", "h", float("nan"), None),
                ("e", "h", float("inf"), 0),
                ("f", "h", float("-inf"), 1),
                ("z", "h", 8, None),
            ]
        )
    )
    duration = profile.duration
    assert (duration.finite_count, duration.null_count, duration.nonfinite_count) == (3, 1, 3)
    assert (duration.negative_count, duration.zero_count) == (1, 1)
    assert (duration.minimum, duration.median, duration.p95, duration.maximum) == (-2, 0, 8, 8)
    facts = {m.name: m for m in profile.measurements}
    assert facts["source_passed_rows"].value == facts["source_failed_rows"].value == 2
    assert facts["source_failure_fraction"].value == 0.5
    assert facts["source_observed_ci_span"].value is None
    assert facts["source_observed_ci_span"].status == "unavailable"
    assert profile.unique_faults is None


def test_empty_known_outcome_population_is_unavailable_not_a_zero_failure_fraction():
    absent = profile_table(normalized([("a", "g", None, None)]))
    observed_zero = profile_table(normalized([("a", "g", 0, 0)]))

    def fraction(p):
        return next(m for m in p.measurements if m.name == "source_failure_fraction")

    assert fraction(absent).value is None
    assert fraction(absent).status == "unavailable"
    assert fraction(observed_zero).value == 0
    assert fraction(observed_zero).status == "observed"
    assert absent.duration.minimum is None
    assert observed_zero.duration.minimum == 0


def test_profile_permutation_duplicate_and_conflict_metamorphic_properties():
    rows = (("a", "g", -0.0, 0), ("b", "g", 2, 1), ("c", "h", None, None))
    expected = profile_table(normalized(rows))
    for order in permutations(rows):
        actual = profile_table(normalized(order))
        assert actual == expected
        assert actual.model_dump_json() == expected.model_dump_json()
    duplicate = profile_table(normalized((*rows, rows[0])))
    assert duplicate.rows == 4
    assert duplicate.unique_observations == 3
    assert duplicate.duplicate_rows == 1
    assert duplicate.conflicting_identities == 0
    conflict = profile_table(normalized((*rows, ("a", "other-group", 3, 1))))
    assert conflict.conflicting_identities == 1
    assert conflict.source_groups == 3


@pytest.mark.parametrize(
    "name,mutation",
    [
        ("source_failed_rows", {"value": 2}),
        ("source_failed_rows", {"unit": "s"}),
        ("source_passed_rows", {"value": 0.5}),
        ("source_failure_fraction", {"value": 0}),
        ("source_observed_ci_span", {"value": 0, "status": "observed"}),
    ],
)
def test_profile_outcome_conservation_and_chronology_cannot_be_forged(name, mutation):
    profile = profile_table(normalized([("a", "g", 0, 0), ("b", "g", 2, 1)]))
    measurements = tuple(changed(item, **mutation) if item.name == name else item for item in profile.measurements)
    with pytest.raises(ValueError):
        changed(profile, measurements=measurements)


def test_signed_zero_measurements_and_constructed_chronology_are_canonical():
    table = normalized([("a", "g", 0, 0)])
    original = profile_table(table)
    signed = tuple(
        changed(item, value=-0.0, minimum=-0.0) if item.value == 0 else item for item in original.measurements
    )
    canonical = changed(original, measurements=signed)
    assert canonical.model_dump_json() == original.model_dump_json()
    constructed = profile_table(table._replace(constructed_history=True))
    span = next(m for m in constructed.measurements if m.name == "source_observed_ci_span")
    assert span.value is None
    assert span.status == "not_applicable"


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
@pytest.mark.parametrize(
    "model", [SourceReview, SourceAssessment, CorpusAdmission, AdmissionCompletion, ProducerVerification]
)
def test_missing_old_future_coerced_versions_reject_before_any_io(model, version):
    valid = valid_contract(model)
    assert model.from_json(valid.to_json()) == valid
    raw = valid.model_dump(mode="json")
    if version is None:
        raw.pop("schema_version")
    else:
        raw["schema_version"] = version
    encoded = json.dumps(raw)
    with pytest.raises(ValueError):
        model.from_json(encoded)


def test_native_variant_execution_groups_preserve_original_declared_groups(tmp_path):
    path = tmp_path / "variant-native.parquet"
    pl.DataFrame(
        {
            "BuildId": [1, 1],
            "Name": ["same", "same"],
            "Variant": ["a", "b"],
            "Duration": [0.0, 2.0],
            "Verdict": [0, 1],
            "LastRun": ["", ""],
        }
    ).write_parquet(path)
    card = local_card(path, variants=True)
    payload = profile_native_source(card, path)
    binding = SourceEvidence(payload, review(payload, status="unverified"), path)
    assessment = assess_source(binding)
    assert assessment.card == card
    assert card.manifest.identity.group_fields == ("BuildId",)
    assert assessment.profile.identity == card.manifest.identity
    assert assessment.profile.execution_group_fields == ("BuildId", "Variant")
    assert assessment.profile.source_groups == 2
    assert payload.logical_builds == 1
    changed_groups = changed(assessment.profile, execution_group_fields=("BuildId",))
    with pytest.raises(ValueError, match="execution-group dimensions"):
        changed(assessment, profile=changed_groups)


@pytest.mark.parametrize(
    "mutation",
    [
        {"raw_packets": (fingerprint("same"),) * 2},
        {"dependencies": ({"name": "Coleman", "version": "owned-v1"}, {"name": "coleman", "version": "owned-v1"})},
        {"dependencies": ({"name": "coleman", "version": "another-version"},)},
        {"dependencies": ({"name": "unrelated", "version": "owned-v1"},)},
    ],
)
def test_original_producer_metadata_cannot_hide_duplicate_packets_or_ambiguous_sdk_versions(mutation):
    valid = producer_metadata()
    with pytest.raises(ValueError):
        changed(valid, **mutation)


def test_review_context_and_source_scope_cannot_be_changed_silently(tmp_path):
    source, _ = lrts_source(tmp_path)
    assessment = assess_source(source)
    assert assessment.scope == "blocked"
    assert assessment.card.history == "constructed"
    assert {fact.status for fact in assessment.reality if fact.axis == "source_execution_model"} == {"synthetic"}
    assert SourceAssessment.from_json(assessment.to_json()) == assessment
    for mutation in [
        {"scope": "synthetic_mechanism"},
        {"admission_reasons": ()},
        {"historical_blockers": ("invented",)},
        {"reality": assessment.reality[:-1]},
        {"reality": (assessment.reality[0],) * 35},
    ]:
        with pytest.raises(ValueError):
            changed(assessment, **mutation)
    incompatible = changed(source.review, history="observed")
    with pytest.raises(ValueError, match="history"):
        changed(assessment, review=incompatible)
    wrong_source = changed(source.review, payload_fingerprint=fingerprint("another-source"))
    with pytest.raises(ValueError, match="identity"):
        changed(assessment, review=wrong_source)


def test_source_units_and_legacy_contracts_keep_their_original_meanings(tmp_path):
    source, _ = lrts_source(tmp_path)
    receipt_before = source.payload.to_json()
    registry_before = load_registry().to_json()
    assessment = assess_source(source)
    assert source.payload.to_json() == receipt_before
    assert load_registry().to_json() == registry_before
    assert assessment.profile.duration.semantics.unit == "s"
    assert assessment.card.manifest.outcomes.duration is not None
    assert (
        assessment.card.manifest.outcomes.duration.meaning
        == "jenkins-source-method-duration-seconds-binary64-projection-v1"
    )
    bad_units = changed(assessment.profile.duration, semantics=UnitSemantics(unit="ms", meaning="another-unit-v1"))
    bad_profile = changed(assessment.profile, duration=bad_units)
    with pytest.raises(ValueError, match="units/meaning"):
        changed(assessment, profile=bad_profile)
    assert type(source.payload).from_json(receipt_before) == source.payload
    assert type(load_registry()).from_json(registry_before).to_json() == registry_before


def test_complete_fact_status_contract_keeps_unknown_and_inapplicable_distinct():
    fact = RealityFact(axis="language", status="missing", values=None, locator="source", limitation="Unverified.")
    assert changed(fact, status="not_applicable") != fact
    for values in [(), ("unknown",), ("a", "a")]:
        with pytest.raises(ValueError):
            changed(fact, values=values)
    observed = changed(fact, status="observed", values=("Python", "C"))
    assert observed.values == ("C", "Python")


def test_context_reviews_have_explicit_date_language_and_no_duplicate_limitations(tmp_path):
    source = controlled_source(tmp_path)
    for mutation in [
        {"reviewed_at": None},
        {"subject_language": "unknown"},
        {"limitations": ("same", "same")},
        {"limitations": ()},
    ]:
        with pytest.raises(ValueError):
            changed(source.review, **mutation)
    unverified = review(source.payload, status="unverified")
    assert unverified.reviewed_at is None


def test_size_and_fixed_completion_role_bounds_fail_closed():
    with pytest.raises(ValueError, match="16-MiB"):
        CorpusAdmission.from_json(b" " * (CorpusAdmission.MAXIMUM_BYTES + 1))
    digest = fingerprint("owned-role")
    roles = tuple(AdmissionFile(path=p, sha256=digest, size_bytes=1) for p in ("admission.json", "report.md"))
    completion = AdmissionCompletion(schema_version=1, report_fingerprint=digest, files=roles)
    assert AdmissionCompletion.from_json(completion.to_json()) == completion
    reversed_roles = tuple(reversed(roles))
    with pytest.raises(ValueError, match="exactly once"):
        changed(completion, files=reversed_roles)
    with pytest.raises(ValueError, match="Duplicate JSON"):
        AdmissionCompletion.from_json('{"schema_version":1,"schema_version":1}')
