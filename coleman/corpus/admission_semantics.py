"""Pure source-purpose and coverage rules; successful ETL never certifies efficacy.

Rules are fixed before an audit. Missing historical clocks remain blockers;
functional negative/inconclusive outcomes never become a reason to discard a
source. Source/component reviews are explicit attestations, separately bound to
actual bytes by the offline publication service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from coleman.corpus.admission_contracts import FactAxis, ProducerVerification, SourceReview
from coleman.corpus.admission_profiles import profile_existing_native
from coleman.corpus.admission_reality import production_facts
from coleman.corpus.admission_sources import (
    source_card,
    source_classification,
    source_description,
    source_duration_semantics,
    source_execution_group_fields,
)
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.profiles import NativeSourceProfile
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_contracts import StressCase, StressReceipt
from coleman.corpus.stress_design import adversarial_cases, covering_design
from coleman.selection.contracts import UnitSemantics

if TYPE_CHECKING:
    from coleman.corpus.admission_contracts import (
        CorpusAdmission,
        EvidencePayload,
        ObservationProfile,
        SourceAssessment,
    )

Scope = Literal["blocked", "controlled_functional", "constructed_functional", "synthetic_mechanism"]


def _controlled_review_blockers(payload: ControlledDataset, review: SourceReview) -> list[str]:
    """Require reviewed subject provenance without deriving a real fault or language from outcomes."""
    reasons = []
    context = payload.spec.context
    if review.subject_language is None:
        reasons.append("subject_language_not_reviewed")
    elif review.subject_language.casefold() != context.language.casefold():
        reasons.append("reviewed_subject_language_conflicts_with_context")
    if context.fault_origin == "real" and review.fault_reference is None:
        reasons.append("curated_real_fault_reference_missing")
    return reasons


def _controlled_blockers(payload: ControlledDataset, review: SourceReview) -> tuple[str, ...]:
    """Require a complete declared capture context, retaining every negative/inconclusive pair."""
    reasons = _controlled_review_blockers(payload, review)
    context = payload.spec.context
    if context.runtime.execution_model == "constructed":
        reasons.append("constructed_executor_not_observed")
    trials = {row.trial_id for row in payload.observations}
    # ControlledDataset already guarantees unique in-universe role/test/trial
    # keys. Cardinality proves completeness without expanding an adversarial
    # trials x declared-tests Cartesian product.
    expected_rows = 2 * len(trials) * len(context.test_ids)
    if not trials or len(payload.observations) != expected_rows:
        reasons.append("incomplete_declared_role_test_trial_matrix")
    if any(row.outcome in {"unavailable", "cancelled", "infrastructure_error"} for row in payload.observations):
        reasons.append("capture_contains_uncompleted_execution")
    return tuple(sorted(reasons))


def qualification(
    payload: EvidencePayload,
    profile: ObservationProfile,
    review: SourceReview,
    verification: ProducerVerification | None = None,
) -> tuple[Scope, tuple[str, ...]]:
    """Return a narrow purpose or explicit blockers, independent of favorable outcomes.

    Historical source receipts currently lack protected availability/universe
    evidence; native profiling alone never supplies it. A constructed numerical
    case qualifies only as a mechanism check, not as empirical diversity.
    Controlled qualification requires reviewed source context, complete selected
    role/test/trial observations and a genuine-fault reference when declared real.
    """
    source_classification(payload, review)
    reasons = []
    if review.status != "reviewed":
        reasons.append("source_context_" + review.status)
    if profile.rows == 0:
        reasons.append("no_retained_source_observations")
    if profile.missing_identity_rows or profile.conflicting_identities:
        reasons.append("source_identity_unresolved")
    if isinstance(payload, ControlledDataset):
        reasons.extend(_controlled_blockers(payload, review))
        if verification is None:
            reasons.append("original_raw_producer_verification_missing")
        scope: Scope = (
            "controlled_functional" if payload.spec.context.fault_origin == "real" else "constructed_functional"
        )
    elif isinstance(payload, StressReceipt):
        scope = "synthetic_mechanism"
    else:
        scope = "blocked"
        reasons.append("descriptive_source_is_not_protected_temporal_admission")
    return ("blocked" if reasons else scope), tuple(sorted(set(reasons)))


def historical_blockers(payload: EvidencePayload) -> tuple[str, ...]:
    """Keep all unresolved historical prerequisites separate from functional/mechanism purposes."""
    if isinstance(payload, NativeSourceProfile):
        return tuple(sorted({*payload.card.manifest.blocking_reasons, "native_profile_is_descriptive_only"}))
    if isinstance(payload, ControlledDataset):
        return ("capture_clocks_are_not_ci_chronology", "controlled_source_is_not_historical_ci")
    if isinstance(payload, StressReceipt):
        return ("constructed_truth_is_not_observed_historical_ci",)
    reasons = {
        "outcome_availability_unverified",
        "feature_availability_unverified",
        "expected_test_universe_unverified",
        "cross_round_test_correspondence_assumed",
        "protected_temporal_protocol_unverified",
        "physical_execution_topology_unverified",
        "cross_source_project_independence_unverified",
    }
    if isinstance(payload, RTPTorrentETLReceipt):
        reasons.add("logical_build_job_mapping_unverified")
    if isinstance(payload, HCSReceipt):
        reasons.update({"upstream_omitted_job_count_unavailable", "upstream_nonpass_statuses_collapsed"})
    return tuple(sorted(reasons))


def retained_rows(payload: EvidencePayload) -> int:
    """Read retained analytical-row counts, never projected native counts or independent sample sizes."""
    if isinstance(payload, NativeSourceProfile):
        return payload.rows
    if isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt)):
        return payload.observed_cases
    if isinstance(payload, RTPTorrentETLReceipt):
        return payload.observed_executions
    if isinstance(payload, HCSReceipt):
        return payload.accounting.selected_rows
    if isinstance(payload, ControlledDataset):
        return len(payload.observations)
    return payload.files[-1].rows


def _chronology_binding(assessment: SourceAssessment) -> None:
    """Keep unavailable observed-history clocks distinct from inapplicable constructed-history clocks."""
    missing = "not_applicable" if assessment.review.history == "constructed" else "unavailable"
    chronology = [
        m for m in assessment.profile.measurements if m.name in {"source_observed_ci_span", "source_observed_churn"}
    ]
    if any(item.status != missing for item in chronology):
        raise ValueError("Admission chronology status disagrees with declared source history")


def _profile_bindings(assessment: SourceAssessment) -> None:
    """Bind retained counts and identity dimensions to their actual receipt family on every read."""
    payload, profile = assessment.payload, assessment.profile
    _chronology_binding(assessment)
    if profile.identity != source_description(payload).identity:
        raise ValueError("Admission profile identity disagrees with source dimensions")
    if profile.execution_group_fields != source_execution_group_fields(payload):
        raise ValueError("Admission execution-group dimensions disagree with source group counts")
    if profile.rows != retained_rows(payload):
        raise ValueError("Admission profile retained rows disagree with source receipt")
    if isinstance(payload, NativeSourceProfile):
        if profile != profile_existing_native(payload):
            raise ValueError("Admission profile disagrees with the unchanged native description")
        return
    original = source_duration_semantics(payload)
    expected = UnitSemantics(unit=original.unit, meaning="binary64-description:" + original.meaning)
    if profile.duration.semantics != expected:
        raise ValueError("Admission profile duration units/meaning disagree with its source projection")


def validate_assessment(assessment: SourceAssessment) -> None:
    """Reapply purpose, source-card and all-axis rules without file access or source execution."""
    _profile_bindings(assessment)
    facts = assessment.reality
    if {fact.axis for fact in facts} != set(get_args(FactAxis)) or len({fact.axis for fact in facts}) != len(facts):
        raise ValueError("Admission reality matrix requires every axis exactly once")
    facts = tuple(sorted(facts, key=lambda fact: fact.axis))
    validate_producer_verification(assessment.payload, assessment.producer_verification)
    scope, reasons = qualification(
        assessment.payload, assessment.profile, assessment.review, assessment.producer_verification
    )
    if (
        assessment.card != source_card(assessment.payload, assessment.profile, assessment.review)
        or facts != production_facts(assessment.payload, assessment.review)
        or assessment.scope != scope
        or assessment.admission_reasons != reasons
        or assessment.historical_blockers != historical_blockers(assessment.payload)
    ):
        raise ValueError("Admission claims disagree with source-bound purpose/evidence rules")
    object.__setattr__(assessment, "reality", facts)


def validate_producer_verification(payload: EvidencePayload, verification: ProducerVerification | None) -> None:
    """Bind a recorded raw-wire attestation to the actual normalized dataset and both declared raw packet members."""
    if verification is None:
        return
    if not isinstance(payload, ControlledDataset) or verification.payload_fingerprint != payload.fingerprint:
        raise ValueError("Raw producer verification does not bind this controlled source dataset")
    members = {member.name: member.sha256 for member in payload.spec.report_materialization.spec.members}
    expected = tuple(members.get(role + "-capture.json") for role in ("buggy", "fixed"))
    if verification.raw_packets != expected:
        raise ValueError("Raw producer verification disagrees with source packet membership/order")


def _stress_family_covered(assessments: tuple[SourceAssessment, ...]) -> bool:
    """Require all declared eight-build covering and named adversarial cases, not one large export.

    Pair coverage is independently certified by the existing design contract.
    Seeds vary deterministic realizations; they never add independent projects.
    Every required case needs its own actual source-verification binding during
    explicit audit/publication. Partial sets cannot satisfy the broad regime.
    """
    expected = {StressCase(schema_version=1, factors=row).fingerprint for row in covering_design().rows} | {
        case.fingerprint for _, case in adversarial_cases()
    }
    verified = {
        item.payload.case.fingerprint
        for item in assessments
        if item.scope == "synthetic_mechanism" and isinstance(item.payload, StressReceipt)
    }
    return expected.issubset(verified)


def admitted_regimes(assessments: tuple[SourceAssessment, ...]) -> tuple[str, ...]:
    """Derive scoped diversity without promoting controlled/synthetic data to historical CI."""
    covered = set()
    for assessment in assessments:
        payload = assessment.payload
        if assessment.scope == "controlled_functional" and isinstance(payload, ControlledDataset):
            language = payload.spec.context.language.casefold()
            covered.add("controlled_java_real_faults" if language == "java" else "controlled_non_java_real_faults")
    if _stress_family_covered(assessments):
        covered.add("deterministic_adversarial_simulation")
    return tuple(sorted(covered))


def observed_models(assessments: tuple[SourceAssessment, ...]) -> tuple[str, ...]:
    """Count reviewed observed source structures, excluding constructed captures and unknown providers.

    Jenkins single/stage views are one model; metadata-only variants do not add
    models. Preserving structures does not establish historical admission or
    operational representativeness. The two-model gate uses Jenkins and GitLab
    receipts with observed, explicitly reviewed histories.
    """
    models = set()
    for assessment in assessments:
        payload = assessment.payload
        if not isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt, HCSReceipt)):
            continue
        if assessment.review.status != "reviewed" or assessment.card.history != "observed":
            continue
        for fact in assessment.reality:
            if fact.axis == "source_execution_model" and fact.status == "observed" and fact.values:
                models.update(fact.values)
    return tuple(sorted(models))


def validate_corpus_admission(report: CorpusAdmission) -> None:
    """Canonicalize identical retries and reject conflicting/unregistered evidence or weakened coverage."""
    unique: dict[str, SourceAssessment] = {}
    for assessment in report.assessments:
        key = assessment.card.manifest.benchmark_id
        if report.registry.card(key) != assessment.card:
            raise ValueError("Admission source card differs from the existing registry entry")
        if unique.setdefault(key, assessment) != assessment:
            raise ValueError("Conflicting evidence for one logical source selection")
    assessments = tuple(unique[key] for key in sorted(unique))
    registered = {card.manifest.benchmark_id for card in report.registry.cards}
    covered = admitted_regimes(assessments)
    uncovered = tuple(sorted(set(report.registry.required_regimes) - set(covered)))
    if (
        report.unassessed_sources != tuple(sorted(registered - unique.keys()))
        or report.covered_regimes != covered
        or report.uncovered_regimes != uncovered
        or report.observed_source_models != observed_models(assessments)
    ):
        raise ValueError("Corpus admission inventory disagrees with source evidence or required diversity")
    object.__setattr__(report, "assessments", assessments)
