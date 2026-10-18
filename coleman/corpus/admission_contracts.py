"""Evidence-backed corpus descriptions, separate from phase-one fixture readiness.

These immutable records describe a verified local snapshot, not execution truth,
legal permission or policy efficacy. Publication requires independent byte
revalidation; deserialization alone cannot certify cached source correspondence.
Legacy registry, profile and receipt versions retain their original readers.
"""

from __future__ import annotations

import re
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.contracts import DataCard, ObservationIdentity, SourceArtifact
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.profiles import NativeSourceProfile, NumericProfile
from coleman.corpus.registry import CorpusRegistry
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_contracts import StressReceipt
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import Fingerprint, Identifier, Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

Count = Annotated[int, Field(strict=True, ge=0, le=2_000_000)]
EvidenceLabel = Annotated[Identifier, Field(max_length=2048)]
FactAxis = Literal[
    "language",
    "build_system",
    "source_execution_model",
    "triggers",
    "test_levels",
    "setup_cost",
    "shared_fixtures",
    "test_order_dependence",
    "serial_execution",
    "parallel_execution",
    "matrix_execution",
    "shards",
    "reruns",
    "runner_type",
    "queue_delay",
    "concurrency",
    "cancellation",
    "retries",
    "quarantine",
    "flakiness",
    "timeouts",
    "infrastructure_failures",
    "vendor_changes",
    "monorepos",
    "renames",
    "change_to_test_traceability",
    "branch_protection",
    "mandatory_suites",
    "human_overrides",
    "partial_outcomes",
    "delayed_outcomes",
    "corrected_outcomes",
    "retention_gaps",
    "outcome_availability",
    "feature_availability",
]


class RealityFact(Immutable):
    """One qualitative axis with explicit epistemic status and a bounded source locator.

    A reported label is evidence of that label, not physical runtime topology.
    Missing/inapplicable values are null, never the string ``false`` or zero.
    The locator names fields or a recipe rule in the embedded payload; it never
    executes a path, URL or import. Source verification rederives every fact.
    """

    axis: FactAxis
    status: Literal["observed", "reconstructed", "missing", "synthetic", "not_applicable"]
    values: Annotated[tuple[EvidenceLabel, ...], Field(min_length=1, max_length=32)] | None
    locator: EvidenceLabel
    limitation: EvidenceLabel

    @model_validator(mode="after")
    def _values(self) -> Self:
        """Keep unavailable axes distinct and canonicalize declared unordered labels."""
        if (self.values is None) != (self.status in {"missing", "not_applicable"}):
            raise ValueError("Reality fact values disagree with their evidence status")
        if self.values is not None:
            if len(set(self.values)) != len(self.values):
                raise ValueError("Duplicate production-reality value")
            object.__setattr__(self, "values", tuple(sorted(self.values)))
        return self


class ObservationProfile(Immutable):
    """Source-row description; counts are observations, never independent trials.

    Duration quantiles retain explicit reported/harness/constructed units, with
    raw negative, nonfinite, null and zero categories. Churn measurements, when
    available, state their label-order assumption. A curated fault label never
    creates a unique-fault count. Reads check conservation, not actual file bytes.
    """

    rows: Count
    unique_observations: Count
    duplicate_rows: Count
    missing_identity_rows: Count
    conflicting_identities: Count
    source_groups: Count
    identity: ObservationIdentity
    execution_group_fields: Annotated[tuple[EvidenceLabel, ...], Field(min_length=1, max_length=16)]
    duration: NumericProfile
    measurements: Annotated[tuple[Measurement, ...], Field(min_length=1, max_length=64)]
    unique_faults: None = None

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Conserve all input rows and reject contradictory identity/group summaries."""
        if self.unique_observations + self.duplicate_rows + self.missing_identity_rows != self.rows:
            raise ValueError("Admission profile identity counts do not conserve rows")
        if self.duration.rows != self.rows:
            raise ValueError("Admission profile duration counts do not conserve rows")
        if self.conflicting_identities > min(self.unique_observations, self.duplicate_rows):
            raise ValueError("Conflicting identities require repeated identified observations")
        if self.source_groups > self.rows:
            raise ValueError("Source execution groups exceed source rows")
        if self.unique_observations and not self.source_groups:
            raise ValueError("Identified source observations require a source execution group")
        if len(set(self.execution_group_fields)) != len(self.execution_group_fields):
            raise ValueError("Duplicate source execution-group dimension")
        object.__setattr__(self, "execution_group_fields", tuple(sorted(self.execution_group_fields)))
        names = [item.name for item in self.measurements]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate admission profile measurement")
        measurements = _canonical_profile_measurements(self.measurements)
        _profile_measurements(measurements, self.rows)
        object.__setattr__(self, "measurements", measurements)
        return self


def _canonical_profile_measurements(measurements: tuple[Measurement, ...]) -> tuple[Measurement, ...]:
    """Give signed binary64 zeros one representation without changing legacy measurements or nonzero values."""
    normalized = []
    for item in measurements:
        fields = item.model_dump()
        if item.value == 0:
            fields["value"] = 0.0
        if item.minimum == 0:
            fields["minimum"] = 0.0
        normalized.append(Measurement.model_validate(fields))
    return tuple(sorted(normalized, key=lambda item: item.name))


def _profile_counter(item: Measurement, rows: int) -> int:
    """Require an exact finite nonnegative observed integer count in explicit source-row units."""
    if (
        item.status != "observed"
        or item.value is None
        or item.unit != "1"
        or item.direction != "none"
        or item.minimum != 0
        or not item.value.is_integer()
        or not 0 <= item.value <= rows
    ):
        raise ValueError("Admission outcome counters require observed nonnegative integral row units")
    return int(item.value)


def _profile_fraction(fraction: Measurement, failed: int, passed: int) -> None:
    """Check a specified binary64 quotient of exact counters; it is not a calibrated probability."""
    known = failed + passed
    expected = failed / known if known else None
    # This is the specified binary64 quotient of exact integral counters, not
    # an approximate mathematical metric comparison or an inference threshold.
    if (
        fraction.unit != "1"
        or fraction.direction != "none"
        or fraction.minimum != 0
        or fraction.value != expected
        or fraction.status != ("observed" if known else "unavailable")
    ):
        raise ValueError("Admission binary-label fraction disagrees with known outcome counters")


def _profile_measurements(measurements: tuple[Measurement, ...], rows: int) -> None:
    """Conserve explicit binary-label counters while retaining unavailable/inapplicable chronology."""
    by_name = {item.name: item for item in measurements}
    counter_names = ("source_failed_rows", "source_passed_rows", "source_unknown_outcomes")
    required = {*counter_names, "source_failure_fraction", "source_observed_ci_span", "source_observed_churn"}
    if not required.issubset(by_name):
        raise ValueError("Admission profile requires its explicit outcome and unavailable chronology measurements")
    counts = [_profile_counter(by_name[name], rows) for name in counter_names]
    if sum(counts) != rows:
        raise ValueError("Admission outcome counters do not conserve retained rows")
    _profile_fraction(by_name["source_failure_fraction"], counts[0], counts[1])
    for name, unit in (("source_observed_ci_span", "s"), ("source_observed_churn", "1")):
        item = by_name[name]
        if item.value is not None or item.status not in {"unavailable", "not_applicable"} or item.unit != unit:
            raise ValueError("Admission source profiles cannot invent observed CI chronology")


EvidencePayload = (
    NativeSourceProfile
    | LRTSETLReceipt
    | LRTSStageReceipt
    | RTPTorrentETLReceipt
    | HCSReceipt
    | ControlledDataset
    | StressReceipt
)


class SourceReview(VersionedContract):
    """Explicit source-context review bound to the complete evidence payload.

    This is a producer attestation, not proof of legal rights or execution
    truth. References are descriptive and never fetched. Existing explicit
    history/origin fields must agree; formats lacking them cannot acquire
    observed-CI status merely by passing a byte validator. A curated real-fault
    reference is separate from a unique-fault mapping and from test outcomes.
    """

    payload_fingerprint: Fingerprint
    status: Literal["reviewed", "unverified", "rejected"]
    reviewed_at: Timestamp | None
    history: Literal["observed", "constructed", "unverified"]
    reference: EvidenceLabel
    fault_reference: EvidenceLabel | None
    subject_language: EvidenceLabel | None
    limitations: Annotated[tuple[EvidenceLabel, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def _declarations(self) -> Self:
        """Keep unknown language null and canonicalize unordered explicit review limitations."""
        if self.status == "reviewed" and self.reviewed_at is None:
            raise ValueError("Reviewed source context requires its explicit UTC review instant")
        if self.subject_language is not None and self.subject_language.casefold() in {
            "unknown",
            "unavailable",
            "unverified",
            "not_applicable",
            "n/a",
        }:
            raise ValueError("Unknown reviewed subject language must be null")
        if len(set(self.limitations)) != len(self.limitations):
            raise ValueError("Duplicate source review limitation")
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))
        return self


class VerificationDependency(Immutable):
    """One installed original-verifier dependency version; metadata is provenance, not a dependency security audit."""

    name: EvidenceLabel
    version: EvidenceLabel


def _verifier_dependencies(
    dependencies: tuple[VerificationDependency, ...], version: str
) -> tuple[VerificationDependency, ...]:
    """Reject ambiguous distribution aliases and require the recorded original SDK version."""
    names = [re.sub(r"[-_.]+", "-", item.name).casefold() for item in dependencies]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate original-verifier dependency identity")
    sdk = [item.version for item, name in zip(dependencies, names, strict=True) if name == "coleman"]
    if sdk != [version]:
        raise ValueError("Original-verifier dependency versions disagree with its pinned SDK version")
    return tuple(sorted(dependencies, key=lambda item: item.name))


class ProducerVerification(VersionedContract):
    """Offline raw-wire correspondence checked by an explicit original-producer verifier.

    This portable record is an attestation until the caller supplies the actual
    verifier binding again. The original installed SDK is checked against its
    pinned wheel; its existing parser revalidates source/recipe/context and both
    raw role packets against normalized observations. This never proves authentic
    physical execution, legal rights, fault independence or policy efficacy.
    """

    method: Literal["original-controlled-capture-wire-v1"] = "original-controlled-capture-wire-v1"
    payload_fingerprint: Fingerprint
    producer_sha256: Fingerprint
    package: SourceArtifact
    capture_receipt_fingerprint: Fingerprint
    capture_plan_fingerprint: Fingerprint
    raw_packets: Annotated[tuple[Fingerprint, ...], Field(min_length=2, max_length=2)]
    verifier_sha256: Fingerprint
    python_version: EvidenceLabel
    dependencies: Annotated[tuple[VerificationDependency, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def _provenance(self) -> Self:
        """Require frozen original-package bytes and canonical unambiguous dependency/packet provenance."""
        if (
            self.package.sha256 is None
            or self.package.size_bytes is None
            or self.package.size_bytes == 0
            or self.package.availability != "verified_local"
        ):
            raise ValueError("Original producer verification requires its pinned verified installed wheel")
        if len(set(self.raw_packets)) != 2:
            raise ValueError("Producer verification requires two distinct original role packets")
        object.__setattr__(self, "dependencies", _verifier_dependencies(self.dependencies, self.package.version))
        return self


class SourceAssessment(VersionedContract):
    """A source-bound assessment whose scope and facts are rederived on validation.

    Historical eligibility, functional reproduction and synthetic mechanisms
    are distinct purposes. A passing purpose never clears another purpose's
    blockers. Embedded receipts preserve component-specific terms, source
    versions, jobs/stages and dependence; no source data is copied into Git.
    """

    payload: EvidencePayload
    review: SourceReview
    producer_verification: ProducerVerification | None = None
    card: DataCard
    profile: ObservationProfile
    reality: Annotated[tuple[RealityFact, ...], Field(min_length=35, max_length=35)]
    scope: Literal["blocked", "controlled_functional", "constructed_functional", "synthetic_mechanism"]
    admission_reasons: Annotated[tuple[Identifier, ...], Field(max_length=32)]
    historical_blockers: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject incomplete axes, reordered duplicates and claims inconsistent with the payload."""
        from coleman.corpus.admission_semantics import validate_assessment

        validate_assessment(self)
        return self


class CorpusAdmission(VersionedContract):
    """One auditable view using the existing registry, never a replacement runner.

    Missing source bindings stay visible. Required diversity and observed CI
    model coverage are derived, not caller-selected success flags. All source
    profiles and purposes remain inspectable; no independence or efficacy claim
    follows from a covered cell. Unknown versions reject without migration.
    """

    MAXIMUM_BYTES: ClassVar[int] = 16 * 1024**2

    producer: Fingerprint
    registry: CorpusRegistry
    assessments: Annotated[tuple[SourceAssessment, ...], Field(max_length=64)]
    unassessed_sources: Annotated[tuple[Identifier, ...], Field(max_length=64)]
    covered_regimes: Annotated[tuple[Identifier, ...], Field(max_length=32)]
    uncovered_regimes: Annotated[tuple[Identifier, ...], Field(max_length=32)]
    observed_source_models: Annotated[tuple[Identifier, ...], Field(max_length=32)]
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Recheck complete registry membership and every derived coverage cell on reads."""
        from coleman.corpus.admission_semantics import validate_corpus_admission

        validate_corpus_admission(self)
        return self

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Reject oversized JSON before decoding, then apply the existing strict version/key/number reader."""
        encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
        if len(encoded) > cls.MAXIMUM_BYTES:
            raise ValueError("Corpus admission JSON exceeds its 16-MiB bound")
        return super().from_json(encoded)

    def to_json(self) -> str:
        """Serialize a revalidated report within its explicit byte bound; never truncate scientific evidence."""
        encoded = super().to_json()
        if len(encoded.encode("utf-8")) > self.MAXIMUM_BYTES:
            raise ValueError("Corpus admission JSON exceeds its 16-MiB bound")
        return encoded


class AdmissionFile(Immutable):
    """One fixed report role; byte size and digest are checked separately against actual local files."""

    path: Literal["admission.json", "report.md"]
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)]


class AdmissionCompletion(VersionedContract):
    """Last-published receipt; individual report files never imply completed verification.

    A read validates metadata, not source bytes or execution truth. Explicit
    completion validation must recheck both files and the complete source
    bindings. V1 never migrates or overwrites legacy registry/profile artifacts.
    """

    recipe: Literal["source-admission-report-completion-v1"] = "source-admission-report-completion-v1"
    report_fingerprint: Fingerprint
    files: Annotated[tuple[AdmissionFile, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def _roles(self) -> Self:
        """Require both deterministic roles exactly once and in canonical publication order."""
        if tuple(item.path for item in self.files) != ("admission.json", "report.md"):
            raise ValueError("Admission completion requires the JSON and human report roles exactly once")
        return self
