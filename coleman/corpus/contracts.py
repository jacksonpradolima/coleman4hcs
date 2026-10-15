"""Versioned source admission metadata for CI benchmark corpora.

Metadata claims are producer attestations, not proof of scientific independence or
legal permission. Materialized bytes require separate digest/schema validation.
Existing native inputs and missing-value conventions are never silently migrated.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.evaluation.protected.temporal import TemporalPlan, Timestamp
from coleman.results.domain import Fingerprint, Identifier, Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

Count = Annotated[int, Field(strict=True, ge=0)]


class SourceArtifact(Immutable):
    """Immutable source identity; null digest/size means not yet verified, never empty.

    URI is descriptive only; no URL, path or import is executed by these models.
    Source version and citation are required even for blocked registrations.
    """

    uri: Identifier
    version: Identifier
    citation: Identifier
    sha256: Fingerprint | None
    size_bytes: Count | None
    availability: Literal["verified_local", "not_verified", "unavailable"]


class LicenseReview(Immutable):
    """Recorded terms review, separate from availability and code-repository licensing.

    Verified requires the specific data's terms URI/digest/date and license ID.
    A code repository's license must not be inferred to cover third-party data.
    Redistribution explicitly distinguishes bundling, recipe-only and prohibition.
    """

    status: Literal["verified", "pending", "prohibited"]
    license_id: Identifier | None
    terms_uri: Identifier | None
    terms_sha256: Fingerprint | None
    reviewed_at: Timestamp | None
    redistribution: Literal["permitted", "recipe_only", "prohibited", "unverified"]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Reject a verified declaration lacking its auditable specific terms evidence."""
        if self.status == "verified" and (
            self.license_id is None or self.terms_uri is None or self.terms_sha256 is None or self.reviewed_at is None
        ):
            raise ValueError("Verified data license requires identifier, terms URI/digest and review date")
        if self.status == "verified" and self.redistribution == "unverified":
            raise ValueError("Verified terms require an explicit redistribution rule")
        return self


class AdapterReference(Immutable):
    """Declared adapter/version and implementation state; names never trigger imports."""

    name: Identifier
    version: Identifier
    implementation: Literal["implemented", "planned"]


class ObservationIdentity(Immutable):
    """Logical observation/group dimensions, including job/stage/variant where relevant.

    A producer must retain every source dimension needed to avoid collisions.
    Sorting column sets canonicalizes metadata, not row values or temporal order.
    Group dimensions are declarations, not proof of independent observations.
    """

    observation_fields: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]
    group_fields: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]

    @field_validator("observation_fields", "group_fields")
    @classmethod
    def _unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate dimensions and canonicalize identity declarations."""
        if len(values) != len(set(values)):
            raise ValueError("Duplicate observation/group dimension")
        return tuple(sorted(values))


class TimeSemantics(Immutable):
    """Event, outcome and feature availability declarations; unknown fields stay null.

    Constructed timestamps remain explicitly constructed, never observed CI history.
    Availability fields must be verified by an adapter before protected evaluation.
    """

    origin: Literal["observed", "constructed", "unverified"]
    event_field: Identifier | None
    outcome_available_field: Identifier | None
    feature_available_field: Identifier | None
    timezone: Literal["UTC", "unverified"]


class VerdictMapping(Immutable):
    """One explicit binary failing-test mapping; booleans are not integer verdicts."""

    source_value: Identifier
    verdict: Annotated[int, Field(strict=True, ge=0, le=1)]


class OutcomeSemantics(Immutable):
    """Failing-test versus mapped-fault outcome units and explicit missingness policy."""

    unit: Literal["failing_test", "unique_fault", "unverified"]
    mapping: Annotated[tuple[VerdictMapping, ...], Field(max_length=32)]
    unique_fault_mapping: Fingerprint | None
    unknown_policy: Literal["reject", "exclude_with_counts"]
    duration: UnitSemantics | None

    @model_validator(mode="after")
    def _mapping(self) -> Self:
        """Reject ambiguous labels and unsupported unique-fault declarations."""
        names = [item.source_value for item in self.mapping]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate verdict label")
        if self.unit == "unique_fault" and self.unique_fault_mapping is None:
            raise ValueError("Unique-fault outcomes require a genuine mapping fingerprint")
        object.__setattr__(self, "mapping", tuple(sorted(self.mapping, key=lambda item: item.source_value)))
        return self


class CorpusResources(Immutable):
    """Byte/row limits for validation, not measured memory or performance claims."""

    maximum_source_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * 1024**3)]
    maximum_rows: Annotated[int, Field(strict=True, ge=1, le=10000000)]


class EvaluationDeclaration(Immutable):
    """Frozen intended evaluation; correctness fixtures do not imply protected efficacy.

    Ratio budgets explicitly use dimensionless fractions of a complete suite.
    Real confirmatory execution requires the protected evaluation services and a
    justified dependence model; an unimplemented declaration cannot run one.
    """

    purpose: Literal["correctness_fixture", "planned_confirmatory"]
    temporal_plan: TemporalPlan | None
    budget_mode: Literal["fixed_runner_seconds", "suite_ratio"]
    budgets: Annotated[tuple[Quantity, ...], Field(min_length=1, max_length=16)]
    seeds: Annotated[tuple[Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)], ...], Field(max_length=16)]
    metric_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def _protocol(self) -> Self:
        """Reject duplicate measures/seeds and ambiguous budget units or ratios."""
        if len(self.metric_ids) != len(set(self.metric_ids)) or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Duplicate declared metric or seed")
        expected = (
            UnitSemantics(unit="s", meaning="test-runner-occupancy-v1")
            if self.budget_mode == "fixed_runner_seconds"
            else UnitSemantics(unit="1", meaning="full-suite-execution-budget-ratio-v1")
        )
        if any(budget.semantics != expected for budget in self.budgets):
            raise ValueError("Incompatible declared budget units/meaning")
        if self.budget_mode == "suite_ratio" and any(budget.value > 1 for budget in self.budgets):
            raise ValueError("Suite budget ratios cannot exceed one")
        if len({budget.value for budget in self.budgets}) != len(self.budgets):
            raise ValueError("Duplicate execution budget")
        return self


class ExclusionCount(Immutable):
    """One primary exclusion reason; counts must not double-count the same input row."""

    reason: Identifier
    rows: Count


class ConversionSummary(Immutable):
    """Explicit row conservation after conversion; unknown counts remain null.

    Complete conversion requires input = output + excluded rows, unique primary
    reasons and known counts. Metadata alone does not verify the converter's work.
    """

    status: Literal["not_validated", "complete"]
    input_rows: Count | None
    output_rows: Count | None
    exclusions: Annotated[tuple[ExclusionCount, ...], Field(max_length=64)]

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Reject duplicate reasons, missing complete counts and broken row accounting."""
        reasons = [item.reason for item in self.exclusions]
        if len(reasons) != len(set(reasons)):
            raise ValueError("Duplicate primary exclusion reason")
        if self.status == "complete":
            if self.input_rows is None or self.output_rows is None:
                raise ValueError("Complete conversion requires known row counts")
            if self.input_rows != self.output_rows + sum(item.rows for item in self.exclusions):
                raise ValueError("Conversion row conservation failed")
        object.__setattr__(self, "exclusions", tuple(sorted(self.exclusions, key=lambda item: item.reason)))
        return self


class CorpusManifest(VersionedContract):
    """Canonical source, provenance, licensing, identity and protocol declaration.

    Registration never means admission. ``blocking_reasons`` exposes unsupported
    adapters and every unresolved foundational prerequisite. Even eligible metadata
    still requires digest, schema and correctness checks on actual bytes.
    All models reject unknown fields/versions and perform no file/network access.
    """

    benchmark_id: Identifier
    dependence_group: Identifier
    layer: Literal["historical_ci", "long_running_ci", "controlled_real_faults", "seeded_faults", "hcs", "synthetic"]
    source: SourceArtifact
    license: LicenseReview
    adapter: AdapterReference
    identity: ObservationIdentity
    time: TimeSemantics
    outcomes: OutcomeSemantics
    evaluation: EvaluationDeclaration
    conversion: ConversionSummary
    resources: CorpusResources
    limitations: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=64)]

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        """Return deterministic metadata blockers, without downloading or assuming rights."""
        checks = (
            (self.source.sha256 is None, "source_digest_unverified"),
            (self.source.size_bytes is None, "source_size_unverified"),
            (self.source.availability != "verified_local", "source_availability_unverified"),
            (self.license.status != "verified", "data_license_unverified_or_prohibited"),
            (self.license.redistribution in {"prohibited", "unverified"}, "data_use_not_permitted"),
            (self.adapter.implementation != "implemented", "adapter_not_implemented"),
            (self.time.origin == "unverified" or self.time.timezone == "unverified", "time_semantics_unverified"),
            (
                self.time.event_field is None or self.time.outcome_available_field is None,
                "outcome_availability_unverified",
            ),
            (self.time.feature_available_field is None, "feature_availability_unverified"),
            (self.outcomes.unit == "unverified" or not self.outcomes.mapping, "outcome_semantics_unverified"),
            (self.outcomes.duration is None, "duration_units_unverified"),
            (self.conversion.status != "complete", "conversion_not_validated"),
            (
                self.evaluation.purpose == "planned_confirmatory" and self.evaluation.temporal_plan is None,
                "temporal_protocol_unverified",
            ),
        )
        return tuple(sorted(reason for blocked, reason in checks if blocked))

    def require_ready(self) -> None:
        """Reject unresolved metadata before materialization/evaluation; no side effects."""
        if self.blocking_reasons:
            raise ValueError("Corpus source blocked: " + ", ".join(self.blocking_reasons))


class DataCard(VersionedContract):
    """Source profile facts with explicit units/status and visible validity limitations.

    Null profile axes are unverified; empty axes are explicitly inapplicable.
    Null measurements retain unknown/unavailable/not-applicable distinctions.
    Descriptive source facts establish no CI representativeness or independence.
    """

    manifest: CorpusManifest
    history: Literal["observed", "constructed", "unverified"]
    fault_origin: Literal["real", "seeded", "synthetic", "unverified"]
    languages: Annotated[tuple[Identifier, ...], Field(max_length=64)] | None
    execution_models: Annotated[tuple[Identifier, ...], Field(max_length=32)] | None
    measurements: Annotated[tuple[Measurement, ...], Field(min_length=1, max_length=128)]
    covered_regimes: Annotated[tuple[Identifier, ...], Field(max_length=64)]
    missing_regimes: Annotated[tuple[Identifier, ...], Field(max_length=64)]

    @model_validator(mode="after")
    def _facts(self) -> Self:
        """Reject conflicting facts/regimes and canonicalize unordered profile declarations."""
        if self.history == "observed" and (
            self.manifest.time.origin != "observed"
            or self.manifest.layer in {"synthetic", "controlled_real_faults", "seeded_faults"}
        ):
            raise ValueError("Constructed/control histories cannot be labeled observed CI")
        if self.manifest.layer == "synthetic" and self.fault_origin != "synthetic":
            raise ValueError("Synthetic sources cannot be labeled real faults")
        names = [item.name for item in self.measurements]
        if len(names) != len(set(names)) or set(self.covered_regimes) & set(self.missing_regimes):
            raise ValueError("Duplicate profile fact or contradictory diversity cell")
        for field in ("languages", "execution_models", "covered_regimes", "missing_regimes"):
            values = getattr(self, field)
            if values is None:
                continue
            if len(values) != len(set(values)):
                raise ValueError("Duplicate profile declaration")
            object.__setattr__(self, field, tuple(sorted(values)))
        object.__setattr__(self, "measurements", tuple(sorted(self.measurements, key=lambda item: item.name)))
        return self
