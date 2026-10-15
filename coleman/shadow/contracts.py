"""Frozen prospective CI protocols and typed source events, without transport or clocks."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.evaluation.protected.temporal import TemporalPlan, Timestamp
from coleman.observations import DecisionSignal, ObservationRevision
from coleman.resources import QUEUE_MS, RUNNER_MS, SETUP_MS, ResourceAmount, ResourceEstimate, ResourceRequest
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, unique_ids

Tick = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Count = Annotated[int, Field(strict=True, ge=1, le=4096)]


class HostClock(Immutable):
    """Explicit host receipt witness; UTC and monotonic nanoseconds are separate clocks.

    A monotonic tick has meaning only within its session. Restarting requires a
    new session identity; clocks from different sessions are never subtracted.
    The host supplies these values. Core code reads no system clock and cannot
    prove clock synchronization or the truth of an upstream timestamp.
    """

    session_id: Annotated[Identifier, Field(max_length=128)]
    tick_ns: Tick
    utc_at: Timestamp


class ShadowSource(Immutable):
    """Authorized source/schema identity and evidence basis, frozen before collection.

    Authorization and clock references are producer attestations, not credentials
    or proof of permission. Owned offline fixtures cannot become prospective
    evidence merely by assigning forward timestamps or repeating seeds.
    """

    source_id: Identifier
    source_version: Identifier
    schema_fingerprint: Fingerprint
    authorization_reference: Identifier
    clock_assumption: Identifier
    license_id: Identifier
    evidence_basis: Literal["functional_offline", "authorized_forward_ci"]


class FrozenCandidate(Immutable):
    """Immutable candidate/model/features/selection/fallback definitions, not executable names.

    Producers bind these fingerprints to reviewed actual configurations. No code
    is imported by identifier or fingerprint. A changed declaration changes the
    experiment fingerprint and requires new untouched evidence. Core selection
    supports only the declared serial prefix v1. Known fallback versions bind
    no-action or estimated-feasible full suite; unknown versions fail closed.
    """

    candidate_id: Identifier
    policy_version: Identifier
    model_fingerprint: Fingerprint
    feature_fingerprint: Fingerprint
    selection_version: Literal["serial-prefix-v1"]
    fallback_version: Literal["no-action-v1", "full-suite-if-estimated-feasible-v1"]
    budget: ResourceAmount
    shared_setup_reserve: ResourceAmount


class ShadowProtocol(VersionedContract):
    """Freeze a bounded prospective protocol without embedding future outcome rows.

    TemporalPlan retains existing chronological split semantics. Forward events
    lie in (validation_end, test_end]; freeze precedes that window. Candidate
    budgets/outcome definitions are identical. Declared independent source groups
    require external justification; distinct seeds/build retries never add groups.
    Research gates are descriptive, not confidence intervals or automatic adoption.
    V1 rejects unknown versions and provides no implicit legacy migration.
    """

    experiment_id: Identifier
    source: ShadowSource
    benchmark_fingerprint: Fingerprint
    frozen_at: Timestamp
    temporal_plan: TemporalPlan
    candidates: Annotated[tuple[FrozenCandidate, ...], Field(min_length=2, max_length=16)]
    baseline_id: Identifier
    independent_groups: Annotated[tuple[Identifier, ...], Field(max_length=128)]
    independence_justification: Identifier | None
    minimum_observation_days: Annotated[int, Field(strict=True, ge=1, le=365)]
    minimum_builds: Annotated[int, Field(strict=True, ge=2, le=4096)]
    minimum_independent_groups: Annotated[int, Field(strict=True, ge=2, le=128)]
    minimum_recall: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    minimum_net_saving_ms: Annotated[int, Field(strict=True, ge=0, le=10**12)]
    actual_build_capacity: ResourceAmount
    maximum_silence_ns: Annotated[int, Field(strict=True, ge=1, le=86400 * 10**9)]
    maximum_source_lag_seconds: Annotated[int, Field(strict=True, ge=0, le=86400)]
    maximum_pending: Annotated[int, Field(strict=True, ge=0, le=128)]
    maximum_events: Count
    maximum_event_bytes: Annotated[int, Field(strict=True, ge=1024, le=262144)]
    maximum_journal_bytes: Annotated[int, Field(strict=True, ge=4096, le=16777216)]
    outcome_unit: Literal["failing_test"] = "failing_test"
    correction_policy: Literal["label_revisions_only_unchanged_occupancy_v1"] = (
        "label_revisions_only_unchanged_occupancy_v1"
    )
    stopping_rule: Literal["stop_recommendations_on_risk_or_source_gap_v1"] = (
        "stop_recommendations_on_risk_or_source_gap_v1"
    )

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        """Reject incomparable candidates, changed units and impossible protected windows."""
        ids = tuple(candidate.candidate_id for candidate in self.candidates)
        unique_ids(ids)
        if self.baseline_id not in ids:
            raise ValueError("Frozen candidates must include the baseline")
        if self.frozen_at > self.temporal_plan.validation_end:
            raise ValueError("Protocol freeze must precede forward observation window")
        span = self.temporal_plan.test_end - self.temporal_plan.validation_end
        if span.total_seconds() < self.minimum_observation_days * 86400:
            raise ValueError("Protected window is shorter than required observation duration")
        budgets = [candidate.budget for candidate in self.candidates]
        if any(budget != budgets[0] or budget.semantics != RUNNER_MS for budget in budgets):
            raise ValueError("Candidates require identical occupied runner millisecond budgets")
        reserves = [candidate.shared_setup_reserve for candidate in self.candidates]
        if any(value != reserves[0] or value.semantics != RUNNER_MS for value in reserves):
            raise ValueError("Candidates require identical explicit shared-setup reservations")
        if self.actual_build_capacity.semantics != RUNNER_MS:
            raise ValueError("Actual CI accounting requires occupied runner milliseconds")
        if self.maximum_event_bytes > self.maximum_journal_bytes:
            raise ValueError("One event cannot exceed the whole journal byte budget")
        groups = unique_ids(self.independent_groups)
        if bool(groups) != (self.independence_justification is not None):
            raise ValueError("Declared independent source groups require an explicit justification")
        object.__setattr__(self, "independent_groups", groups)
        object.__setattr__(self, "candidates", tuple(sorted(self.candidates, key=lambda item: item.candidate_id)))
        return self

    @property
    def logical_id(self) -> str:
        """Bind experiment identity to the complete predeclared protocol, excluding future data."""
        return self.fingerprint


class CandidateDecision(Immutable):
    """Pre-outcome sealed recommendation, with model identity and optional as-of signals.

    ResourceRequest contains no realized duration/outcome. Signals retain their
    existing availability/fit/freeze checks. Their computation still requires the
    reviewed producer/model verifier; metadata alone cannot prove prediction code.
    """

    candidate_id: Identifier
    model_fingerprint: Fingerprint
    request: ResourceRequest
    decision_overhead: ResourceAmount | None
    shared_setup_reserve: ResourceAmount
    overhead_interpretation: Literal["occupied_runner", "elapsed_one_runner_proxy", "unavailable"]
    signals: Annotated[tuple[DecisionSignal, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def _overhead(self) -> Self:
        """Keep unknown inference cost separate from observed zero and elapsed proxies."""
        if (self.decision_overhead is None) != (self.overhead_interpretation == "unavailable"):
            raise ValueError("Decision overhead availability and interpretation disagree")
        if self.decision_overhead is not None and self.decision_overhead.semantics != RUNNER_MS:
            raise ValueError("Decision overhead requires occupied runner milliseconds")
        if self.shared_setup_reserve.semantics != RUNNER_MS:
            raise ValueError("Shared setup reservation requires occupied runner milliseconds")
        return self


class BuildOpened(Immutable):
    """Upstream required CI build plus recommendations sealed before feedback.

    Required tests execute independently of shadow selections. An empty decision
    tuple records observation-only operation when recommendations are stopped.
    Full-build reservation is analytical accounting, never authority to dispatch
    or cancel the upstream CI job. Every declared test identity must be unique.
    """

    kind: Literal["build_opened"] = "build_opened"
    event_id: Identifier
    build_id: Identifier
    logical_build_id: Identifier
    retry_of: Identifier | None = None
    group_id: Identifier
    decided_at: Timestamp
    target_end: Timestamp
    required_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=256)]
    actual_estimate: ResourceEstimate
    decisions: Annotated[tuple[CandidateDecision, ...], Field(max_length=16)]

    @model_validator(mode="after")
    def _ids(self) -> Self:
        """Normalize the required universe without repairing duplicate physical tests."""
        if self.target_end < self.decided_at:
            raise ValueError("Build target interval ends before its decision")
        object.__setattr__(self, "required_ids", unique_ids(self.required_ids))
        ids = tuple(decision.candidate_id for decision in self.decisions)
        unique_ids(ids)
        object.__setattr__(self, "decisions", tuple(sorted(self.decisions, key=lambda item: item.candidate_id)))
        return self


class TestFeedback(Immutable):
    """Provider-normalized outcome revision and separately reported actual occupancy.

    ObservationRevision preserves original upstream timestamps/first receipt. The
    application journals its own admission witness separately. Corrections may
    change the label, never occupied work; changed occupancy is quarantined because
    resource-ledger v1 supports no refunds or retrospective capacity adjustment.
    Null occupancy is unknown, not zero, and prevents complete cost evidence.
    """

    kind: Literal["test_feedback"] = "test_feedback"
    event_id: Identifier
    build_id: Identifier
    test_id: Identifier
    revision: ObservationRevision
    occupancy: ResourceAmount | None

    @model_validator(mode="after")
    def _outcome(self) -> Self:
        """Require a raw failure indicator rather than a reward/ranking/probability."""
        measurement = self.revision.measurement
        if (
            measurement.unit != "1"
            or measurement.meaning != "ci-test-failure-indicator-v1"
            or measurement.direction != "none"
        ):
            raise ValueError("Feedback requires an explicitly interpreted CI failure indicator")
        if measurement.value is not None and measurement.value not in {0.0, 1.0}:
            raise ValueError("Observed CI failure indicators must be exactly zero or one")
        if self.occupancy is not None and self.occupancy.semantics != RUNNER_MS:
            raise ValueError("Feedback occupancy requires occupied runner milliseconds")
        return self


class BuildClosed(Immutable):
    """Explicit upstream closure; missing/unknown results remain incomplete/censored."""

    kind: Literal["build_closed"] = "build_closed"
    event_id: Identifier
    build_id: Identifier
    available_at: Timestamp
    dispatch_occupancy: ResourceAmount | None
    shared_setup_occupancy: ResourceAmount | None
    queue_elapsed: ResourceAmount | None
    setup_elapsed: ResourceAmount | None

    @model_validator(mode="after")
    def _units(self) -> Self:
        """Keep unknown resource charges and elapsed queue/setup delays distinct."""
        for value, semantics in (
            (self.dispatch_occupancy, RUNNER_MS),
            (self.shared_setup_occupancy, RUNNER_MS),
            (self.queue_elapsed, QUEUE_MS),
            (self.setup_elapsed, SETUP_MS),
        ):
            if value is not None and value.semantics != semantics:
                raise ValueError("Incompatible closure occupancy or elapsed-time semantics")
        return self


class SourceStatus(Immutable):
    """Fresh authorized connectivity witness; duplicates do not refresh the local clock."""

    kind: Literal["source_status"] = "source_status"
    event_id: Identifier
    available_at: Timestamp
    connected: Annotated[bool, Field(strict=True)]


ShadowEvent = Annotated[BuildOpened | TestFeedback | BuildClosed | SourceStatus, Field(discriminator="kind")]


class AdmittedEvent(Immutable):
    """Accepted source payload and its first immutable application admission witness.

    Source first receipt remains in the original revision. Application admission
    defines when this process could use validated information. Retrying the same
    source payload retains this witness rather than stamping another arrival.
    """

    event: ShadowEvent
    clock: HostClock
    protocol_fingerprint: Fingerprint
    disposition: Literal["accepted", "quarantined"]
    reason: Identifier | None
