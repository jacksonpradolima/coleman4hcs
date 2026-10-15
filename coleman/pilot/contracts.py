"""Frozen offline CI-pilot inputs; no policy, transport or scheduling side effects."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.evaluation.protected.temporal import TemporalPlan, Timestamp
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

SECONDS = UnitSemantics(unit="s", meaning="test-runner-occupancy-v1")
WALL_SECONDS = UnitSemantics(unit="s", meaning="execution-elapsed-excluding-queue-setup-overhead-v1")
OVERHEAD_SECONDS = UnitSemantics(unit="s", meaning="native-prioritization-overhead-rounded-up-ns-v1")
RATE = UnitSemantics(unit="credit/s", meaning="modeled-variable-runner-price-v1")
CREDITS = UnitSemantics(unit="credit", meaning="modeled-variable-runner-cost-excluding-fixed-costs-v1")
Count = Annotated[int, Field(strict=True, ge=0, le=1000000)]


class PilotTest(Immutable):
    """One logical test's occupied runner time and observed outcome.

    Seconds include timeout occupancy. Unknown/timeout are never inferred failures
    or passes; they make complete recall and adoption evidence unavailable. Zero
    duration is valid. Durations represent declared replay inputs, not a promise
    that future executions have the same duration. No unique-fault mapping exists.
    """

    test_id: Identifier
    duration: Quantity
    outcome: Literal["passed", "failed", "unknown", "timeout"]

    @model_validator(mode="after")
    def _units(self) -> Self:
        """Reject mismatched duration meanings, even for zero."""
        if self.duration.semantics != SECONDS:
            raise ValueError("Pilot duration requires test runner occupancy seconds")
        return self


class PilotBuild(Immutable):
    """Full test universe for one build, including budget-excluded failures.

    Base order is the declared comparison, never reconstructed from chosen tests.
    IDs are unique within a build; suite tuple order is immaterial after validation.
    Outcomes must be available by the evaluation cutoff, never by implication at
    decision time. This offline contract does not expose features to a policy.
    """

    build_id: Identifier
    decided_at: Timestamp
    outcomes_available_at: Timestamp
    tests: Annotated[tuple[PilotTest, ...], Field(min_length=1, max_length=10000)]
    baseline_order: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=10000)]

    @model_validator(mode="after")
    def _universe(self) -> Self:
        """Require full unambiguous baseline and chronological outcome availability."""
        ids = [test.test_id for test in self.tests]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(self.baseline_order):
            raise ValueError("Baseline must contain every unique logical test exactly once")
        if self.outcomes_available_at < self.decided_at:
            raise ValueError("Outcome availability precedes decision")
        object.__setattr__(self, "tests", tuple(sorted(self.tests, key=lambda test: test.test_id)))
        return self


class AdoptionThresholds(Immutable):
    """Predeclared per-build safety gates, not statistical significance thresholds.

    Promotion additionally requires observed CI, complete outcomes, positive
    baseline failure evidence and enough distinct builds. Multiple seeds do not
    increase that count. Passing these descriptive gates is not causal inference.
    """

    minimum_builds: Annotated[int, Field(strict=True, ge=2, le=10000)]
    minimum_recall: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    minimum_runner_saving_fraction: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    maximum_overhead: Quantity

    @model_validator(mode="after")
    def _units(self) -> Self:
        """Require an explicit overhead budget in native prioritization seconds."""
        if self.maximum_overhead.semantics != OVERHEAD_SECONDS:
            raise ValueError("Incompatible overhead threshold units")
        return self


class PilotManifest(VersionedContract):
    """Freeze identity, complete outcomes, time splits and adoption rules before runs.

    Source digest identifies immutable source bytes; dataset fingerprint identifies
    this complete interpreted build universe. These claims require producer review
    of license and provenance. The manifest is a replay protocol, not an access
    control boundary. Persist it before execution; never tune using its test rows.
    V1 reads only explicit version 1 and performs no legacy artifact migration.
    """

    pilot_id: Identifier
    source_uri: Identifier
    source_sha256: Fingerprint
    license_id: Identifier
    dataset_fingerprint: Fingerprint
    evidence_basis: Literal["synthetic_replay", "observed_ci"]
    temporal_plan: TemporalPlan
    frozen_at: Timestamp
    candidates: Annotated[tuple[Identifier, ...], Field(min_length=2, max_length=64)]
    baseline_id: Identifier
    budget: Quantity
    variable_price: Quantity
    thresholds: AdoptionThresholds
    builds: Annotated[tuple[PilotBuild, ...], Field(min_length=1, max_length=10000)]
    scheduling: Literal["serial_stop_at_first_nonfit_v1"] = "serial_stop_at_first_nonfit_v1"
    outcome_unit: Literal["failing_test"] = "failing_test"

    @model_validator(mode="after")
    def _protocol(self) -> Self:
        """Bind interpreted data, units, candidates and protected test period."""
        from coleman.results.domain import fingerprint

        if self.budget.semantics != SECONDS or self.variable_price.semantics != RATE:
            raise ValueError("Incompatible budget or variable-price units")
        if len(set(self.candidates)) != len(self.candidates) or self.baseline_id not in self.candidates:
            raise ValueError("Unique candidates must include baseline")
        ids = [build.build_id for build in self.builds]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate build identity")
        ordered = tuple(sorted(self.builds, key=lambda build: (build.decided_at, build.build_id)))
        object.__setattr__(self, "builds", ordered)
        object.__setattr__(self, "candidates", tuple(sorted(self.candidates)))
        if self.dataset_fingerprint != fingerprint([build.model_dump(mode="json") for build in ordered]):
            raise ValueError("Interpreted dataset fingerprint mismatch")
        if self.frozen_at > self.temporal_plan.validation_end:
            raise ValueError("Freeze must precede protected test period")
        for build in self.builds:
            if not self.temporal_plan.validation_end < build.decided_at <= self.temporal_plan.test_end:
                raise ValueError("Pilot builds must lie within protected test period")
            if build.outcomes_available_at > self.temporal_plan.test_end:
                raise ValueError("Outcomes unavailable at evaluation cutoff")
        return self


class ExecutionEvent(Immutable):
    """Completed test occupancy on one worker, relative to the first dispatch.

    Offsets use elapsed seconds. Completion-start must equal source occupancy;
    negative offsets and nonfinite numbers fail. Parallel replay may overlap
    workers, never tasks on the same worker. Setup, queue and scheduling overhead
    are excluded explicitly, not inferred removable from test durations.
    """

    test_id: Identifier
    worker_id: Identifier
    started: Quantity
    completed: Quantity

    @model_validator(mode="after")
    def _interval(self) -> Self:
        """Validate explicit elapsed-time units and nonreversed intervals."""
        if self.started.semantics != WALL_SECONDS or self.completed.semantics != WALL_SECONDS:
            raise ValueError("Event offsets require elapsed seconds")
        if self.completed.value < self.started.value:
            raise ValueError("Event completes before dispatch")
        return self


class PilotRun(Immutable):
    """One completed candidate/build attempt, with full ranking and replayable events.

    Seed is a reproducibility label only. V1 admits one run per candidate/build;
    repeated seeds cannot pretend to be independent builds. Overhead is measured
    separately and never subtracted from occupied runner time.
    """

    candidate_id: Identifier
    build_id: Identifier
    seed: Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
    ranking: Annotated[tuple[Identifier, ...], Field(max_length=10000)]
    events: Annotated[tuple[ExecutionEvent, ...], Field(max_length=10000)]
    prioritization_overhead: Quantity

    @model_validator(mode="after")
    def _overhead(self) -> Self:
        """Reject ambiguous overhead measurements and duplicate execution identities."""
        if self.prioritization_overhead.semantics != OVERHEAD_SECONDS:
            raise ValueError("Incompatible measured overhead units")
        if len({event.test_id for event in self.events}) != len(self.events):
            raise ValueError("Duplicate logical test execution")
        object.__setattr__(
            self, "events", tuple(sorted(self.events, key=lambda event: (event.started.value, event.test_id)))
        )
        return self
