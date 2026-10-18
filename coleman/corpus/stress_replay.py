"""Explicit modeled prefix/list schedules; no native policy or real executor is replaced."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.corpus.reference import RatioReference
from coleman.corpus.stress_contracts import StressCase
from coleman.corpus.stress_world import generate_build
from coleman.pilot.contracts import SECONDS, WALL_SECONDS
from coleman.results.domain import Fingerprint, Identifier, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, decimal_value

Offset = Annotated[int, Field(strict=True, ge=0, le=19200000)]


class StressEvent(Immutable):
    """Constructed nonpreemptive occupancy in integer elapsed seconds, excluding setup/queue."""

    name: Identifier
    worker: Annotated[int, Field(strict=True, ge=0, le=3)]
    started_seconds: Offset
    completed_seconds: Offset

    @model_validator(mode="after")
    def _occupancy(self) -> Self:
        """Require ordered offsets and the generator's <=128-second occupancy bound."""
        if not 0 <= self.completed_seconds - self.started_seconds <= 128:
            raise ValueError("Stress event occupancy outside [0, 128] seconds")
        return self


class StressTrace(VersionedContract):
    """Ranking-dependent synthetic schedule with exact accounting and full-universe denominator.

    The declared budget limits summed occupancy, even with four workers. Selected
    tests are the fitting ranking prefix, stopping at first nonfit. Elapsed is
    modeled earliest-free-worker list scheduling, not measured parallel speedup.
    Unselected failures stay in recall denominator; zero-failure recall is null.
    This DTO alone does not certify correspondence; regenerate with schedule_build.
    """

    case_fingerprint: Fingerprint
    build: Annotated[int, Field(strict=True, ge=0, le=15)]
    ranking_fingerprint: Fingerprint
    workers: Literal[1, 4]
    events: Annotated[tuple[StressEvent, ...], Field(max_length=150000)]
    budget: Quantity
    runner_work: Quantity
    full_suite_work: Quantity
    modeled_elapsed: Quantity
    first_failure_elapsed: Quantity | None
    first_failure_status: Literal["detected", "right_censored"]
    failing_executions: Annotated[int, Field(strict=True, ge=0, le=150000)]
    detected_executions: Annotated[int, Field(strict=True, ge=0, le=150000)]
    recall: RatioReference
    independence: Literal["variant-executions-dependent-not-unique-faults"] = (
        "variant-executions-dependent-not-unique-faults"
    )

    @field_validator("workers", mode="before")
    @classmethod
    def _strict_workers(cls, value):
        """Prevent booleans/floats from impersonating a declared worker count."""
        if type(value) is not int:
            raise ValueError("Stress worker count requires a strict integer")
        return value

    @model_validator(mode="after")
    def _accounting(self) -> Self:
        """Validate units, event conservation and null/zero states without asserting truth correspondence."""
        _check_units(self)
        occupied, elapsed = _event_totals(self.events, self.workers)
        if self.runner_work.value != occupied or self.modeled_elapsed.value != elapsed:
            raise ValueError("Stress event/work/elapsed conservation failed")
        if self.runner_work.value > min(self.budget.value, self.full_suite_work.value):
            raise ValueError("Stress work exceeds its budget/universe")
        _check_quality(self)
        return self


def _check_units(trace: StressTrace) -> None:
    """Keep runner occupancy and modeled elapsed meanings distinct, including zero."""
    if any(q.semantics != SECONDS for q in (trace.budget, trace.runner_work, trace.full_suite_work)):
        raise ValueError("Stress work/budget requires runner occupancy seconds")
    if trace.modeled_elapsed.semantics != WALL_SECONDS or (
        trace.first_failure_elapsed is not None and trace.first_failure_elapsed.semantics != WALL_SECONDS
    ):
        raise ValueError("Stress elapsed requires modeled elapsed seconds")


def _event_totals(events: tuple[StressEvent, ...], workers: int) -> tuple[int, int]:
    """Reject repeated identity/overlap and count exact declared integer occupancy."""
    ids, worker_end, occupied = set(), {}, 0
    if events and min(event.started_seconds for event in events) != 0:
        raise ValueError("Stress execution offsets must begin at zero")
    for event in events:
        if event.name in ids or event.worker >= workers:
            raise ValueError("Duplicate stress event or unsupported worker")
        if event.started_seconds < worker_end.get(event.worker, 0):
            raise ValueError("Overlapping stress worker occupancy")
        ids.add(event.name)
        worker_end[event.worker] = event.completed_seconds
        occupied += event.completed_seconds - event.started_seconds
    return occupied, max(worker_end.values(), default=0)


def _check_quality(trace: StressTrace) -> None:
    """Validate full-universe recall and distinguish zero detection time from null censoring."""
    if trace.detected_executions > min(trace.failing_executions, len(trace.events)):
        raise ValueError("Stress detected count exceeds its universe")
    expected = Fraction(trace.detected_executions, trace.failing_executions) if trace.failing_executions else None
    if trace.recall.meaning != "failing-test-recall-v1" or trace.recall.value != expected:
        raise ValueError("Stress recall differs from full-universe denominator")
    if (trace.first_failure_elapsed is not None) != bool(trace.detected_executions):
        raise ValueError("Stress first failure must be null exactly when no failure was detected")
    if trace.first_failure_status != ("detected" if trace.detected_executions else "right_censored"):
        raise ValueError("Stress first-failure status disagrees with detection")
    if trace.first_failure_elapsed is not None and trace.first_failure_elapsed.value > trace.modeled_elapsed.value:
        raise ValueError("Stress first failure exceeds modeled execution elapsed")


def schedule_build(case: StressCase, build: int, ranking: tuple[str, ...] | None = None) -> StressTrace:
    """Generate exact post-hoc prefix selection and bounded-worker list events.

    Full unique execution ranking is required; default is the declared source
    order, not a learned/oracle candidate. Uses integer costs and dyadic budgets,
    so sums/prefix boundaries also remain exactly representable natively. No
    execution, I/O, real latency assertion or candidate score promotion occurs.
    Invalid identity/duplicates/partial ranking raises ValueError.
    """
    case = StressCase.model_validate(case)
    rows = generate_build(case, build)
    tests = {row.execution_name: row for row in rows}
    if ranking is None:
        ranking = tuple(tests)
    if len(ranking) != len(tests) or set(ranking) != set(tests):
        raise ValueError("Stress ranking must contain every execution exactly once")
    total = sum(row.duration_seconds for row in rows)
    budget = Fraction(total, 16 if case.factors.budget == "severe" else 1)
    workers = [0] * (1 if case.factors.workers == "serial" else 4)
    occupied, detected, failures_at = 0, 0, []
    events = []
    for name in ranking:
        row = tests[name]
        if occupied + row.duration_seconds > budget:
            break
        worker = min(range(len(workers)), key=lambda k: (workers[k], k))
        start = workers[worker]
        end = start + row.duration_seconds
        events.append(StressEvent(name=name, worker=worker, started_seconds=start, completed_seconds=end))
        workers[worker] = end
        occupied += row.duration_seconds
        if row.verdict:
            detected += 1
            failures_at.append(end)
    failures = sum(row.verdict for row in rows)
    return StressTrace(
        schema_version=1,
        case_fingerprint=case.fingerprint,
        build=build,
        ranking_fingerprint=fingerprint(list(ranking)),
        workers=1 if case.factors.workers == "serial" else 4,
        events=tuple(events),
        budget=Quantity(value=decimal_value(budget.numerator * (10**12 // budget.denominator)), semantics=SECONDS),
        runner_work=Quantity(value=Decimal(occupied), semantics=SECONDS),
        full_suite_work=Quantity(value=Decimal(total), semantics=SECONDS),
        modeled_elapsed=Quantity(value=Decimal(max(workers)), semantics=WALL_SECONDS),
        first_failure_elapsed=Quantity(value=Decimal(min(failures_at)), semantics=WALL_SECONDS)
        if failures_at
        else None,
        first_failure_status="detected" if failures_at else "right_censored",
        failing_executions=failures,
        detected_executions=detected,
        recall=RatioReference(
            meaning="failing-test-recall-v1",
            value=Fraction(detected, failures) if failures else None,
            status="defined" if failures else "not_applicable",
        ),
    )
