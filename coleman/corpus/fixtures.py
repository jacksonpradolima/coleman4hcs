"""Bounded repository-authored CI fixtures and independently declared reference values."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.reference import RatioReference, TinyCIReference
from coleman.pilot.contracts import SECONDS, WALL_SECONDS, ExecutionEvent, PilotBuild
from coleman.results.domain import Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, decimal_value, ticks


class SerialFixtureCase(Immutable):
    """A full declared ranking with an explicit stop-at-first-nonfit occupancy budget."""

    kind: Literal["serial"] = "serial"
    case_id: Identifier
    ranking: Annotated[tuple[Identifier, ...], Field(max_length=16)]
    budget: Quantity
    expected: TinyCIReference


class ParallelReference(Immutable):
    """Known complete simultaneous-worker accounting, with no serial APFD implication."""

    total_failing_tests: Annotated[int, Field(strict=True, ge=0, le=16)]
    detected_failing_tests: Annotated[int, Field(strict=True, ge=0, le=16)]
    runner_work: Quantity
    execution_elapsed: Quantity
    first_failure_elapsed: Quantity | None
    first_failure_status: Literal["detected", "right_censored"]
    recall: RatioReference


class ParallelFixtureCase(Immutable):
    """Complete simultaneous trace; one distinct worker for each of at most sixteen tests."""

    kind: Literal["parallel"] = "parallel"
    case_id: Identifier
    events: Annotated[tuple[ExecutionEvent, ...], Field(max_length=16)]
    expected: ParallelReference


FixtureCase = Annotated[SerialFixtureCase | ParallelFixtureCase, Field(discriminator="kind")]


class CIReferenceFixture(VersionedContract):
    """One complete tiny synthetic build with hand-calculated correctness cases.

    Integer-second durations up to one million keep the native fixture projection
    exact in Float32/Float64. This restriction belongs to correctness fixtures,
    not to the generic source manifest or exact oracle. No real CI is inferred.
    Parallel cases cover all tests at time zero on distinct workers; partial,
    queued or shared-worker traces require different correctness fixtures.
    """

    fixture_id: Identifier
    history: Literal["constructed"] = "constructed"
    build: PilotBuild
    cases: Annotated[tuple[FixtureCase, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """Require complete, integer, collision-free tiny source observations."""
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)) or len(self.build.tests) > 16:
            raise ValueError("Tiny fixtures require unique cases and at most sixteen tests")
        for test in self.build.tests:
            numerator, denominator = test.duration.value.as_integer_ratio()
            if test.outcome not in {"passed", "failed"} or denominator != 1 or numerator > 1000000:
                raise ValueError("Native fixture requires complete binary outcomes and bounded integer seconds")
        for case in self.cases:
            if isinstance(case, SerialFixtureCase):
                _serial_input(self.build, case)
            else:
                _parallel_input(self.build, case)
        object.__setattr__(self, "cases", tuple(sorted(self.cases, key=lambda case: case.case_id)))
        return self


def _serial_input(build: PilotBuild, case: SerialFixtureCase) -> None:
    """Reject partial rankings, unit mismatches and inexact native fixture budgets."""
    numerator, denominator = case.budget.value.as_integer_ratio()
    if (
        sorted(case.ranking) != sorted(build.baseline_order)
        or case.budget.semantics != SECONDS
        or denominator != 1
        or numerator > 1000000
    ):
        raise ValueError("Serial fixture requires full ranking and bounded integer runner-second budget")


def _parallel_input(build: PilotBuild, case: ParallelFixtureCase) -> None:
    """Limit the independent parallel oracle to full immediate distinct-worker execution."""
    tests = {test.test_id: test for test in build.tests}
    ids = [event.test_id for event in case.events]
    workers = [event.worker_id for event in case.events]
    if sorted(ids) != sorted(tests) or len(set(workers)) != len(workers):
        raise ValueError("Parallel fixture requires every unique test on distinct workers")
    if any(
        event.started.value != 0 or event.completed.value != tests[event.test_id].duration.value
        for event in case.events
    ):
        raise ValueError("Parallel fixture requires immediate complete occupancy")


def parallel_reference(build: PilotBuild) -> ParallelReference:
    """Compute the complete simultaneous-case reference without production event replay.

    Work is sum(duration); elapsed is max(duration); first failure is the minimum
    failing duration. Bounds/outcomes are validated by CIReferenceFixture before
    invoking this reference. No-failure recall is inapplicable, never perfect.
    """
    build = PilotBuild.model_validate(build)
    if len(build.tests) > 16 or any(test.outcome not in {"passed", "failed"} for test in build.tests):
        raise ValueError("Parallel reference requires complete binary outcomes and at most sixteen tests")
    durations = [ticks(test.duration.value) for test in build.tests]
    failures = [ticks(test.duration.value) for test in build.tests if test.outcome == "failed"]
    return ParallelReference(
        total_failing_tests=len(failures),
        detected_failing_tests=len(failures),
        runner_work=Quantity(value=decimal_value(sum(durations)), semantics=SECONDS),
        execution_elapsed=Quantity(value=decimal_value(max(durations)), semantics=WALL_SECONDS),
        first_failure_elapsed=Quantity(value=decimal_value(min(failures)), semantics=WALL_SECONDS)
        if failures
        else None,
        first_failure_status="detected" if failures else "right_censored",
        recall=RatioReference(
            unit="1",
            meaning="failing-test-recall-v1",
            value=Fraction(1) if failures else None,
            status="defined" if failures else "not_applicable",
        ),
    )


def fixture_bytes(fixture: CIReferenceFixture) -> bytes:
    """Return the canonical UTF-8 source format, including one terminating newline."""
    return (CIReferenceFixture.model_validate(fixture).to_json() + "\n").encode("utf-8")
