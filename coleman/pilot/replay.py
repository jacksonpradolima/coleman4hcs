"""Independent CI event accounting; never imports production metrics or schedulers."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

from coleman.pilot.contracts import (
    CREDITS,
    SECONDS,
    WALL_SECONDS,
    Count,
    ExecutionEvent,
    PilotBuild,
    PilotManifest,
    PilotRun,
)
from coleman.selection.contracts import Immutable, Quantity, decimal_value, ticks


def _amount(value: int, semantics, *, places: int = 12) -> Quantity:
    """Represent exact ticks in declared units without Decimal-context rounding."""
    return Quantity(value=decimal_value(value, places=places), semantics=semantics)


class ReplayMetrics(Immutable):
    """Exact work/cost and descriptive failing-test quality from complete source rows.

    Null recall denotes unknown/timeout outcomes or no known failing tests. Null
    first-failure time means no detected failure, including zero-failure builds;
    genuine detection at time zero remains zero. Modeled currency excludes fixed,
    queue/setup and overhead costs. No elapsed savings are inferred from work.
    """

    selected_ids: tuple[str, ...]
    failing_tests: Count
    detected_failing_tests: Count
    unresolved_tests: Count
    failing_test_recall: float | None
    first_failure_elapsed: Quantity | None
    first_failure_status: Literal["detected", "right_censored"]
    runner_work: Quantity
    full_suite_runner_work: Quantity
    execution_elapsed: Quantity
    modeled_variable_cost: Quantity


def serial_events(build: PilotBuild, ranking: tuple[str, ...], budget: Quantity) -> tuple[ExecutionEvent, ...]:
    """Construct serial stop-at-first-nonfit replay events from declared durations.

    This pure fixture helper executes no tests and changes no native behavior.
    Full ranking, explicit runner-second budget and valid IDs are mandatory.
    At the first nonfitting test all later tests remain excluded, even if shorter.
    """
    build = PilotBuild.model_validate(build)
    budget = Quantity.model_validate(budget)
    if budget.semantics != SECONDS or sorted(ranking) != sorted(build.baseline_order):
        raise ValueError("Expected compatible budget and full unique ranking")
    tests = {test.test_id: test for test in build.tests}
    elapsed = 0
    events = []
    for test_id in ranking:
        duration = ticks(tests[test_id].duration.value)
        if elapsed + duration > ticks(budget.value):
            break
        events.append(
            ExecutionEvent(
                test_id=test_id,
                worker_id="serial",
                started=_amount(elapsed, WALL_SECONDS),
                completed=_amount(elapsed + duration, WALL_SECONDS),
            )
        )
        elapsed += duration
    return tuple(events)


def _verified_events(build: PilotBuild, events: tuple[ExecutionEvent, ...]) -> tuple[ExecutionEvent, ...]:
    """Validate event identity and the common first-dispatch origin before accounting."""
    verified = tuple(ExecutionEvent.model_validate(event) for event in events)
    if verified and min(event.started.value for event in verified) != 0:
        raise ValueError("Event offsets must start at the first dispatch (zero)")
    ids = [event.test_id for event in verified]
    if len(set(ids)) != len(ids) or not set(ids).issubset(test.test_id for test in build.tests):
        raise ValueError("Duplicate or unknown execution identity")
    return verified


def replay_events(build: PilotBuild, events: tuple[ExecutionEvent, ...], price: Quantity) -> ReplayMetrics:
    """Verify explicit serial/parallel events against complete immutable source rows.

    Reject duplicates, unknown tests, occupancy mismatches and worker overlaps.
    Does not infer a schedule from a ranking; caller must verify that separately.
    All excluded known failures remain in recall denominator. Unknown outcomes
    make recall unavailable, never spuriously perfect. No file or RNG side effects.
    """
    from coleman.pilot.contracts import RATE

    build = PilotBuild.model_validate(build)
    price = Quantity.model_validate(price)
    if price.semantics != RATE:
        raise ValueError("Incompatible variable price")
    tests = {test.test_id: test for test in build.tests}
    verified = _verified_events(build, events)
    worker_end: dict[str, int] = {}
    work = 0
    failures_at = []
    for event in sorted(verified, key=lambda item: (item.started.value, item.completed.value, item.test_id)):
        start, end = ticks(event.started.value), ticks(event.completed.value)
        test = tests[event.test_id]
        if end - start != ticks(test.duration.value) or start < worker_end.get(event.worker_id, 0):
            raise ValueError("Occupancy mismatch or overlapping worker events")
        worker_end[event.worker_id] = end
        work += end - start
        if test.outcome == "failed":
            failures_at.append(end)
    failing = sum(test.outcome == "failed" for test in tests.values())
    unresolved = sum(test.outcome in {"timeout", "unknown"} for test in tests.values())
    # Price and work each have 12 places. Quantity's authoritative v1 12-place
    # contract deliberately rejects an unrepresentable product instead of rounding.
    cost = decimal_value(work * ticks(price.value), places=24)
    return ReplayMetrics(
        selected_ids=tuple(
            event.test_id for event in sorted(verified, key=lambda item: (item.started.value, item.test_id))
        ),
        failing_tests=failing,
        detected_failing_tests=len(failures_at),
        unresolved_tests=unresolved,
        failing_test_recall=None if unresolved or not failing else float(Fraction(len(failures_at), failing)),
        first_failure_elapsed=_amount(min(failures_at), WALL_SECONDS) if failures_at else None,
        first_failure_status="detected" if failures_at else "right_censored",
        runner_work=_amount(work, SECONDS),
        full_suite_runner_work=_amount(sum(ticks(test.duration.value) for test in tests.values()), SECONDS),
        execution_elapsed=_amount(max(worker_end.values(), default=0), WALL_SECONDS),
        modeled_variable_cost=Quantity(value=cost, semantics=CREDITS),
    )


def verify_run(manifest: PilotManifest, run: PilotRun) -> ReplayMetrics:
    """Require native serial prefix semantics before independent event accounting.

    Unknown candidates/builds, altered baseline order or incomplete/different event
    schedules raise ValueError. Parallel events are supported by replay_events,
    but cannot silently claim this serial pilot protocol. No native metrics used.
    """
    manifest = PilotManifest.model_validate(manifest)
    run = PilotRun.model_validate(run)
    builds = {build.build_id: build for build in manifest.builds}
    if run.candidate_id not in manifest.candidates or run.build_id not in builds:
        raise ValueError("Unknown candidate or build")
    build = builds[run.build_id]
    if run.candidate_id == manifest.baseline_id and run.ranking != build.baseline_order:
        raise ValueError("Baseline order differs from frozen protocol")
    expected = serial_events(build, run.ranking, manifest.budget)
    # Native collectors may call their single worker by another stable label.
    actual = tuple((event.test_id, event.started, event.completed) for event in run.events)
    wanted = tuple(
        (event.test_id, event.started, event.completed)
        for event in sorted(expected, key=lambda e: (e.started.value, e.test_id))
    )
    if actual != wanted or len({event.worker_id for event in run.events}) > 1:
        raise ValueError("Events differ from frozen serial prefix schedule")
    return replay_events(build, run.events, manifest.variable_price)
