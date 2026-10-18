"""Exact independent integer accounting, bounded workers and strict trace semantics."""

from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.corpus.reference import tiny_ci_reference
from coleman.corpus.stress_replay import StressEvent, StressTrace, schedule_build
from coleman.corpus.stress_world import generate_build
from coleman.pilot.contracts import RATE, SECONDS, WALL_SECONDS, ExecutionEvent
from coleman.pilot.replay import replay_events
from coleman.selection.contracts import Quantity
from tests.corpus.stress_support import case, pilot


def test_versioned_truth_and_trace_goldens_with_hand_counted_worker_answer():
    from coleman.corpus.stress_contracts import StressCase

    root = Path(__file__).parent / "fixtures"
    value = StressCase.from_json((root / "stress-case-v1.json").read_bytes())
    truth = generate_build(value, 0)
    assert b"".join(r.to_bytes() for r in truth) == (root / "stress-truth-v1.jsonl").read_bytes()
    trace = schedule_build(value, 0)
    assert trace.to_json() == (root / "stress-trace-v1.json").read_text().strip()
    assert [r.duration_seconds for r in truth] == [0, 2, 4, 8, 1, 2, 4, 8]
    assert trace.runner_work.value == 29
    assert trace.modeled_elapsed.value == 11
    # Independently hand-listed earliest-worker dispatch, with deterministic ties.
    assert [(e.worker, e.started_seconds, e.completed_seconds) for e in trace.events] == [
        (0, 0, 0),
        (0, 0, 2),
        (1, 0, 4),
        (2, 0, 8),
        (3, 0, 1),
        (3, 1, 3),
        (0, 2, 6),
        (3, 3, 11),
    ]


@pytest.mark.parametrize("budget", ["severe", "full"])
@pytest.mark.parametrize("workers", ["serial", "bounded"])
def test_list_schedule_against_independent_event_replay_and_exact_tiny_oracle(budget, workers):
    value = case(budget=budget, workers=workers, variants="changing", duration="heavy_tail")
    build = pilot(value, 0)
    trace = schedule_build(value, 0, build.baseline_order)
    events = tuple(
        ExecutionEvent(
            test_id=e.name,
            worker_id=f"w{e.worker}",
            started=Quantity(value=Decimal(e.started_seconds), semantics=WALL_SECONDS),
            completed=Quantity(value=Decimal(e.completed_seconds), semantics=WALL_SECONDS),
        )
        for e in trace.events
    )
    metrics = replay_events(build, events, Quantity(value=Decimal(1), semantics=RATE))
    assert trace.runner_work == metrics.runner_work
    assert trace.modeled_elapsed == metrics.execution_elapsed
    assert trace.first_failure_elapsed == metrics.first_failure_elapsed
    assert trace.detected_executions == metrics.detected_failing_tests
    assert trace.failing_executions == metrics.failing_tests
    if workers == "serial":
        exact = tiny_ci_reference(build, build.baseline_order, trace.budget)
        assert trace.runner_work == exact.runner_work
        assert trace.recall == exact.recall
        assert trace.first_failure_elapsed == exact.first_failure_elapsed
    assert StressTrace.from_json(trace.to_json()) == trace


def test_worker_and_budget_metamorphisms_preserve_selected_denominator_and_separate_elapsed():
    base = dict(variants="changing", duration="constant", failures="rare")
    serial = schedule_build(case(**base, workers="serial", budget="full"), 0)
    parallel = schedule_build(case(**base, workers="bounded", budget="full"), 0)
    severe = schedule_build(case(**base, workers="bounded", budget="severe"), 0)
    assert serial.runner_work.value == parallel.runner_work.value == 7
    assert serial.modeled_elapsed.value == 7
    assert parallel.modeled_elapsed.value == 2
    assert serial.failing_executions == parallel.failing_executions == severe.failing_executions == 1
    assert severe.runner_work.value <= parallel.runner_work.value
    assert severe.detected_executions <= parallel.detected_executions
    assert [e.name for e in parallel.events[: len(severe.events)]] == [e.name for e in severe.events]


def test_zero_null_and_detected_at_zero_are_distinct():
    zero = schedule_build(case(failures="zero", budget="full"), 0)
    assert zero.recall.value is None
    assert zero.recall.status == "not_applicable"
    assert zero.first_failure_elapsed is None
    assert zero.first_failure_status == "right_censored"
    # Seed 4 puts the rare failure on the explicitly zero-duration logical test.
    detected = None
    for seed in range(32):
        candidate = schedule_build(case(seed=seed, failures="rare", budget="full"), 0)
        if candidate.first_failure_elapsed is not None and candidate.first_failure_elapsed.value == 0:
            detected = candidate
            break
    assert detected is not None
    assert detected.first_failure_status == "detected"
    assert detected.recall.value == Fraction(1)
    excluded = schedule_build(case(variants="changing", failures="rare", budget="severe"), 0)
    assert excluded.failing_executions == 1
    assert excluded.recall.value == 0
    assert excluded.first_failure_elapsed is None


def test_partial_duplicate_and_unknown_ranking_fail_before_scheduling():
    value = case()
    full = tuple(r.execution_name for r in generate_build(value, 0))
    for bad in [full[:-1], full[:-1] + full[:1], full[:-1] + ("unknown",)]:
        with pytest.raises(ValueError, match="every execution"):
            schedule_build(value, 0, bad)
    reversed_trace = schedule_build(value, 0, tuple(reversed(full)))
    assert reversed_trace.ranking_fingerprint != schedule_build(value, 0).ranking_fingerprint


@pytest.mark.parametrize(
    "field,semantics",
    [
        ("budget", WALL_SECONDS),
        ("runner_work", WALL_SECONDS),
        ("full_suite_work", WALL_SECONDS),
        ("modeled_elapsed", SECONDS),
        ("first_failure_elapsed", SECONDS),
    ],
)
def test_incompatible_units_reject(field, semantics):
    value = schedule_build(case(budget="full"), 0).model_dump()
    value[field] = {"value": Decimal(0), "semantics": semantics}
    with pytest.raises(ValueError, match="requires|requires|seconds"):
        StressTrace.model_validate(value)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "-1"])
def test_nonfinite_and_negative_occupancy_reject(bad):
    value = schedule_build(case(), 0).model_dump()
    value["budget"]["value"] = bad
    with pytest.raises(ValueError):
        StressTrace.model_validate(value)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"runner_work": {"value": "999", "semantics": SECONDS}}, "conservation"),
        ({"modeled_elapsed": {"value": "999", "semantics": WALL_SECONDS}}, "conservation"),
        ({"budget": {"value": "0", "semantics": SECONDS}}, "exceeds"),
        ({"detected_executions": 150000}, "detected count"),
        ({"first_failure_elapsed": None}, "must be null"),
        ({"first_failure_status": "right_censored"}, "status"),
        ({"first_failure_elapsed": {"value": "999", "semantics": WALL_SECONDS}}, "exceeds"),
    ],
)
def test_accounting_mutations_reject(mutation, error):
    value = schedule_build(case(budget="full"), 0).model_dump()
    with pytest.raises(ValueError, match=error):
        StressTrace.model_validate({**value, **mutation})


def test_duplicate_overlapping_workers_recall_denominator_and_invalid_occupancy_reject():
    value = schedule_build(case(budget="full"), 0).model_dump()
    with pytest.raises(ValueError, match="Duplicate"):
        StressTrace.model_validate({**value, "events": value["events"] + value["events"][:1]})
    events = [dict(e) for e in value["events"]]
    events[-1]["started_seconds"] = 0
    events[-1]["completed_seconds"] = 1
    with pytest.raises(ValueError, match="Overlapping"):
        StressTrace.model_validate({**value, "events": events})
    events[-1]["worker"] = 3
    with pytest.raises(ValueError, match="unsupported worker"):
        StressTrace.model_validate({**value, "events": events})
    value["recall"]["value"] = "0"
    with pytest.raises(ValueError, match="denominator"):
        StressTrace.model_validate(value)
    for start, end in [(2, 1), (0, 129), (-1, 0)]:
        with pytest.raises(ValueError):
            StressEvent(name="t0", worker=0, started_seconds=start, completed_seconds=end)


@pytest.mark.parametrize("workers", [True, 1.0, "1", 2, 5])
def test_worker_count_cannot_coerce_or_escape_declared_topologies(workers):
    value = schedule_build(case(), 0).model_dump()
    with pytest.raises(ValueError):
        StressTrace.model_validate({**value, "workers": workers})


def test_execution_cannot_claim_unmodeled_leading_queue_time():
    value = schedule_build(case(budget="full"), 0).model_dump()
    for event in value["events"]:
        event["started_seconds"] += 1
        event["completed_seconds"] += 1
    with pytest.raises(ValueError, match="begin at zero"):
        StressTrace.model_validate(value)


def test_zero_work_rare_build_retains_real_detection_at_zero_without_an_apfdc_score():
    value = case(churn="additions", builds=8, failures="rare", budget="full", variants="changing")
    trace = schedule_build(value, 0)
    assert len(trace.events) == 1
    assert trace.failing_executions == trace.detected_executions == 1
    assert trace.runner_work.value == trace.full_suite_work.value == trace.modeled_elapsed.value == 0
    assert trace.first_failure_elapsed is not None
    assert trace.first_failure_elapsed.value == 0
    assert trace.first_failure_status == "detected"
    assert trace.recall.value == 1
    build = pilot(value, 0)
    reference = tiny_ci_reference(build, build.baseline_order, trace.budget)
    assert reference.apfdc.value is None
    assert reference.apfdc.status == "not_applicable"
    with pytest.raises(ValueError, match="positive full-suite work"):
        value.require_native()
