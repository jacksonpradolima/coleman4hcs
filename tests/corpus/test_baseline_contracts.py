"""Strict candidate/source contracts and independent simple-baseline references."""

import hashlib
import json
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

from coleman.corpus.baseline_contracts import BaselinePlan
from coleman.corpus.baseline_eligibility import baseline_eligibility
from coleman.corpus.baseline_heuristics import HistoricalBaseline, ObservedFeedback, decision_views, unattainable_order
from coleman.spec.producer_identity import installed_producer_identity, package_pins
from tests.corpus.baseline_support import baseline_plan, change


def test_versioned_baseline_golden_and_roundtrip():
    """A versioned complete plan is stable across row/candidate/seed/budget permutations."""
    plan = baseline_plan()
    golden = Path(__file__).parent / "fixtures/baseline-plan-v1.json"
    assert plan.to_json() + "\n" == golden.read_text()
    assert BaselinePlan.from_json(plan.to_json()) == plan
    reversed_builds = tuple(change(b, tests=tuple(reversed(b.tests))) for b in reversed(plan.source.builds))
    other = change(
        plan,
        source=change(plan.source, builds=reversed_builds),
        methods=tuple(reversed(plan.methods)),
        budgets=tuple(reversed(plan.budgets)),
        seeds=tuple(reversed(plan.seeds)),
    )
    assert other.to_json() == plan.to_json()
    assert other.fingerprint == plan.fingerprint
    assert change(plan, seeds=(12, 42)).fingerprint != plan.fingerprint
    assert not any(
        hasattr(test, "failure") or hasattr(test, "duration") for b in decision_views(plan) for test in b.tests
    )


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_missing_old_unknown_or_coerced_baseline_version_rejects(version):
    payload = json.loads(baseline_plan().to_json())
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        BaselinePlan.from_json(json.dumps(payload))


@pytest.mark.parametrize("field", ["methods", "seeds", "budgets"])
def test_duplicate_repetition_dimensions_reject(field):
    plan = baseline_plan()
    with pytest.raises(ValueError, match="Duplicate"):
        change(plan, **{field: (getattr(plan, field)[0], getattr(plan, field)[0])})


def test_baseline_clock_and_unit_semantics_fail_closed():
    plan = baseline_plan()
    source, build = plan.source, plan.source.builds[0]
    test = build.tests[0]
    with pytest.raises(ValueError, match="occupancy seconds"):
        change(test, duration=change(test.duration, unit="ms"))
    with pytest.raises(ValueError, match="binary"):
        change(test, failure=change(test.failure, value=0.5))
    with pytest.raises(ValueError, match="future"):
        change(build, decided_at=test.context[0].available_at - timedelta(seconds=1))
    with pytest.raises(ValueError, match="before its decision step"):
        change(
            build, tests=(change(test, context=(change(test.context[0], available_before_step=1),)), *build.tests[1:])
        )
    with pytest.raises(ValueError, match="original decision"):
        change(source, builds=(change(build, source_metadata=None), *source.builds[1:]))
    with pytest.raises(ValueError, match="contiguous"):
        change(source, builds=(source.builds[1],))
    with pytest.raises(ValueError, match="sibling source groups"):
        change(
            source,
            builds=(build, change(source.builds[1], source_metadata=build.source_metadata), source.builds[2]),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_physical_costs_reject_nonfinite_or_negative(value):
    with pytest.raises(ValueError):
        change(baseline_plan().source.builds[0].tests[0].duration, value=value)


def test_eligibility_keeps_missing_context_order_and_external_methods_visible():
    plan = baseline_plan()
    complete = {item.method: item for item in baseline_eligibility(plan)}
    assert len(complete) == 13
    assert all(item.compatible for name, item in complete.items() if name != "external_dante_prioritization")
    assert complete["external_dante_prioritization"].reasons
    builds = tuple(
        change(b, original_order=None, tests=tuple(change(t, context=()) for t in b.tests)) for b in plan.source.builds
    )
    changed = change(plan, source=change(plan.source, builds=builds))
    result = {item.method: item for item in baseline_eligibility(changed)}
    assert result["original_order_v1"].reasons == ("meaningful_original_order_missing",)
    assert result["linucb_alpha05_v1"].reasons == ("context_features_missing",)
    assert result["seeded_random_v1"].compatible


def test_zero_truth_is_valid_but_unavailable_truth_and_known_late_labels_block_native():
    plan = baseline_plan()
    first = plan.source.builds[0]
    zero = change(first.tests[0], duration=change(first.tests[0].duration, value=0.0))
    source = change(plan.source, builds=(change(first, tests=(zero, *first.tests[1:])), *plan.source.builds[1:]))
    assert next(
        x for x in baseline_eligibility(change(plan, source=source)) if x.method == "seeded_random_v1"
    ).compatible
    missing = change(zero, failure=change(zero.failure, status="unavailable", value=None))
    source = change(plan.source, builds=(change(first, tests=(missing, *first.tests[1:])), *plan.source.builds[1:]))
    assert all(not item.compatible for item in baseline_eligibility(change(plan, source=source)))
    delayed = change(first.source_metadata, available_at=plan.source.builds[1].decided_at + timedelta(hours=1))
    source = change(plan.source, builds=(change(first, source_metadata=delayed), *plan.source.builds[1:]))
    assert all(
        "native_immediate_feedback_contradicts_known_late_labels" in item.reasons
        for item in baseline_eligibility(change(plan, source=source))
    )


def test_independent_heuristic_answers_and_real_zero_cost():
    plan = baseline_plan()
    views = decision_views(plan)
    original = HistoricalBaseline("original_order_v1", views, point_in_time=True)
    assert original.rank(1, ("b", "a", "c")) == ["c", "a", "b"]
    shortest = HistoricalBaseline("shortest_prior_v1", views, point_in_time=True)
    assert shortest.rank(1, ("a", "b", "c")) == ["b", "c", "a"]  # Prior costs 1/2/4, not current 2/4/1.
    for method in ("recent_failure_v1", "failure_frequency_v1", "recency_duration_v1"):
        policy = HistoricalBaseline(method, views, point_in_time=True)
        assert policy.rank(1, ("c", "b", "a")) == ["a", "b", "c"]
        assert policy.fallback_count == 3
        policy.accept(1, (ObservedFeedback("c", True, 0.0), ObservedFeedback("a", False, 2.0)))
        assert policy.rank(2, ("a", "b", "c")) == ["c", "a", "b"]
        assert policy.fallback_count == 1  # Unobserved b is distinct from observed passing a.
        assert policy.history["c"].duration_sum == 0
        assert policy.history["a"].duration_sum == Fraction(2)


def test_current_truth_mutation_cannot_change_attainable_views_or_rankings():
    plan = baseline_plan()
    builds = tuple(
        change(
            b,
            tests=tuple(
                change(t, failure=change(t.failure, value=1.0), duration=change(t.duration, value=99.0))
                for t in b.tests
            ),
        )
        for b in plan.source.builds
    )
    other = change(plan, source=change(plan.source, builds=builds))
    assert plan.fingerprint != other.fingerprint
    assert decision_views(plan) == decision_views(other)
    for method in ("shortest_prior_v1", "recent_failure_v1", "failure_frequency_v1", "recency_duration_v1"):
        left = HistoricalBaseline(method, decision_views(plan), point_in_time=True)
        right = HistoricalBaseline(method, decision_views(other), point_in_time=True)
        assert left.rank(1, ("a", "b", "c")) == right.rank(1, ("a", "b", "c"))
    assert unattainable_order(plan.source.builds[1]) == ["b", "a", "c"]


@pytest.mark.parametrize(
    "feedback",
    [
        (ObservedFeedback("a", False, -1.0),),
        (ObservedFeedback("a", False, float("inf")),),
        (ObservedFeedback("a", False, True),),
        (ObservedFeedback("a", cast(bool, 1), 1.0),),
        (ObservedFeedback("unknown", True, 1.0),),
        (ObservedFeedback("a", False, 1.0), ObservedFeedback("a", True, 1.0)),
    ],
)
def test_invalid_feedback_rejects_before_mutation(feedback):
    policy = HistoricalBaseline("failure_frequency_v1", decision_views(baseline_plan()), point_in_time=True)
    with pytest.raises(ValueError):
        policy.accept(1, feedback)
    assert policy.history == {} and policy.committed_step == 0


def test_empty_feedback_advances_without_inventing_passes_and_duplicates_reject():
    policy = HistoricalBaseline("shortest_prior_v1", decision_views(baseline_plan()), point_in_time=True)
    policy.accept(1, ())
    assert policy.committed_step == 1 and not policy.history
    with pytest.raises(ValueError, match="out-of-order"):
        policy.accept(1, ())
    with pytest.raises(ValueError, match="next unobserved"):
        policy.rank(1, ("a", "b", "c"))
    with pytest.raises(ValueError, match="universe"):
        policy.rank(2, ("a", "b"))


def test_shared_core_producer_identity_preserves_legacy_import_and_resource_algorithm(tmp_path):
    """The baseline application reuses the existing identity without depending on a transport module."""
    from coleman.transports import producer_identity as legacy

    assert legacy.installed_producer_identity is installed_producer_identity
    assert legacy.package_pins is package_pins
    (tmp_path / "component.py").write_bytes(b"owned implementation\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "resource.dat").write_bytes(b"owned resource")
    for name in (".hidden", "compiled.pyc", "compiled.pyo"):
        (tmp_path / name).write_bytes(b"excluded")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_bytes(b"excluded subtree")
    expected = {
        "test/component.py": "sha256:" + hashlib.sha256(b"owned implementation\n").hexdigest(),
        "test/data/resource.dat": "sha256:" + hashlib.sha256(b"owned resource").hexdigest(),
    }
    assert package_pins(tmp_path, "test") == expected


def test_signed_context_retains_negative_values_without_becoming_a_probability():
    original = baseline_plan().source.builds[0].tests[0]
    feature = original.context[0]
    signed = change(feature, measurement=change(feature.measurement, value=-2.0, minimum=None))
    test = change(original, context=(signed,))
    assert test.context[0].measurement.value == -2.0
    assert test.context[0].measurement.unit == "1"
    assert test.context[0].measurement.direction == "none"
