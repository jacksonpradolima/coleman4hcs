"""Scientific edge cases: eligibility, whole-universe references, physical units and source identity."""

import pytest

from coleman.corpus.baseline_eligibility import baseline_eligibility
from coleman.corpus.baseline_heuristics import HistoricalBaseline, decision_views, unattainable_order
from coleman.corpus.baseline_reference import baseline_reference
from tests.corpus.baseline_support import baseline_plan, change


@pytest.mark.parametrize(
    "quantity,field,value",
    [
        ("duration", "minimum", None),
        ("duration", "direction", "none"),
        ("failure", "direction", "maximize"),
        ("failure", "unit", "probability"),
    ],
)
def test_source_truth_requires_exact_physical_contract(quantity, field, value):
    original = baseline_plan().source.builds[0].tests[0]
    with pytest.raises(ValueError):
        change(original, **{quantity: change(getattr(original, quantity), **{field: value})})


@pytest.mark.parametrize("field,value", [("meaning", "wrong-v1"), ("minimum", None), ("direction", "none")])
def test_prior_cost_must_remain_a_distinct_estimate(field, value):
    original = baseline_plan().source.builds[0].tests[0]
    with pytest.raises(ValueError):
        change(
            original,
            prior_cost=change(
                original.prior_cost, measurement=change(original.prior_cost.measurement, **{field: value})
            ),
        )


def test_duplicate_context_original_order_and_test_universe_reject():
    build = baseline_plan().source.builds[0]
    first = build.tests[0]
    with pytest.raises(ValueError, match="Duplicate baseline context"):
        change(first, context=(first.context[0], first.context[0]))
    with pytest.raises(ValueError, match="original order"):
        change(build, original_order=("a", "b"))
    with pytest.raises(ValueError, match="Duplicate logical"):
        change(build, tests=(first, first))


def test_source_member_identity_and_real_clocks_cannot_be_rewritten():
    source = baseline_plan().source
    with pytest.raises(ValueError, match="Duplicate baseline source member"):
        change(
            source,
            builds=(source.builds[0], change(source.builds[1], member_id=source.builds[0].member_id), source.builds[2]),
        )
    with pytest.raises(ValueError, match="increase strictly"):
        change(
            source,
            builds=(
                source.builds[0],
                change(source.builds[1], decided_at=source.builds[0].decided_at),
                source.builds[2],
            ),
        )
    with pytest.raises(ValueError, match="already available"):
        change(
            source,
            builds=(
                change(
                    source.builds[0],
                    source_metadata=change(
                        source.builds[0].source_metadata,
                        event_at=source.builds[0].decided_at.replace(year=2025),
                        available_at=source.builds[0].decided_at.replace(year=2025),
                    ),
                ),
                *source.builds[1:],
            ),
        )


@pytest.mark.parametrize(
    "value,reason",
    [
        (1e40, "native_binary32_duration_overflow"),
        (0.0, "native_requires_finite_positive_full_suite_work"),
        (1e-100, "native_requires_finite_positive_projected_work"),
    ],
)
def test_unrepresentable_or_empty_work_stays_visible_as_incompatible(value, reason):
    plan = baseline_plan()
    builds = tuple(
        change(build, tests=tuple(change(test, duration=change(test.duration, value=value)) for test in build.tests))
        for build in plan.source.builds
    )
    plan = change(plan, source=change(plan.source, builds=builds))
    assert all(reason in result.reasons for result in baseline_eligibility(plan))


def test_context_clock_unknown_is_incompatible_with_point_in_time_execution():
    plan = baseline_plan()
    builds = tuple(
        change(
            build,
            tests=tuple(
                change(test, context=tuple(change(feature, available_at=None) for feature in test.context))
                for test in build.tests
            ),
        )
        for build in plan.source.builds
    )
    plan = change(plan, source=change(plan.source, builds=builds))
    contextual = next(value for value in baseline_eligibility(plan) if value.method == "linucb_alpha05_v1")
    assert contextual.reasons == ("context_features_unavailable_at_decision",)


def test_reference_requires_observed_truth_and_finite_projected_work():
    build = baseline_plan().source.builds[0]
    first = build.tests[0]
    missing = change(
        build, tests=(change(first, failure=change(first.failure, value=None, status="unknown")), *build.tests[1:])
    )
    with pytest.raises(ValueError, match="complete observed"):
        baseline_reference(missing, ("a", "b", "c"), 7.0)
    with pytest.raises(ValueError, match="complete observed"):
        unattainable_order(missing)
    overflow = change(build, tests=(change(first, duration=change(first.duration, value=1e40)), *build.tests[1:]))
    with pytest.raises(ValueError, match="binary32"):
        baseline_reference(overflow, ("a", "b", "c"), 7.0)
    zeros = change(build, tests=tuple(change(test, duration=change(test.duration, value=0.0)) for test in build.tests))
    with pytest.raises(ValueError, match="positive"):
        baseline_reference(zeros, ("a", "b", "c"), 7.0)


def test_heuristic_rejects_wrong_membership_and_missing_meaningful_order():
    plan = baseline_plan()
    with pytest.raises(ValueError, match="Unsupported"):
        HistoricalBaseline("seeded_random_v1", decision_views(plan), point_in_time=True)
    heuristic = HistoricalBaseline("recent_failure_v1", decision_views(plan), point_in_time=True)
    with pytest.raises(ValueError, match="universe"):
        heuristic.rank(1, ("a", "b"))
    plan = change(
        plan,
        source=change(plan.source, builds=tuple(change(build, original_order=None) for build in plan.source.builds)),
    )
    heuristic = HistoricalBaseline("original_order_v1", decision_views(plan), point_in_time=True)
    with pytest.raises(ValueError, match="unavailable"):
        heuristic.rank(1, ("a", "b", "c"))
