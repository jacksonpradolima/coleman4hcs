"""Deterministic constructed channels, churn identities, scale and mechanism metamorphisms."""

import random
from dataclasses import asdict

import pytest

from coleman.corpus.stress_feedback import decision_inputs
from coleman.corpus.stress_world import generate_build
from tests.corpus.stress_support import case


@pytest.mark.parametrize("build", [-1, 4, True, 0.0])
def test_build_bounds_before_allocation(build):
    value = case()
    with pytest.raises(ValueError, match="index"):
        generate_build(value, build)


def test_build_order_and_ambient_randomness_do_not_change_truth():
    value = case(duration="changing", context="noisy")
    forward = [generate_build(value, i) for i in range(4)]
    random.seed(987)
    assert list(reversed([generate_build(value, i) for i in reversed(range(4))])) == forward
    assert all(r.duration_seconds >= 0 for rows in forward for r in rows)
    assert {r.context_score for rows in forward for r in rows} == {-1, 0, 1}
    other = [generate_build(case(seed=29, duration="changing", context="noisy"), i) for i in range(4)]
    assert forward != other


@pytest.mark.parametrize("regime", ["independent", "history", "environment"])
def test_zero_failure_regime_overrides_flakiness_and_variants_are_dependent(regime):
    value = case(failures="zero", flakiness=regime)
    for build in range(4):
        rows = generate_build(value, build)
        assert {r.verdict for r in rows} == {0}
        assert len({r.test_id for r in rows}) * 3 == len(rows)
        for logical in {r.test_id for r in rows}:
            assert len({(r.verdict, r.duration_seconds, r.context_score) for r in rows if r.test_id == logical}) == 1


def test_rare_moderate_and_bursty_are_distinct_declared_regimes():
    rare = case(failures="rare", variants="changing")
    assert all(sum(r.verdict for r in generate_build(rare, b)) == 1 for b in range(4))
    burst = case(failures="bursty", variants="changing")
    assert [sum(r.verdict for r in generate_build(burst, b)) for b in range(4)] == [0, 0, 4, 0]
    assert 0 < sum(r.verdict for r in generate_build(case(variants="changing"), 0)) < 8


@pytest.mark.parametrize("duration", ["constant", "heavy_tail", "failure_correlated", "changing"])
def test_duration_mechanisms_use_exact_bounded_seconds(duration):
    value = case(size="medium", duration=duration)
    rows = generate_build(value, 0)
    assert {r.duration_seconds for r in rows if r.test_id == "t00000"} == {0}
    assert all(type(r.duration_seconds) is int and 0 <= r.duration_seconds <= 128 for r in rows)
    if duration == "heavy_tail":
        assert {1, 2, 4, 8, 16, 32, 64, 128}.issuperset(r.duration_seconds for r in rows if r.duration_seconds)
        assert max(r.duration_seconds for r in rows) >= 32
    if duration == "failure_correlated":
        assert all(r.duration_seconds == (16 if r.verdict else 1) for r in rows if r.test_id != "t00000")
    if duration == "changing":
        assert [r.duration_seconds for r in rows] != [r.duration_seconds for r in generate_build(value, 1)]


@pytest.mark.parametrize("churn", ["stable", "additions", "removals", "renames", "collisions"])
def test_churn_and_display_collisions_preserve_logical_identity(churn):
    value = case(churn=churn, variants="changing")
    builds = [generate_build(value, b) for b in range(4)]
    ids = [{r.test_id for r in rows} for rows in builds]
    if churn == "additions":
        assert ids[0] < ids[1] < ids[2] < ids[3]
    elif churn == "removals":
        assert ids[0] > ids[1] > ids[2] > ids[3]
    else:
        assert all(v == ids[0] for v in ids)
    if churn == "renames":
        assert {r.display_name for r in builds[0]}.isdisjoint(r.display_name for r in builds[-1])
    if churn == "collisions":
        assert len({r.display_name for r in builds[0]}) == 4
        assert len({r.execution_name for r in builds[0]}) == 8


@pytest.mark.parametrize("drift", ["sudden", "gradual", "recurring", "contextual"])
def test_latent_risk_drift_has_declared_temporal_structure(drift):
    value = case(drift=drift, builds=8)
    risks = [[r.latent_risk for r in generate_build(value, b)] for b in range(8)]
    assert len({tuple(v) for v in risks}) > 1
    if drift == "sudden":
        assert risks[0] == risks[3]
        assert risks[3] != risks[4]
    if drift == "recurring":
        assert risks[0] == risks[4]


def test_visible_context_signed_missing_adversarial_and_outcome_leakage_mutation(monkeypatch):
    informative = decision_inputs(case(), 0)
    adverse = decision_inputs(case(context="adversarial"), 0)
    assert all(a.context_score == -b.context_score for a, b in zip(informative, adverse, strict=True))
    assert all(r.context_score is None for r in decision_inputs(case(context="missing"), 0))
    assert all(set(r.model_dump()) == {"name", "display_name", "context_score"} for r in informative)
    # Deliberately mutate *all* hidden current outcomes. The view must remain byte-identical.
    import coleman.corpus.stress_world as world

    before = generate_build(case(), 0)
    with monkeypatch.context() as mutation:
        mutation.setattr(world, "_verdict", lambda *_: 1)
        assert decision_inputs(case(), 0) == informative
        assert generate_build(case(), 0) != before


def test_large_build_has_50000_distinct_logical_tests_and_sparse_variant_membership():
    rows = generate_build(case(size="large", variants="sparse", duration="heavy_tail"), 0)
    assert len(rows) == 50000
    assert len({r.test_id for r in rows}) == 50000
    assert len({r.execution_name for r in rows}) == 50000
    assert {r.variant for r in rows} == {"v0", "v1", "v2"}
    assert max(r.duration_seconds for r in rows) == 128
    assert asdict(rows[0])["duration_seconds"] == 0
    assert rows[0].to_bytes().endswith(b"\n")


def test_history_flakiness_has_hand_counted_nonzero_memory_effect():
    value = case(flakiness="history", variants="changing")
    failures = [sorted(r.test_id for r in generate_build(value, b) if r.verdict) for b in range(4)]
    assert failures == [["t00000", "t00004"], [], ["t00001", "t00005"], ["t00001", "t00005"]]


def test_environment_flips_are_shared_with_both_states_and_independent_flips_are_not_shared():
    environment = case(size="medium", flakiness="environment", variants="changing", builds=16)
    shared = []
    for build in range(16):
        flips = {r.verdict ^ r.latent_risk for r in generate_build(environment, build)}
        assert len(flips) == 1
        shared.extend(flips)
    assert shared == [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    independent = case(size="medium", flakiness="independent", variants="changing")
    assert {r.verdict ^ r.latent_risk for r in generate_build(independent, 0)} == {0, 1}


def test_feedback_budget_worker_and_context_corruption_do_not_change_hidden_world():
    def truth(value):
        return [(r.test_id, r.variant, r.duration_seconds, r.verdict, r.latent_risk) for r in generate_build(value, 0)]

    original = truth(case())
    for settings in [{"budget": "full"}, {"feedback": "corrected"}, {"workers": "bounded"}, {"context": "adversarial"}]:
        assert truth(case(**settings)) == original


@pytest.mark.parametrize("churn", ["stable", "additions", "removals", "renames", "collisions"])
@pytest.mark.parametrize("builds", [4, 16])
def test_rare_failures_remain_one_active_logical_test_after_churn(churn, builds):
    value = case(failures="rare", variants="stable", churn=churn, builds=builds)
    for build in range(builds):
        rows = generate_build(value, build)
        failing = {row.test_id for row in rows if row.verdict}
        assert len(failing) == 1
        assert sum(row.verdict for row in rows) == 3  # three dependent variants, not three faults
        assert failing.issubset(row.test_id for row in rows)


def test_generated_rare_churn_covering_rows_preserve_nonzero_prevalence():
    from coleman.corpus.stress_contracts import StressCase
    from coleman.corpus.stress_design import covering_design

    checked = []
    for index, factors in enumerate(covering_design().rows):
        if factors.failures != "rare" or factors.churn not in {"additions", "removals"}:
            continue
        value = StressCase(schema_version=1, factors=factors)
        counts = [
            len({row.test_id for row in generate_build(value, build) if row.verdict}) for build in range(value.builds)
        ]
        assert counts == [1] * value.builds
        checked.append(index)
    assert checked == [2, 13, 23]
