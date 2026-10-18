"""Actual native learners and simple baselines, including checkpointed policy RNG continuation."""

from typing import cast

import numpy as np
import pytest

from coleman.application.baseline_agents import BaselineAgent
from coleman.bandit import EvaluationMetricBandit
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.evaluation import NAPFDVerdictMetric
from coleman.policy import base as policy_base
from tests.corpus.baseline_support import baseline_plan


def bandit_for(build, available_time=7.0):
    """Create the actual native bandit and metric from the owned source, without policy/metric mocks."""
    metric = NAPFDVerdictMetric()
    metric.update_available_time(available_time)
    rows = [
        {
            "Name": test.test_id,
            "Duration": test.duration.value,
            "Verdict": int(cast(float, test.failure.value)),
            "CalcPrio": 0,
            "LastRun": "",
        }
        for test in build.tests
    ]
    return EvaluationMetricBandit(rows, metric)


@pytest.mark.parametrize(
    "method",
    [
        "seeded_random_v1",
        "original_order_v1",
        "shortest_prior_v1",
        "recent_failure_v1",
        "failure_frequency_v1",
        "recency_duration_v1",
        "ucb_c1_v1",
        "thompson_beta11_v1",
        "discounted_ucb_095_c1_v1",
        "linucb_alpha05_v1",
        "portfolio_random_ucb_v1",
        "unattainable_failure_first_v1",
    ],
)
def test_real_native_candidate_complete_order_and_observed_phase_costs(method):
    plan = baseline_plan()
    agent = BaselineAgent(plan, method, 42)
    outer_rng = policy_base._rng
    for build in plan.source.builds:
        bandit = bandit_for(build)
        agent.update_bandit(bandit)
        ranking = agent.choose()
        assert sorted(ranking) == ["a", "b", "c"]
        if build.step == 1 and method == "original_order_v1":
            assert ranking == ["c", "a", "b"]
        if build.step == 1 and method == "shortest_prior_v1":
            assert ranking == ["b", "c", "a"]
        agent.observe(bandit.pull(ranking))
        assert policy_base._rng is outer_rng
        assert agent.timing.preparation_seconds >= 0
        assert agent.timing.inference_seconds >= 0
        assert agent.timing.learning_seconds >= 0
    assert agent.native.t == 3
    assert str(agent) == method


@pytest.mark.parametrize(
    "method", ["seeded_random_v1", "thompson_beta11_v1", "portfolio_random_ucb_v1", "recent_failure_v1"]
)
@pytest.mark.parametrize("seed", [11, 42])
def test_real_checkpoint_continuation_preserves_sampling_and_observed_history(tmp_path, method, seed):
    plan = baseline_plan()
    agent = BaselineAgent(plan, method, seed)
    for build in plan.source.builds[:2]:
        bandit = bandit_for(build)
        agent.update_bandit(bandit)
        ranking = agent.choose()
        agent.observe(bandit.pull(ranking))
    store = LocalCheckpointStore(str(tmp_path))
    store.save(CheckpointPayload(run_id="owned", experiment=1, step=2, agents=[agent], bandit=bandit))
    final = bandit_for(plan.source.builds[2])
    agent.update_bandit(final)
    uninterrupted = agent.choose()
    payload = store.load("owned", 1)
    assert payload is not None and payload.step == 2
    restored = payload.agents[0]
    previous = policy_base._rng
    try:
        policy_base._rng = np.random.default_rng(987654321)  # A resumed worker has a new process-global RNG.
        restored.update_bandit(bandit_for(plan.source.builds[2]))
        assert restored.choose() == uninterrupted
    finally:
        policy_base._rng = previous


def test_cost_history_never_observes_unselected_test_as_a_pass():
    plan = baseline_plan()
    agent = BaselineAgent(plan, "shortest_prior_v1", 42)
    native = bandit_for(plan.source.builds[0], available_time=2.5)
    agent.update_bandit(native)
    ranking = agent.choose()
    agent.observe(native.pull(ranking))
    assert native.evaluation_metric.scheduled_testcases == ["b"]
    assert set(agent.native.policy.historical.history) == {"b"}
    history = agent.native.policy.historical.history["b"]
    assert history.observations == 1 and history.failures == 0 and history.duration_sum == 2


def test_out_of_source_forecast_cannot_invent_an_extra_decision():
    plan = baseline_plan()
    agent = BaselineAgent(plan, "shortest_prior_v1", 42)
    agent.native.t = len(plan.source.builds)
    with pytest.raises(ValueError, match="no declared decision"):
        agent.update_bandit(bandit_for(plan.source.builds[-1]))


def test_context_feature_name_cannot_replace_native_test_identity():
    """A valid source feature called Name maps to a private numeric column, preserving native test IDs."""
    from tests.corpus.baseline_support import change

    plan = baseline_plan()
    builds = tuple(
        change(
            build,
            tests=tuple(
                change(
                    test,
                    context=tuple(
                        change(feature, measurement=change(feature.measurement, name="Name"))
                        for feature in test.context
                    ),
                )
                for test in build.tests
            ),
        )
        for build in plan.source.builds
    )
    plan = change(plan, source=change(plan.source, builds=builds))
    agent = BaselineAgent(plan, "linucb_alpha05_v1", 42)
    for build in plan.source.builds:
        bandit = bandit_for(build)
        agent.update_bandit(bandit)
        ranking = agent.choose()
        assert set(ranking) == {test.test_id for test in build.tests}
        bandit.pull(ranking)
        agent.observe(bandit.evaluation_metric)
