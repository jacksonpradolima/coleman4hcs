"""Native agent adapters for frozen CI baselines; existing policies and checkpoints remain authoritative."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from coleman.agent import Agent, ContextualAgent, RewardAgent
from coleman.corpus.baseline_contracts import BaselineBuild, BaselinePlan, Method
from coleman.corpus.baseline_eligibility import baseline_eligibility
from coleman.corpus.baseline_heuristics import (
    HEURISTICS,
    DecisionBuild,
    HistoricalBaseline,
    ObservedFeedback,
    decision_views,
    unattainable_order,
)
from coleman.evaluation import EvaluationMetric
from coleman.policy import Policy
from coleman.policy import base as policy_base
from coleman.reward import RNFailReward
from coleman.runner import build_agents_from_config


@contextmanager
def _policy_generator(generator: np.random.Generator):
    """Temporarily bind the unchanged native policies to a checkpointed generator.

    Calls are process-isolated and serialized, not thread-safe. The previous
    global is restored even on failure; unrelated native calls keep their legacy
    RNG semantics. The matrix agent owns/pickles the supplied generator because
    the legacy native checkpoint does not capture the process-global generator.
    """
    previous = policy_base._rng
    policy_base._rng = generator
    try:
        yield
    finally:
        policy_base._rng = previous


@dataclass(frozen=True)
class AgentTiming:
    """Observed elapsed seconds per policy phase, distinct from modeled test occupancy."""

    preparation_seconds: float = 0.0
    inference_seconds: float = 0.0
    learning_seconds: float = 0.0


class _HistoricalPolicy(Policy):
    """Delegate only separated predecision inputs and actually observed feedback to a simple heuristic."""

    def __init__(self, method: Method, views: tuple[DecisionBuild, ...], point_in_time: bool):
        """Create candidate-local state; no source truth or external feature extraction is retained."""
        self.historical = HistoricalBaseline(method, views, point_in_time=point_in_time)

    def choose_all(self, agent: Agent) -> list[str]:
        """Apply the declared heuristic immediately, including its first build."""
        return self.historical.rank(agent.t + 1, tuple(agent.actions["Name"].to_list()))

    def credit_assignment(self, agent: Agent) -> None:
        """Keep heuristic feedback separate from native RNFail's accumulated reward columns."""


class _UnattainablePolicy(Policy):
    """Explicit diagnostic with current truth; it is never an attainable policy or promotion candidate."""

    def __init__(self, builds: tuple[BaselineBuild, ...]):
        """Retain the complete declared universe only for this labeled reference."""
        self.builds = builds

    def choose_all(self, agent: Agent) -> list[str]:
        """Use current known failures/costs in the declared next build."""
        return unattainable_order(self.builds[agent.t])

    def credit_assignment(self, agent: Agent) -> None:
        """No online learning can make current-outcome oracle access attainable."""


def _native_agent(method: Method, seed: int) -> RewardAgent:
    """Reuse the real native factory and its exact versioned representative parameters."""
    policies: dict[str, tuple[str, dict[str, Any]]] = {
        "seeded_random_v1": ("Random", {}),
        "ucb_c1_v1": ("UCB", {"c": 1.0}),
        "thompson_beta11_v1": ("ThompsonSampling", {"alpha_prior": 1.0, "beta_prior": 1.0}),
        "discounted_ucb_095_c1_v1": ("DiscountedUCB", {"gamma": 0.95, "c": 1.0}),
        "linucb_alpha05_v1": ("LinUCB", {"alpha": 0.5}),
        "portfolio_random_ucb_v1": ("PortfolioUCB", {"policies": ["Random", "UCB"], "c": 1.0, "window_size": 20}),
    }
    if method not in policies:
        raise ValueError("Method has no qualified native agent")
    name, parameters = policies[method]
    configuration = {name.lower(): {"rnfail": parameters}}
    if name == "PortfolioUCB":
        configuration["ucb"] = {"rnfail": {"c": 1.0}}
    agents = build_agents_from_config(configuration, [name], ["RNFail"], seed=seed)
    if len(agents) != 1 or not isinstance(agents[0], RewardAgent):
        raise ValueError("Frozen baseline method must resolve to exactly one native reward agent")
    return agents[0]


class BaselineAgent:
    """Decorate one actual native agent without modifying legacy global behavior.

    The existing Environment consumes this small duck-typed agent boundary.
    Contextual inputs are injected here from their verified predecision view;
    the environment's legacy exact-type contextual dispatch is intentionally not
    used. Native bandits retain their seeded first shuffle. Declared heuristics
    and the unattainable reference execute their own first ordering immediately.

    Policy preparation, inference and reward/history update are timed separately
    using the monotonic process clock. These are elapsed measurements, not CPU,
    training of a calibrated model or observed CI savings. Native pickle
    checkpoints retain the decorated agent, original policy and heuristic history.
    Only trusted local checkpoints may be loaded, as with the existing runner.
    """

    def __init__(self, plan: BaselinePlan, method: Method, seed: int):
        """Require a declared method; callers must check its full-source eligibility before execution."""
        plan = BaselinePlan.model_validate(plan)
        if method not in plan.methods or type(seed) is not int or seed not in plan.seeds:
            raise ValueError("Baseline method or seed is absent from the frozen plan")
        eligibility = next(item for item in baseline_eligibility(plan) if item.method == method)
        if not eligibility.compatible:
            raise ValueError("Baseline candidate is incompatible: " + ",".join(eligibility.reasons))
        self.method = method
        self.policy_generator = np.random.default_rng(seed)
        self.views = decision_views(plan)
        if method in HEURISTICS:
            policy = _HistoricalPolicy(method, self.views, plan.source.clock_basis == "source_point_in_time")
            self.native = RewardAgent(policy, RNFailReward(), seed=seed)
        elif method == "unattainable_failure_first_v1":
            self.native = RewardAgent(_UnattainablePolicy(plan.source.builds), RNFailReward(), seed=seed)
        else:
            self.native = _native_agent(method, seed)
        self.timing = AgentTiming()

    def __str__(self) -> str:
        """Use the frozen candidate ID in native rows, rather than conflating similarly named methods."""
        return self.method

    @property
    def last_reward(self):
        """Expose the unchanged native reward for native monitoring."""
        return self.native.last_reward

    @property
    def bandit(self):
        """Expose the same native bandit for Environment.pull; the wrapper creates no alternate executor."""
        return self.native.bandit

    @property
    def fallback_count(self) -> int:
        """Count explicit missing-input fallbacks only for the current heuristic decision."""
        policy = self.native.policy
        return policy.historical.fallback_count if isinstance(policy, _HistoricalPolicy) else 0

    def get_reward_function(self):
        """Expose the original native reward definition, including budget-censored RNFail semantics."""
        return self.native.get_reward_function()

    def reset(self) -> None:
        """Reset before a new native execution; checkpoint restore subsequently replaces the saved agent."""
        self.native.reset()
        policy = self.native.policy
        if isinstance(policy, _HistoricalPolicy):
            previous = policy.historical
            policy.historical = HistoricalBaseline(
                previous.method, previous.views, point_in_time=previous.point_in_time
            )
        self.timing = AgentTiming()

    def update_bandit(self, bandit) -> None:
        """Bind the actual current native universe, injecting only eligible contextual inputs.

        The source ordinal is the native agent's completed-step count, including
        after checkpoint recovery. A final out-of-source forecast is unsupported;
        the matrix environment disables that optional native forecast explicitly.
        """
        start = time.perf_counter()
        try:
            if not 0 <= self.native.t < len(self.views):
                raise ValueError("Baseline agent has no declared decision view for this step")
            if isinstance(self.native, ContextualAgent):
                view = self.views[self.native.t]
                features = [f"baseline_context_{index}" for index in range(len(view.tests[0].context))]
                rows = [
                    {
                        "Name": test.test_id,
                        **{
                            name: feature.measurement.value
                            for name, feature in zip(features, test.context, strict=True)
                        },
                    }
                    for test in view.tests
                ]
                self.native.update_features(features)
                self.native.update_context(pl.DataFrame(rows))
            with _policy_generator(self.policy_generator):
                self.native.update_bandit(bandit)
        finally:
            self.timing = AgentTiming(preparation_seconds=time.perf_counter() - start)

    def choose(self) -> list[str]:
        """Time actual policy inference; heuristic/oracle cold starts bypass only the legacy initial shuffle."""
        start = time.perf_counter()
        try:
            with _policy_generator(self.policy_generator):
                if isinstance(self.native.policy, _HistoricalPolicy | _UnattainablePolicy):
                    action = self.native.policy.choose_all(self.native)
                    self.native.last_prioritization = action
                    return action
                return self.native.choose()
        finally:
            self.timing = AgentTiming(self.timing.preparation_seconds, time.perf_counter() - start)

    def observe(self, metric: EvaluationMetric) -> None:
        """Update the original learner and commit only selected feedback to heuristic history.

        Failure identities come from actual native detection ranks. Realized
        durations are read only after execution and only for selected tests.
        Unselected tests do not enter heuristic counts or duration means. Native
        RNFail itself keeps its existing zero reward for unselected actions.
        """
        start = time.perf_counter()
        try:
            with _policy_generator(self.policy_generator):
                self.native.observe(metric)
            policy = self.native.policy
            if isinstance(policy, _HistoricalPolicy):
                if self.native.bandit is None:
                    raise ValueError("Observed native baseline has no bandit")
                failures = {self.native.last_prioritization[index - 1] for index in metric.detection_ranks}
                selected = self.native.bandit.arms.filter(pl.col("Name").is_in(metric.scheduled_testcases))
                feedback = tuple(
                    ObservedFeedback(str(name), name in failures, float(cost))
                    for name, cost in selected.select(["Name", "Duration"]).iter_rows()
                )
                policy.historical.accept(self.native.t, feedback)
        finally:
            self.timing = AgentTiming(
                self.timing.preparation_seconds, self.timing.inference_seconds, time.perf_counter() - start
            )
