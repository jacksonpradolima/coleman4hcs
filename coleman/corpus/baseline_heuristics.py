"""Simple CI rankings over separated decision views and genuinely observed past feedback."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from coleman.corpus.baseline_contracts import BaselineBuild, BaselineFeature, BaselinePlan, Method
from coleman.corpus.baseline_eligibility import _available
from coleman.evaluation.protected.temporal import Timestamp


@dataclass(frozen=True)
class DecisionTest:
    """A test's predecision fields; deliberately contains no current duration or verdict."""

    test_id: str
    prior_cost: BaselineFeature | None
    context: tuple[BaselineFeature, ...]


@dataclass(frozen=True)
class DecisionBuild:
    """Immutable policy view, separated from the source's current realized outcomes."""

    step: int
    member_id: str
    decided_at: Timestamp | None
    original_order: tuple[str, ...] | None
    tests: tuple[DecisionTest, ...]


def decision_views(plan: BaselinePlan) -> tuple[DecisionBuild, ...]:
    """Drop current truth before passing source declarations to attainable policies.

    These are process-local typed views, not a second serialized artifact contract.
    Feature eligibility remains explicit in the frozen plan and source contract.
    No I/O, fitting, date inference or mutation of source records occurs.
    """
    return tuple(
        DecisionBuild(
            build.step,
            build.member_id,
            build.decided_at,
            build.original_order,
            tuple(DecisionTest(test.test_id, test.prior_cost, test.context) for test in build.tests),
        )
        for build in plan.source.builds
    )


@dataclass(frozen=True)
class ObservedFeedback:
    """One actually observed selected test, after execution; seconds use the native cost lattice."""

    test_id: str
    failed: bool
    duration_seconds: float


@dataclass
class _History:
    """Candidate-local observed counts; unselected tests never add a passing observation."""

    observations: int = 0
    failures: int = 0
    last_failure: int | None = None
    duration_sum: Fraction = Fraction(0)


HEURISTICS = frozenset(
    {"original_order_v1", "shortest_prior_v1", "recent_failure_v1", "failure_frequency_v1", "recency_duration_v1"}
)


class HistoricalBaseline:
    """Stateful ranking with explicit cold-start/churn fallback and no current truth access.

    Recent-failure ranks the last observed failure step. Frequency is failures /
    genuinely observed attempts, not all elapsed builds. Shortest-first uses an
    available declared prior estimate or the mean of this candidate's selected
    past durations. The hybrid divides frequency + reciprocal failure distance
    by max(prior seconds, 1e-9 s); it is a priority score, never a probability.
    Missing history/cost sorts after known scores with stable-ID tie breaking;
    observed zero frequency and zero cost remain distinct from missing values.

    ``accept`` mutates candidate-local history after a completed decision; native
    checkpoints serialize it with the agent. ``rank`` changes only the most recent
    fallback count. No filesystem, RNG, transport or global history is used.
    """

    def __init__(self, method: Method, views: tuple[DecisionBuild, ...], *, point_in_time: bool):
        """Require one supported heuristic; native and oracle candidates use their own implementations."""
        if method not in HEURISTICS:
            raise ValueError("Unsupported historical baseline method")
        self.method: Method = method
        self.views = views
        self.point_in_time = point_in_time
        self.history: dict[str, _History] = {}
        self.fallback_count = 0
        self.committed_step = 0

    def _cost(self, test: DecisionTest, view: DecisionBuild) -> Fraction | None:
        """Prefer an eligible frozen estimate; otherwise use only this candidate's selected past costs."""
        feature = test.prior_cost
        if feature is not None and _available(feature, view.decided_at, self.point_in_time):
            value = feature.measurement.value
            if value is not None:
                return Fraction(value)
        history = self.history.get(test.test_id)
        if history is not None and history.observations:
            return history.duration_sum / history.observations
        return None

    def _score(self, test: DecisionTest, view: DecisionBuild) -> Fraction | None:
        """Return a descending priority or explicit missing state; never inspect target outcomes."""
        cost = self._cost(test, view)
        if self.method == "shortest_prior_v1":
            return None if cost is None else -cost
        history = self.history.get(test.test_id)
        if history is None or not history.observations:
            return None
        if self.method == "recent_failure_v1":
            return Fraction(history.last_failure or 0)
        frequency = Fraction(history.failures, history.observations)
        if self.method == "failure_frequency_v1":
            return frequency
        if cost is None:
            return None
        recency = Fraction(0) if history.last_failure is None else Fraction(1, view.step - history.last_failure)
        return (frequency + recency) / max(cost, Fraction(1, 10**9))

    def rank(self, step: int, ids: tuple[str, ...]) -> list[str]:
        """Rank the complete declared current universe, rejecting stale/future steps or changed membership."""
        if type(step) is not int or step != self.committed_step + 1 or not 1 <= step <= len(self.views):
            raise ValueError("Baseline ranking requires the next unobserved source step")
        view = self.views[step - 1]
        if len(ids) != len(set(ids)) or set(ids) != {test.test_id for test in view.tests}:
            raise ValueError("Native action universe differs from frozen baseline decision view")
        if self.method == "original_order_v1":
            if view.original_order is None:
                raise ValueError("Meaningful original order is unavailable")
            self.fallback_count = 0
            return list(view.original_order)
        scores = {test.test_id: self._score(test, view) for test in view.tests}
        self.fallback_count = sum(value is None for value in scores.values())
        return sorted(ids, key=lambda name: (scores[name] is None, -(scores[name] or Fraction(0)), name))

    def accept(self, step: int, feedback: tuple[ObservedFeedback, ...]) -> None:
        """Commit one observed feedback frontier; reject duplicate/invalid deliveries before mutation.

        Empty feedback is valid when the budget selected no tests. The ordinal
        still advances but no pass/failure/cost is fabricated. Current observed
        costs may be zero; negative/nonfinite costs and nonboolean labels reject.
        """
        import math

        if type(step) is not int or step != self.committed_step + 1 or not 1 <= step <= len(self.views):
            raise ValueError("Duplicate, missing or out-of-order baseline feedback step")
        allowed = {test.test_id for test in self.views[step - 1].tests}
        names = [row.test_id for row in feedback]
        if len(set(names)) != len(names) or set(names) - allowed:
            raise ValueError("Observed baseline feedback has duplicate or unknown tests")
        if any(
            type(row.failed) is not bool
            or type(row.duration_seconds) not in (int, float)
            or not math.isfinite(row.duration_seconds)
            or row.duration_seconds < 0
            for row in feedback
        ):
            raise ValueError("Observed feedback requires a boolean outcome and finite nonnegative seconds")
        for row in feedback:
            history = self.history.setdefault(row.test_id, _History())
            history.observations += 1
            history.failures += int(row.failed)
            if row.failed:
                history.last_failure = step
            history.duration_sum += Fraction(row.duration_seconds)
        self.committed_step = step


def unattainable_order(build: BaselineBuild) -> list[str]:
    """Rank known failures before passes, then by realized cost/ID; explicitly unavailable in a real decision.

    Requires complete observed binary truth. This is a labeled diagnostic ordering,
    not a claim of global optimality for every budget, metric or fault-cost model.
    It deliberately uses current realized outcomes/costs, unlike attainable views.
    """
    build = BaselineBuild.model_validate(build)
    if any(test.failure.status != "observed" or test.duration.status != "observed" for test in build.tests):
        raise ValueError("Unattainable reference requires complete observed current outcomes and costs")
    return [
        test.test_id
        for test in sorted(
            build.tests,
            key=lambda test: (-cast(float, test.failure.value), cast(float, test.duration.value), test.test_id),
        )
    ]
