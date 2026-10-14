"""Exact signed scenario moments and separate feasibility/risk checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from coleman.allocation.contracts import AllocationProblem, DiscreteAllocation, JointScenario
from coleman.allocation.quantities import ExactPayoff, NonnegativeMoment, variance_semantics
from coleman.results.domain import Identifier
from coleman.selection.contracts import SCALE, Immutable, UnitSemantics, decimal_value, ticks
from coleman.selection.resources import WorkCounter


class NetOutcome(Immutable):
    """One preserved signed net increment and terminal balance for a declared joint state."""

    scenario_id: Identifier
    net: ExactPayoff
    terminal_balance: ExactPayoff


class ActionForecast(Immutable):
    """Conditional distribution summaries in explicit payoff/squared units, without clipping.

    Gross is the entire allocation's increment before one deterministic fee. Net
    deducts that fee once. Variance is E[net^2]-E[net]^2. Target shortfall is the
    unconditional E[max(target-net,0)], distinct from CVaR. Utility is evaluated
    pointwise before expectation. These are modeled forecasts, not CI rewards.
    """

    action_id: Identifier
    outcomes: Annotated[tuple[NetOutcome, ...], Field(min_length=1, max_length=256)]
    expected_gross: ExactPayoff
    expected_net: ExactPayoff
    variance: NonnegativeMoment
    expected_target_shortfall: NonnegativeMoment
    utility: ExactPayoff


class ActionEvaluation(Immutable):
    """A complete forecast plus explicit conditional feasibility reasons; no selection yet."""

    forecast: ActionForecast
    feasible: Annotated[bool, Field(strict=True)]
    rejection_reasons: tuple[Identifier, ...]


@dataclass(frozen=True)
class _Inputs:
    """Signed and nonnegative source amounts in exact 10^-12 ticks."""

    gross: dict[str, int]
    fee: int
    balance: int
    target: int


@dataclass
class _Moments:
    """Invocation-local integer numerators; means/shortfall use scale squared, second moments cubed."""

    gross: int = 0
    net: int = 0
    second: int = 0
    shortfall: int = 0
    shortfall_square: int = 0

    def add(self, scenario: JointScenario, values: _Inputs, semantics: UnitSemantics) -> NetOutcome:
        """Accumulate one whole joint outcome and preserve its unnormalized signed net."""
        probability = ticks(scenario.probability)
        net = values.gross[scenario.scenario_id] - values.fee
        shortage = max(values.target - net, 0)
        self.gross += probability * values.gross[scenario.scenario_id]
        self.net += probability * net
        self.second += probability * net * net
        self.shortfall += probability * shortage
        self.shortfall_square += probability * shortage * shortage
        return NetOutcome(
            scenario_id=scenario.scenario_id,
            net=ExactPayoff(value=decimal_value(net), semantics=semantics),
            terminal_balance=ExactPayoff(value=decimal_value(values.balance + net), semantics=semantics),
        )


def _forecast_action(
    problem: AllocationProblem, action: DiscreteAllocation, counter: WorkCounter | None = None
) -> ActionForecast:
    """Compute exact moments with 12/24/48-place integer arithmetic; perform no I/O.

    Point weights sum exactly to one. The optional counter polls the shared
    deadline/cancellation between scenario operations; no partial forecast escapes
    if interrupted. Signed losses remain present even when constraints later fail.
    """
    values = _Inputs(
        {payoff.scenario_id: ticks(payoff.gross.value) for payoff in action.gross_payoffs},
        ticks(action.transaction_cost.value),
        ticks(problem.initial_balance.value),
        ticks(problem.risk.downside_target.value),
    )
    moments = _Moments()
    rows = []
    for scenario in problem.scenarios:
        if counter is not None:
            counter.poll()
        rows.append(moments.add(scenario, values, problem.value_semantics))
    variance = moments.second * SCALE - moments.net * moments.net
    utility = moments.net * SCALE * SCALE
    if problem.utility.coefficient is not None:
        utility -= ticks(problem.utility.coefficient.value) * moments.shortfall_square
    return ActionForecast(
        action_id=action.action_id,
        outcomes=tuple(rows),
        expected_gross=ExactPayoff(value=decimal_value(moments.gross, places=24), semantics=problem.value_semantics),
        expected_net=ExactPayoff(value=decimal_value(moments.net, places=24), semantics=problem.value_semantics),
        variance=NonnegativeMoment(
            value=decimal_value(variance, places=48), semantics=variance_semantics(problem.value_semantics)
        ),
        expected_target_shortfall=NonnegativeMoment(
            value=decimal_value(moments.shortfall, places=24), semantics=problem.value_semantics
        ),
        utility=ExactPayoff(value=decimal_value(utility, places=48), semantics=problem.value_semantics),
    )


def _feasibility_reasons(
    problem: AllocationProblem, action: DiscreteAllocation, forecast: ActionForecast
) -> tuple[str, ...]:
    """Check reservations, total resources and point-weight risk separately from forecasts.

    These checks are conditional on the declared point distribution. Estimated
    probabilities never establish safe new allocation decisions in this v1 API.
    """
    reasons = []
    if action.capital_required.value > problem.initial_balance.value:
        reasons.append("capital_constraint")
    capacities = {resource.name: resource.capacity.value for resource in problem.resources}
    if any(resource.amount.value > capacities[resource.name] for resource in action.resource_use):
        reasons.append("resource_constraint")
    risk = problem.risk
    if risk.maximum_variance is not None and forecast.variance.value > risk.maximum_variance.value:
        reasons.append("variance_constraint")
    if (
        risk.maximum_expected_shortfall is not None
        and forecast.expected_target_shortfall.value > risk.maximum_expected_shortfall.value
    ):
        reasons.append("target_shortfall_constraint")
    support = {scenario.scenario_id for scenario in problem.scenarios if scenario.probability > 0}
    if risk.minimum_terminal_balance is not None and any(
        outcome.terminal_balance.value < risk.minimum_terminal_balance.value
        for outcome in forecast.outcomes
        if outcome.scenario_id in support
    ):
        reasons.append("terminal_balance_constraint")
    return tuple(reasons)


def _evaluate_action(
    problem: AllocationProblem, action: DiscreteAllocation, counter: WorkCounter | None = None
) -> ActionEvaluation:
    """Return a whole forecast and independent constraint checks, never a partial result."""
    forecast = _forecast_action(problem, action, counter)
    reasons = _feasibility_reasons(problem, action, forecast)
    return ActionEvaluation(forecast=forecast, feasible=not reasons, rejection_reasons=reasons)


def _certified_no_action(problem: AllocationProblem, evaluation: ActionEvaluation) -> bool:
    """Conservatively certify only the no-change fallback over declared uncertain support.

    Capital/resource checks are distribution-free. For learned weights, variance
    is bounded by squared payoff range and expected shortfall by the largest
    supported shortfall. These loose bounds apply to every normalized distribution
    on that support; failure to certify is not proof of infeasibility. Supplied
    probability bounds do not themselves establish external statistical coverage.
    """
    if evaluation.forecast.action_id != problem.no_action_id:
        return False
    if problem.probability_evidence.kind == "assumed":
        return evaluation.feasible
    if "capital_constraint" in evaluation.rejection_reasons or "resource_constraint" in evaluation.rejection_reasons:
        return False
    support = {
        scenario.scenario_id for scenario in problem.scenarios if scenario.upper is not None and scenario.upper > 0
    }
    outcomes = [outcome for outcome in evaluation.forecast.outcomes if outcome.scenario_id in support]
    values = [ticks(outcome.net.value) for outcome in outcomes]
    spread = max(values) - min(values)
    maximum_shortfall = max(max(ticks(problem.risk.downside_target.value) - value, 0) for value in values)
    risk = problem.risk
    if risk.maximum_variance is not None and decimal_value(spread * spread, places=24) > risk.maximum_variance.value:
        return False
    if (
        risk.maximum_expected_shortfall is not None
        and decimal_value(maximum_shortfall) > risk.maximum_expected_shortfall.value
    ):
        return False
    floor = risk.minimum_terminal_balance
    return floor is None or all(outcome.terminal_balance.value >= floor.value for outcome in outcomes)


def evaluate_allocation_action(problem: AllocationProblem, action_id: str) -> ActionEvaluation:
    """Validate a frozen problem and forecast/check one declared action without selecting or I/O.

    Unknown action or malformed/version-incompatible input raises ValueError or
    Pydantic ValidationError. No external data, inference or probability calibration
    is invented. Returned feasibility is conditional on declared point weights.
    """
    problem = AllocationProblem.model_validate(problem)
    return _evaluate_action(problem, problem.action(action_id))
