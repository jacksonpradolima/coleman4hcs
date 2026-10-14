"""Independent tiny exhaustive rational oracles; no production solver or mathematical helpers.

Only frozen input contracts are consumed. Enumeration and every moment/constraint
are implemented here with fractions and combinations, separately from production
integer ticks, search, certificates, forecast and feasibility implementations.
These references establish mathematical correctness for bounded declared models,
not independence of real observations or the validity of supplied scenario weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from coleman.allocation.contracts import AllocationProblem, DiscreteAllocation
from coleman.benchmarks.contracts import ReferenceAnswer
from coleman.selection.contracts import CoverageProblem


@dataclass(frozen=True)
class CoverageValue:
    """An exact selected union with costs and objective in their source units."""

    ids: tuple[str, ...]
    cost: Fraction
    objective: Fraction
    feasible: bool


def coverage_value(problem: CoverageProblem, ids: tuple[str, ...]) -> CoverageValue:
    """Independently recompute scope, mandatory/cardinality/budget constraints and union utility."""
    candidates = {row.candidate_id: row for row in problem.candidates}
    if len(ids) != len(set(ids)) or not set(ids).issubset(candidates):
        raise ValueError("Unknown or duplicate benchmark selection")
    cost = sum((Fraction(candidates[name].cost.value) for name in ids), Fraction())
    covered = set().union(*(candidates[name].covers for name in ids))
    objective = sum((Fraction(row.weight.value) for row in problem.outcomes if row.outcome_id in covered), Fraction())
    maximum = problem.constraints.maximum_selected
    feasible = (
        set(ids).issubset(problem.search_ids or ())
        and set(problem.constraints.required_ids).issubset(ids)
        and (maximum is None or len(ids) <= maximum)
        and cost <= Fraction(problem.constraints.budget.value)
    )
    return CoverageValue(tuple(sorted(ids)), cost, objective, feasible)


def coverage_oracle(problem: CoverageProblem) -> ReferenceAnswer:
    """Enumerate all subsets up to twelve declared candidates; reject larger oracle workloads."""
    problem = CoverageProblem.model_validate(problem)
    if len(problem.candidates) > 12 or len(problem.outcomes) > 256:
        raise ValueError("Independent coverage oracle size limit exceeded")
    names = sorted(problem.search_ids or ())
    best = None
    for count in range(len(names) + 1):
        for selected in combinations(names, count):
            row = coverage_value(problem, selected)
            if row.feasible and (best is None or (-row.objective, row.ids) < (-best.objective, best.ids)):
                best = row
    return ReferenceAnswer(
        objective_semantics=problem.objective_semantics,
        cost_semantics=problem.constraints.budget.semantics,
        selected_ids=None if best is None else best.ids,
        cost=None if best is None else best.cost,
        objective=None if best is None else best.objective,
    )


@dataclass(frozen=True)
class _AllocationForecast:
    """Independent complete signed forecast and conditional feasibility in explicit source units."""

    action_id: str
    outcomes: tuple[tuple[str, Fraction, Fraction], ...]
    gross: Fraction
    net: Fraction
    variance: Fraction
    shortfall: Fraction
    utility: Fraction


@dataclass(frozen=True)
class AllocationValue(_AllocationForecast):
    """Independent full forecast with conditional feasibility computed separately."""

    feasible: bool


def _allocation_forecast(problem: AllocationProblem, action: DiscreteAllocation) -> _AllocationForecast:
    """Compute rational distribution moments using direct deviations, without production arithmetic."""
    probabilities = {row.scenario_id: Fraction(row.probability) for row in problem.scenarios}
    target = Fraction(problem.risk.downside_target.value)
    balance = Fraction(problem.initial_balance.value)
    coefficient = Fraction(problem.utility.coefficient.value) if problem.utility.coefficient is not None else Fraction()
    gross = {row.scenario_id: Fraction(row.gross.value) for row in action.gross_payoffs}
    net = {name: value - Fraction(action.transaction_cost.value) for name, value in gross.items()}
    mean = sum((probabilities[name] * value for name, value in net.items()), Fraction())
    variance = sum((probabilities[name] * (value - mean) ** 2 for name, value in net.items()), Fraction())
    shortfall = sum((probabilities[name] * max(target - value, 0) for name, value in net.items()), Fraction())
    utility = sum(
        (probabilities[name] * (value - coefficient * max(target - value, 0) ** 2) for name, value in net.items()),
        Fraction(),
    )
    return _AllocationForecast(
        action_id=action.action_id,
        outcomes=tuple((name, net[name], balance + net[name]) for name in sorted(net)),
        gross=sum((probabilities[name] * value for name, value in gross.items()), Fraction()),
        net=mean,
        variance=variance,
        shortfall=shortfall,
        utility=utility,
    )


def _allocation_feasible(problem: AllocationProblem, action: DiscreteAllocation, forecast: _AllocationForecast) -> bool:
    """Check reservation, total resource and original distribution constraints independently."""
    if Fraction(action.capital_required.value) > Fraction(problem.initial_balance.value):
        return False
    capacities = {row.name: Fraction(row.capacity.value) for row in problem.resources}
    if any(Fraction(row.amount.value) > capacities[row.name] for row in action.resource_use):
        return False
    risk = problem.risk
    if risk.maximum_variance is not None and forecast.variance > Fraction(risk.maximum_variance.value):
        return False
    if risk.maximum_expected_shortfall is not None and forecast.shortfall > Fraction(
        risk.maximum_expected_shortfall.value
    ):
        return False
    if risk.minimum_terminal_balance is None:
        return True
    positive = {row.scenario_id for row in problem.scenarios if row.probability > 0}
    return all(
        terminal >= Fraction(risk.minimum_terminal_balance.value)
        for name, _, terminal in forecast.outcomes
        if name in positive
    )


def allocation_values(problem: AllocationProblem) -> tuple[AllocationValue, ...]:
    """Independently compute whole joint distributions and conditional feasibility, with explicit caps.

    Variance uses sum(p*(x-mean)^2), separately from production E[x^2]-E[x]^2.
    Capital is reserved, fee charged once, and risk applies to the original
    distribution. No optimizer or production mathematical helper is called.
    """
    problem = AllocationProblem.model_validate(problem)
    if len(problem.actions) > 32 or len(problem.scenarios) > 64:
        raise ValueError("Independent allocation oracle size limit exceeded")
    rows = []
    for action in problem.actions:
        forecast = _allocation_forecast(problem, action)
        rows.append(
            AllocationValue(
                action_id=forecast.action_id,
                outcomes=forecast.outcomes,
                gross=forecast.gross,
                net=forecast.net,
                variance=forecast.variance,
                shortfall=forecast.shortfall,
                utility=forecast.utility,
                feasible=_allocation_feasible(problem, action, forecast),
            )
        )
    return tuple(rows)


def allocation_oracle(problem: AllocationProblem) -> ReferenceAnswer:
    """Choose independently among feasible declared allocations, favoring no-action then lexical IDs."""
    if problem.probability_evidence.kind != "assumed":
        raise ValueError("An estimated model cannot establish a benchmark optimal allocation")
    feasible = [row for row in allocation_values(problem) if row.feasible]
    best = min(
        feasible,
        key=lambda row: (-row.utility, row.action_id != problem.no_action_id, row.action_id),
        default=None,
    )
    return ReferenceAnswer(
        objective_semantics=problem.value_semantics,
        selected_ids=None if best is None else (best.action_id,),
        objective=None if best is None else best.utility,
    )
