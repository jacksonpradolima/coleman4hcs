"""Independent exact correctness gates before production artifact validation."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field

from coleman.allocation import AllocationResult
from coleman.allocation.forecasts import ActionEvaluation
from coleman.benchmarks.contracts import AllocationCase, CoverageCase, ExactFraction, ReferenceAnswer
from coleman.benchmarks.oracles import (
    AllocationValue,
    CoverageValue,
    allocation_oracle,
    allocation_values,
    coverage_oracle,
    coverage_value,
)
from coleman.selection import CoverageResult, UnitSemantics
from coleman.selection.contracts import Immutable


class GateSummary(Immutable):
    """Quality against a tiny independent oracle, not statistical performance evidence.

    Method and outcome classes differ: a heuristic can carry a valid bound, and
    exact search can stop with only a bounded incumbent. Current adapters produce
    no sampled claims. Objective, optimum and regret use source objective units;
    null means unavailable. All arithmetic gates have zero rational tolerance.
    """

    objective_semantics: UnitSemantics
    method_class: Literal["exact_search", "heuristic", "sampled"]
    evidence: Literal["exact", "bounded", "heuristic", "sampled", "unavailable"]
    status: str
    selected_ids: tuple[str, ...] | None
    feasible: Annotated[bool, Field(strict=True)]
    objective: ExactFraction | None
    upper_bound: ExactFraction | None
    independent_optimum: ExactFraction | None
    absolute_regret: ExactFraction | None
    uncertainty: Literal["declared_finite_model_only"] = "declared_finite_model_only"


def reference(case: CoverageCase | AllocationCase) -> ReferenceAnswer:
    """Check committed expected answers against fresh independent reference enumeration."""
    case = type(case).model_validate(case)
    expected = coverage_oracle(case.problem) if isinstance(case, CoverageCase) else allocation_oracle(case.problem)
    if expected != case.expected:
        raise ValueError(f"Committed reference answer differs from independent oracle: {case.case_id}")
    return expected


def _coverage_request(case: CoverageCase, result: CoverageResult) -> None:
    """Bind source, requested methods and declared compute/exactness requirements."""
    if result.problem != case.problem or result.limits != case.limits or result.method not in case.methods:
        raise ValueError("Coverage benchmark result changed frozen problem, method or budget")
    if result.work_used > case.limits.max_work:
        raise ValueError("Coverage benchmark work budget exceeded")
    if result.method in case.exact_methods and result.status not in {"optimal", "infeasible"}:
        raise ValueError("Coverage benchmark required a completed exact result")


def _coverage_incumbent(case: CoverageCase, result: CoverageResult, expected: ReferenceAnswer) -> CoverageValue | None:
    """Recompute selected unions and constraints independently, preserving unavailable incumbents."""
    if result.selected_ids is None:
        if result.feasible or (result.status == "infeasible" and expected.selected_ids is not None):
            raise ValueError("Coverage benchmark falsely claims infeasibility")
        return None
    row = coverage_value(case.problem, result.selected_ids)
    if not row.feasible or not result.feasible:
        raise ValueError("Coverage benchmark selection violates scope or budget constraints")
    if result.cost is None or Fraction(result.cost.value) != row.cost:
        raise ValueError("Coverage benchmark cost differs from independent sum")
    if result.objective is None or Fraction(result.objective.value) != row.objective:
        raise ValueError("Coverage benchmark objective differs from independent union")
    return row


def _coverage_bound(result: CoverageResult, row: CoverageValue | None, expected: ReferenceAnswer) -> Fraction | None:
    """Require valid maximizing bounds and independently optimal IDs, including lexical ties."""
    bound = None if result.upper_bound is None else Fraction(result.upper_bound.value)
    if bound is not None and expected.objective is not None and bound < expected.objective:
        raise ValueError("Coverage maximizing upper bound is below the independent optimum")
    if result.status == "optimal" and (
        row is None or row.ids != expected.selected_ids or row.objective != expected.objective
    ):
        raise ValueError("Coverage benchmark false optimal status or tie selection")
    return bound


def check_coverage(case: CoverageCase, result: CoverageResult) -> GateSummary:
    """Independently check feasibility, quantities, bounds and exactness, then check native proof.

    Primary quantities have zero tolerance, rather than rounded binary64 equality.
    A maximizing bound must dominate the independent optimum. Full production proof
    replay is additional evidence, not a substitute for the combinations oracle.
    Unchecked fake instances demonstrate that these independent gates fail closed.
    """
    case = CoverageCase.model_validate(case)
    expected = reference(case)
    _coverage_request(case, result)
    row = _coverage_incumbent(case, result, expected)
    bound = _coverage_bound(result, row, expected)
    CoverageResult.model_validate(result)
    evidence = _coverage_evidence(result.status, row, bound)
    return GateSummary(
        objective_semantics=expected.objective_semantics,
        method_class="heuristic" if result.method == "greedy_density_v1" else "exact_search",
        evidence=evidence,
        status=result.status,
        selected_ids=None if row is None else row.ids,
        feasible=row is not None,
        objective=None if row is None else row.objective,
        upper_bound=bound,
        independent_optimum=expected.objective,
        absolute_regret=None if row is None or expected.objective is None else expected.objective - row.objective,
    )


def _coverage_evidence(status, row, bound):
    """Keep absent incumbents unavailable even when a valid global upper bound exists."""
    if status in {"optimal", "infeasible"}:
        return "exact"
    if row is None:
        return "unavailable"
    return "bounded" if bound is not None else "heuristic"


def _allocation_request(case: AllocationCase, result: AllocationResult) -> None:
    """Bind a fresh benchmark attempt to its source and work budget, excluding outside cached search."""
    if result.problem != case.problem or result.limits != case.limits or result.checkpoint_input is not None:
        raise ValueError("Allocation benchmark changed frozen problem/budget or reused outside work")
    if result.work_used > case.limits.max_work:
        raise ValueError("Allocation benchmark work budget exceeded")
    if case.require_exact and result.choice.status not in {"optimal", "infeasible"}:
        raise ValueError("Allocation benchmark required a completed exact result")


def _allocation_forecast(evaluation: ActionEvaluation, row: AllocationValue) -> None:
    """Compare every signed joint outcome, terminal value, moment and constraint to rational reference."""
    forecast = evaluation.forecast
    outcomes = tuple(
        (item.scenario_id, Fraction(item.net.value), Fraction(item.terminal_balance.value))
        for item in forecast.outcomes
    )
    moments = tuple(
        Fraction(quantity.value)
        for quantity in (
            forecast.expected_gross,
            forecast.expected_net,
            forecast.variance,
            forecast.expected_target_shortfall,
            forecast.utility,
        )
    )
    if outcomes != row.outcomes or moments != (row.gross, row.net, row.variance, row.shortfall, row.utility):
        raise ValueError("Allocation benchmark forecast differs from independent joint distribution")
    if evaluation.feasible != row.feasible:
        raise ValueError("Allocation benchmark feasibility differs from independent constraints")


def _allocation_rows(case: AllocationCase, result: AllocationResult) -> dict[str, AllocationValue]:
    """Check unique completed evaluations without borrowing production forecast/feasibility helpers."""
    independent = {row.action_id: row for row in allocation_values(case.problem)}
    completed = {}
    for evaluation in result.completed:
        row = independent.get(evaluation.forecast.action_id)
        if row is None or row.action_id in completed:
            raise ValueError("Allocation benchmark unknown or duplicate completed action")
        _allocation_forecast(evaluation, row)
        completed[row.action_id] = row
    return completed


def _allocation_choice(result, completed, expected) -> AllocationValue | None:
    """Verify full-scope optimality/infeasibility and feasibility of any returned incumbent."""
    selected = result.choice.selected_action_id
    chosen = completed.get(selected)
    if selected is not None and (chosen is None or not chosen.feasible):
        raise ValueError("Allocation benchmark selected an unverified or infeasible action")
    if result.choice.status == "optimal" and (
        len(completed) != len(result.problem.actions) or chosen is None or (chosen.action_id,) != expected.selected_ids
    ):
        raise ValueError("Allocation benchmark false optimal status")
    if result.choice.status == "infeasible" and expected.selected_ids is not None:
        raise ValueError("Allocation benchmark falsely claims infeasibility")
    return chosen


def check_allocation(case: AllocationCase, result: AllocationResult) -> GateSummary:
    """Check every complete signed distribution, all moments and constraints before native replay."""
    case = AllocationCase.model_validate(case)
    expected = reference(case)
    _allocation_request(case, result)
    completed = _allocation_rows(case, result)
    chosen = _allocation_choice(result, completed, expected)
    AllocationResult.model_validate(result)
    objective = None if chosen is None else chosen.utility
    return GateSummary(
        objective_semantics=expected.objective_semantics,
        method_class="exact_search",
        evidence=_allocation_evidence(result.choice.status, chosen),
        status=result.choice.status,
        selected_ids=None if chosen is None else (chosen.action_id,),
        feasible=chosen is not None,
        objective=objective,
        upper_bound=None,
        independent_optimum=expected.objective,
        absolute_regret=None if objective is None or expected.objective is None else expected.objective - objective,
    )


def _allocation_evidence(status, chosen):
    """Classify completed enumeration separately from a feasible partial incumbent."""
    if status in {"optimal", "infeasible"}:
        return "exact"
    return "heuristic" if chosen is not None else "unavailable"


def same_mathematics(first: str, second: str) -> None:
    """Fail repeated/permuted streams when full canonical mathematical artifacts differ.

    Supply validated canonical result JSON, excluding runtime/environment metadata.
    Comparing only objectives or IDs misses changed proofs, source, risk forecasts,
    bounds, work evidence or statuses. Equality does not establish empirical efficacy.
    """
    if first != second:
        raise ValueError("Repeated or permuted benchmark mathematical artifacts differ")
