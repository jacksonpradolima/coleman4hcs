"""Deterministic density greedy and bounded branch-and-bound coverage selection."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, TypedDict

from coleman.selection.contracts import CoverageProblem, Quantity, decimal_value, ticks
from coleman.selection.mathematics import (
    SelectionValue,
    constraints_feasible,
    infeasibility_reason,
    optional_ids,
    selection_value,
    union_upper_bound,
)
from coleman.selection.resources import (
    Method,
    Preflight,
    SearchLimits,
    SearchStoppedError,
    WorkCounter,
    inspect_workload,
    preflight_reason,
)
from coleman.selection.results import CoverageCheckpoint, CoverageResult


class _ResultContext(TypedDict):
    """Typed construction boundary shared by the three validated result outcomes."""

    schema_version: int
    problem: CoverageProblem
    method: Method
    limits: SearchLimits
    preflight: Preflight
    scope: Literal["full", "restricted"]
    scope_fingerprint: str
    work_used: int


@dataclass(frozen=True)
class _Prepared:
    """Scoped immutable arithmetic tables; allocated only after resource preflight."""

    costs: dict[str, int]
    covers: dict[str, frozenset[str]]
    weights: dict[str, int]
    budget: int
    maximum_selected: int | None
    required: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass
class _SearchState:
    """Invocation-local incumbent and proof; a choice commits only after full calculation."""

    data: _Prepared
    counter: WorkCounter
    best: SelectionValue
    proof: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Node:
    """One prefix subtree with an exact, possibly infeasible, fixed partial selection."""

    prefix: str
    value: SelectionValue
    covered: frozenset[str]


def _covered(data: _Prepared, selected: tuple[str, ...]) -> frozenset[str]:
    """Take the scoped union without looking at outcomes, rewards or native agent state."""
    return frozenset().union(*(data.covers[name] for name in selected))


def _gain(state: _SearchState, name: str, covered: frozenset[str]) -> int:
    """Calculate marginal nonnegative union weight, polling the overall deadline."""
    gain = 0
    for outcome in state.data.covers[name]:
        state.counter.poll()
        if outcome not in covered:
            gain += state.data.weights[outcome]
    return gain


def _fits(data: _Prepared, value: SelectionValue, name: str) -> bool:
    """Check adding one indivisible candidate against inclusive budget/cardinality."""
    return value.cost + data.costs[name] <= data.budget and (
        data.maximum_selected is None or len(value.selected_ids) < data.maximum_selected
    )


def _greedy(state: _SearchState) -> None:
    """Add positive marginal density, free positive coverage first; ties use candidate IDs.

    This particular budget/cardinality/mandatory heuristic carries no approximation
    guarantee. Zero-gain candidates are left out. It never worsens a resumed hint.
    """
    data = state.data
    covered = _covered(data, state.best.selected_ids)
    while True:
        choice, choice_gain = _greedy_choice(state, covered)
        if choice is None:
            return
        state.best = SelectionValue(
            tuple(sorted((*state.best.selected_ids, choice))),
            state.best.cost + data.costs[choice],
            state.best.objective + choice_gain,
        )
        covered |= data.covers[choice]


def _denser(first: tuple[int, Fraction], second: tuple[int, Fraction]) -> bool:
    """Compare free-priority and exact rational density explicitly."""
    return first[0] > second[0] or (first[0] == second[0] and first[1] > second[1])


def _greedy_choice(state: _SearchState, covered: frozenset[str]) -> tuple[str | None, int]:
    """Scan once in canonical ID order, completing a choice before committing it."""
    choice = None
    best_density = (0, Fraction(0))
    choice_gain = 0
    selected = set(state.best.selected_ids)
    for name in state.data.optional:
        if name in selected:
            continue
        state.counter.step()
        if not _fits(state.data, state.best, name):
            continue
        gain = _gain(state, name, covered)
        if gain <= 0:
            continue
        cost = state.data.costs[name]
        density = (1, Fraction(0)) if cost == 0 else (0, Fraction(gain, cost))
        if choice is None or _denser(density, best_density):
            choice, best_density, choice_gain = name, density, gain
    return choice, choice_gain


def _proof_leaf(state: _SearchState, prefix: str, limit: int) -> None:
    """Record a terminal subtree only while the bounded proof can still be retained."""
    if len(state.proof) >= limit:
        raise SearchStoppedError("certificate_limit")
    state.proof.append(prefix)


def _branch_bound(state: _SearchState, certificate_limit: int) -> None:
    """Explore bounded DFS with strict union pruning, retaining global lexicographic ties."""
    data = state.data
    covered = _covered(data, data.required)
    required = SelectionValue(
        data.required, sum(data.costs[name] for name in data.required), sum(data.weights[name] for name in covered)
    )
    frontier = [_Node("", required, covered)]
    suffix = [frozenset()] * (len(data.optional) + 1)
    for index in reversed(range(len(data.optional))):
        suffix[index] = suffix[index + 1] | data.covers[data.optional[index]]
    while frontier:
        state.counter.step()
        node = frontier.pop()
        _visit(state, node, frontier, suffix, certificate_limit)


def _visit(
    state: _SearchState, node: _Node, frontier: list[_Node], suffix: list[frozenset[str]], certificate_limit: int
) -> None:
    """Prune/prove a node or expand both children; equal bounds are never discarded."""
    data, value = state.data, node.value
    if value.cost > data.budget or (
        data.maximum_selected is not None and len(value.selected_ids) > data.maximum_selected
    ):
        _proof_leaf(state, node.prefix, certificate_limit)
        return
    if (-value.objective, value.selected_ids) < (-state.best.objective, state.best.selected_ids):
        state.best = value
    index = len(node.prefix)
    bound = sum(data.weights[name] for name in node.covered | suffix[index])
    state.counter.poll()
    if bound < state.best.objective or index == len(data.optional):
        _proof_leaf(state, node.prefix, certificate_limit)
        return
    name = data.optional[index]
    included = SelectionValue(
        tuple(sorted((*value.selected_ids, name))),
        value.cost + data.costs[name],
        value.objective + _gain(state, name, node.covered),
    )
    frontier.append(_Node(node.prefix + "0", value, node.covered))
    frontier.append(_Node(node.prefix + "1", included, node.covered | data.covers[name]))


def _prepare(problem: CoverageProblem) -> _Prepared:
    """Build sparse scoped tables only, avoiding a dense candidate/outcome matrix."""
    candidates = problem.scoped_candidates()
    return _Prepared(
        costs={item.candidate_id: ticks(item.cost.value) for item in candidates},
        covers={item.candidate_id: frozenset(item.covers) for item in candidates},
        weights={item.outcome_id: ticks(item.weight.value) for item in problem.outcomes},
        budget=ticks(problem.constraints.budget.value),
        maximum_selected=problem.constraints.maximum_selected,
        required=problem.constraints.required_ids,
        optional=optional_ids(problem),
    )


def solve_coverage(
    problem: CoverageProblem,
    *,
    method: Method = "greedy_density_v1",
    limits: SearchLimits | None = None,
    checkpoint: CoverageCheckpoint | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> CoverageResult:
    """Select finite weighted coverage, preserving truthful scope, bounds and incumbents.

    No RNG, filesystem, learning, runner or transport is involved. Input validation
    errors propagate. Preflight refusal returns limited with null incumbent;
    mandatory infeasibility is proved separately. After initialization, cooperative
    limits/cancellation/KeyboardInterrupt return the last complete feasible choice,
    a union bound and no optimality proof. Exact completion requires a bounded full
    prefix certificate, rechecked by the result contract. No hard real-time or
    measured RSS promise is made; validation/output/certificate overhead is bounded
    as documented in SearchLimits. Checkpoints retain a verified incumbent only.
    """
    started = time.monotonic()
    problem = CoverageProblem.model_validate(problem)
    limits = SearchLimits.model_validate(limits or SearchLimits())
    if method not in {"greedy_density_v1", "branch_bound_v1"}:
        raise ValueError("Unknown coverage method")
    checkpoint = _validated_checkpoint(problem, checkpoint)
    facts = inspect_workload(problem, method, limits)
    payload: _ResultContext = {
        "schema_version": 1,
        "problem": problem,
        "method": method,
        "limits": limits,
        "preflight": facts,
        "scope": problem.search_scope,
        "scope_fingerprint": problem.fingerprint,
        "work_used": 0,
    }
    impossible = infeasibility_reason(problem)
    rejected = preflight_reason(problem, method, limits, facts)
    stopping_reason = impossible or rejected
    if stopping_reason is not None:
        return CoverageResult(
            **payload,
            selected_ids=None,
            cost=None,
            objective=None,
            feasible=False,
            status="infeasible" if impossible else "limited",
            termination_reason=stopping_reason,
            upper_bound=None,
            absolute_gap=None,
        )
    state = _SearchState(
        _prepare(problem), WorkCounter(limits, started, cancelled), _initial_incumbent(problem, checkpoint)
    )
    reason = _search(state, method, limits)
    optimal = reason == "search_complete"
    bound = state.best.objective if optimal else union_upper_bound(problem)
    payload["work_used"] = state.counter.used
    return CoverageResult(
        **payload,
        selected_ids=state.best.selected_ids,
        cost=Quantity(value=decimal_value(state.best.cost), semantics=problem.constraints.budget.semantics),
        objective=Quantity(value=decimal_value(state.best.objective), semantics=problem.objective_semantics),
        feasible=True,
        status="optimal" if optimal else "feasible",
        termination_reason=reason,
        upper_bound=Quantity(value=decimal_value(bound), semantics=problem.objective_semantics),
        absolute_gap=Quantity(value=decimal_value(bound - state.best.objective), semantics=problem.objective_semantics),
        proof_prefixes=tuple(sorted(state.proof)) if optimal else (),
    )


def _validated_checkpoint(problem: CoverageProblem, checkpoint: CoverageCheckpoint | None) -> CoverageCheckpoint | None:
    """Check hint integrity before preflight/infeasibility returns, without coverage allocation."""
    if checkpoint is None:
        return None
    checkpoint = CoverageCheckpoint.model_validate(checkpoint)
    if checkpoint.problem_fingerprint != problem.fingerprint:
        raise ValueError("Checkpoint differs from the frozen coverage problem/scope")
    costs = {item.candidate_id: ticks(item.cost.value) for item in problem.candidates}
    if not set(checkpoint.selected_ids).issubset(costs):
        raise ValueError("Unknown checkpoint candidate")
    if not constraints_feasible(problem, checkpoint.selected_ids, sum(costs[name] for name in checkpoint.selected_ids)):
        raise ValueError("Checkpoint incumbent is not feasible")
    return checkpoint


def _initial_incumbent(problem: CoverageProblem, checkpoint: CoverageCheckpoint | None) -> SelectionValue:
    """Recompute arithmetic for mandatory or previously validated incumbent identities."""
    if checkpoint is None:
        return selection_value(problem, problem.constraints.required_ids)
    return selection_value(problem, checkpoint.selected_ids)


def _search(state: _SearchState, method: Method, limits: SearchLimits) -> str:
    """Share the entire search budget across greedy warm start and DFS; preserve on stop."""
    try:
        state.counter.poll()
        _greedy(state)
        if method == "greedy_density_v1":
            return "greedy_complete"
        _branch_bound(state, limits.max_certificate_leaves)
        return "search_complete"
    except SearchStoppedError as exc:
        return str(exc)
    except KeyboardInterrupt:
        return "interrupted"
