"""Pure integer coverage arithmetic and independently checkable prefix certificates."""

from __future__ import annotations

from dataclasses import dataclass

from coleman.selection.contracts import CoverageProblem, ticks


@dataclass(frozen=True)
class SelectionValue:
    """Exact cost/union weight in 10^-12 ticks for a canonical selected-ID tuple."""

    selected_ids: tuple[str, ...]
    cost: int
    objective: int


def selection_value(problem: CoverageProblem, selected_ids: tuple[str, ...]) -> SelectionValue:
    """Recompute cost and nonnegative union objective; reject unknown/duplicate identities."""
    candidates = {candidate.candidate_id: candidate for candidate in problem.candidates}
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(candidates):
        raise ValueError("Unknown or duplicate selected candidate")
    cost = sum(ticks(candidates[name].cost.value) for name in selected_ids)
    covered = set().union(*(candidates[name].covers for name in selected_ids))
    objective = sum(ticks(outcome.weight.value) for outcome in problem.outcomes if outcome.outcome_id in covered)
    return SelectionValue(tuple(sorted(selected_ids)), cost, objective)


def feasible(problem: CoverageProblem, value: SelectionValue) -> bool:
    """Check inclusive budget/cardinality, mandatory identities and declared search scope."""
    return constraints_feasible(problem, value.selected_ids, value.cost)


def constraints_feasible(problem: CoverageProblem, selected_ids: tuple[str, ...], cost: int) -> bool:
    """Check validated identities and exact cost ticks without calculating coverage/forecasts."""
    selected = set(selected_ids)
    ceiling = problem.constraints.maximum_selected
    return (
        len(selected) == len(selected_ids)
        and selected.issubset(problem.search_ids or ())
        and set(problem.constraints.required_ids).issubset(selected)
        and 0 <= cost <= ticks(problem.constraints.budget.value)
        and (ceiling is None or len(selected) <= ceiling)
    )


def union_upper_bound(problem: CoverageProblem) -> int:
    """Bound every scoped feasible objective by its entire nonnegative coverage union.

    Ignores budget/cardinality, so the bound is valid but may be loose. It is not
    an approximation ratio or an assertion about excluded/outside candidates.
    """
    covered = set().union(*(candidate.covers for candidate in problem.scoped_candidates()))
    return sum(ticks(outcome.weight.value) for outcome in problem.outcomes if outcome.outcome_id in covered)


def infeasibility_reason(problem: CoverageProblem) -> str | None:
    """Prove infeasibility from mandatory identities; the empty extension otherwise works."""
    required = set(problem.constraints.required_ids)
    costs = {candidate.candidate_id: ticks(candidate.cost.value) for candidate in problem.candidates}
    if not required.issubset(problem.search_ids or ()):
        return "mandatory_outside_scope"
    if sum(costs[name] for name in required) > ticks(problem.constraints.budget.value):
        return "mandatory_budget_exceeded"
    maximum = problem.constraints.maximum_selected
    if maximum is not None and len(required) > maximum:
        return "mandatory_cardinality_exceeded"
    return None


def optional_ids(problem: CoverageProblem) -> tuple[str, ...]:
    """Name the sorted binary decision coordinates, excluding fixed mandatory members."""
    required = set(problem.constraints.required_ids)
    return tuple(name for name in problem.search_ids or () if name not in required)


@dataclass(frozen=True)
class _ProofDomain:
    """Compile bounded scoped arithmetic once, independently of solver search state."""

    costs: dict[str, int]
    covers: dict[str, frozenset[str]]
    weights: dict[str, int]
    required: tuple[str, ...]
    budget: int
    maximum_selected: int | None

    def value(self, selected: tuple[str, ...]) -> SelectionValue:
        """Recompute exact union arithmetic over at most twenty scoped candidates."""
        covered = set().union(*(self.covers[name] for name in selected))
        return SelectionValue(
            selected, sum(self.costs[name] for name in selected), sum(self.weights[name] for name in covered)
        )


def verify_optimality(problem: CoverageProblem, value: SelectionValue, prefixes: tuple[str, ...]) -> None:
    """Verify a bounded complete branch partition without executing an optimizer.

    A prefix fixes include/exclude bits in sorted optional-ID order. Its entire
    subtree must be infeasible, have a strict union bound below the incumbent, or
    be a fully assigned leaf no better under (negative objective, selected IDs).
    Disjoint adjacent dyadic intervals must cover the full binary cube. This
    proves both objective optimality and global lexicographic tie-breaking within
    the supplied scope. Hard v1 caps: 20 candidates, 256 touched outcomes, 4096
    memberships, 1024 prefixes. Invalid/unbounded proofs raise ValueError.
    """
    candidates = problem.scoped_candidates()
    touched = set().union(*(candidate.covers for candidate in candidates))
    if len(candidates) > 20 or len(touched) > 256 or sum(len(item.covers) for item in candidates) > 4096:
        raise ValueError("Exact certificate exceeds the v1 verification scope")
    if not prefixes or len(prefixes) > 1024:
        raise ValueError("Missing or oversized optimality certificate")
    if value != selection_value(problem, value.selected_ids) or not feasible(problem, value):
        raise ValueError("Optimality proof requires an actual feasible incumbent")
    domain = _ProofDomain(
        costs={item.candidate_id: ticks(item.cost.value) for item in candidates},
        covers={item.candidate_id: frozenset(item.covers) for item in candidates},
        weights={item.outcome_id: ticks(item.weight.value) for item in problem.outcomes if item.outcome_id in touched},
        required=problem.constraints.required_ids,
        budget=ticks(problem.constraints.budget.value),
        maximum_selected=problem.constraints.maximum_selected,
    )
    ids = optional_ids(problem)
    cursor = 0
    for prefix in sorted(prefixes):
        start, width = _prefix_interval(prefix, len(ids))
        if start != cursor:
            raise ValueError("Certificate prefixes overlap or leave an unexplored branch")
        cursor += width
        _verify_subtree(domain, value, ids, prefix)
    if cursor != 1 << len(ids):
        raise ValueError("Certificate does not cover the complete search scope")


def _prefix_interval(prefix: str, dimensions: int) -> tuple[int, int]:
    """Validate a binary prefix and map it to its exact dyadic cube interval."""
    if len(prefix) > dimensions or any(bit not in "01" for bit in prefix):
        raise ValueError("Invalid certificate prefix")
    width = 1 << (dimensions - len(prefix))
    return (int(prefix, 2) if prefix else 0) * width, width


def _verify_subtree(domain: _ProofDomain, best: SelectionValue, ids: tuple[str, ...], prefix: str) -> None:
    """Check one leaf/pruned subtree against exact nonnegative union arithmetic."""
    selected = tuple(sorted((*domain.required, *(name for name, bit in zip(ids, prefix, strict=False) if bit == "1"))))
    current = domain.value(selected)
    if current.cost > domain.budget or (
        domain.maximum_selected is not None and len(selected) > domain.maximum_selected
    ):
        return
    possible = tuple(sorted((*selected, *ids[len(prefix) :])))
    bound = domain.value(possible).objective
    if bound < best.objective:
        return
    if len(prefix) < len(ids):
        raise ValueError("Certificate prunes a potentially better or lexicographically earlier branch")
    if (-current.objective, current.selected_ids) < (-best.objective, best.selected_ids):
        raise ValueError("Certificate contains a better feasible selection")
