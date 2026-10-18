"""Deterministic pair covering without enumerating the factor Cartesian product."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product

from coleman.corpus.stress_contracts import StressCase, StressDesign, StressFactors, factor_levels

Interaction = tuple[int, str, int, str]


def _required() -> set[Interaction]:
    """Enumerate pair-level requirements only (quadratic in the declared axes)."""
    axes = factor_levels()
    return {(i, a, j, b) for i, j in combinations(range(len(axes)), 2) for a, b in product(axes[i][1], axes[j][1])}


def _pairs(values: tuple[str, ...]) -> set[Interaction]:
    """Return one row's static pair interactions, excluding temporal interpretation."""
    return {(i, values[i], j, values[j]) for i, j in combinations(range(len(values)), 2)}


def certify_design(rows: tuple[StressFactors, ...]) -> tuple[Interaction, ...]:
    """Return one private pair witness per row; reject incomplete/redundant designs.

    Input row order is significant; factor dictionary insertion order is not.
    This is an independent exhaustive pair certificate, not a proof of optimal
    row count, temporal interactions, candidate advantage or fault coverage.
    """
    if len(rows) > 256:
        raise ValueError("Stress design exceeds its 256-row resource bound")
    checked = tuple(StressFactors.model_validate(row) for row in rows)
    pairs = [_pairs(tuple(row.model_dump()[name] for name, _ in factor_levels())) for row in checked]
    counts = Counter(pair for collection in pairs for pair in collection)
    if set(counts) != _required():
        raise ValueError("Stress design omits required two-factor level interactions")
    witnesses = []
    for collection in pairs:
        private = sorted(pair for pair in collection if counts[pair] == 1)
        if not private:
            raise ValueError("Stress design contains a redundant row without a coverage witness")
        witnesses.append(private[0])
    return tuple(witnesses)


def covering_design() -> StressDesign:
    """Greedily cover outstanding pairs, then delete redundant rows deterministically.

    Ties use the declared factor/level order. At most one row per outstanding
    pair is added; the v1 result must fit the explicit 256-row contract. No RNG,
    Cartesian experiment expansion, filesystem, runner or external tool is used.
    """
    axes, uncovered = factor_levels(), _required()
    rows: list[tuple[str, ...]] = []
    while uncovered:
        i, a, j, b = min(uncovered)
        chosen = {i: a, j: b}
        for index, (_, levels) in enumerate(axes):
            if index in chosen:
                continue

            def score(level: str, axis=index, fixed=chosen, missing=uncovered) -> int:
                """Count newly covered pairs with already fixed factors."""
                return sum(
                    (min(axis, k), level if axis < k else v, max(axis, k), v if axis < k else level) in missing
                    for k, v in fixed.items()
                )

            chosen[index] = max(levels, key=score)
        row = tuple(chosen[k] for k in range(len(axes)))
        rows.append(row)
        uncovered -= _pairs(row)
    counts = Counter(pair for row in rows for pair in _pairs(row))
    for index in range(len(rows) - 1, -1, -1):
        pairs = _pairs(rows[index])
        if all(counts[pair] > 1 for pair in pairs):
            counts.subtract(pairs)
            del rows[index]
    return StressDesign(
        schema_version=1,
        rows=tuple(StressFactors.model_validate(dict(zip((a[0] for a in axes), row, strict=True))) for row in rows),
    )


def adversarial_cases() -> tuple[tuple[str, StressCase], ...]:
    """Declare adverse sequences separately from static covering rows; no run is launched."""
    settings = (
        ("long-zero-failure", {"failures": "zero", "budget": "full"}),
        ("rename-label-collision", {"churn": "collisions", "variants": "changing"}),
        ("delayed-abrupt-drift", {"feedback": "delayed", "drift": "sudden"}),
        ("late-correction", {"feedback": "corrected", "drift": "recurring"}),
        ("missing-adversarial-context", {"feedback": "missing", "context": "adversarial"}),
        ("shared-flaky-parallel", {"flakiness": "environment", "workers": "bounded"}),
        ("large-heavy-tail", {"size": "large", "duration": "heavy_tail", "variants": "sparse"}),
    )
    return tuple(
        (name, StressCase(schema_version=1, factors=StressFactors.model_validate(values))) for name, values in settings
    )
