"""Independent linear-time native CI ranking checks; no production metric is imported."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from coleman.corpus.baseline_contracts import BaselineBuild


@dataclass(frozen=True)
class BaselineReference:
    """Binary32-cost reference with explicit binary64 native prefix and full-universe denominators.

    Costs/first-failure are modeled serial runner seconds, not measured CI elapsed
    savings. Zero-failure ratios are null. No-detection time is censored/null even
    though the old native metric retains its own fallback scalar. Budget excess
    exposes an exact projected-work discrepancy caused by native prefix rounding.
    """

    selected: tuple[str, ...]
    failures: int
    detected: int
    napfd: float | None
    apfdc: float | None
    recall: float | None
    selected_work: float
    projected_full_work: float
    first_failure: float | None
    budget_excess: float


def baseline_reference(build: BaselineBuild, ranking: tuple[str, ...], budget_seconds: float) -> BaselineReference:
    """Check a complete binary test ranking using independent Python/rational arithmetic.

    Duration projection is exactly IEEE binary32 carried into binary64. Selection
    repeats the declared sequential binary64 prefix sum, independently of NumPy
    and the production bandit/metric. Subsequent areas/ratios use exact fractions
    of projected costs before one final binary64 conversion. Full-universe NAPFD
    retains the legacy denominator convention; it is not the paper's alternative
    selected-horizon measure. APFDc is the midpoint-cost variant, without an
    external extra correction. Missing/nonfinite/negative input and changed or
    repeated ranking membership raise ValueError; no I/O or state mutation occurs.
    """
    build = BaselineBuild.model_validate(build)
    if type(budget_seconds) not in (float, int) or not math.isfinite(budget_seconds) or budget_seconds < 0:
        raise ValueError("Native reference budget requires finite nonnegative seconds")
    tests, costs = _projected_inputs(build, ranking)
    total = sum(costs.values(), Fraction(0))
    if not total:
        raise ValueError("Native reference full-suite projected work must be positive")
    selected, ranks, prefix_float = [], [], 0.0
    occupied, area, first = Fraction(0), Fraction(0), None
    for position, name in enumerate(ranking, start=1):
        prefix_float += float(costs[name])
        if prefix_float > budget_seconds:
            break
        selected.append(name)
        if tests[name].failure.value == 1:
            ranks.append(position)
            area += total - occupied - costs[name] / 2
            if first is None:
                first = occupied + costs[name]
        occupied += costs[name]
    failures = sum(test.failure.value == 1 for test in tests.values())
    napfd, apfdc, recall = _quality_ratios(len(tests), failures, ranks, area, total)
    return BaselineReference(
        selected=tuple(selected),
        failures=failures,
        detected=len(ranks),
        napfd=napfd,
        apfdc=apfdc,
        recall=recall,
        selected_work=float(occupied),
        projected_full_work=float(total),
        first_failure=None if first is None else float(first),
        budget_excess=float(max(Fraction(0), occupied - Fraction(budget_seconds))),
    )


def _projected_inputs(build: BaselineBuild, ranking: tuple[str, ...]):
    """Validate complete truth and ranking membership before projecting finite native costs."""
    tests = {test.test_id: test for test in build.tests}
    if len(set(ranking)) != len(ranking) or set(ranking) != set(tests):
        raise ValueError("Native reference requires the complete unique test ranking")
    if any(test.duration.status != "observed" or test.failure.status != "observed" for test in tests.values()):
        raise ValueError("Native reference requires complete observed costs and binary outcomes")
    costs = {}
    for name in ranking:
        value = cast(float, tests[name].duration.value)
        try:
            projected = struct.unpack("!f", struct.pack("!f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("Native reference cost cannot be represented as binary32") from error
        if not math.isfinite(projected):
            raise ValueError("Native reference cost cannot be represented as finite binary32")
        costs[name] = Fraction(projected)
    return tests, costs


def _quality_ratios(
    test_count: int, failures: int, ranks: list[int], area: Fraction, total: Fraction
) -> tuple[float | None, float | None, float | None]:
    """Convert exact full-universe ratios once; a zero failure denominator remains unavailable."""
    if not failures:
        return None, None, None
    recall = Fraction(len(ranks), failures)
    napfd = recall - Fraction(sum(ranks), failures * test_count) + recall / (2 * test_count)
    return float(napfd), float(area / (total * failures)), float(recall)
