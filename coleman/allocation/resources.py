"""Bounded finite-action allocation work and memory preflight using shared work accounting."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from coleman.allocation.contracts import AllocationProblem
from coleman.results.domain import Finite
from coleman.selection.contracts import Immutable


class AllocationLimits(Immutable):
    """Limit new action examinations, cooperative overall seconds and estimated working bytes.

    A unit is one attempted action evaluation, with polling within joint scenarios.
    Validation, checkpoint verification, bounded output and serialization add
    overhead; this is neither a hard real-time kill nor a measured RSS promise.
    Already verified checkpoint rows do not consume new search units. No phase
    resets the timer. Input action/scenario/component caps remain non-overridable.
    """

    max_work: Annotated[int, Field(strict=True, ge=0, le=64)] = 64
    deadline_seconds: Annotated[Finite, Field(ge=0)] | None = 10.0
    max_working_bytes: Annotated[int, Field(strict=True, ge=0, le=1073741824)] = 67108864


class AllocationPreflight(Immutable):
    """Reproducible finite dimensions and estimated sparse additional allocation bytes."""

    actions: Annotated[int, Field(strict=True, ge=1)]
    scenarios: Annotated[int, Field(strict=True, ge=1)]
    estimated_working_bytes: Annotated[int, Field(strict=True, ge=0)]


def inspect_allocation(problem: AllocationProblem) -> AllocationPreflight:
    """Estimate complete finite forecast storage without constructing new scenario matrices."""
    points = len(problem.actions) * len(problem.scenarios)
    components = sum(len(action.resource_use) + len(action.holdings) for action in problem.actions)
    return AllocationPreflight(
        actions=len(problem.actions),
        scenarios=len(problem.scenarios),
        estimated_working_bytes=2048 * points + 1024 * components + 4096 * len(problem.actions),
    )
