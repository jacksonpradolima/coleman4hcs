"""Selection preflight and cooperative overall search-work/deadline accounting."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field

from coleman.results.domain import Finite
from coleman.selection.contracts import CoverageProblem, Immutable

Method = Literal["greedy_density_v1", "branch_bound_v1"]


class SearchLimits(Immutable):
    """Bound shared search work, deadline and estimated additional working memory.

    One work unit is one greedy candidate examination or one branch-node pop;
    warm start and exact search share the same counter/deadline. Setup, validated
    output and certificate verification have bounded linear/certificate overhead.
    Deadline is cooperative, measured from API entry including setup, not a hard
    real-time process kill. Memory estimates exclude the already supplied input,
    interpreter and artifact bytes; they are a conservative allocation preflight,
    not a measured RSS guarantee. Exact certificates have non-overridable v1 caps.
    """

    max_work: Annotated[int, Field(strict=True, ge=0, le=10000000)] = 100000
    deadline_seconds: Annotated[Finite, Field(ge=0)] | None = 10.0
    max_working_bytes: Annotated[int, Field(strict=True, ge=0, le=1073741824)] = 67108864
    max_candidates: Annotated[int, Field(strict=True, ge=0, le=10000)] = 10000
    max_outcomes: Annotated[int, Field(strict=True, ge=0, le=10000)] = 10000
    max_memberships: Annotated[int, Field(strict=True, ge=0, le=1000000)] = 1000000
    max_certificate_leaves: Annotated[int, Field(strict=True, ge=1, le=1024)] = 1024


class Preflight(Immutable):
    """Reproducible request-size and conservative solver-allocation estimates in bytes."""

    candidates: Annotated[int, Field(strict=True, ge=0)]
    outcomes: Annotated[int, Field(strict=True, ge=0)]
    memberships: Annotated[int, Field(strict=True, ge=0)]
    estimated_working_bytes: Annotated[int, Field(strict=True, ge=0)]


def inspect_workload(problem: CoverageProblem, method: Method, limits: SearchLimits) -> Preflight:
    """Count input without materializing candidate/outcome incidence matrices."""
    memberships = sum(len(candidate.covers) for candidate in problem.candidates)
    size = 512 * (len(problem.candidates) + len(problem.outcomes)) + 96 * memberships
    if method == "branch_bound_v1":
        scoped = problem.scoped_candidates()
        # The exact-scope cap is checked before any frontier allocation. The
        # product bounds its repeated coverage sets without constructing them.
        size += 128 * len(scoped) * min(256, len(problem.outcomes)) + 128 * limits.max_certificate_leaves
    return Preflight(
        candidates=len(problem.candidates),
        outcomes=len(problem.outcomes),
        memberships=memberships,
        estimated_working_bytes=size,
    )


def preflight_reason(problem: CoverageProblem, method: Method, limits: SearchLimits, facts: Preflight) -> str | None:
    """Reject oversized work before building mutable search state; never infer infeasibility."""
    for actual, maximum, reason in (
        (facts.candidates, limits.max_candidates, "candidate_limit"),
        (facts.outcomes, limits.max_outcomes, "outcome_limit"),
        (facts.memberships, limits.max_memberships, "membership_limit"),
        (facts.estimated_working_bytes, limits.max_working_bytes, "memory_limit"),
    ):
        if actual > maximum:
            return reason
    if method == "branch_bound_v1":
        scoped = problem.scoped_candidates()
        if len(scoped) > 20 or sum(len(candidate.covers) for candidate in scoped) > 4096:
            return "exact_scope_limit"
        if len(set().union(*(candidate.covers for candidate in scoped))) > 256:
            return "exact_scope_limit"
    return None


class SearchStoppedError(Exception):
    """Internal cooperative stop that preserves the last complete feasible incumbent."""


class WorkCounter:
    """Per-invocation work/deadline/cancellation state; no global RNG or transport."""

    def __init__(self, limits: SearchLimits, started: float, cancelled: Callable[[], bool] | None) -> None:
        """Start accounting at caller API entry, before setup and warm start."""
        self.limits = limits
        self.started = started
        self.cancelled = cancelled
        self.used = 0

    def poll(self) -> None:
        """Check cancellation and elapsed overall time between bounded search operations."""
        if self.cancelled is not None and self.cancelled():
            raise SearchStoppedError("cancelled")
        if self.limits.deadline_seconds is not None and time.monotonic() - self.started >= self.limits.deadline_seconds:
            raise SearchStoppedError("deadline")

    def step(self) -> None:
        """Consume one shared search unit; reaching a limit never partially commits a choice."""
        self.poll()
        if self.used >= self.limits.max_work:
            raise SearchStoppedError("work_limit")
        self.used += 1
