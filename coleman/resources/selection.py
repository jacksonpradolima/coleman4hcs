"""Predispatch feasibility receives estimates and identities, never realized outcomes."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.evaluation.protected.temporal import Timestamp
from coleman.resources.contracts import ResourceAmount, ResourceEstimate
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, UnitSemantics


class TestEstimate(Immutable):
    """A declared test with an explicitly available or unavailable estimate.

    Missing estimates are null, never zero. An expired or future estimate can be
    retained for audit but cannot reserve resources. Values include no verdicts.
    """

    test_id: Identifier
    estimate: ResourceEstimate | None


class ResourceRequest(VersionedContract):
    """Frozen predecision information for a serial stop-at-first-nonfit selection.

    Ranking covers the full unique test universe, including unselected tests.
    Priority status concerns the ranking signal, independently of cost freshness.
    A full-suite fallback requires eligible estimates for every test and enough
    capacity; otherwise fallback is no-action. This modeled selection never
    overrides a CI system's required suite or changes native scheduling.
    """

    decision_id: Identifier
    configuration_fingerprint: Fingerprint
    decided_at: Timestamp
    capacity: ResourceAmount
    tests: Annotated[tuple[TestEstimate, ...], Field(max_length=1000)]
    ranking: Annotated[tuple[Identifier, ...], Field(max_length=1000)]
    priority_status: Literal["available", "stale", "unavailable"] = "available"
    fallback: Literal["no_action", "full_suite"] = "no_action"

    @model_validator(mode="after")
    def _universe(self) -> Self:
        """Reject incomplete rankings, duplicates and incompatible cost semantics."""
        ids = [test.test_id for test in self.tests]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(self.ranking):
            raise ValueError("Ranking must contain every unique test exactly once")
        for test in self.tests:
            if test.estimate is not None and test.estimate.amount.semantics != self.capacity.semantics:
                raise ValueError("Incompatible estimate/capacity units")
        object.__setattr__(self, "tests", tuple(sorted(self.tests, key=lambda test: test.test_id)))
        return self


class ResourceSelection(Immutable):
    """Deterministic estimated feasibility; no accuracy, runtime or payoff guarantee."""

    selected_ids: tuple[Identifier, ...]
    estimated_units: int
    semantics: UnitSemantics
    disposition: Literal["ranked_prefix", "full_suite_fallback", "no_action"]
    reason: Literal["eligible", "ineligible_estimate", "priority_fallback", "fallback_infeasible"]


def select_resources(request: ResourceRequest) -> ResourceSelection:
    """Select a bounded serial prefix using only eligible predecision estimates.

    Zero-cost tests fit a zero budget. The first nonfit stops the prefix; later
    cheap tests are not silently substituted. Inclusive availability/expiry ties
    are eligible. No clock, RNG, actual outcomes or source I/O is accessed.
    """
    request = ResourceRequest.model_validate(request)
    estimates = {test.test_id: test.estimate for test in request.tests}
    if any(
        estimate is None or not estimate.available_at <= request.decided_at <= estimate.valid_until
        for estimate in estimates.values()
    ):
        return ResourceSelection(
            selected_ids=(),
            estimated_units=0,
            semantics=request.capacity.semantics,
            disposition="no_action",
            reason="ineligible_estimate",
        )
    costs = {key: estimate.amount.value for key, estimate in estimates.items() if estimate is not None}
    if request.priority_status != "available":
        if request.fallback == "full_suite" and sum(costs.values()) <= request.capacity.value:
            return ResourceSelection(
                selected_ids=request.ranking,
                estimated_units=sum(costs.values()),
                semantics=request.capacity.semantics,
                disposition="full_suite_fallback",
                reason="priority_fallback",
            )
        return ResourceSelection(
            selected_ids=(),
            estimated_units=0,
            semantics=request.capacity.semantics,
            disposition="no_action",
            reason="fallback_infeasible",
        )
    selected = []
    occupied = 0
    for test_id in request.ranking:
        if occupied + costs[test_id] > request.capacity.value:
            break
        selected.append(test_id)
        occupied += costs[test_id]
    return ResourceSelection(
        selected_ids=tuple(selected),
        estimated_units=occupied,
        semantics=request.capacity.semantics,
        disposition="ranked_prefix",
        reason="eligible",
    )
