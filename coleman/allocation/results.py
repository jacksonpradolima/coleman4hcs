"""Auditable finite allocation decisions with replayed forecasts and checkpoint evidence."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.allocation.contracts import AllocationProblem, ordered
from coleman.allocation.forecasts import ActionEvaluation, _certified_no_action, _evaluate_action
from coleman.allocation.resources import AllocationLimits, AllocationPreflight, inspect_allocation
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

DecisionStatus = Literal["optimal", "feasible", "infeasible", "limited", "insufficient_evidence"]
Termination = Literal["enumeration_complete", "work_limit", "deadline", "cancelled", "interrupted", "memory_limit"]


class AllocationChoice(Immutable):
    """A conditional finite-model choice, separate from forecasts and execution status."""

    selected_action_id: Identifier | None
    status: DecisionStatus
    decision_reason: Identifier
    conservative_fallback: Annotated[bool, Field(strict=True)]


def choose_completed(problem: AllocationProblem, completed: tuple[ActionEvaluation, ...]) -> AllocationChoice:
    """Choose exact utility among feasible completed actions; abstain on learned weights.

    Ties favor unchanged allocation, then ascending action ID. Decimal comparison
    is exact; no context-sensitive unary negation or float ranking is used.
    Estimated point weights retain conditional forecasts but never justify a new
    allocation. Only a conservatively certified no-action fallback may be returned.
    """
    if problem.probability_evidence.kind == "estimated":
        unchanged = next((item for item in completed if item.forecast.action_id == problem.no_action_id), None)
        safe = unchanged is not None and _certified_no_action(problem, unchanged)
        return AllocationChoice(
            selected_action_id=problem.no_action_id if safe else None,
            status="insufficient_evidence",
            decision_reason="estimated_weights_require_protected_evidence",
            conservative_fallback=safe,
        )
    feasible = [item for item in completed if item.feasible]
    complete = {item.forecast.action_id for item in completed} == {action.action_id for action in problem.actions}
    if not feasible:
        return AllocationChoice(
            selected_action_id=None,
            status="infeasible" if complete else "limited",
            decision_reason="no_feasible_action" if complete else "no_complete_feasible_incumbent",
            conservative_fallback=False,
        )
    maximum = max(item.forecast.utility.value for item in feasible)
    ties = [item.forecast.action_id for item in feasible if item.forecast.utility.value == maximum]
    selected = problem.no_action_id if problem.no_action_id in ties else min(ties)
    return AllocationChoice(
        selected_action_id=selected,
        status="optimal" if complete else "feasible",
        decision_reason="finite_model_optimum" if complete else "best_completed_incumbent",
        conservative_fallback=False,
    )


class AllocationCheckpoint(VersionedContract):
    """Complete forecast rows tied to a frozen source; integrity is checked on binding/resume.

    Store JSON in existing CheckpointPayload.extra. Cached forecasts (including
    rejected actions) are recomputed against source before use, so no invented
    utility, risk or completion flag can bypass a fresh budget. Validation has
    bounded overhead; cached rows are not new search work or independent evidence.
    """

    problem_fingerprint: Fingerprint
    completed: Annotated[tuple[ActionEvaluation, ...], Field(max_length=64)]
    solver_version: Literal["allocation-v1"] = "allocation-v1"

    @field_validator("completed")
    @classmethod
    def _completed(cls, values: tuple[ActionEvaluation, ...]) -> tuple[ActionEvaluation, ...]:
        """Canonicalize cached actions without collapsing duplicate evaluations."""
        return ordered(values, lambda value: value.forecast.action_id)


def verify_completed(problem: AllocationProblem, completed: tuple[ActionEvaluation, ...]) -> None:
    """Recompute complete rows from the declared source, rejecting forged statistics/risk."""
    for row in completed:
        expected = _evaluate_action(problem, problem.action(row.forecast.action_id))
        if row != expected:
            raise ValueError("Allocation forecast or feasibility differs from frozen joint-scenario evidence")


def bind_checkpoint(problem: AllocationProblem, checkpoint: AllocationCheckpoint | None) -> AllocationCheckpoint | None:
    """Check cache schema, source and every row before any early limit/infeasibility return."""
    if checkpoint is None:
        return None
    checkpoint = AllocationCheckpoint.model_validate(checkpoint)
    if checkpoint.problem_fingerprint != problem.fingerprint:
        raise ValueError("Allocation checkpoint source differs from this problem")
    verify_completed(problem, checkpoint.completed)
    return checkpoint


class AllocationResult(VersionedContract):
    """Self-contained exact finite-model result, preserving negative and inconclusive outcomes.

    Full declared actions/scenarios and completed forecasts permit arithmetic/risk
    replay without running a selector. Optimal means every finite action was
    checked under supplied assumed weights, not empirical efficacy or continuous
    optimality. Estimated weights remain insufficient evidence with at most a
    certified no-action fallback. Null selection means no certified incumbent,
    never an invented zero payoff. No external action or policy activation occurs.
    """

    problem: AllocationProblem
    problem_fingerprint: Fingerprint
    limits: AllocationLimits
    preflight: AllocationPreflight
    checkpoint_input: AllocationCheckpoint | None = None
    completed: Annotated[tuple[ActionEvaluation, ...], Field(max_length=64)]
    choice: AllocationChoice
    termination_reason: Termination
    work_used: Annotated[int, Field(strict=True, ge=0)]
    solver_version: Literal["allocation-v1"] = "allocation-v1"

    @field_validator("completed")
    @classmethod
    def _completed(cls, values: tuple[ActionEvaluation, ...]) -> tuple[ActionEvaluation, ...]:
        """Make action-row ordering deterministic and reject duplicated progress."""
        return ordered(values, lambda value: value.forecast.action_id)

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Verify source, dimensions and checkpoint integrity before accepting any choice."""
        if self.problem_fingerprint != self.problem.fingerprint or self.preflight != inspect_allocation(self.problem):
            raise ValueError("Allocation result source or preflight differs from its problem")
        bind_checkpoint(self.problem, self.checkpoint_input)
        verify_completed(self.problem, self.completed)
        self._verify_work()
        return self

    def _verify_work(self) -> None:
        """Count complete new attempts against the shared limit, retaining truthful partial progress."""
        if self.work_used > self.limits.max_work:
            raise ValueError("Allocation work exceeds its overall limit")
        cached = (
            {row.forecast.action_id: row for row in self.checkpoint_input.completed} if self.checkpoint_input else {}
        )
        current = {row.forecast.action_id: row for row in self.completed}
        if self.termination_reason == "memory_limit":
            if (
                self.completed
                or self.work_used
                or self.preflight.estimated_working_bytes <= self.limits.max_working_bytes
            ):
                raise ValueError("Unsupported allocation memory-limit result")
            return
        if self.preflight.estimated_working_bytes > self.limits.max_working_bytes:
            raise ValueError("Allocation result bypasses its memory preflight")
        if any(current.get(name) != value for name, value in cached.items()):
            raise ValueError("Allocation result discarded or altered validated checkpoint rows")
        self._counter_evidence(len(current) - len(cached), set(current))

    def _counter_evidence(self, new: int, completed_ids: set[str]) -> None:
        """Check complete-attempt counts and verifiable work-limit/completion claims."""
        if not new <= self.work_used <= new + 1:
            raise ValueError("Allocation work cannot account for completed evaluations")
        if self.termination_reason == "work_limit" and (
            self.work_used != self.limits.max_work or self.work_used != new
        ):
            raise ValueError("Allocation work-limit stop does not match its completed budget")
        if self.termination_reason == "enumeration_complete" and (
            completed_ids != {action.action_id for action in self.problem.actions} or self.work_used != new
        ):
            raise ValueError("Allocation enumeration is falsely marked complete")

    @model_validator(mode="after")
    def _choice(self) -> Self:
        """Recompute conditional choice, no-action tie/fallback and complete-scope claims."""
        if self.termination_reason == "memory_limit":
            expected = AllocationChoice(
                selected_action_id=None,
                status="limited",
                decision_reason="preflight_refused_work",
                conservative_fallback=False,
            )
        else:
            expected = choose_completed(self.problem, self.completed)
        if self.choice != expected:
            raise ValueError("Allocation choice differs from verified finite-model evidence")
        return self

    def checkpoint(self) -> AllocationCheckpoint:
        """Export complete verified progress to existing checkpoint infrastructure, without I/O."""
        validated = type(self).model_validate(self)
        if not validated.completed:
            raise ValueError("No completed allocation evaluation to checkpoint")
        return AllocationCheckpoint(
            schema_version=1, problem_fingerprint=validated.problem_fingerprint, completed=validated.completed
        )
