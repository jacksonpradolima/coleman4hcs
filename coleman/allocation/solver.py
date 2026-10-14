"""Bounded enumeration of declared discrete allocations, with conservative evidence handling."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import cast

from coleman.allocation.contracts import AllocationProblem
from coleman.allocation.forecasts import ActionEvaluation, _evaluate_action
from coleman.allocation.resources import AllocationLimits, inspect_allocation
from coleman.allocation.results import (
    AllocationCheckpoint,
    AllocationChoice,
    AllocationResult,
    Termination,
    bind_checkpoint,
    choose_completed,
)
from coleman.selection.resources import SearchStoppedError, WorkCounter


def select_allocation(
    problem: AllocationProblem,
    *,
    limits: AllocationLimits | None = None,
    checkpoint: AllocationCheckpoint | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AllocationResult:
    """Evaluate finite actions under one deadline/work budget; never execute an allocation.

    No-action is examined first, then canonical action IDs. Interrupted attempts
    publish no partial forecast; all completed evidence and the best certified
    incumbent survive. Zero work can yield no incumbent, not fake infeasibility.
    Checkpoint integrity is validated before preflight returns. Estimated weights
    preserve uncertainty and allow at most a certified unchanged fallback. No RNG,
    reward normalization, continuous optimization, external data or I/O is used.
    """
    started = time.monotonic()
    problem = AllocationProblem.model_validate(problem)
    limits = AllocationLimits.model_validate(limits or AllocationLimits())
    checkpoint = bind_checkpoint(problem, checkpoint)
    facts = inspect_allocation(problem)
    if facts.estimated_working_bytes > limits.max_working_bytes:
        return AllocationResult(
            schema_version=1,
            problem=problem,
            problem_fingerprint=problem.fingerprint,
            limits=limits,
            preflight=facts,
            checkpoint_input=checkpoint,
            completed=(),
            work_used=0,
            termination_reason="memory_limit",
            choice=AllocationChoice(
                selected_action_id=None,
                status="limited",
                decision_reason="preflight_refused_work",
                conservative_fallback=False,
            ),
        )
    completed = list(checkpoint.completed) if checkpoint is not None else []
    counter = WorkCounter(limits, started, cancelled)
    reason = _enumerate(problem, completed, counter)
    return AllocationResult(
        schema_version=1,
        problem=problem,
        problem_fingerprint=problem.fingerprint,
        limits=limits,
        preflight=facts,
        checkpoint_input=checkpoint,
        completed=tuple(completed),
        work_used=counter.used,
        termination_reason=reason,
        choice=choose_completed(problem, tuple(completed)),
    )


def _enumerate(problem: AllocationProblem, completed: list[ActionEvaluation], counter: WorkCounter) -> Termination:
    """Commit whole evaluations only, preserving completed rows on cooperative stops."""
    cached = {row.forecast.action_id for row in completed}
    actions = sorted(problem.actions, key=lambda action: (action.action_id != problem.no_action_id, action.action_id))
    try:
        for action in actions:
            if action.action_id in cached:
                continue
            counter.step()
            completed.append(_evaluate_action(problem, action, counter))
        return "enumeration_complete"
    except SearchStoppedError as exc:
        return cast(Termination, str(exc))
    except KeyboardInterrupt:
        return "interrupted"
