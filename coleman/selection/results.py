"""Self-contained versioned coverage results, proofs and verified incumbent checkpoints."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import CoverageProblem, Quantity, decimal_value, unique_ids
from coleman.selection.mathematics import (
    feasible,
    infeasibility_reason,
    selection_value,
    union_upper_bound,
    verify_optimality,
)
from coleman.selection.resources import Method, Preflight, SearchLimits, inspect_workload, preflight_reason


class CoverageCheckpoint(VersionedContract):
    """A verified incumbent hint, never a restored proof of previously searched branches.

    Save these JSON bytes in the existing CheckpointPayload.extra/CheckpointStore.
    Resume checks scope/input identity and recomputes feasibility/objective. Search
    restarts at the root with that incumbent; discarded frontier/work never becomes
    an optimality claim. No I/O is performed by this checkpoint contract.
    """

    problem_fingerprint: Fingerprint
    selected_ids: Annotated[tuple[Identifier, ...], Field(max_length=10000)]
    solver_version: Literal["coverage-v1"] = "coverage-v1"

    @field_validator("selected_ids")
    @classmethod
    def _selected(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize the hint, retaining errors for duplicate identities."""
        return unique_ids(values)


class CoverageResult(VersionedContract):
    """Exact cost/objective and a checked certificate for the supplied finite scope.

    Null selected_ids/cost/objective means no incumbent was produced; an empty
    tuple is a real zero-cost selection. Infeasible means a mandatory-constraint
    proof; limited means preflight refused work, with no infeasibility claim.
    Feasible interrupted results retain their incumbent and a loose union bound.
    Optimal requires a complete independently checked prefix proof, including
    lexicographic ties, and has bound=objective/gap=0. Bounds concern this search
    scope only. Deserialization recomputes arithmetic/proofs; no optimizer or I/O
    runs. Verification/output work has the documented bounded overhead.
    """

    problem: CoverageProblem
    method: Method
    limits: SearchLimits
    preflight: Preflight
    solver_version: Literal["coverage-v1"] = "coverage-v1"
    scope: Literal["full", "restricted"]
    scope_fingerprint: Fingerprint
    selected_ids: Annotated[tuple[Identifier, ...], Field(max_length=10000)] | None
    cost: Quantity | None
    objective: Quantity | None
    feasible: Annotated[bool, Field(strict=True)]
    status: Literal["feasible", "optimal", "infeasible", "limited"]
    termination_reason: Identifier
    work_used: Annotated[int, Field(strict=True, ge=0)]
    upper_bound: Quantity | None
    absolute_gap: Quantity | None
    proof_prefixes: Annotated[
        tuple[Annotated[str, Field(pattern="^[01]*$", max_length=20)], ...], Field(max_length=1024)
    ] = ()

    @field_validator("selected_ids")
    @classmethod
    def _selected(cls, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Normalize an existing selection; null never becomes an empty selection."""
        return None if values is None else unique_ids(values)

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Verify source/scope and request-size evidence before accepting claims."""
        if self.scope_fingerprint != self.problem.fingerprint or self.scope != self.problem.search_scope:
            raise ValueError("Coverage result does not match its full declared scope")
        if self.preflight != inspect_workload(self.problem, self.method, self.limits):
            raise ValueError("Coverage preflight differs from the request")
        if self.feasible and preflight_reason(self.problem, self.method, self.limits, self.preflight) is not None:
            raise ValueError("Coverage result claims work refused by its preflight")
        if self.work_used > self.limits.max_work:
            raise ValueError("Coverage work counter exceeds the overall limit")
        if len(self.proof_prefixes) > self.limits.max_certificate_leaves:
            raise ValueError("Coverage certificate exceeds the declared proof limit")
        return self

    @model_validator(mode="after")
    def _selection(self) -> Self:
        """Recompute feasible selections; reject unknown or invented quantities."""
        if not self.feasible:
            self._no_incumbent()
            return self
        if self.selected_ids is None or self.status not in {"feasible", "optimal"}:
            raise ValueError("Feasible result requires a selected incumbent")
        value = selection_value(self.problem, self.selected_ids)
        if not feasible(self.problem, value):
            raise ValueError("Coverage selection violates constraints or scope")
        expected_cost = Quantity(value=decimal_value(value.cost), semantics=self.problem.constraints.budget.semantics)
        expected_objective = Quantity(value=decimal_value(value.objective), semantics=self.problem.objective_semantics)
        if self.cost != expected_cost or self.objective != expected_objective:
            raise ValueError("Coverage quantities differ from the selected union")
        return self

    def _no_incumbent(self) -> None:
        """Separate proven infeasibility from rejected computation, without fake zeros."""
        if any(
            item is not None
            for item in (self.selected_ids, self.cost, self.objective, self.upper_bound, self.absolute_gap)
        ):
            raise ValueError("A result without an incumbent must retain null quantities")
        if self.proof_prefixes or self.work_used:
            raise ValueError("No-incumbent result cannot contain search/proof evidence")
        if self.status == "infeasible" and self.termination_reason == infeasibility_reason(self.problem):
            return
        rejected = preflight_reason(self.problem, self.method, self.limits, self.preflight)
        if self.status != "limited" or rejected is None or self.termination_reason != rejected:
            raise ValueError("Unsupported infeasibility or computation-limit claim")

    @model_validator(mode="after")
    def _certificate(self) -> Self:
        """Check exact optimality proofs or the scope-union upper bound and exact gap."""
        if not self.feasible:
            return self
        if self.selected_ids is None:
            raise ValueError("A certificate requires a selected incumbent")
        value = selection_value(self.problem, self.selected_ids)
        if self.status == "optimal":
            if self.method != "branch_bound_v1" or self.termination_reason != "search_complete":
                raise ValueError("Optimality requires the exact method and completed proof")
            if self.work_used < 2 * len(self.proof_prefixes) - 1:
                raise ValueError("Coverage work cannot account for the complete proof tree")
            verify_optimality(self.problem, value, self.proof_prefixes)
            bound = value.objective
        else:
            self._stopping_reason()
            if self.proof_prefixes:
                raise ValueError("Partial search must not publish a complete proof")
            bound = union_upper_bound(self.problem)
        expected_bound = Quantity(value=decimal_value(bound), semantics=self.problem.objective_semantics)
        expected_gap = Quantity(
            value=decimal_value(bound - value.objective), semantics=self.problem.objective_semantics
        )
        if self.upper_bound != expected_bound or self.absolute_gap != expected_gap:
            raise ValueError("Unsupported coverage bound or absolute gap")
        return self

    def _stopping_reason(self) -> None:
        """Accept completion only for greedy, and explicit resource/interrupt stops otherwise."""
        if self.termination_reason in {"work_limit", "deadline", "cancelled", "interrupted"}:
            return
        if self.termination_reason == "greedy_complete" and self.method == "greedy_density_v1":
            return
        if self.termination_reason == "certificate_limit" and self.method == "branch_bound_v1":
            return
        raise ValueError("Unsupported feasible-search stopping reason")

    def checkpoint(self) -> CoverageCheckpoint:
        """Return a revalidated feasible hint suitable for the existing checkpoint store."""
        validated = type(self).model_validate(self)
        if not validated.feasible or validated.selected_ids is None:
            raise ValueError("Cannot checkpoint a missing feasible incumbent")
        return CoverageCheckpoint(
            schema_version=1, problem_fingerprint=validated.scope_fingerprint, selected_ids=validated.selected_ids
        )
