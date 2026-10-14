"""Bounded benchmark manifests with exact reference answers and explicit compute protocols."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, model_validator

from coleman.allocation import AllocationLimits, AllocationProblem
from coleman.results.domain import Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection import CoverageProblem, SearchLimits, UnitSemantics
from coleman.selection.contracts import Immutable


def _exact_rational(value):
    """Reject implicit binary64 rounding and boolean-as-number coercion in exact expected values."""
    if isinstance(value, (float, bool)):
        raise ValueError("Exact reference numbers require integers, decimals or rational strings")
    return value


ExactFraction = Annotated[Fraction, BeforeValidator(_exact_rational)]


class ReferenceAnswer(Immutable):
    """Independent expected optimum, conditional on the complete declared finite model.

    Fractions serialize as rational strings, retaining exact units in the source
    problem. Null IDs/objective mean proven infeasibility, never a zero optimum.
    Allocation cost is unavailable here: its distinct accounting fields remain
    in the full action forecasts instead of being collapsed into one expense.
    """

    objective_semantics: UnitSemantics
    cost_semantics: UnitSemantics | None = None
    selected_ids: tuple[Identifier, ...] | None
    objective: ExactFraction | None
    cost: ExactFraction | None = None


class CoverageCase(VersionedContract):
    """Tiny complete coverage instance; reference enumeration is capped at twelve candidates."""

    kind: Literal["coverage"] = "coverage"
    case_id: Identifier
    numeric_tolerance: Literal["0"] = "0"
    exact_methods: tuple[Literal["branch_bound_v1"], ...] = ("branch_bound_v1",)
    problem: CoverageProblem
    expected: ReferenceAnswer
    methods: tuple[Literal["greedy_density_v1", "branch_bound_v1"], ...] = ("greedy_density_v1", "branch_bound_v1")
    limits: SearchLimits = Field(default_factory=lambda: SearchLimits(deadline_seconds=None))

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """Keep exhaustive references small and correctness independent of machine speed."""
        if len(self.problem.candidates) > 12 or len(self.problem.outcomes) > 256:
            raise ValueError("Coverage benchmarks allow at most twelve candidates and 256 outcomes")
        if sum(len(item.covers) for item in self.problem.candidates) > 4096:
            raise ValueError("Coverage benchmark membership limit exceeded")
        _protocol(self.methods, self.limits.deadline_seconds)
        if not set(self.exact_methods).issubset(self.methods) or len(self.exact_methods) != len(
            set(self.exact_methods)
        ):
            raise ValueError("Required exact methods must be unique declared methods")
        return self


class AllocationCase(VersionedContract):
    """Finite assumed-weight allocation benchmark; estimates require separate protected evidence."""

    kind: Literal["allocation"] = "allocation"
    case_id: Identifier
    numeric_tolerance: Literal["0"] = "0"
    require_exact: Annotated[bool, Field(strict=True)] = True
    problem: AllocationProblem
    expected: ReferenceAnswer
    methods: tuple[Literal["enumerate_allocation_v1"], ...] = ("enumerate_allocation_v1",)
    limits: AllocationLimits = Field(default_factory=lambda: AllocationLimits(deadline_seconds=None))

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """Bound oracle work and exclude statistical claims from synthetic model checking."""
        if len(self.problem.actions) > 32 or len(self.problem.scenarios) > 64:
            raise ValueError("Allocation benchmarks allow at most 32 actions and 64 scenarios")
        if self.problem.probability_evidence.kind != "assumed":
            raise ValueError("Mathematical allocation benchmarks require assumed weights")
        _protocol(self.methods, self.limits.deadline_seconds)
        return self


def _protocol(methods: tuple[str, ...], deadline: float | None) -> None:
    """Reject duplicate/empty methods and noisy wall-clock correctness cutoffs."""
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Benchmark methods must be nonempty and unique")
    if deadline is not None:
        raise ValueError("Deterministic benchmark protocols require deadline_seconds=None")


BenchmarkCase = Annotated[CoverageCase | AllocationCase, Field(discriminator="kind")]


class BenchmarkSuite(VersionedContract):
    """Frozen cases and work limits; seeds are execution labels, not independent observations.

    Current production solvers use no RNG. Repeated labels explicitly test complete
    mathematical-artifact equality; a suite without a repeated label must report
    reproducibility as untested. Maximum 128 solver invocations bounds the suite.
    No empirical efficacy or runtime equivalence follows from equal work counters:
    coverage counts candidate examinations/node pops; allocation counts actions.
    """

    benchmark_id: Identifier
    cases: Annotated[tuple[BenchmarkCase, ...], Field(min_length=1, max_length=16)]
    seeds: Annotated[
        tuple[Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)], ...], Field(min_length=1, max_length=16)
    ]

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Keep case identities unique and preflight the full number of invocations."""
        names = [case.case_id for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate benchmark case identity")
        if sum(len(case.methods) for case in self.cases) * len(self.seeds) > 128:
            raise ValueError("Benchmark invocation limit exceeded")
        return self
