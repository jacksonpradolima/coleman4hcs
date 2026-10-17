"""Versioned optional Java matrix declarations; the controlled domain remains transport independent."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.controlled_contracts import Label
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.transports.junit_contracts import JunitCaptureRecipe, ProducerElapsed

Count = Annotated[int, Field(strict=True, ge=0, le=10000)]


class JunitMatrixCell(Immutable):
    """Predeclare one trial of one focal fix, with passive curation and fixture-selection references.

    The recipe already pins both source views, common fixture, JARs, runtime and
    harness. References are citations, never executable commands or acquisition
    instructions. A curated label does not establish an independent unique fault.
    """

    cell_id: Label
    trial_id: Label
    source_reference: Label
    fixture_selection: Label
    reconstruction: Literal["direct-upstream-common-fixed-fixture-v1"] = "direct-upstream-common-fixed-fixture-v1"
    recipe: JunitCaptureRecipe


class JunitMatrixRecipe(VersionedContract):
    """Freeze at most sixteen bounded source trials before observing their outcomes.

    Cells sort by logical ID; physical declaration order does not affect identity.
    Duplicate recipe/trial collections reject rather than inflating evidence.
    Version 1 requires explicit schema version 1; legacy/missing/future versions
    reject without implicit migration. This model performs no I/O or execution.
    """

    MAXIMUM_BYTES: ClassVar[int] = 4 * 1024**2
    recipe: Literal["junit3-controlled-matrix-v1"] = "junit3-controlled-matrix-v1"
    matrix_id: Label
    cells: Annotated[tuple[JunitMatrixCell, ...], Field(min_length=1, max_length=16)]
    limitations: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _identities(self) -> Self:
        """Reject duplicate logical collections and canonicalize declaration order only."""
        if len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("Duplicate Java matrix cell identity")
        keys = {(cell.recipe.fingerprint, cell.trial_id) for cell in self.cells}
        if len(keys) != len(self.cells):
            raise ValueError("Duplicate Java matrix recipe/trial collection")
        object.__setattr__(self, "cells", tuple(sorted(self.cells, key=lambda cell: cell.cell_id)))
        return self


class JunitPairCounts(Immutable):
    """Numbers of same-test/trial pairs, not numbers of faults or independent observations."""

    expected: Count = 0
    both_pass: Count = 0
    both_fail: Count = 0
    reversed: Count = 0
    inconclusive: Count = 0

    @property
    def total(self) -> int:
        """Count observed pairs, including one-sided/inconclusive comparisons."""
        return self.expected + self.both_pass + self.both_fail + self.reversed + self.inconclusive


class JunitMatrixRole(Immutable):
    """Bind a raw role packet and its process accounting; unavailable elapsed is null, never zero.

    Producer elapsed nanoseconds include container observation overhead. They are
    distinct from harness case durations, peak memory, runner work and CI latency.
    """

    role: Literal["buggy", "fixed"]
    packet_sha256: Fingerprint
    status: Literal["collected", "process_error", "timeout", "output_limit", "protocol_error", "interrupted"]
    producer_elapsed: ProducerElapsed

    @model_validator(mode="after")
    def _elapsed(self) -> Self:
        """Keep interrupted elapsed unavailable and measured process statuses observed; launch errors allow either."""
        if self.status == "interrupted" and self.producer_elapsed.status != "unavailable":
            raise ValueError("Interrupted Java matrix role requires unavailable elapsed")
        if self.status not in {"interrupted", "process_error"} and self.producer_elapsed.status != "observed":
            raise ValueError("Observed Java matrix role requires observed elapsed")
        return self


class JunitMatrixCellResult(Immutable):
    """Reference a separately validated controlled dataset without copying its source/observation envelope.

    Entirely absent test pairs remain explicitly unobserved; one-sided pairs are
    inconclusive. A complete collection can contain failures/errors and never
    implies successful scientific admission or a wholly correct fixed revision.
    """

    cell_id: Label
    dataset_fingerprint: Fingerprint
    context_fingerprint: Fingerprint
    planned_pairs: Count
    unobserved_pairs: Count
    pairs: JunitPairCounts
    roles: Annotated[tuple[JunitMatrixRole, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def _accounting(self) -> Self:
        """Conserve declared tests and require exactly one packet per role."""
        if {role.role for role in self.roles} != {"buggy", "fixed"}:
            raise ValueError("Java matrix requires both distinct source roles")
        if self.planned_pairs != self.pairs.total + self.unobserved_pairs:
            raise ValueError("Java matrix pair conservation failed")
        if self.collected and self.unobserved_pairs:
            raise ValueError("Collected Java matrix cannot omit declared pairs")
        object.__setattr__(self, "roles", tuple(sorted(self.roles, key=lambda role: role.role)))
        return self

    @property
    def collected(self) -> bool:
        """Return process collection completeness, independently of assertion/test outcomes."""
        return all(role.status == "collected" for role in self.roles)


class JunitMatrixReceipt(VersionedContract):
    """Deterministic bounded matrix index; actual datasets/raw packets must be independently revalidated.

    A receipt may describe incomplete evidence. Only fully collected receipts may
    acquire a completion marker. Version 1 preserves all existing dataset/producer
    contracts and grants no historical-CI or statistical-independence claim.
    """

    MAXIMUM_BYTES: ClassVar[int] = 8 * 1024**2
    plan: JunitMatrixRecipe
    producer_sha256: Fingerprint
    cells: Annotated[tuple[JunitMatrixCellResult, ...], Field(min_length=1, max_length=16)]
    status: Literal["collected", "incomplete"]
    history: Literal["constructed"] = "constructed"
    admission: Literal["not_evaluated"] = "not_evaluated"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Require every planned cell exactly once and honest completion/coverage accounting."""
        expected = {cell.cell_id: len(cell.recipe.test_ids) for cell in self.plan.cells}
        if len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("Duplicate Java matrix result cell")
        if {cell.cell_id for cell in self.cells} != set(expected):
            raise ValueError("Java matrix results do not cover planned cells")
        if any(cell.planned_pairs != expected[cell.cell_id] for cell in self.cells):
            raise ValueError("Java matrix planned test count disagrees with recipe")
        status = "collected" if all(cell.collected for cell in self.cells) else "incomplete"
        if self.status != status:
            raise ValueError("Java matrix collection status disagrees with role evidence")
        object.__setattr__(self, "cells", tuple(sorted(self.cells, key=lambda cell: cell.cell_id)))
        return self

    @property
    def declared_projects(self) -> int:
        """Count distinct declared project labels, without asserting statistical independence."""
        return len({cell.recipe.project for cell in self.plan.cells})

    @property
    def curated_focal_fixes(self) -> int:
        """Count distinct project/fault labels; tests and repeated trials never add focal fixes."""
        return len({(cell.recipe.project, cell.recipe.fault_id) for cell in self.plan.cells})
