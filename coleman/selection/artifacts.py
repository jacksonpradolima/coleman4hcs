"""Coverage artifact publication and explicit binary64 projections into existing sinks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from coleman.artifacts import ArtifactWriter
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.selection.contracts import Quantity, UnitSemantics
from coleman.selection.results import CoverageResult


def publish_coverage_result(result: CoverageResult, writer: ArtifactWriter) -> Path:
    """Publish revalidated exact JSON immutably through the existing ArtifactWriter.

    Identical replay is idempotent; conflicting content/I/O errors propagate and
    interrupted writes never expose partial completed artifacts. Limited/feasible
    reports keep their explicit status; publication does not assert optimality.
    Local hard-link/fsync durability limits are those of ArtifactWriter.
    """
    result = CoverageResult.model_validate(result)
    path = writer.run_root / "coverage-results" / f"{result.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, result.to_json() + "\n")
    return path


def coverage_record(result: CoverageResult, *, run_id: str, available_at: datetime) -> DomainRecord:
    """Project exact decimal quantities explicitly to #251 binary64 measurement values.

    The complete result artifact is authoritative for arithmetic/proofs. Projected
    meanings include `binary64-projection-v1`; this is a declared lossy plotting/
    analysis adapter, never used for optimization or bound verification. Null means
    unavailable (no incumbent/bound), not zero. No learning reward is fabricated.
    Caller supplies actual zoned result availability after computation and must
    retain that timestamp on retries. Existing local sinks provide idempotence.
    """
    result = CoverageResult.model_validate(result)
    values: tuple[tuple[str, Quantity | None, UnitSemantics, Literal["minimize", "maximize", "none"]], ...] = (
        ("selection_cost", result.cost, result.problem.constraints.budget.semantics, "minimize"),
        ("coverage_objective", result.objective, result.problem.objective_semantics, "maximize"),
        ("coverage_upper_bound", result.upper_bound, result.problem.objective_semantics, "none"),
        ("coverage_absolute_gap", result.absolute_gap, result.problem.objective_semantics, "none"),
    )
    measurements = tuple(
        Measurement(
            name=name,
            value=float(quantity.value) if quantity is not None else None,
            unit=semantics.unit,
            meaning=semantics.meaning + ":binary64-projection-v1",
            direction=direction,
            status="observed" if quantity is not None else "unavailable",
            minimum=0.0,
        )
        for name, quantity, semantics, direction in values
    )
    return DomainRecord(
        schema_version=1,
        namespace="decision.coverage",
        run_id=run_id,
        execution_id=result.scope_fingerprint,
        decision_id=result.fingerprint,
        data_fingerprint=result.problem.data_fingerprint,
        available_at=available_at,
        provenance=ResultProvenance(
            producer="coleman.coverage",
            producer_version=result.solver_version,
            spec_fingerprint=fingerprint({"method": result.method, "limits": result.limits.model_dump(mode="json")}),
            evaluation_fingerprint=result.scope_fingerprint,
        ),
        measurements=measurements,
    )
