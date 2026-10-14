"""Immutable exact allocation artifacts and explicit projections into existing domain sinks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from coleman.allocation.quantities import variance_semantics
from coleman.allocation.results import AllocationResult
from coleman.artifacts import ArtifactWriter
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.selection.contracts import UnitSemantics


def publish_allocation_result(result: AllocationResult, writer: ArtifactWriter) -> Path:
    """Publish revalidated exact evidence atomically and idempotently through ArtifactWriter.

    Limited/negative/inconclusive results retain their status. Conflicting replay
    and I/O errors propagate; incomplete writes never expose completed artifacts.
    Existing local hard-link/fsync and power-loss limitations apply.
    """
    result = AllocationResult.model_validate(result)
    path = writer.run_root / "allocation-results" / f"{result.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, result.to_json() + "\n")
    return path


def allocation_record(result: AllocationResult, *, run_id: str, available_at: datetime) -> DomainRecord:
    """Project selected conditional forecasts into #251 records, preserving signed losses.

    Exact JSON is authoritative; binary64 projection is explicit in each meaning
    and is never used by selection or evidence validation. No incumbent yields
    unavailable measurements, not zeros. Supply actual zoned result availability
    after calculation and retain it on retry. No learning reward or policy
    activation is fabricated, including for estimated-weight fallback decisions.
    """
    result = AllocationResult.model_validate(result)
    selected = next(
        (row.forecast for row in result.completed if row.forecast.action_id == result.choice.selected_action_id), None
    )
    semantics = result.problem.value_semantics
    values: tuple[tuple[str, str, UnitSemantics, Literal["minimize", "maximize", "none"]], ...] = (
        ("expected_net_payoff", "expected_net", semantics, "maximize"),
        ("net_payoff_variance", "variance", variance_semantics(semantics), "minimize"),
        ("expected_target_shortfall", "expected_target_shortfall", semantics, "minimize"),
        ("scenario_utility", "utility", semantics, "maximize"),
    )
    measurements = tuple(
        Measurement(
            name=name,
            value=float(getattr(selected, field).value) if selected is not None else None,
            unit=unit.unit,
            meaning=unit.meaning + ":modeled-" + name + ":binary64-projection-v1",
            direction=direction,
            status="observed" if selected is not None else "unavailable",
        )
        for name, field, unit, direction in values
    )
    return DomainRecord(
        schema_version=1,
        namespace="decision.scenario_allocation",
        run_id=run_id,
        execution_id=result.problem_fingerprint,
        decision_id=result.fingerprint,
        data_fingerprint=result.problem.data_fingerprint,
        available_at=available_at,
        provenance=ResultProvenance(
            producer="coleman.scenario-allocation",
            producer_version=result.solver_version,
            spec_fingerprint=fingerprint(
                {
                    "limits": result.limits.model_dump(mode="json"),
                    "utility": result.problem.utility.model_dump(mode="json"),
                    "risk": result.problem.risk.model_dump(mode="json"),
                }
            ),
            evaluation_fingerprint=result.problem_fingerprint,
        ),
        measurements=measurements,
    )
