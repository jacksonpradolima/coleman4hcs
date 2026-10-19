"""Typed paired inputs, immutable raw-record evidence and finite inference outputs."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from coleman.evaluation.protected.temporal import TemporalObservation
from coleman.results.domain import DomainRecord, Fingerprint, Finite, Identifier


class PairedObservation(BaseModel):  # pylint: disable=too-few-public-methods
    """One candidate's raw result for a declared independent group/unit.

    ``metadata.group_id`` is the independent unit, not a seed/worker identity.
    Exactly one observation per candidate/unit is accepted; repeated seed rows
    require a predeclared upstream aggregation and never count as new units.
    Outcome availability must agree with the versioned result record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    candidate_id: Identifier
    metadata: TemporalObservation
    record: DomainRecord

    @model_validator(mode="after")
    def _availability(self) -> Self:
        """Prevent metadata from making a later result appear available sooner."""
        if self.metadata.available_at != self.record.available_at:
            raise ValueError("Metadata and result availability disagree")
        return self


class ComparisonResult(BaseModel):  # pylint: disable=too-few-public-methods
    """Signed estimates/uncertainty in raw metric units, with conservative conclusions.

    Raw difference is candidate minus baseline. Improvement reverses that sign
    for minimization. Family-wise intervals use Bonferroni; p-values use Holm.
    Insufficient, incomplete, degenerate or numerically unstable data returns null
    uncertainty rather than fabricated finite evidence. Reused holdout estimates
    are visible but never classified as confirmatory benefit/harm.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    hypothesis_id: Identifier
    independent_units: tuple[Identifier, ...]
    observed_pairs: Annotated[int, Field(strict=True, ge=0)]
    raw_mean_difference: Finite | None
    improvement: Finite | None
    ci_lower: Finite | None
    ci_upper: Finite | None
    p_value: Annotated[Finite, Field(ge=0, le=1)] | None
    adjusted_p_value: Annotated[Finite, Field(ge=0, le=1)] | None
    conclusion: Literal[
        "evidence_of_benefit",
        "evidence_of_harm",
        "evidence_of_practical_equivalence",
        "inconclusive",
        "insufficient_evidence",
    ]
    reason: Identifier

    @field_validator("independent_units")
    @classmethod
    def _ordered_units(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize unit declarations without creating extra observations."""
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _shape(self) -> Self:
        """Validate unit counts and interval completeness independently of claims."""
        result = self
        if result.observed_pairs > len(result.independent_units):
            raise ValueError("Observed pairs exceed independent units")
        if len(set(result.independent_units)) != len(result.independent_units):
            raise ValueError("Repeated independent unit in report")
        if (result.ci_lower is None) != (result.ci_upper is None):
            raise ValueError("An interval needs both finite endpoints")
        if result.ci_lower is not None and result.ci_upper is not None and result.ci_lower > result.ci_upper:
            raise ValueError("Reversed comparison interval")
        if (result.raw_mean_difference is None) != (result.improvement is None):
            raise ValueError("Raw difference and improvement must be jointly available")
        if result.adjusted_p_value is not None and (result.p_value is None or result.adjusted_p_value < result.p_value):
            raise ValueError("Adjusted p-value cannot be below its raw p-value")
        return self


class ObservationReference(BaseModel):  # pylint: disable=too-few-public-methods
    """Embedded immutable raw result with verified join/digest and temporal metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    candidate_id: Identifier
    metadata: TemporalObservation
    record_key: tuple[Identifier, Identifier, Identifier, Identifier]
    record_digest: Fingerprint
    record: DomainRecord

    @field_serializer("record")
    def _serialized_record(self, record: DomainRecord) -> dict[str, Any]:
        """Preserve absent versus explicit-null fields in the original #251 digest."""
        return json.loads(record.to_json())

    @model_validator(mode="after")
    def _record_identity(self) -> Self:
        """Require the embedded immutable record to match both source references."""
        if self.record.logical_key != self.record_key or self.record.digest != self.record_digest:
            raise ValueError("Embedded raw record differs from its referenced identity or digest")
        return self
