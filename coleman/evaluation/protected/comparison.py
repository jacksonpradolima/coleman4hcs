"""Audited paired comparisons on independent units with explicit normal assumptions."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Annotated, Any, Literal, NamedTuple, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy import stats

from coleman.evaluation.protected.audit import AccessReceipt, EvaluationAudit
from coleman.evaluation.protected.contracts import (
    ComparisonHypothesis,
    EvaluationContract,
    EvaluationProtocol,
    FrozenEvaluation,
)
from coleman.evaluation.protected.temporal import TemporalObservation
from coleman.results.domain import (
    DomainRecord,
    Fingerprint,
    Finite,
    Identifier,
    Measurement,
    ResultProvenance,
    fingerprint,
)


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
    conclusion: Literal["evidence_of_benefit", "evidence_of_harm", "inconclusive", "insufficient_evidence"]
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
    """Immutable join/digest for one raw result, retaining point-in-time metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    candidate_id: Identifier
    metadata: TemporalObservation
    record_key: tuple[Identifier, Identifier, Identifier, Identifier]
    record_digest: Fingerprint


class FamilyReport(EvaluationContract):
    """Versioned report tied to an immutable freeze and pre-load access receipt.

    Receipt history determines confirmatory eligibility, independently of point
    estimates. This report never activates a policy. The frozen protocol declares
    the metric units, practical threshold, independent-unit meaning and assumptions.
    """

    frozen: FrozenEvaluation
    access: AccessReceipt
    results: tuple[ComparisonResult, ...]
    inputs: tuple[ObservationReference, ...]

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Require exact freeze/scope/data identity and the complete predeclared family."""
        if (
            self.access.evaluation_fingerprint != self.frozen.fingerprint
            or self.access.holdout_id != self.frozen.protocol.holdout_id
            or self.access.data_fingerprint != self.frozen.data_fingerprint
        ):
            raise ValueError("Report receipt does not match its frozen evaluation")
        if tuple(item.hypothesis_id for item in self.results) != tuple(
            item.hypothesis_id for item in self.frozen.comparison_family
        ):
            raise ValueError("Report must contain the complete frozen comparison family")
        references: dict[tuple[str, str], ObservationReference] = {}
        for reference in self.inputs:
            self.frozen.candidate(reference.candidate_id)
            key = reference.candidate_id, reference.metadata.group_id
            if key in references:
                raise ValueError("Duplicate report input unit/candidate")
            decision = self.frozen.plan.decide(reference.metadata)
            if decision.partition != "test" or not decision.eligible:
                raise ValueError("Ineligible report input metadata")
            references[key] = reference
        for hypothesis, result in zip(self.frozen.comparison_family, self.results, strict=True):
            for candidate_id in (hypothesis.candidate_id, hypothesis.baseline_id):
                units = {unit for name, unit in references if name == candidate_id}
                if units != set(result.independent_units):
                    raise ValueError("Report unit counts do not match its raw-input references")
            for unit in result.independent_units:
                left = references[hypothesis.candidate_id, unit]
                right = references[hypothesis.baseline_id, unit]
                if left.metadata != right.metadata:
                    raise ValueError("Report input metadata does not match across paired units")
        return self

    @model_validator(mode="after")
    def _inference(self) -> Self:
        """Reject forged confirmatory claims on reused scopes or unsupported assumptions."""
        for result in self.results:
            if result.conclusion in {"evidence_of_benefit", "evidence_of_harm"}:
                self._check_claim(result)
        return self

    def _check_claim(self, result: ComparisonResult) -> None:
        """Validate access, independent observations and family-wise practical evidence."""
        protocol = self.frozen.protocol
        if (
            not self.confirmatory_eligible
            or protocol.dependence != "independent"
            or not protocol.normal_differences_assumed
        ):
            raise ValueError("Unsupported confirmatory access or assumptions")
        if result.observed_pairs != len(result.independent_units) or result.observed_pairs < protocol.minimum_units:
            raise ValueError("Insufficient independent observations for confirmatory conclusion")
        if (
            result.adjusted_p_value is None
            or result.adjusted_p_value > protocol.alpha
            or result.ci_lower is None
            or result.ci_upper is None
        ):
            raise ValueError("Unsupported confirmatory uncertainty")
        if result.conclusion == "evidence_of_benefit" and result.ci_lower <= protocol.practical_effect:
            raise ValueError("Benefit does not exceed the practical threshold")
        if result.conclusion == "evidence_of_harm" and result.ci_upper >= -protocol.practical_effect:
            raise ValueError("Harm does not exceed the practical threshold")

    @property
    def confirmatory_eligible(self) -> bool:
        """Return whether this API recorded the first protected confirmatory access."""
        return (
            self.access.partition == "test"
            and self.access.purpose == "confirmatory"
            and self.access.prior_test_accesses == 0
        )

    def domain_records(self, run_id: str, *, namespace: str = "ci.protected_comparisons") -> tuple[DomainRecord, ...]:
        """Convert estimates to #251 records, retaining nulls, signs, units and provenance.

        Each access/hypothesis has a distinct logical decision identity; repeating
        serialization or sink publication is idempotent. Unknown uncertainty is
        explicitly unavailable. Access IDs/sequences never count as independent
        evidence. No transformed learning reward is created from inference.
        """
        validated = type(self).model_validate(self)
        unit = validated.frozen.protocol.unit
        output = []
        for result in validated.results:
            quantities = (
                ("raw_mean_difference", result.raw_mean_difference),
                ("improvement", result.improvement),
                ("ci_lower", result.ci_lower),
                ("ci_upper", result.ci_upper),
            )
            measurements = tuple(
                Measurement(
                    name=name,
                    value=value,
                    unit=unit,
                    meaning=f"{validated.frozen.protocol.method}:{name}",
                    direction="none",
                    status="observed" if value is not None else "unavailable",
                )
                for name, value in quantities
            )
            measurements += (
                Measurement(
                    name="adjusted_p_value",
                    value=result.adjusted_p_value,
                    unit="1",
                    meaning="holm-family-wise-p-value-v1",
                    direction="none",
                    status="observed" if result.adjusted_p_value is not None else "unavailable",
                ),
            )
            measurements += (
                Measurement(
                    name="independent_units",
                    value=float(len(result.independent_units)),
                    unit="count",
                    meaning=validated.frozen.protocol.independent_unit,
                    direction="none",
                    status="observed",
                    minimum=0.0,
                ),
            )
            output.append(
                DomainRecord(
                    schema_version=1,
                    namespace=namespace,
                    run_id=run_id,
                    execution_id=validated.frozen.fingerprint,
                    decision_id=fingerprint([validated.access.sequence, result.hypothesis_id]),
                    data_fingerprint=validated.frozen.data_fingerprint,
                    available_at=validated.access.requested_at,
                    provenance=ResultProvenance(
                        producer="coleman.protected-evaluation",
                        producer_version="1",
                        spec_fingerprint=validated.frozen.fingerprint,
                        evaluation_fingerprint=validated.frozen.fingerprint,
                    ),
                    measurements=measurements,
                )
            )
        return tuple(output)


def holm_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Return Holm-Bonferroni adjusted p-values controlling family-wise error.

    Uses all predeclared hypotheses, with unavailable tests supplied as p=1 by
    the evaluator. Finite values in [0,1] are required. Sorting is internal and
    original ordering is preserved; empty input returns an empty tuple. No RNG,
    independence-between-hypotheses or additional optional dependency is needed.
    """
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or value > 1
        for value in p_values
    ):
        raise ValueError("P-values must be finite and within [0,1]")
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda index: p_values[index])):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


def _estimate(
    frozen: FrozenEvaluation,
    hypothesis: ComparisonHypothesis,
    observations: dict[str, dict[str, PairedObservation]],
) -> ComparisonResult:
    """Align independent units and estimate one pair without complete-case selection."""
    left = observations[hypothesis.candidate_id]
    right = observations[hypothesis.baseline_id]
    if set(left) != set(right):
        raise ValueError("Candidates must have the same paired independent units")
    units = tuple(sorted(left))
    differences = []
    for unit in units:
        if left[unit].metadata != right[unit].metadata:
            raise ValueError("Paired unit metadata must agree across candidates")
        first = left[unit].record.measurement(frozen.protocol.metric_name)
        second = right[unit].record.measurement(frozen.protocol.metric_name)
        left[unit].record.require_compatible(right[unit].record, frozen.protocol.metric_name)
        if first.value is not None and second.value is not None:
            difference = first.value - second.value
            if not math.isfinite(difference):
                raise ValueError("Paired difference exceeds finite numeric range")
            differences.append(difference)
    summary = _summarize(tuple(differences), frozen.protocol, len(frozen.comparison_family), len(units))
    return ComparisonResult(
        hypothesis_id=hypothesis.hypothesis_id,
        independent_units=units,
        observed_pairs=len(differences),
        raw_mean_difference=summary.mean,
        improvement=summary.improvement,
        ci_lower=summary.low,
        ci_upper=summary.high,
        p_value=summary.p_value,
        adjusted_p_value=None,
        conclusion="insufficient_evidence",
        reason=summary.reason,
    )


class _Statistics(NamedTuple):
    """Finite scalar summary in metric units; unavailable uncertainty remains null."""

    mean: float | None = None
    improvement: float | None = None
    low: float | None = None
    high: float | None = None
    p_value: float | None = None
    reason: str = "too_few_independent_units"


def _summarize(
    differences: tuple[float, ...],
    protocol: EvaluationProtocol,
    family_size: int,
    total_units: int,
) -> _Statistics:
    """Estimate scaled paired moments with Bonferroni simultaneous t intervals."""
    if len(differences) != total_units:
        return _Statistics(reason="incomplete_outcomes")
    if not differences:
        return _Statistics()
    scale = max(abs(value) for value in differences) or 1.0
    normalized = np.asarray(differences, dtype=float) / scale
    mean = float(np.mean(normalized)) * scale
    improvement = mean if protocol.direction == "maximize" else -mean
    if len(differences) < protocol.minimum_units:
        return _Statistics(mean=mean, improvement=improvement)
    if len(set(normalized.tolist())) == 1:
        return _Statistics(mean=mean, improvement=improvement, reason="degenerate_difference_variance")
    test = stats.ttest_1samp(normalized if protocol.direction == "maximize" else -normalized, popmean=0.0)
    interval = test.confidence_interval(confidence_level=1 - protocol.alpha / family_size)
    low, high, p_value = float(interval.low) * scale, float(interval.high) * scale, float(cast(Any, test).pvalue)
    if not all(math.isfinite(value) for value in (low, high, p_value)):
        return _Statistics(mean=mean, improvement=improvement, reason="numerically_unstable_uncertainty")
    return _Statistics(mean=mean, improvement=improvement, low=low, high=high, p_value=p_value, reason="estimated")


def _validate_observations(
    frozen: FrozenEvaluation,
    rows: tuple[PairedObservation, ...],
) -> dict[str, dict[str, PairedObservation]]:
    """Check full frozen provenance and temporal eligibility before calculating values."""
    by_candidate: dict[str, dict[str, PairedObservation]] = {
        candidate.candidate_id: {} for candidate in frozen.candidates
    }
    metadata: dict[tuple[str, str], TemporalObservation] = {}
    for raw in rows:
        row = PairedObservation.model_validate(raw)
        candidate = frozen.candidate(row.candidate_id)
        if row.record.data_fingerprint != frozen.data_fingerprint:
            raise ValueError("Observation dataset differs from frozen data")
        if row.record.provenance.evaluation_fingerprint != frozen.fingerprint:
            raise ValueError("Observation evaluation context was not frozen")
        if row.record.provenance.spec_fingerprint != candidate.configuration_fingerprint:
            raise ValueError("Observation candidate configuration differs from freeze")
        decision = frozen.plan.decide(row.metadata)
        if decision.partition != "test" or not decision.eligible:
            raise ValueError(f"Ineligible protected observation: {decision.reason}")
        key = row.metadata.observation_id, row.metadata.revision_id
        previous = metadata.setdefault(key, row.metadata)
        if previous != row.metadata:
            raise ValueError("Observation revision metadata differs across candidates")
        unit = row.metadata.group_id
        if unit in by_candidate[row.candidate_id]:
            raise ValueError("Repeated candidate/unit: seeds or revisions are not independent evidence")
        measurement = row.record.measurement(frozen.protocol.metric_name)
        if (measurement.unit, measurement.meaning, measurement.direction) != (
            frozen.protocol.unit,
            frozen.protocol.meaning,
            frozen.protocol.direction,
        ):
            raise ValueError("Measurement differs from the frozen metric contract")
        by_candidate[row.candidate_id][unit] = row
    frozen.plan.split(tuple(metadata.values()))
    return by_candidate


def evaluate(
    frozen: FrozenEvaluation,
    audit: EvaluationAudit,
    load_observations: Callable[[], tuple[PairedObservation, ...]],
    *,
    actor: str,
) -> FamilyReport:
    """Reserve protected access, load typed records and evaluate the full frozen family.

    Paired t inference requires declared independent units and normal differences;
    serial/clustered observations and undeclared normality assumptions raise
    ``NotImplementedError`` before outcome loading. Complete pairing is required;
    repeated seed rows are rejected. Missing outcomes produce insufficient evidence
    rather than selective complete-case tests. Holm p correction and Bonferroni
    simultaneous intervals use the entire predeclared family. Reuse is durably
    recorded and prevents a confirmatory conclusion, even after candidate changes.
    Loader/validation failures leave access reservations visible for honest retries.
    """
    frozen = FrozenEvaluation.model_validate(frozen)
    if frozen.protocol.dependence != "independent" or not frozen.protocol.normal_differences_assumed:
        raise NotImplementedError("Paired t inference requires independent units and declared normal differences")
    receipt = audit.access(frozen, partition="test", purpose="confirmatory", actor=actor)
    observations = _validate_observations(frozen, load_observations())
    results = [_estimate(frozen, hypothesis, observations) for hypothesis in frozen.comparison_family]
    adjusted = holm_adjust(tuple(result.p_value if result.p_value is not None else 1.0 for result in results))
    updated = []
    for result, corrected in zip(results, adjusted, strict=True):
        payload = result.model_dump()
        if result.p_value is not None and result.ci_lower is not None and result.ci_upper is not None:
            payload["adjusted_p_value"] = corrected
            payload["conclusion"] = "inconclusive"
            if receipt.prior_test_accesses:
                payload["reason"] = "reused_holdout_no_confirmatory_claim"
            elif corrected <= frozen.protocol.alpha:
                if result.ci_lower > frozen.protocol.practical_effect:
                    payload["conclusion"] = "evidence_of_benefit"
                elif result.ci_upper < -frozen.protocol.practical_effect:
                    payload["conclusion"] = "evidence_of_harm"
        updated.append(ComparisonResult.model_validate(payload))
    references = tuple(
        ObservationReference(
            candidate_id=candidate_id,
            metadata=row.metadata,
            record_key=row.record.logical_key,
            record_digest=row.record.digest,
        )
        for candidate_id, units in sorted(observations.items())
        for _, row in sorted(units.items())
    )
    return FamilyReport(schema_version=1, frozen=frozen, access=receipt, results=tuple(updated), inputs=references)
