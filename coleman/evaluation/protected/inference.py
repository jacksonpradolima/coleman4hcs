"""Pure paired calculations over a predeclared finite hypothesis family."""

from __future__ import annotations

import math
from typing import Any, NamedTuple, cast

import numpy as np
from scipy import stats

from coleman.evaluation.protected.audit import AccessReceipt
from coleman.evaluation.protected.contracts import ComparisonHypothesis, EvaluationProtocol, FrozenEvaluation
from coleman.evaluation.protected.observations import ComparisonResult, PairedObservation
from coleman.evaluation.protected.temporal import TemporalObservation


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
    configuration_digests = {
        candidate.candidate_id: candidate.configuration_fingerprint for candidate in frozen.candidates
    }
    evaluation_digest = frozen.fingerprint
    metadata: dict[tuple[str, str], TemporalObservation] = {}
    record_keys: set[tuple[str, tuple[str, str, str, str]]] = set()
    for raw in rows:
        row = PairedObservation.model_validate(raw)
        if row.candidate_id not in configuration_digests:
            raise ValueError("Observation candidate is not frozen")
        _validate_record(frozen, row, evaluation_digest, configuration_digests[row.candidate_id])
        key = row.metadata.observation_id, row.metadata.revision_id
        previous = metadata.setdefault(key, row.metadata)
        if previous != row.metadata:
            raise ValueError("Observation revision metadata differs across candidates")
        unit = row.metadata.group_id
        if unit in by_candidate[row.candidate_id]:
            raise ValueError("Repeated candidate/unit: seeds or revisions are not independent evidence")
        record_key = row.candidate_id, row.record.logical_key
        if record_key in record_keys:
            raise ValueError("Repeated raw result identity across independent units")
        record_keys.add(record_key)
        by_candidate[row.candidate_id][unit] = row
    frozen.plan.split(tuple(metadata.values()))
    return by_candidate


def _validate_record(
    frozen: FrozenEvaluation, row: PairedObservation, evaluation_digest: str, configuration_digest: str
) -> None:
    """Validate one result against cached frozen identities and the temporal metric contract."""
    if row.record.data_fingerprint != frozen.data_fingerprint:
        raise ValueError("Observation dataset differs from frozen data")
    if row.record.provenance.evaluation_fingerprint != evaluation_digest:
        raise ValueError("Observation evaluation context was not frozen")
    if row.record.provenance.spec_fingerprint != configuration_digest:
        raise ValueError("Observation candidate configuration differs from freeze")
    decision = frozen.plan.decide(row.metadata)
    if decision.partition != "test" or not decision.eligible:
        raise ValueError(f"Ineligible protected observation: {decision.reason}")
    measurement = row.record.measurement(frozen.protocol.metric_name)
    if (measurement.unit, measurement.meaning, measurement.direction) != (
        frozen.protocol.unit,
        frozen.protocol.meaning,
        frozen.protocol.direction,
    ):
        raise ValueError("Measurement differs from the frozen metric contract")


def _conclude(
    frozen: FrozenEvaluation, result: ComparisonResult, receipt: AccessReceipt, corrected: float
) -> ComparisonResult:
    """Apply full-family corrected evidence only to a first protected access."""
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
    return ComparisonResult.model_validate(payload)


def _compute_results(
    frozen: FrozenEvaluation, observations: dict[str, dict[str, PairedObservation]], receipt: AccessReceipt
) -> tuple[ComparisonResult, ...]:
    """Calculate the complete frozen family without access reservations or publication."""
    results = tuple(_estimate(frozen, hypothesis, observations) for hypothesis in frozen.comparison_family)
    adjusted = holm_adjust(tuple(result.p_value if result.p_value is not None else 1.0 for result in results))
    return tuple(
        _conclude(frozen, result, receipt, corrected) for result, corrected in zip(results, adjusted, strict=True)
    )
