"""Independent bounded paired inference, frozen bounds, exact fixtures and conservative protected access."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest

from coleman.evaluation.protected import (
    BoundedMeanProtocol,
    ComparisonHypothesis,
    EvaluationAudit,
    FamilyReport,
    FrozenCandidate,
    FrozenEvaluation,
    evaluate,
    freeze,
    holm_adjust,
)
from coleman.evaluation.protected.bounded_inference import (
    bounded_radius,
    bounded_tail_upper,
    outward_binary64,
    summarize_bounded,
)
from coleman.results.domain import DomainRecord, ResultProvenance
from tests.evaluation.test_protected import FIXTURES, frozen_contract, observations


def bounded_contract(**overrides):
    """Reuse authored independent mathematical units, not empirical projects or repeated seeds."""
    legacy = frozen_contract()
    values = legacy.protocol.model_dump()
    values.pop("method")
    values.update(
        {
            "schema_version": 1,
            "normal_differences_assumed": False,
            "outcome_lower": 0,
            "outcome_upper": 1,
            "minimum_units": 20,
            **overrides,
        }
    )
    protocol = BoundedMeanProtocol.model_validate(values)
    return freeze(legacy.candidates, legacy.data_fingerprint, legacy.plan, protocol, legacy.comparison_family)


def evaluate_owned(tmp_path, differences=(0.5,) * 100, *, baseline=0.25, **overrides):
    """Exercise the actual protected audit, record validation, inference and report recomputation."""
    frozen = bounded_contract(**overrides)
    rows = observations(frozen, differences, baseline=baseline)
    audit = EvaluationAudit(tmp_path)
    return evaluate(frozen, audit, lambda: rows, actor="owned-statistical-correctness"), audit, rows


def test_legacy_frozen_and_report_golden_bytes_are_unchanged():
    for contract, name in ((FrozenEvaluation, "freeze-v1.json"), (FamilyReport, "report-v1.json")):
        payload = (FIXTURES / name).read_text()
        original = contract.from_json(payload)
        assert original.to_json() + "\n" == payload
        protocol = original.frozen.protocol if isinstance(original, FamilyReport) else original.protocol
        assert protocol.method == "paired_t_holm_bonferroni_ci_v1"


def test_known_positive_negative_zero_and_minimization_effects(tmp_path):
    positive, _, _ = evaluate_owned(tmp_path / "positive")
    result = positive.results[0]
    assert result.raw_mean_difference == result.improvement == 0.5
    assert result.conclusion == "evidence_of_benefit"
    assert result.ci_lower > 0.01 and result.ci_upper < 1
    negative, _, _ = evaluate_owned(tmp_path / "negative", differences=(-0.5,) * 100, baseline=0.75)
    opposite = negative.results[0]
    assert opposite.conclusion == "evidence_of_harm"
    assert opposite.raw_mean_difference == -result.raw_mean_difference
    assert opposite.p_value == result.p_value
    assert (opposite.ci_lower, opposite.ci_upper) == (-result.ci_upper, -result.ci_lower)
    minimize, _, _ = evaluate_owned(tmp_path / "minimize", direction="minimize")
    assert minimize.results[0].conclusion == "evidence_of_harm"
    assert minimize.results[0].improvement == -0.5
    null, _, _ = evaluate_owned(tmp_path / "null", differences=(0.0,) * 100)
    assert null.results[0].p_value == 1
    assert null.results[0].conclusion == "inconclusive"
    assert null.results[0].ci_lower < 0 < null.results[0].ci_upper


def test_practical_equivalence_uses_simultaneous_interval_without_rejecting_zero(tmp_path):
    report, audit, rows = evaluate_owned(tmp_path, differences=(0.0,) * 100, practical_effect=0.3)
    result = report.results[0]
    assert result.conclusion == "evidence_of_practical_equivalence"
    assert result.p_value == result.adjusted_p_value == 1
    assert -0.3 <= result.ci_lower <= result.ci_upper <= 0.3
    path = audit.save_report(report)
    assert audit.save_report(report) == path
    assert FamilyReport.from_json(path.read_bytes()) == report
    retry = evaluate(report.frozen, audit, lambda: rows, actor="owned-retry")
    assert retry.results[0].conclusion == "inconclusive"
    assert retry.results[0].reason == "reused_holdout_no_confirmatory_claim"
    assert len(audit.history(report.frozen.protocol.holdout_id)) == 2


def test_missing_pair_is_not_zero_or_selective_complete_case_mean(tmp_path):
    frozen = bounded_contract()
    rows = observations(frozen, (0.5,) * 100, baseline=0.25, unavailable_index=17)
    report = evaluate(frozen, EvaluationAudit(tmp_path), lambda: rows, actor="owned-missing")
    result = report.results[0]
    assert result.observed_pairs == 99 and len(result.independent_units) == 100
    assert result.raw_mean_difference is result.improvement is result.p_value is None
    assert result.ci_lower is result.ci_upper is None
    assert result.reason == "incomplete_outcomes"
    assert result.conclusion == "insufficient_evidence"


def test_too_few_units_negative_metric_values_and_empty_evidence(tmp_path):
    few, _, _ = evaluate_owned(tmp_path / "few", differences=(0.5,) * 2)
    assert few.results[0].improvement == 0.5
    assert few.results[0].p_value is None
    assert few.results[0].conclusion == "insufficient_evidence"
    signed, _, _ = evaluate_owned(
        tmp_path / "signed", differences=(0.5,) * 100, baseline=-0.75, outcome_lower=-1, outcome_upper=0
    )
    assert signed.results[0].conclusion == "evidence_of_benefit"
    frozen = bounded_contract()
    empty = evaluate(frozen, EvaluationAudit(tmp_path / "empty"), tuple, actor="owned-empty")
    assert empty.results[0].observed_pairs == 0
    assert empty.results[0].conclusion == "insufficient_evidence"


def test_frozen_bounds_reject_before_statistics_and_retain_failed_access(tmp_path):
    frozen = bounded_contract()
    rows = observations(frozen, (1.0,) * 20, baseline=0.25)
    audit = EvaluationAudit(tmp_path)
    with pytest.raises(ValueError, match="frozen metric outcome bounds"):
        evaluate(frozen, audit, lambda: rows, actor="owned-outside-bounds")
    assert len(audit.history(frozen.protocol.holdout_id)) == 1
    assert not list(tmp_path.rglob("evaluation-reports/*.json"))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"outcome_lower": 1},
        {"outcome_upper": float("inf")},
        {"outcome_lower": float("nan")},
        {"dependence": "clustered"},
        {"dependence": "serial"},
        {"normal_differences_assumed": True},
        {"normal_differences_assumed": 0},
        {"minimum_units": True},
        {"arithmetic": "rational-taylor64-v2"},
        {"method": "unknown-future-method"},
    ],
)
def test_invalid_future_or_unsupported_protocol_fails_without_access(changes):
    with pytest.raises(ValueError):
        bounded_contract(**changes)


def test_protocol_roundtrip_new_golden_and_unknown_future_method():
    frozen = bounded_contract()
    assert FrozenEvaluation.from_json(frozen.to_json()) == frozen
    assert isinstance(frozen.protocol, BoundedMeanProtocol)
    assert BoundedMeanProtocol.from_json(frozen.protocol.to_json()) == frozen.protocol
    golden = Path(__file__).with_name("fixtures") / "bounded-freeze-v1.json"
    assert frozen.to_json() + "\n" == golden.read_text()
    values = json.loads(frozen.to_json())
    values["protocol"]["schema_version"] = 2
    with pytest.raises(ValueError):
        FrozenEvaluation.model_validate(values)


def test_new_bounded_report_golden_roundtrip_and_future_rejection():
    path = Path(__file__).with_name("fixtures") / "bounded-report-v1.json"
    payload = path.read_text()
    report = FamilyReport.from_json(payload)
    assert isinstance(report.frozen.protocol, BoundedMeanProtocol)
    assert report.to_json() + "\n" == payload
    assert report.results[0].raw_mean_difference == 1
    assert report.results[0].conclusion == "evidence_of_benefit"
    assert len(report.results[0].independent_units) == 10
    values = json.loads(payload)
    values["schema_version"] = 2
    with pytest.raises(ValueError):
        FamilyReport.model_validate(values)


def test_rational_exponential_upper_bound_against_independent_decimal_reference():
    with localcontext() as context:
        context.prec = 100
        for exponent in (Fraction(0), Fraction(1, 10), Fraction(3), Fraction(81, 10), Fraction(50), Fraction(1000)):
            reference = min(Decimal(1), 2 * (-Decimal(exponent.numerator) / Decimal(exponent.denominator)).exp())
            rational = bounded_tail_upper(exponent)
            actual = Decimal(rational.numerator) / Decimal(rational.denominator)
            assert actual >= reference
            rounded = outward_binary64(rational, upper=True)
            assert Fraction(rounded) >= rational


@pytest.mark.parametrize("n", [2, 10, 20, 100, 200])
@pytest.mark.parametrize("family", [1, 11])
def test_exact_rademacher_null_rejection_probability_respects_family_error(n, family):
    # This finite-law enumeration is independent of the production policy/runner.
    # Signs model independent mathematical units, not empirical project copies.
    alpha = Fraction(1, 20)
    probability = Fraction(0)
    for positive in range(n + 1):
        average = Fraction(2 * positive - n, n)
        if bounded_tail_upper(n * average * average / 2) <= alpha / family:
            probability += Fraction(math.comb(n, positive), 2**n)
    assert probability <= alpha / family
    radius = bounded_radius(n, family, alpha)
    assert 0 < radius <= 2
    if radius < 2:
        assert bounded_tail_upper(n * radius * radius / 2) <= alpha / family
        smaller = radius - Fraction(1, 2**31)
        assert bounded_tail_upper(n * smaller * smaller / 2) > alpha / family


def test_radius_and_tail_metamorphisms_and_invalid_exact_inputs():
    alpha = Fraction(1, 20)
    assert bounded_radius(200, 1, alpha) < bounded_radius(20, 1, alpha)
    assert bounded_radius(20, 11, alpha) > bounded_radius(20, 1, alpha)
    assert bounded_radius(20, 1, Fraction(1, 100)) > bounded_radius(20, 1, alpha)
    assert bounded_radius(2, 11, alpha) == 2
    assert bounded_tail_upper(Fraction(3)) > bounded_tail_upper(Fraction(4))
    for bad in (float("nan"), float("inf"), True, -1):
        with pytest.raises(ValueError):
            bounded_tail_upper(cast(Any, bad))
    for counts in ((0, 1), (1, 0), (True, 1), (1, False)):
        with pytest.raises(ValueError):
            bounded_radius(*counts, alpha)
    with pytest.raises(ValueError):
        bounded_radius(20, 1, cast(Any, 0.05))
    with pytest.raises(ValueError):
        outward_binary64(cast(Any, 0.5), upper=True)


def test_outward_subnormal_zero_and_overflow_behavior():
    tiny = Fraction.from_float(math.ulp(0.0)) / 2
    assert outward_binary64(tiny, upper=True) == math.ulp(0.0)
    assert outward_binary64(tiny, upper=False) == 0.0
    assert outward_binary64(-tiny, upper=False) == -math.ulp(0.0)
    assert math.copysign(1.0, outward_binary64(Fraction(0), upper=True)) == 1.0
    with pytest.raises(OverflowError):
        outward_binary64(Fraction(10**400), upper=True)


def test_exact_mean_extremes_and_numerically_unrepresentable_uncertainty():
    frozen = bounded_contract(outcome_lower=-1e308, outcome_upper=1e308, minimum_units=2)
    protocol = frozen.protocol
    assert isinstance(protocol, BoundedMeanProtocol)
    differences = (Fraction(1e308) * 2, Fraction(-1e308) * 2)
    result = summarize_bounded(differences, protocol, 1, 2)
    assert result.mean == 0.0
    assert result.low is result.high is result.p_value is None
    assert result.reason == "numerically_unstable_uncertainty"
    with pytest.raises(ValueError, match="mean difference exceeds"):
        summarize_bounded((differences[0],) * 2, protocol, 1, 2)
    with pytest.raises(ValueError, match="frozen outcome bounds"):
        summarize_bounded((Fraction(10),), bounded_contract().protocol, 1, 1)


def test_report_recomputation_rejects_even_one_ulp_narrowed_bound_and_forged_equivalence(tmp_path):
    report, _, _ = evaluate_owned(tmp_path)
    payload = report.model_dump()
    payload["results"][0]["ci_lower"] = math.nextafter(report.results[0].ci_lower, math.inf)
    with pytest.raises(ValueError, match="recomputed raw-record"):
        FamilyReport.model_validate(payload)
    payload = report.model_dump()
    payload["results"][0]["conclusion"] = "evidence_of_practical_equivalence"
    with pytest.raises(ValueError, match="practical equivalence"):
        FamilyReport.model_validate(payload)


def test_full_paired_family_keeps_shared_baseline_and_controls_multiplicity(tmp_path):
    original = bounded_contract()
    third = FrozenCandidate.from_configuration("C", {"policy": "owned-third-fixture"})
    frozen = freeze(
        (*original.candidates, third),
        original.data_fingerprint,
        original.plan,
        original.protocol,
        (*original.comparison_family, ComparisonHypothesis(hypothesis_id="C-B", candidate_id="C", baseline_id="B")),
    )
    rows = observations(frozen, (0.5,) * 100, baseline=0.25)
    extra = []
    for row in rows:
        if row.candidate_id != "A":
            continue
        value = row.record.model_dump()
        value["run_id"] = "run-C"
        value["provenance"] = ResultProvenance(
            producer="owned-family-fixture",
            producer_version="1",
            spec_fingerprint=third.configuration_fingerprint,
            evaluation_fingerprint=frozen.fingerprint,
        )
        value["measurements"][0]["value"] = 0.5
        extra.append(type(row)(candidate_id="C", metadata=row.metadata, record=DomainRecord.model_validate(value)))
    report = evaluate(frozen, EvaluationAudit(tmp_path), lambda: (*rows, *extra), actor="owned-full-family")
    assert len(report.results) == 2
    assert all(len(result.independent_units) == 100 for result in report.results)
    assert all(result.p_value is not None for result in report.results)
    adjusted = holm_adjust(tuple(cast(float, result.p_value) for result in report.results), outward=True)
    assert tuple(result.adjusted_p_value for result in report.results) == adjusted
    single, _, _ = evaluate_owned(tmp_path / "single")
    assert report.results[0].ci_lower < single.results[0].ci_lower
    assert report.results[0].ci_upper > single.results[0].ci_upper
    assert report.results[0].conclusion == "evidence_of_benefit"
    assert report.results[1].conclusion == "inconclusive"


def test_repeated_seed_or_build_labels_cannot_manufacture_independent_units(tmp_path):
    frozen = bounded_contract()
    rows = observations(frozen, (0.5,) * 100, baseline=0.25)
    repeated = tuple(
        type(row).model_validate(
            {
                **row.model_dump(),
                "metadata": {
                    **row.metadata.model_dump(),
                    "group_id": "one-owned-related-family",
                },
            }
        )
        for row in rows
    )
    audit = EvaluationAudit(tmp_path)
    with pytest.raises(ValueError, match="seeds or revisions are not independent"):
        evaluate(frozen, audit, lambda: repeated, actor="owned-seed-replication-guard")
    assert len(audit.history(frozen.protocol.holdout_id)) == 1


def test_exact_affine_units_and_row_permutations_preserve_normalized_evidence():
    original = bounded_contract(outcome_lower=-1, outcome_upper=1).protocol
    assert isinstance(original, BoundedMeanProtocol)
    differences = (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 4)) * 25
    first = summarize_bounded(differences, original, 3, len(differences))
    assert first.mean is not None and first.low is not None and first.high is not None
    permuted = summarize_bounded(tuple(reversed(differences)), original, 3, len(differences))
    assert first == permuted
    scaled = BoundedMeanProtocol.model_validate(
        {
            **original.model_dump(),
            "unit": "owned-scaled-score",
            "outcome_lower": 24,
            "outcome_upper": 40,
            "practical_effect": original.practical_effect * 8,
        }
    )
    second = summarize_bounded(tuple(value * 8 for value in differences), scaled, 3, len(differences))
    assert second.p_value == first.p_value
    assert second.mean == first.mean * 8
    assert second.low == first.low * 8
    assert second.high == first.high * 8
    with localcontext() as context:
        context.prec = 2
        assert summarize_bounded(differences, original, 3, len(differences)) == first


def test_new_method_keeps_strict_units_and_old_reports_cannot_gain_equivalence(tmp_path):
    frozen = bounded_contract()
    rows = list(observations(frozen, (0.5,) * 20, baseline=0.25))
    invalid = rows[0].model_dump()
    invalid["record"]["measurements"][0]["unit"] = "s"
    rows[0] = type(rows[0]).model_validate(invalid)
    audit = EvaluationAudit(tmp_path / "wrong-unit")
    with pytest.raises(ValueError, match="frozen metric contract"):
        evaluate(frozen, audit, lambda: tuple(rows), actor="owned-unit-check")
    old = frozen_contract(practical_effect=1)
    old_rows = observations(old)
    report = evaluate(old, EvaluationAudit(tmp_path / "legacy"), lambda: old_rows, actor="owned-legacy")
    assert report.results[0].conclusion == "inconclusive"
    payload = report.model_dump()
    payload["results"][0]["conclusion"] = "evidence_of_practical_equivalence"
    with pytest.raises(ValueError, match="practical equivalence"):
        FamilyReport.model_validate(payload)


def test_bounded_holm_cannot_round_a_conservative_probability_downward():
    probabilities = (0.3, 0.4, 0.9)
    legacy = holm_adjust(probabilities)
    adjusted = holm_adjust(probabilities, outward=True)
    exact_first = 3 * Fraction(probabilities[0])
    assert Fraction(legacy[0]) < exact_first
    assert Fraction(adjusted[0]) >= exact_first
    assert adjusted == (0.9, 0.9, 0.9)
    assert holm_adjust((), outward=True) == ()
    with pytest.raises(ValueError, match="selector"):
        holm_adjust(probabilities, outward=cast(Any, "true"))
