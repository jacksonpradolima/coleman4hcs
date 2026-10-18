"""Independent corpus metric denominators, native projection and mutation-sensitive guards."""

import itertools
import struct
from datetime import datetime
from decimal import Decimal
from fractions import Fraction

import pytest

from coleman.bandit.metric_bandit import EvaluationMetricBandit
from coleman.corpus import CIReferenceFixture, bundled_fixture_bytes, tiny_ci_reference, validate_bundled_source
from coleman.evaluation import NAPFDVerdictMetric
from coleman.pilot import SECONDS, PilotBuild, PilotTest
from coleman.selection import Quantity


def q(value):
    return Quantity(value=Decimal(str(value)), semantics=SECONDS)


def fixture():
    return CIReferenceFixture.from_json(bundled_fixture_bytes("ci-serial-boundary-v1"))


def test_apfd_full_universe_and_selected_horizon_napfd_are_distinct_estimands():
    build = fixture().build
    full = tiny_ci_reference(build, ("t1", "t2", "t3"), q(7))
    partial = tiny_ci_reference(build, ("t1", "t2", "t3"), q(3))
    # APFD: 1 - (2 + 3)/(2*3) + 1/(2*3). The unit is a failing test,
    # not a mapped unique fault. Full execution makes native NAPFD equal APFD.
    assert full.napfd.value == Fraction(1, 3)
    assert partial.napfd.value == Fraction(1, 4)
    # Qu et al. use a declared executed horizon; renormalizing n to two here
    # defines a different 1/8 estimand. Old COLEMAN results retain n=3.
    selected_horizon = Fraction(1, 2) - Fraction(2, 2 * 2) + Fraction(1, 2) / (2 * 2)
    assert selected_horizon == Fraction(1, 8)
    assert selected_horizon != partial.napfd.value
    # APFDc's full-universe cost denominator includes the excluded failing t3.
    assert partial.apfdc.value == Fraction(5, 14)
    assert full.apfdc.value == Fraction(1, 2)
    assert partial.recall.value == Fraction(1, 2)


@pytest.mark.parametrize("seconds", ["0.1", "0.2", "0.3", "37.11"])
def test_actual_bandit_projection_and_original_decimal_boundary_are_both_retained(seconds):
    metric = NAPFDVerdictMetric()
    metric.update_budget("fixed_time", float(seconds))
    metric.update_available_time(float(seconds))
    arms = [{"Name": "failing-test", "Duration": float(seconds), "CalcPrio": 0, "Verdict": 1}]
    bandit = EvaluationMetricBandit(arms, metric)
    projected = struct.unpack("!f", struct.pack("!f", float(seconds)))[0]
    assert bandit.arms["Duration"].to_list() == [projected]
    metric = bandit.pull(["failing-test"])
    assert metric.scheduled_testcases == (["failing-test"] if projected <= float(seconds) else [])
    source = PilotBuild(
        build_id="decimal-source",
        decided_at=datetime.fromisoformat("2026-01-05T00:00:00Z"),
        outcomes_available_at=datetime.fromisoformat("2026-01-05T01:00:00Z"),
        tests=(PilotTest(test_id="failing-test", duration=q(seconds), outcome="failed"),),
        baseline_order=("failing-test",),
    )
    exact = tiny_ci_reference(source, source.baseline_order, q(seconds))
    assert exact.selected_ids == ("failing-test",)
    assert exact.runner_work.value == Decimal(seconds)
    # A boundary disagreement is evidence of different quantity definitions,
    # never hidden by tolerances or by rewriting the original source decimals.
    assert metric.detected_failures == int(projected <= float(seconds))


def test_all_orders_and_integer_budget_boundaries_match_real_native_metric():
    build = fixture().build
    tests = {test.test_id: test for test in build.tests}
    for order, budget in itertools.product(itertools.permutations(tests), range(8)):
        reference = tiny_ci_reference(build, order, q(budget))
        metric = NAPFDVerdictMetric()
        metric.update_budget("fixed_time", float(budget))
        metric.update_available_time(float(budget))
        metric.evaluate(
            [
                {
                    "Name": name,
                    "Duration": float(tests[name].duration.value),
                    "Verdict": int(tests[name].outcome == "failed"),
                }
                for name in order
            ]
        )
        assert reference.napfd.value is not None
        assert reference.apfdc.value is not None
        assert tuple(metric.scheduled_testcases) == reference.selected_ids
        assert metric.fitness == pytest.approx(float(reference.napfd.value), rel=0, abs=1e-15)
        assert metric.cost == pytest.approx(float(reference.apfdc.value), rel=0, abs=1e-15)


@pytest.mark.parametrize("mutation", ["zero", "selected_only_denominator", "censored_as_observed_zero"])
def test_actual_production_mutations_fail_existing_independent_fixture_guards(monkeypatch, mutation):
    original = NAPFDVerdictMetric.compute_metrics

    def fake(self, *args):
        original(self, *args)
        if mutation == "zero":
            self.fitness = 0
        elif mutation == "selected_only_denominator":
            self.cost = self.cost * 2
        else:
            self.ttf_duration = 0

    # Positive control: real component passes before replacing its computation.
    report = validate_bundled_source("ci-serial-boundary-v1")
    assert len(report.observations) > 0
    monkeypatch.setattr(NAPFDVerdictMetric, "compute_metrics", fake)
    with pytest.raises(ValueError, match="differs|disagree"):
        validate_bundled_source("ci-serial-boundary-v1")
