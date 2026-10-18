"""Owned stress cases and independent integer event checks; no native metric is called."""

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest

from coleman.corpus.stress_contracts import StressCase, StressFactors
from coleman.corpus.stress_world import generate_build
from coleman.pilot.contracts import SECONDS, PilotBuild, PilotTest
from coleman.selection.contracts import Quantity


def case(*, seed=11, builds=4, **factors):
    return StressCase(schema_version=1, seed=seed, builds=builds, factors=StressFactors.model_validate(factors))


def pilot(case_value, build):
    """Project <=16 complete truth rows only; never relax the existing tiny/pilot bounds."""
    from coleman.corpus.stress_feedback import decision_time

    rows = generate_build(case_value, build)
    assert len(rows) <= 16
    return PilotBuild(
        build_id=f"stress-{build}",
        decided_at=decision_time(case_value, build),
        outcomes_available_at=decision_time(case_value, build + 1),
        tests=tuple(
            PilotTest(
                test_id=r.execution_name,
                duration=Quantity(value=Decimal(r.duration_seconds), semantics=SECONDS),
                outcome="failed" if r.verdict else "passed",
            )
            for r in rows
        ),
        baseline_order=tuple(r.execution_name for r in rows),
    )


def check_native(result, case_value):
    """Independently verify every actual native ranking, prefix, denominator and exact tiny metric."""
    from coleman.corpus.reference import tiny_ci_reference

    rows = pl.read_parquet(str(Path(result.artifacts_dir) / "results/**/*.parquet"), hive_partitioning=True)
    for record in rows.iter_rows(named=True):
        build = record["step"] - 1
        tests = {r.execution_name: r for r in generate_build(case_value, build)}
        order = json.loads(record["prioritization_order_top_k"])
        assert len(order) == len(set(order)) == len(tests)
        assert set(order) == tests.keys()
        total = sum(r.duration_seconds for r in tests.values())
        ratio = Fraction(1, 16) if case_value.factors.budget == "severe" else Fraction(1)
        occupied, selected = 0, []
        for name in order:
            duration = tests[name].duration_seconds
            if occupied + duration > total * ratio:
                break
            occupied += duration
            selected.append(name)
        detected = sum(tests[name].verdict for name in selected)
        failing = sum(r.verdict for r in tests.values())
        assert record["tests_ran"] == len(selected)
        assert record["tests_not_ran"] == len(tests) - len(selected)
        assert record["detected"] == detected
        assert record["missed"] == failing - detected
        assert record["total_build_duration"] == total
        if len(tests) <= 16:
            reference = tiny_ci_reference(
                pilot(case_value, build),
                tuple(order),
                Quantity(value=Decimal(total) / Decimal(ratio.denominator), semantics=SECONDS),
            )
            assert reference.runner_work.value == occupied
            if failing:
                assert reference.napfd.value is not None
                assert reference.apfdc.value is not None
                # The reference stays rational; native final metric arithmetic is binary64.
                assert record["fitness"] == pytest.approx(float(reference.napfd.value), rel=0, abs=1e-15)
                assert record["cost"] == pytest.approx(float(reference.apfdc.value), rel=0, abs=1e-15)
    coordinates = [(r["scenario"], r["policy"], r["step"], r["budget_value"]) for r in rows.iter_rows(named=True)]
    assert len(set(coordinates)) == rows.height
    return rows
