"""Versioned golden grid and nested domain serialization preserve replay bytes."""

from pathlib import Path

from coleman.corpus.baseline_eligibility import baseline_eligibility
from coleman.corpus.baseline_results import BaselineCellResult, BaselineMatrixReport, baseline_coordinates
from tests.corpus.baseline_support import baseline_plan, change


def incompatible_report():
    """Construct an owned all-unavailable source; no actual execution or provenance is asserted."""
    original = baseline_plan()
    builds = tuple(
        change(
            build,
            tests=tuple(
                change(test, failure=change(test.failure, value=None, status="unavailable")) for test in build.tests
            ),
        )
        for build in original.source.builds
    )
    plan = change(original, source=change(original.source, builds=builds), seeds=(42,), budgets=(original.budgets[0],))
    eligibility = {value.method: value for value in baseline_eligibility(plan)}
    cells = tuple(
        BaselineCellResult(
            **coordinate.model_dump(),
            state="incompatible",
            reasons=eligibility[coordinate.method].reasons,
            native_run_id=None,
            observations=(),
            mathematical_observations=(),
            attempts=(),
            unclosed_attempts=0,
            resource_evidence="not_applicable",
        )
        for coordinate in baseline_coordinates(plan)
    )
    return BaselineMatrixReport(schema_version=1, plan=plan, producer_fingerprint="sha256:" + "a" * 64, cells=cells)


def test_versioned_complete_incompatible_grid_golden():
    report = incompatible_report()
    golden = Path(__file__).parent / "fixtures/baseline-matrix-v1.json"
    assert report.to_json() + "\n" == golden.read_text()
    assert BaselineMatrixReport.from_json(golden.read_bytes()) == report
    assert report.complete
    assert (
        report.mathematical_fingerprint == BaselineMatrixReport.from_json(golden.read_bytes()).mathematical_fingerprint
    )
