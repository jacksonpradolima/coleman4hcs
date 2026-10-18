"""Independent quantity/reference contracts and complete-grid rejection cases."""

from fractions import Fraction

import pytest

from coleman.corpus.baseline_reference import baseline_reference
from coleman.corpus.baseline_results import (
    BaselineAttempt,
    BaselineFilePin,
    BaselineMatrixReport,
    baseline_cell_id,
    baseline_coordinates,
    baseline_measurement,
)
from tests.corpus.baseline_support import baseline_plan, change


@pytest.mark.parametrize("value", [0.0, 2.0, None])
def test_quantity_roundtrip_preserves_zero_and_unavailable(value):
    quantity = baseline_measurement("selected_runner_seconds", value)
    assert quantity.value == value
    assert quantity.status == ("unavailable" if value is None else "observed")
    assert quantity.unit == "s"
    assert quantity.minimum == 0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_physical_quantity_rejects(value):
    with pytest.raises(ValueError):
        baseline_measurement("selected_runner_seconds", value)


def test_unknown_quantity_and_cell_dimensions_reject():
    plan = baseline_plan()
    cell = baseline_coordinates(plan)[0]
    assert baseline_cell_id(plan, cell.method, cell.seed, cell.budget) == cell.cell_id
    assert len({value.cell_id for value in baseline_coordinates(plan)}) == 52
    with pytest.raises(ValueError):
        baseline_measurement("undeclared", 0.0)
    with pytest.raises(ValueError):
        baseline_cell_id(plan, cell.method, 999, cell.budget)


@pytest.mark.parametrize("path", ["/absolute", "../outside", "a/../b", "a//b", "a/./b", "a\\b", "a:b", "a\x00b", "."])
def test_native_pin_rejects_unsafe_path(path):
    with pytest.raises(ValueError):
        BaselineFilePin(path=path, sha256="sha256:" + "a" * 64, size_bytes=1)


def test_independent_midpoint_reference_and_censored_time():
    build = baseline_plan().source.builds[1]
    tests = tuple(
        change(test, failure=change(test.failure, value=float(test.test_id in {"b", "c"}))) for test in build.tests
    )
    build = change(build, tests=tests)
    result = baseline_reference(build, ("a", "b", "c"), 7.0)
    assert result.apfdc == 0.5
    assert result.napfd == float(Fraction(1, 3))
    assert result.first_failure == 3.0
    assert result.selected_work == 7.0
    censored = baseline_reference(build, ("a", "b", "c"), 1.0)
    assert censored.first_failure is None
    assert censored.recall == censored.apfdc == censored.napfd == 0.0
    assert censored.selected == ("a",)


def test_zero_failure_ratios_are_unavailable_and_zero_detection_time_is_valid():
    build = baseline_plan().source.builds[0]
    absent = baseline_reference(build, ("a", "b", "c"), 7.0)
    assert absent.napfd is absent.apfdc is absent.recall is absent.first_failure is None
    first = build.tests[0]
    zero = change(first, duration=change(first.duration, value=0.0), failure=change(first.failure, value=1.0))
    build = change(build, tests=(zero, *build.tests[1:]))
    result = baseline_reference(build, ("a", "b", "c"), 0.0)
    assert result.first_failure == 0.0
    assert result.selected_work == 0.0
    assert result.selected == ("a",)


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf"), True])
def test_reference_invalid_budget_rejects(budget):
    with pytest.raises(ValueError):
        baseline_reference(baseline_plan().source.builds[0], ("a", "b", "c"), budget)


@pytest.mark.parametrize("ranking", [("a", "a", "c"), ("a", "b"), ("a", "b", "unknown")])
def test_reference_membership_rejects(ranking):
    with pytest.raises(ValueError):
        baseline_reference(baseline_plan().source.builds[0], ranking, 7.0)


def test_binary_cost_scaling_metamorphism():
    build = baseline_plan().source.builds[1]
    reference = baseline_reference(build, ("c", "b", "a"), 6.0)
    assert reference.first_failure is not None
    for power in (-10, -1, 1, 12):
        scale = 2.0**power
        tests = tuple(
            change(test, duration=change(test.duration, value=test.duration.value * scale)) for test in build.tests
        )
        scaled = baseline_reference(change(build, tests=tests), ("c", "b", "a"), 6.0 * scale)
        assert scaled.selected == reference.selected
        assert (scaled.napfd, scaled.apfdc, scaled.recall) == (reference.napfd, reference.apfdc, reference.recall)
        assert scaled.selected_work == reference.selected_work * scale
        assert scaled.first_failure == reference.first_failure * scale


@pytest.mark.parametrize("contract", [BaselineMatrixReport, BaselineAttempt])
@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_evidence_unknown_versions_reject_before_use(contract, version):
    with pytest.raises(ValueError):
        contract.model_validate({"schema_version": version})


def valid_attempt():
    """Owned physical invocation contract fixture, not evidence of a real execution."""
    return BaselineAttempt(
        schema_version=1,
        cell_id="sha256:" + "a" * 64,
        attempt_id="owned-attempt",
        event="terminal",
        state="completed",
        recorded_at=baseline_plan().source.builds[0].decided_at,
        reasons=(),
        host_elapsed=baseline_measurement("host_attempt_elapsed_seconds", 0.0),
        worker_elapsed=baseline_measurement("native_process_elapsed_seconds", 0.0),
        worker_peak=baseline_measurement("native_process_peak_mib", 100.0),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"event": "started"},
        {"reasons": ("false_failure",)},
        {"state": "failed"},
        {"state": "aborted"},
        {"state": "blocked"},
        {"host_elapsed": baseline_measurement("host_attempt_elapsed_seconds", None)},
        {"worker_peak": baseline_measurement("selected_runner_seconds", 1.0)},
    ],
)
def test_attempt_event_and_resource_semantics_reject_ambiguous_claims(mutation):
    with pytest.raises(ValueError):
        change(valid_attempt(), **mutation)


def test_started_attempt_cannot_claim_terminal_cost_or_failure_reason():
    terminal = valid_attempt()
    started = change(
        terminal,
        event="started",
        state="running",
        host_elapsed=baseline_measurement("host_attempt_elapsed_seconds", None),
        worker_elapsed=baseline_measurement("native_process_elapsed_seconds", None),
        worker_peak=baseline_measurement("native_process_peak_mib", None),
    )
    assert BaselineAttempt.from_json(started.to_json()) == started
    with pytest.raises(ValueError, match="no terminal cost"):
        change(started, worker_peak=terminal.worker_peak)
    with pytest.raises(ValueError, match="no terminal cost"):
        change(started, reasons=("invented_failure",))


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_fully_valid_attempt_and_report_reject_unknown_versions(version):
    from tests.corpus.test_baseline_serialization import incompatible_report

    for value in (valid_attempt(), incompatible_report()):
        data = value.model_dump()
        data["schema_version"] = version
        with pytest.raises(ValueError, match="schema_version"):
            type(value).model_validate(data)


def test_complete_grid_never_silently_drops_incompatible_cells():
    from tests.corpus.test_baseline_serialization import incompatible_report

    report = incompatible_report()
    with pytest.raises(ValueError, match="entire unique"):
        change(report, cells=report.cells[:-1])
    with pytest.raises(ValueError, match="entire unique"):
        change(report, cells=(*report.cells, report.cells[0]))
    cell = report.cells[0]
    with pytest.raises(ValueError, match="exact eligibility"):
        change(report, cells=(change(cell, reasons=("invented_reason",)), *report.cells[1:]))
    with pytest.raises(ValueError, match="frozen candidate or budget|declared dimensions"):
        change(report, cells=(change(cell, seed=999), *report.cells[1:]))
    with pytest.raises(ValueError, match="native run"):
        change(cell, native_run_id="invented-native-run")
    with pytest.raises(ValueError, match="resource observations"):
        change(cell, resource_evidence="observed")
