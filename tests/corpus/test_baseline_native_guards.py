"""Independent native-result guards reject incorrect rankings, denominators, clocks and cost projections."""

import math
from dataclasses import replace

import pytest

from coleman.application.baseline_execution import baseline_input_bytes
from coleman.application.baseline_observations import _check_native
from coleman.budget import BudgetMode
from coleman.corpus.baseline_results import baseline_coordinates
from coleman.scenarios import ScenarioLoader
from coleman.utils.monitor_params import CollectParams
from tests.corpus.baseline_support import baseline_plan
from tests.corpus.test_baseline_agents import bandit_for


def native_case(tmp_path, step):
    """Use actual native loader, bandit and metric on the owned source; no principal component is mocked."""
    plan = baseline_plan()
    coordinate = next(
        value
        for value in baseline_coordinates(plan)
        if value.method == "seeded_random_v1" and value.budget.value == 0.5
    )
    build = plan.source.builds[step - 1]
    path = tmp_path / "features-engineered.parquet"
    path.write_bytes(baseline_input_bytes(plan))
    loader = ScenarioLoader(str(path), 0.5, BudgetMode.RATIO, 0.5)
    bandit = bandit_for(build, 3.5)
    metric = bandit.pull(["a", "b", "c"])
    params = CollectParams(
        scenario_provider=loader,
        available_time=3.5,
        experiment=1,
        t=step,
        policy=coordinate.method,
        reward_function="RNFail",
        metric=metric,
        total_build_duration=7.0,
        prioritization_time=0.0,
        rewards=0.0,
        prioritization_order=["a", "b", "c"],
        execution_id="owned-guard-attempt",
        budget_mode=BudgetMode.RATIO,
        budget_value=0.5,
    )
    return params, build, coordinate


@pytest.mark.parametrize("step", [1, 2])
def test_real_native_result_satisfies_independent_scientific_reference(tmp_path, step):
    params, build, cell = native_case(tmp_path, step)
    result = _check_native(params, build, cell)
    assert result.selected == ("a", "b")
    assert result.selected_work == 3.0
    assert result.failures == int(step == 2)


@pytest.mark.parametrize(
    "field,value", [("policy", "wrong"), ("budget_mode", None), ("budget_value", 1.0), ("available_time", 7.0)]
)
def test_native_candidate_or_budget_cannot_change_after_freezing(tmp_path, field, value):
    params, build, cell = native_case(tmp_path, 2)
    prepared_input = replace(params, **{field: value})
    with pytest.raises(ValueError, match="frozen candidate or budget"):
        _check_native(prepared_input, build, cell)


def test_native_full_suite_duration_must_match_actual_source(tmp_path):
    params, build, cell = native_case(tmp_path, 2)
    prepared_input = replace(params, available_time=4.0, total_build_duration=8.0)
    with pytest.raises(ValueError, match="full-suite work"):
        _check_native(prepared_input, build, cell)


def test_native_source_total_cannot_drift_one_binary64_ulp(tmp_path):
    params, build, cell = native_case(tmp_path, 2)
    work = math.nextafter(params.total_build_duration, math.inf)
    changed = replace(params, total_build_duration=work, available_time=work * float(cell.budget.value))
    with pytest.raises(ValueError, match="full-suite work"):
        _check_native(changed, build, cell)


@pytest.mark.parametrize("attribute,value", [("detected_failures", 0), ("undetected_failures", 1)])
def test_native_failure_denominator_cannot_be_changed(tmp_path, attribute, value):
    params, build, cell = native_case(tmp_path, 2)
    setattr(params.metric, attribute, value)
    with pytest.raises(ValueError, match="failure denominator"):
        _check_native(params, build, cell)


@pytest.mark.parametrize("attribute,value", [("fitness", 0.9), ("cost", 0.9), ("recall", 0.0)])
def test_native_quality_corruption_cannot_certify_mathematical_evidence(tmp_path, attribute, value):
    params, build, cell = native_case(tmp_path, 2)
    setattr(params.metric, attribute, value)
    with pytest.raises(ValueError, match="quality metric"):
        _check_native(params, build, cell)


def test_native_legacy_zero_failure_fallback_is_checked_without_becoming_a_scientific_ratio(tmp_path):
    params, build, cell = native_case(tmp_path, 1)
    assert _check_native(params, build, cell).napfd is None
    params.metric.fitness = 0.0
    with pytest.raises(ValueError, match="zero-failure fallback"):
        _check_native(params, build, cell)


def test_native_first_failure_cost_cannot_change_independently_of_the_ranking(tmp_path):
    params, build, cell = native_case(tmp_path, 2)
    assert _check_native(params, build, cell).first_failure == 3.0
    params.metric.ttf_duration = 4.0
    with pytest.raises(ValueError, match="first-failure time"):
        _check_native(params, build, cell)


def completed_cell():
    """Owned schema-only cell fixture; hashes identify synthetic references, not actual executions."""
    from coleman.corpus.baseline_results import BaselineCellResult

    coordinate = baseline_coordinates(baseline_plan())[0]
    return BaselineCellResult(
        **coordinate.model_dump(),
        state="completed",
        reasons=(),
        native_run_id="owned-native-run",
        observations=tuple("sha256:" + value * 64 for value in "abc"),
        mathematical_observations=tuple("sha256:" + value * 64 for value in "def"),
        attempts=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
        unclosed_attempts=0,
        resource_evidence="observed",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"reasons": ("invented_failure",)},
        {"state": "failed"},
        {"observations": ("sha256:" + "a" * 64,)},
        {"observations": tuple("sha256:" + "a" * 64 for _ in range(3))},
        {"mathematical_observations": tuple("sha256:" + "d" * 64 for _ in range(3))},
        {"resource_evidence": "not_applicable"},
        {"unclosed_attempts": 1},
    ],
)
def test_cell_contract_rejects_duplicate_incomplete_or_unobserved_evidence(mutation):
    from tests.corpus.baseline_support import change

    prepared_input = completed_cell()
    with pytest.raises(ValueError):
        change(prepared_input, **mutation)


@pytest.mark.parametrize("state", ["aborted", "unavailable"])
def test_unfinished_cell_cannot_mark_the_grid_complete(state):
    from coleman.corpus.baseline_results import BaselineCellResult, BaselineMatrixReport
    from tests.corpus.baseline_support import change

    original = baseline_plan()
    plan = change(original, methods=("seeded_random_v1",), seeds=(42,), budgets=(original.budgets[0],))
    cell = BaselineCellResult(
        **baseline_coordinates(plan)[0].model_dump(),
        state=state,
        reasons=("owned-interruption",),
        native_run_id=None,
        observations=(),
        mathematical_observations=(),
        attempts=(),
        unclosed_attempts=0,
        resource_evidence="unavailable",
    )
    report = BaselineMatrixReport(schema_version=1, plan=plan, producer_fingerprint="sha256:" + "a" * 64, cells=(cell,))
    assert not report.complete
    incompatible = change(cell, state="incompatible", resource_evidence="not_applicable")
    with pytest.raises(ValueError, match="eligible baseline"):
        change(report, cells=(incompatible,))


def native_completion():
    """Owned completion contract fixture, explicitly not an actual-byte verification receipt."""
    from coleman.corpus.baseline_results import BaselineFilePin, BaselineNativeCompletion, baseline_measurement

    cell = completed_cell()
    return BaselineNativeCompletion(
        schema_version=1,
        cell_id=cell.cell_id,
        plan_fingerprint="sha256:" + "a" * 64,
        runtime_fingerprint="sha256:" + "b" * 64,
        native_run_id="owned-native-run",
        recorded_at=baseline_plan().source.builds[0].decided_at,
        completed_steps=1,
        observations=cell.observations[:1],
        mathematical_observations=cell.mathematical_observations[:1],
        files=(BaselineFilePin(path="checkpoints/owned.pkl", sha256="sha256:" + "c" * 64, size_bytes=0),),
        worker_elapsed=baseline_measurement("native_process_elapsed_seconds", 1.0),
        worker_cpu=baseline_measurement("native_process_cpu_seconds", 0.5),
        worker_peak=baseline_measurement("native_process_peak_mib", 100.0),
    )


def test_native_completion_requires_complete_membership_unique_file_roles_and_resource_units():
    from coleman.corpus.baseline_results import BaselineNativeCompletion, baseline_measurement
    from tests.corpus.baseline_support import change

    receipt = native_completion()
    assert BaselineNativeCompletion.from_json(receipt.to_json()) == receipt
    with pytest.raises(ValueError, match="duplicated evidence"):
        change(receipt, completed_steps=2)
    with pytest.raises(ValueError, match="duplicated evidence"):
        change(receipt, files=(*receipt.files, *receipt.files))
    prepared_worker_peak = baseline_measurement("native_process_elapsed_seconds", 1.0)
    with pytest.raises(ValueError, match="resource semantics"):
        change(receipt, worker_peak=prepared_worker_peak)


def test_runtime_identity_is_canonical_without_equating_different_recorded_environments():
    from coleman.spec.runtime_identity import InstalledRuntime
    from tests.corpus.baseline_support import change

    runtime = InstalledRuntime.capture()
    assert change(runtime, distributions=tuple(reversed(runtime.distributions))).fingerprint == runtime.fingerprint
    assert change(runtime, machine="owned-other-machine").fingerprint != runtime.fingerprint
    with pytest.raises(ValueError, match="unique canonical"):
        change(runtime, distributions=(*runtime.distributions, runtime.distributions[0]))
    prepared_distributions = (change(runtime.distributions[0], name="Non_Canonical"),)
    with pytest.raises(ValueError, match="unique canonical"):
        change(runtime, distributions=prepared_distributions)
