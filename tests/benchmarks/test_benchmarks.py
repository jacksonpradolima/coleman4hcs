"""Independent oracles, deliberately broken solvers, deterministic streams and native persistence."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from coleman.artifacts import ArtifactWriter
from coleman.benchmarks import (
    AllocationCase,
    BenchmarkCheckpoint,
    BenchmarkEnvironment,
    BenchmarkExecution,
    BenchmarkFailure,
    BenchmarkReport,
    BenchmarkRunError,
    BenchmarkSuite,
    CoverageCase,
    ReferenceAnswer,
    check_allocation,
    check_coverage,
    execute_native,
    load_suite,
    publish_benchmark_report,
    reference,
    run_benchmarks,
    same_mathematics,
)
from coleman.benchmarks.bundled import Profile
from coleman.benchmarks.oracles import allocation_oracle, allocation_values, coverage_oracle, coverage_value
from coleman.checkpoint import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.manifest import generate_manifest
from coleman.selection import CoverageProblem, CoverageResult, Quantity

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/benchmarks"


def coverage_case(suite) -> CoverageCase:
    case = suite.cases[0]
    assert isinstance(case, CoverageCase)
    return case


def allocation_case(suite, index=1) -> AllocationCase:
    case = suite.cases[index]
    assert isinstance(case, AllocationCase)
    return case


def failed_attempt(report) -> BenchmarkFailure:
    assert report.failure is not None
    return report.failure


@pytest.fixture(scope="module")
def environment():
    return BenchmarkEnvironment(
        python="synthetic-fixture",
        platform="synthetic-fixture",
        machine="synthetic-fixture",
        processor="synthetic-fixture",
        coleman_version="synthetic-fixture",
        pydantic_version="synthetic-fixture",
    )


@pytest.fixture(scope="module")
def suite():
    return load_suite()


@pytest.fixture(scope="module")
def small(suite):
    case = coverage_case(suite)
    case = case.model_copy(update={"methods": ("branch_bound_v1",)})
    return BenchmarkSuite(schema_version=1, benchmark_id="tiny-ci-replay-v1", cases=(case,), seeds=(11, 11, 29))


@pytest.fixture(scope="module")
def report(small, environment):
    return run_benchmarks(small, execution=BenchmarkExecution(environment=environment))


def adapted(case, data, **changes):
    problem = type(case.problem).model_validate(data)
    expected = coverage_oracle(problem) if isinstance(case, CoverageCase) else allocation_oracle(problem)
    return type(case).model_validate({**case.model_dump(), "problem": problem, "expected": expected, **changes})


def clone_greedy_case(suite):
    base = coverage_case(suite)
    data = json.loads(base.problem.to_json())
    data["candidates"] = [
        {
            "candidate_id": name,
            "cost": {"value": "1", "semantics": data["constraints"]["budget"]["semantics"]},
            "covers": ["x"],
        }
        for name in ("A", "B")
    ]
    data["outcomes"] = [{"outcome_id": "x", "weight": {"value": "1", "semantics": data["objective_semantics"]}}]
    data["constraints"]["budget"]["value"] = "1"
    data["search_ids"] = None
    return adapted(base, data, methods=("greedy_density_v1",), exact_methods=())


def alternative_greedy(case):
    original = execute_native(case, "greedy_density_v1", 11)
    return CoverageResult.model_validate(original.model_copy(update={"selected_ids": ("B",)}))


def test_issue_hand_fixture_and_classifications(suite, environment):
    result = run_benchmarks(suite, execution=BenchmarkExecution(environment=environment))
    assert result.status == "complete"
    assert len(result.observations) == 18
    assert result.reproducibility == "verified_repeated_labels"
    first = result.observations[0]
    assert first.summary.selected_ids == ("A", "C")
    assert first.summary.objective == 1
    assert first.summary.independent_optimum == 1
    assert first.summary.upper_bound is not None
    assert first.summary.upper_bound >= first.summary.independent_optimum
    assert first.summary.method_class == "heuristic"
    assert first.summary.evidence == "bounded"
    exact = result.observations[3]
    assert exact.summary.evidence == "exact"
    assert exact.summary.absolute_regret == 0
    old = CoverageResult.from_json((ROOT / "tests/fixtures/coverage/optimal-v1.json").read_bytes())
    assert check_coverage(suite.cases[0], old).objective == 1
    allocation = result.observations[6]
    assert allocation.summary.selected_ids == ("stable",)
    assert allocation.summary.objective == 2
    limited = result.observations[-1]
    assert limited.summary.evidence == "heuristic"
    assert limited.summary.objective == 0
    assert limited.summary.absolute_regret == 2
    assert allocation.summary.uncertainty == "declared_finite_model_only"


def test_signed_joint_moments_and_nonlinear_distribution_gate(suite):
    case = allocation_case(suite)
    rows = {row.action_id: row for row in allocation_values(case.problem)}
    assert rows["variable"].net == 1
    assert rows["variable"].variance == 81
    assert rows["variable"].shortfall == 4
    assert rows["no_action"].net == 0
    nonlinear = allocation_case(suite, 3)
    rows = {row.action_id: row for row in allocation_values(nonlinear.problem)}
    assert rows["variable"].net == 3
    assert rows["variable"].utility == Fraction(6, 5)
    assert allocation_oracle(nonlinear.problem).selected_ids == ("stable",)
    native = execute_native(nonlinear, "enumerate_allocation_v1", 11)
    assert check_allocation(nonlinear, native).objective == 2


def test_independent_gates_catch_invalid_bounds_and_budget_overruns(suite):
    case = coverage_case(suite)
    native = execute_native(case, "branch_bound_v1", 11)
    bad_bound = native.model_copy(
        update={
            "upper_bound": Quantity(value=Decimal("0.5"), semantics=case.problem.objective_semantics),
        }
    )
    with pytest.raises(ValueError, match="upper bound is below"):
        check_coverage(case, bad_bound)
    exceeded = native.model_copy(update={"selected_ids": ("A", "B")})
    with pytest.raises(ValueError, match="violates scope or budget"):
        check_coverage(case, exceeded)
    work = native.model_copy(update={"work_used": case.limits.max_work + 1})
    with pytest.raises(ValueError, match="work budget exceeded"):
        check_coverage(case, work)
    assert native.cost is not None
    fake_cost = native.model_copy(update={"cost": Quantity(value=Decimal(0), semantics=native.cost.semantics)})
    with pytest.raises(ValueError, match="independent sum"):
        check_coverage(case, fake_cost)


def test_false_optimal_status_is_caught_by_independent_optimum(suite):
    base = coverage_case(suite)
    data = json.loads(base.problem.to_json())
    data["candidates"] = data["candidates"][:2]
    data["candidates"][0]["cost"]["value"] = "1"
    data["candidates"][1]["covers"] = ["x", "y", "z"]
    data["constraints"]["budget"]["value"] = "2"
    data["search_ids"] = None
    case = adapted(base, data)
    greedy = execute_native(case, "greedy_density_v1", 11)
    assert greedy.objective is not None
    assert greedy.objective.value == Decimal("0.8")
    assert reference(case).selected_ids == ("B",)
    forged = greedy.model_copy(
        update={
            "method": "branch_bound_v1",
            "status": "optimal",
            "termination_reason": "search_complete",
        }
    )
    with pytest.raises(ValueError, match="false optimal"):
        check_coverage(case, forged)


def test_full_math_stream_catches_order_dependence_despite_identical_objectives(suite, environment):
    case = clone_greedy_case(suite)
    first = execute_native(case, "greedy_density_v1", 11)
    second = alternative_greedy(case)
    assert first.objective is not None
    assert first.objective == second.objective
    assert first.cost == second.cost
    assert check_coverage(case, second).objective == 1
    first_json, second_json = first.to_json(), second.to_json()
    with pytest.raises(ValueError, match="mathematical artifacts differ"):
        same_mathematics(first_json, second_json)
    turns = 0

    def input_order_dependent(request, method, seed):
        nonlocal turns
        turns += 1
        # A faulty stream changes the first examined candidate on the repeated seed.
        return execute_native(request, method, seed) if turns == 1 else alternative_greedy(request)

    request = BenchmarkSuite(schema_version=1, benchmark_id="order-mutation-v1", cases=(case,), seeds=(11, 11, 29))
    failed_execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(request, execution=failed_execution, adapter=input_order_dependent)
    failed = caught.value.report
    assert isinstance(caught.value.__cause__, ValueError)
    assert failed.status == "failed"
    assert failed_attempt(failed).phase == "verification"
    assert failed_attempt(failed).seed_index == 1
    assert len(failed.observations) == 1
    assert "Failed/aborted attempt" in failed.markdown()
    assert BenchmarkReport.from_json(failed.to_json()) == failed


def test_gate_rejects_forged_allocation_forecasts_and_risk(suite):
    case = allocation_case(suite)
    native = execute_native(case, "enumerate_allocation_v1", 11)
    original = native.completed[0]
    value = original.forecast.utility.model_copy(update={"value": Decimal(999)})
    forecast = original.forecast.model_copy(update={"utility": value})
    row = original.model_copy(update={"forecast": forecast})
    corrupt = native.model_copy(update={"completed": (row, *native.completed[1:])})
    with pytest.raises(ValueError, match="independent joint distribution"):
        check_allocation(case, corrupt)
    row = original.model_copy(update={"feasible": not original.feasible})
    corrupt = native.model_copy(update={"completed": (row, *native.completed[1:])})
    with pytest.raises(ValueError, match="independent constraints"):
        check_allocation(case, corrupt)


def test_zero_negative_and_unavailable_remain_distinct(suite):
    coverage = coverage_case(suite)
    coverage = coverage.model_copy(
        update={"exact_methods": (), "limits": coverage.limits.model_copy(update={"max_working_bytes": 0})}
    )
    no_incumbent = check_coverage(coverage, execute_native(coverage, "branch_bound_v1", 11))
    assert no_incumbent.objective is None
    assert no_incumbent.upper_bound is None
    assert no_incumbent.evidence == "unavailable"
    case = allocation_case(suite, -1)
    observed = check_allocation(case, execute_native(case, "enumerate_allocation_v1", 11))
    assert observed.objective == 0
    unavailable = case.model_copy(update={"limits": case.limits.model_copy(update={"max_work": 0})})
    missing = check_allocation(unavailable, execute_native(unavailable, "enumerate_allocation_v1", 11))
    assert missing.objective is None
    assert not missing.feasible
    assert missing.evidence == "unavailable"
    data = json.loads(case.problem.to_json())
    for action in data["actions"]:
        for payoff in action["gross_payoffs"]:
            payoff["gross"]["value"] = "-5" if action["action_id"] == "no_action" else "-2"
    negative = adapted(allocation_case(suite), data)
    signed = check_allocation(negative, execute_native(negative, "enumerate_allocation_v1", 11))
    assert signed.objective == -2
    assert signed.objective_semantics.unit == "credit"


@pytest.mark.parametrize("value", [0.0, True, float("nan"), float("inf"), "NaN", "Infinity"])
def test_exact_reference_rejects_nonfinite_and_implicit_float_or_boolean_numbers(suite, value):
    fields = suite.cases[0].expected.model_dump()
    fields["objective"] = value
    with pytest.raises(ValidationError):
        ReferenceAnswer.model_validate(fields)


def test_incompatible_units_and_wrong_expected_answers_fail_before_solver(suite, environment):
    case = coverage_case(suite)
    wrong = case.expected.model_copy(update={"objective": Fraction(999)})
    request = suite.model_copy(update={"cases": (case.model_copy(update={"expected": wrong}),)})
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("No solver should start")

    failed_execution = BenchmarkExecution(environment=environment)
    with pytest.raises(ValueError, match="independent oracle"):
        run_benchmarks(request, execution=failed_execution, adapter=forbidden)
    assert calls == []
    wrong_unit = case.expected.objective_semantics.model_copy(update={"unit": "different"})
    wrong = case.expected.model_copy(update={"objective_semantics": wrong_unit})
    request = suite.model_copy(update={"cases": (case.model_copy(update={"expected": wrong}),)})
    with pytest.raises(ValueError, match="independent oracle"):
        run_benchmarks(request, adapter=forbidden)
    assert calls == []


def test_round_trip_full_identity_ignores_only_observational_metadata(report):
    assert BenchmarkReport.from_json(report.to_json()) == report
    data = json.loads(report.to_json())
    data["completed_at"] = "2026-02-02T00:00:00Z"
    data["observations"][0]["elapsed_ns"] += 1
    second = BenchmarkReport.model_validate(data)
    assert second.mathematical_identity == report.mathematical_identity
    assert second.fingerprint != report.fingerprint
    data["observations"][0]["mathematics_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="different digest"):
        BenchmarkReport.model_validate(data)


@pytest.mark.parametrize("value", [None, 0, 2, True, "1"])
def test_unknown_versions_fail_closed_in_all_new_artifacts(report, value):
    for model in (report, report.suite, report.checkpoint()):
        data = json.loads(model.to_json())
        if value is None:
            del data["schema_version"]
        else:
            data["schema_version"] = value
        reader, encoded = type(model), json.dumps(data)
        with pytest.raises(ValidationError):
            reader.from_json(encoded)
    with pytest.raises(ValueError, match="Duplicate"):
        BenchmarkSuite.from_json('{"schema_version":1,"schema_version":1}')


@pytest.mark.parametrize("when", [0, 1.0, True, "2026-01-01", "2026-01-01T00:00:00"])
def test_availability_rejects_implicit_epoch_units_and_naive_times(report, when):
    data = json.loads(report.to_json())
    data["completed_at"] = when
    with pytest.raises(ValidationError, match="timezone-aware"):
        BenchmarkReport.model_validate(data)


def test_no_repeated_label_is_explicitly_reproducibility_untested(small, environment):
    request = small.model_copy(update={"seeds": (11, 29)})
    result = run_benchmarks(request, execution=BenchmarkExecution(environment=environment))
    assert result.reproducibility == "untested"
    data = json.loads(result.to_json())
    data["reproducibility"] = "verified_repeated_labels"
    with pytest.raises(ValidationError, match="reproducibility"):
        BenchmarkReport.model_validate(data)


def test_native_checkpoint_continuation_and_corruption_before_zero_work(small, report, environment, tmp_path):
    execution = BenchmarkExecution(max_new_invocations=1, environment=environment)
    partial = run_benchmarks(small, execution=execution)
    assert partial.status == "interrupted"
    store = LocalCheckpointStore(str(tmp_path))
    checkpoint = partial.checkpoint()
    store.save(CheckpointPayload(run_id="ci-benchmark", step=1, extra={"benchmark_v1": checkpoint.to_json()}))
    payload = store.load("ci-benchmark", 0)
    assert payload is not None
    checkpoint = BenchmarkCheckpoint.from_json(payload.extra["benchmark_v1"])
    resumed = run_benchmarks(small, checkpoint=checkpoint, execution=BenchmarkExecution(environment=environment))
    assert resumed.status == "complete"
    assert resumed.observations[0] == partial.observations[0]
    assert resumed.mathematical_identity == report.mathematical_identity
    data = json.loads(checkpoint.to_json())
    data["observations"][0]["seed_index"] = 1
    corrupt = BenchmarkCheckpoint.model_validate(data)
    no_work = BenchmarkExecution(max_new_invocations=0, environment=environment)
    with pytest.raises(ValueError, match="prefix"):
        run_benchmarks(small, checkpoint=corrupt, execution=no_work)
    stale = checkpoint.model_copy(update={"suite_fingerprint": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="differs"):
        run_benchmarks(small, checkpoint=stale, execution=no_work)


def test_fully_cached_calls_still_poll_cancellation_and_retain_known_math(report, environment):
    result = run_benchmarks(
        report.suite,
        checkpoint=report.checkpoint(),
        execution=BenchmarkExecution(max_new_invocations=0, environment=environment),
        cancelled=lambda: True,
    )
    assert result.status == "interrupted"
    assert result.observations == report.observations
    assert result.mathematical_identity == report.mathematical_identity
    assert result.reproducibility == "verified_repeated_labels"


@pytest.mark.parametrize("error", [RuntimeError("injected"), KeyboardInterrupt()])
def test_failed_and_aborted_attempts_are_not_completed_observations(small, environment, error):
    def broken(*_args):
        raise error

    execution = BenchmarkExecution(environment=environment)
    if isinstance(error, KeyboardInterrupt):
        result = run_benchmarks(small, execution=execution, adapter=broken)
        assert result.status == "interrupted"
        assert failed_attempt(result).phase == "interrupted"
    else:
        with pytest.raises(BenchmarkRunError) as caught:
            run_benchmarks(small, execution=execution, adapter=broken)
        result = caught.value.report
        assert result.status == "failed"
        assert caught.value.__cause__ is error
        assert failed_attempt(result).phase == "solver"
    assert result.observations == ()
    failure = failed_attempt(result)
    assert failure.elapsed_ns is not None
    assert failure.python_peak_bytes is not None
    assert failure.elapsed_ns >= 0
    assert failure.python_peak_bytes >= 0
    assert not tracemalloc.is_tracing()
    assert BenchmarkReport.from_json(result.to_json()) == result


def test_existing_tracing_is_preserved_and_admission_recovers(small, environment):
    tracemalloc.start()
    try:
        failed_execution = BenchmarkExecution(environment=environment)
        with pytest.raises(BenchmarkRunError) as caught:
            run_benchmarks(small, execution=failed_execution)
        assert tracemalloc.is_tracing()
        failure = failed_attempt(caught.value.report)
        assert failure.phase == "instrumentation"
        assert failure.elapsed_ns is None
        assert failure.python_peak_bytes is None
    finally:
        tracemalloc.stop()
    assert run_benchmarks(small, execution=BenchmarkExecution(environment=environment)).status == "complete"


def test_concurrent_measurements_reject_overlap_without_corrupting_the_owner(small, environment):
    started, release = threading.Event(), threading.Event()

    def holding(case, method, seed):
        started.set()
        if not release.wait(10):
            raise RuntimeError("Test coordination timed out")
        return execute_native(case, method, seed)

    execution = BenchmarkExecution(environment=environment)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_benchmarks, small, execution=execution, adapter=holding)
        assert started.wait(10)
        try:
            with pytest.raises(BenchmarkRunError) as caught:
                run_benchmarks(small, execution=execution)
            assert failed_attempt(caught.value.report).phase == "instrumentation"
        finally:
            release.set()
        assert future.result(timeout=20).status == "complete"
    assert not tracemalloc.is_tracing()


def test_atomic_publication_duplicates_output_caps_and_manifest(report, tmp_path):
    writer = ArtifactWriter(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda _: publish_benchmark_report(report, writer), range(6)))
    assert len(set(paths)) == 1
    assert BenchmarkReport.from_json(paths[0].read_bytes()) == report
    manifest = json.loads(generate_manifest(tmp_path, run_id="ci-benchmark").read_text())
    assert [entry["type"] for entry in manifest["artifacts"]] == ["benchmark_report"]
    with pytest.raises(ValueError, match="output byte limit"):
        publish_benchmark_report(report, writer, max_output_bytes=0)


def test_interrupted_json_publish_is_not_a_completed_artifact_and_can_resume(report, tmp_path, monkeypatch):
    real_link = os.link

    def interrupted(source, destination):
        if str(destination).endswith(".json"):
            raise OSError("injected before JSON completion")
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", interrupted)
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(OSError, match="injected"):
        publish_benchmark_report(report, writer)
    assert list(tmp_path.rglob("*.json")) == []
    monkeypatch.setattr(os, "link", real_link)
    path = publish_benchmark_report(report, writer)
    assert path.exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_report_output_does_not_follow_a_symlink_directory(report, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "run"
    root.mkdir()
    (root / "benchmark-reports").symlink_to(outside, target_is_directory=True)
    writer = ArtifactWriter(root)
    with pytest.raises(ValueError, match="symlink"):
        publish_benchmark_report(report, writer)
    assert list(outside.iterdir()) == []


def test_metamorphic_candidate_order_relabeling_budget_and_duplicate_coverage(suite):
    base = coverage_case(suite)
    native = execute_native(base, "branch_bound_v1", 11)
    data = json.loads(base.problem.to_json())
    data["candidates"].reverse()
    data["outcomes"].reverse()
    for row in data["candidates"]:
        row["covers"].reverse()
    reordered = adapted(base, data)
    same_mathematics(native.to_json(), execute_native(reordered, "branch_bound_v1", 11).to_json())
    mapping = {"A": "ci-z", "B": "ci-x", "C": "ci-y"}
    for row in data["candidates"]:
        row["candidate_id"] = mapping[row["candidate_id"]]
    data["search_ids"] = None
    relabeled = adapted(base, data)
    selected = execute_native(relabeled, "branch_bound_v1", 11)
    assert set(selected.selected_ids) == {"ci-z", "ci-y"}
    assert selected.objective == native.objective
    assert selected.cost == native.cost
    enlarged = json.loads(base.problem.to_json())
    enlarged["constraints"]["budget"]["value"] = "4"
    old_optimum = reference(base).objective
    new_optimum = reference(adapted(base, enlarged)).objective
    assert old_optimum is not None
    assert new_optimum is not None
    assert new_optimum >= old_optimum
    clones = clone_greedy_case(suite)
    assert coverage_value(clones.problem, ("A", "B")).objective == 1
    malformed = json.loads(base.problem.to_json())
    malformed["candidates"][0]["covers"].append("x")
    with pytest.raises(ValidationError, match="Duplicate"):
        CoverageProblem.model_validate(malformed)


def test_additive_joint_payoff_and_target_shift_preserves_variance_shortfall_and_choice(suite):
    case = allocation_case(suite, 3)
    before = {row.action_id: row for row in allocation_values(case.problem)}
    data = json.loads(case.problem.to_json())
    shift = Decimal(2)
    data["risk"]["downside_target"]["value"] = "2"
    for action in data["actions"]:
        for row in action["gross_payoffs"]:
            row["gross"]["value"] = str(Decimal(row["gross"]["value"]) + shift)
    changed = adapted(case, data)
    after = {row.action_id: row for row in allocation_values(changed.problem)}
    for name, original in before.items():
        assert after[name].variance == original.variance
        assert after[name].shortfall == original.shortfall
        assert after[name].utility == original.utility + 2
        assert after[name].net == original.net + 2
    with localcontext() as context:
        context.prec = 4
        result = execute_native(changed, "enumerate_allocation_v1", 11)
        summary = check_allocation(changed, result)
    assert summary.selected_ids == ("stable",)
    assert summary.objective == 4


def test_true_native_cancellation_is_an_aborted_attempt_not_a_failed_exactness_claim(small, environment):
    from coleman.selection import solve_coverage

    def cancelled_native(case, method, _seed):
        return solve_coverage(case.problem, method=method, limits=case.limits, cancelled=lambda: True)

    result = run_benchmarks(
        small,
        execution=BenchmarkExecution(environment=environment),
        adapter=cancelled_native,
    )
    assert result.status == "interrupted"
    assert result.observations == ()
    assert failed_attempt(result).phase == "interrupted"
    assert failed_attempt(result).error_type == "native_cancelled"


def test_manifest_preflight_duplicate_ids_unbounded_protocols_and_bad_profile(suite):
    data = json.loads(suite.to_json())
    data["cases"][0]["limits"]["deadline_seconds"] = 1
    with pytest.raises(ValidationError, match="deadline_seconds=None"):
        BenchmarkSuite.model_validate(data)
    data = json.loads(suite.to_json())
    data["cases"][0]["methods"] = ["greedy_density_v1", "greedy_density_v1"]
    with pytest.raises(ValidationError, match="unique"):
        BenchmarkSuite.model_validate(data)
    data = json.loads(suite.to_json())
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValidationError, match="Duplicate benchmark case"):
        BenchmarkSuite.model_validate(data)
    base = coverage_case(suite)
    too_many = tuple(base.model_copy(update={"case_id": f"case-{i}"}) for i in range(8))
    fields = {"schema_version": 1, "benchmark_id": "bounded", "cases": too_many, "seeds": tuple(range(16))}
    with pytest.raises(ValidationError, match="invocation limit"):
        BenchmarkSuite.model_validate(fields)
    unknown_profile = cast(Profile, "../../outside")
    with pytest.raises(ValueError, match="Unknown bundled"):
        load_suite(unknown_profile)


def test_complete_status_and_report_summaries_cannot_be_forged(report):
    data = json.loads(report.to_json())
    data["observations"] = data["observations"][:-1]
    with pytest.raises(ValidationError, match="complete execution"):
        BenchmarkReport.model_validate(data)
    data = json.loads(report.to_json())
    data["observations"][0]["summary"]["absolute_regret"] = "-1"
    with pytest.raises(ValidationError, match="independently verified"):
        BenchmarkReport.model_validate(data)
    data = json.loads(report.to_json())
    data["observations"].append(data["observations"][0])
    with pytest.raises(ValidationError, match="excess or duplicate"):
        BenchmarkReport.model_validate(data)


def test_golden_v1_exact_math_and_producer_metadata_remain_readable(small):
    golden = BenchmarkReport.from_json((FIXTURES / "report-v1.json").read_bytes())
    checkpoint = BenchmarkCheckpoint.from_json((FIXTURES / "checkpoint-v1.json").read_bytes())
    expected = json.loads((FIXTURES / "digests-v1.json").read_text())
    assert golden.suite == small
    assert golden.fingerprint == expected["report"]
    assert golden.mathematical_identity == expected["mathematics"]
    assert checkpoint.fingerprint == expected["checkpoint"]
    assert golden.to_json() + "\n" == (FIXTURES / "report-v1.json").read_text()
    assert golden.environment.python == "synthetic-fixture"
    assert golden.observations[0].elapsed_ns == 0
    assert golden.observations[0].python_peak_bytes == 0
    assert BenchmarkCheckpoint.from_json(checkpoint.to_json()) == checkpoint


def test_runtime_and_ast_import_boundaries_and_hash_seed_replay(tmp_path):
    directory = ROOT / "coleman/benchmarks"
    for path in directory.glob("*.py"):
        if path.name == "__main__.py":
            continue
        tree = ast.parse(path.read_text())
        modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(module.startswith(("coleman.cli", "coleman.benchmarks.__main__", "mcp")) for module in modules)
    tree = ast.parse((directory / "oracles.py").read_text())
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls.intersection({"solve_coverage", "select_allocation", "_evaluate_action", "selection_value"})
    program = """from coleman.benchmarks import load_suite,run_benchmarks,BenchmarkSuite
import sys
s=load_suite()
s=BenchmarkSuite(schema_version=1,benchmark_id="hash-replay-v1",cases=(s.cases[0],),seeds=(11,11))
r=run_benchmarks(s)
assert "coleman.cli" not in sys.modules
assert not any(name=="mcp" or name.startswith("mcp.") for name in sys.modules)
print(r.mathematical_identity)
"""
    values = []
    for seed in (1, 734):
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        values.append(result.stdout.strip())
    assert values[0] == values[1]


def test_cli_real_profile_and_invalid_inputs(suite, tmp_path, monkeypatch, capsys):
    from coleman.benchmarks.__main__ import main

    monkeypatch.chdir(tmp_path)
    assert main(["--output", "artifacts"]) == 0
    first = json.loads(capsys.readouterr().out)
    artifact = Path(first["artifact"])
    parsed = BenchmarkReport.from_json(artifact.read_bytes())
    assert parsed.suite == suite
    assert len(parsed.observations) == 18
    assert parsed.reproducibility == "verified_repeated_labels"
    assert main(["--source-commit", "invalid", "--output", "invalid"]) == 2
    assert not (tmp_path / "invalid").exists()
    assert "input/publication error" in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        main(["--output", "../escape"])
    assert caught.value.code == 2


def test_cli_persists_failed_reports_instead_of_reporting_success(small, environment, tmp_path, monkeypatch, capsys):
    from coleman.benchmarks import __main__ as transport

    def broken(*_args):
        raise RuntimeError("test-only solver failure")

    failed_execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(small, execution=failed_execution, adapter=broken)
    error = caught.value

    def failed(*_args, **_kwargs):
        raise error

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(transport, "run_benchmarks", failed)
    assert transport.main(["--output", "failed"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    artifact = BenchmarkReport.from_json(Path(output["artifact"]).read_bytes())
    assert failed_attempt(artifact).error_type == "RuntimeError"


def test_oracle_caps_estimated_weights_and_required_exactness_are_explicit(suite):
    case = coverage_case(suite)
    data = json.loads(case.problem.to_json())
    for i in range(10):
        row = json.loads(json.dumps(data["candidates"][0]))
        row["candidate_id"] = f"more-{i}"
        data["candidates"].append(row)
    data["search_ids"] = None
    larger = CoverageProblem.model_validate(data)
    with pytest.raises(ValueError, match="oracle size limit"):
        coverage_oracle(larger)
    oversized_case = {**case.model_dump(), "problem": larger}
    with pytest.raises(ValidationError, match="twelve candidates"):
        CoverageCase.model_validate(oversized_case)
    short = case.model_copy(update={"limits": case.limits.model_copy(update={"max_work": 0})})
    incomplete = execute_native(short, "branch_bound_v1", 11)
    with pytest.raises(ValueError, match="required a completed exact"):
        check_coverage(short, incomplete)
    allocation = allocation_case(suite)
    data = json.loads(allocation.problem.to_json())
    data["probability_evidence"] = {
        "kind": "estimated",
        "method_id": "fixture-estimator",
        "evidence_fingerprint": "sha256:" + "1" * 64,
    }
    for row in data["scenarios"]:
        row["lower"], row["upper"] = "0", "1"
    from coleman.allocation import AllocationProblem

    estimated = AllocationProblem.model_validate(data)
    with pytest.raises(ValueError, match="estimated model"):
        allocation_oracle(estimated)
    estimated_case = {**allocation.model_dump(), "problem": estimated}
    with pytest.raises(ValidationError, match="assumed weights"):
        AllocationCase.model_validate(estimated_case)


def test_failed_report_and_checkpoint_cannot_invent_or_skip_attempts(small, environment):
    def broken(*_args):
        raise ValueError("fixture")

    failed_execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(small, execution=failed_execution, adapter=broken)
    failed = caught.value.report
    data = json.loads(failed.to_json())
    data["failure"]["seed_index"] = 1
    with pytest.raises(ValidationError, match="does not follow"):
        BenchmarkReport.model_validate(data)
    data = json.loads(failed.to_json())
    data["failure"]["elapsed_ns"] = None
    with pytest.raises(ValidationError, match="both be present"):
        BenchmarkReport.model_validate(data)
    data = json.loads(failed.to_json())
    data["failure"] = None
    with pytest.raises(ValidationError, match="explicit failed invocation"):
        BenchmarkReport.model_validate(data)
    assert failed.checkpoint().observations == ()


def test_wrong_native_result_kind_is_a_typed_verification_failure(suite, small, environment):
    payload = execute_native(allocation_case(suite), "enumerate_allocation_v1", 11)

    def wrong_kind(*_args):
        return payload

    failed_execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(small, execution=failed_execution, adapter=wrong_kind)
    assert failed_attempt(caught.value.report).phase == "verification"
    assert isinstance(caught.value.__cause__, TypeError)


def test_exact_infeasibility_and_empty_valid_selection_are_distinct(suite):
    base = coverage_case(suite)
    data = json.loads(base.problem.to_json())
    data["constraints"]["budget"]["value"] = "0"
    empty = adapted(base, data)
    result = execute_native(empty, "branch_bound_v1", 11)
    summary = check_coverage(empty, result)
    assert summary.selected_ids == ()
    assert summary.objective == 0
    assert summary.feasible
    data["constraints"]["required_ids"] = ["A"]
    impossible = adapted(base, data)
    result = execute_native(impossible, "branch_bound_v1", 11)
    summary = check_coverage(impossible, result)
    assert summary.selected_ids is None
    assert summary.objective is None
    assert summary.evidence == "exact"
    assert summary.status == "infeasible"
    assert not summary.feasible


def test_resuming_a_failed_run_retains_attempt_history_without_relabeling_unknown_work(small, report, environment):
    def broken(*_args):
        raise RuntimeError("one failed attempt")

    execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(small, execution=execution, adapter=broken)
    failed = caught.value.report
    assert failed_attempt(failed).work_used is None
    checkpoint = BenchmarkCheckpoint.from_json(failed.checkpoint().to_json())
    assert len(checkpoint.prior_failures) == 1
    result = run_benchmarks(small, checkpoint=checkpoint, execution=execution)
    assert result.status == "complete"
    assert result.failure is None
    assert result.prior_failures == (failed.failure,)
    assert result.prior_failures[0].attempt_index == 0
    assert result.mathematical_identity == report.mathematical_identity
    assert "Failed/aborted attempt" in result.markdown()
    assert BenchmarkReport.from_json(result.to_json()) == result
    data = json.loads(checkpoint.to_json())
    data["prior_failures"].append(data["prior_failures"][0])
    corrupt = BenchmarkCheckpoint.model_validate(data)
    zero = BenchmarkExecution(environment=environment, max_new_invocations=0)
    with pytest.raises(ValueError, match="duplicate attempt"):
        run_benchmarks(small, checkpoint=corrupt, execution=zero)


def test_failure_history_cap_and_future_availability_fail_closed_before_another_attempt(small, environment):
    def broken(*_args):
        raise RuntimeError("fixture")

    execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(small, execution=execution, adapter=broken)
    failed = caught.value.report
    failure = failed_attempt(failed)
    checkpoint = failed.checkpoint()
    checkpoint = checkpoint.model_copy(
        update={
            "prior_failures": tuple(failure.model_copy(update={"attempt_index": index}) for index in range(128)),
        }
    )
    with pytest.raises(ValueError, match="history exhausted"):
        run_benchmarks(small, checkpoint=checkpoint, execution=execution)
    data = json.loads(failed.to_json())
    data["failure"]["finished_at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(ValidationError, match="availability predates"):
        BenchmarkReport.model_validate(data)
    future = failed.checkpoint().model_copy(
        update={"prior_failures": (failure.model_copy(update={"finished_at": datetime(2099, 1, 1, tzinfo=UTC)}),)}
    )
    with pytest.raises(ValueError, match="unavailable future"):
        run_benchmarks(small, checkpoint=future, execution=execution)
    data = json.loads(failed.to_json())
    del data["failure"]["finished_at"]
    with pytest.raises(ValidationError):
        BenchmarkReport.model_validate(data)


def test_counter_units_are_replayed_and_keyboard_cancellation_preserves_the_prefix(report, environment):
    data = json.loads(report.to_json())
    data["observations"][0]["work_unit"] = "allocation_action_attempt"
    with pytest.raises(ValidationError, match="work unit differs"):
        BenchmarkReport.model_validate(data)

    def keyboard():
        raise KeyboardInterrupt

    stopped = run_benchmarks(
        report.suite,
        checkpoint=report.checkpoint(),
        execution=BenchmarkExecution(environment=environment),
        cancelled=keyboard,
    )
    assert stopped.status == "interrupted"
    assert stopped.observations == report.observations
    assert stopped.failure is None


@pytest.mark.parametrize("phase", ["solver", "verification"])
@pytest.mark.parametrize("invented_work", [0, 7])
def test_failed_attempts_reject_invented_work_in_reports_and_checkpoints(small, environment, phase, invented_work):
    def broken(*_args):
        raise RuntimeError("fixture")

    execution = BenchmarkExecution(environment=environment)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmarks(small, execution=execution, adapter=broken)
    data = json.loads(caught.value.report.to_json())
    data["failure"]["phase"] = phase
    valid = BenchmarkReport.model_validate(data)
    assert failed_attempt(valid).work_used is None
    assert BenchmarkReport.from_json(valid.to_json()) == valid
    data["failure"]["work_used"] = invented_work
    with pytest.raises(ValidationError, match="unavailable work"):
        BenchmarkReport.model_validate(data)
    checkpoint = json.loads(valid.checkpoint().to_json())
    checkpoint["prior_failures"][0]["work_used"] = invented_work
    with pytest.raises(ValidationError, match="unavailable work"):
        BenchmarkCheckpoint.model_validate(checkpoint)
