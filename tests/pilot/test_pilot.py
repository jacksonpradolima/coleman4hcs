"""Independent arithmetic, replay, protocol and immutable-recovery gates for CI pilots."""

import ast
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.pilot import (
    OVERHEAD_SECONDS,
    RATE,
    SECONDS,
    WALL_SECONDS,
    ExecutionEvent,
    PilotManifest,
    PilotReport,
    PilotRun,
    PilotTest,
    publish_pilot_report,
    render_pilot_report,
    replay_events,
    serial_events,
    verify_run,
)
from coleman.results.domain import fingerprint
from coleman.selection import Quantity, UnitSemantics

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples/operational-pilot/run.py"
SPEC = importlib.util.spec_from_file_location("pilot_example", EXAMPLE)
assert SPEC is not None
assert SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def q(value, semantics=SECONDS):
    return Quantity(value=Decimal(str(value)), semantics=semantics)


@pytest.fixture
def manifest():
    return example.fixture()[0]


def run(manifest, candidate="base-order", build=None, ranking=("t1", "t2", "t3")):
    build = build or manifest.builds[0]
    return PilotRun(
        candidate_id=candidate,
        build_id=build.build_id,
        seed=42,
        ranking=ranking,
        events=serial_events(build, ranking, manifest.budget),
        prioritization_overhead=q("0.001", OVERHEAD_SECONDS),
    )


def changed(model, **kwargs):
    return type(model).model_validate({**model.model_dump(), **kwargs})


def complete(manifest):
    return PilotReport(
        schema_version=1,
        manifest=manifest,
        runs=tuple(run(manifest, candidate, build) for candidate in manifest.candidates for build in manifest.builds),
    )


def test_hand_calculated_fixture_and_reverse_feedback(manifest):
    original = verify_run(manifest, run(manifest))
    faster = verify_run(manifest, run(manifest, "shortest-first", ranking=("t2", "t1", "t3")))
    assert original.runner_work.value == faster.runner_work.value == 3
    assert original.full_suite_runner_work.value == 7
    assert original.failing_tests == 2
    assert original.detected_failing_tests == 1
    assert original.failing_test_recall == faster.failing_test_recall == 0.5
    assert original.first_failure_elapsed is not None
    assert faster.first_failure_elapsed is not None
    assert original.first_failure_elapsed.value == 3
    assert faster.first_failure_elapsed.value == 2
    assert original.modeled_variable_cost.value == Decimal("0.03")


def test_stop_nonfit_no_shorter_skip_and_denominator_cannot_disappear(manifest):
    no_fit = run(manifest, "shortest-first", ranking=("t3", "t1", "t2"))
    metric = verify_run(manifest, no_fit)
    assert metric.runner_work.value == 0
    assert metric.failing_test_recall == 0
    assert metric.first_failure_elapsed is None
    assert metric.first_failure_status == "right_censored"
    with pytest.raises(ValueError, match="full unique ranking"):
        serial_events(manifest.builds[0], ("t1", "t2"), manifest.budget)
    report = complete(manifest)
    replaced = tuple(
        no_fit if (item.candidate_id, item.build_id) == (no_fit.candidate_id, no_fit.build_id) else item
        for item in report.runs
    )
    finding = next(item for item in changed(report, runs=replaced).findings() if item.candidate_id == "shortest-first")
    assert finding.gate == "fail"
    assert finding.runner_saving_fraction_range == (0, 1)
    assert "detection_regression_or_threshold_failure" in finding.reasons
    assert finding.adoption == "insufficient_evidence"


@pytest.mark.parametrize("outcome", ["unknown", "timeout", "passed"])
def test_unresolved_zero_failure_and_zero_time_are_distinct(manifest, outcome):
    build = changed(
        manifest.builds[0], tests=(PilotTest(test_id="z", duration=q(0), outcome=outcome),), baseline_order=("z",)
    )
    events = serial_events(build, ("z",), q(0))
    metric = replay_events(build, events, manifest.variable_price)
    assert metric.failing_test_recall is None
    assert metric.unresolved_tests == int(outcome != "passed")
    assert metric.first_failure_elapsed is None
    failed = changed(build, tests=(changed(build.tests[0], outcome="failed"),))
    metric = replay_events(failed, events, manifest.variable_price)
    assert metric.first_failure_elapsed is not None
    assert metric.first_failure_elapsed.value == 0
    assert metric.failing_test_recall == 1


def test_parallel_replay_distinguishes_work_from_elapsed_and_rejects_overlap(manifest):
    build = manifest.builds[0]
    events = tuple(
        ExecutionEvent(
            test_id=test.test_id,
            worker_id=test.test_id,
            started=q(0, WALL_SECONDS),
            completed=q(test.duration.value, WALL_SECONDS),
        )
        for test in build.tests
    )
    metric = replay_events(build, events, manifest.variable_price)
    assert metric.runner_work.value == 7
    assert metric.execution_elapsed.value == 4
    assert metric.first_failure_elapsed is not None
    assert metric.first_failure_elapsed.value == 2
    overlapping = tuple(changed(event, worker_id="one") for event in events)
    with pytest.raises(ValueError, match="overlapping"):
        replay_events(build, overlapping, manifest.variable_price)
    parallel = changed(run(manifest), events=events)
    with pytest.raises(ValueError, match="serial prefix"):
        verify_run(manifest, parallel)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0.0000000000001"])
def test_invalid_nonfinite_negative_and_unrepresentable_quantities(value):
    with pytest.raises(ValueError):
        q(value)


def test_incompatible_units_null_duplicate_ids_and_unknown_test(manifest):
    milliseconds = q(0, UnitSemantics(unit="ms", meaning="test-duration"))
    with pytest.raises(ValueError, match="occupancy seconds"):
        PilotTest(test_id="x", duration=milliseconds, outcome="passed")
    with pytest.raises(ValueError):
        PilotTest.model_validate({"test_id": "x", "duration": None, "outcome": "unknown"})
    with pytest.raises(ValueError, match="unique logical"):
        changed(manifest.builds[0], baseline_order=("t1", "t1", "t3"))
    event = run(manifest).events[0]
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        replay_events(manifest.builds[0], (event, event), manifest.variable_price)
    absent = changed(event, test_id="absent")
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        replay_events(manifest.builds[0], (absent,), manifest.variable_price)
    bad_duration = changed(event, completed=q(2, WALL_SECONDS))
    with pytest.raises(ValueError, match="Occupancy mismatch"):
        replay_events(manifest.builds[0], (bad_duration,), manifest.variable_price)
    wrong_price = q(1)
    with pytest.raises(ValueError, match="variable price"):
        replay_events(manifest.builds[0], (event,), wrong_price)


def test_protocol_rejects_changed_source_future_labels_and_baseline(manifest):
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        changed(manifest, dataset_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="Freeze"):
        changed(manifest, frozen_at="2026-01-02T00:00:00Z")
    builds = tuple(changed(build, outcomes_available_at="2027-01-01T00:00:00Z") for build in manifest.builds)
    data_fingerprint = fingerprint([build.model_dump(mode="json") for build in builds])
    with pytest.raises(ValueError, match="evaluation cutoff"):
        changed(manifest, builds=builds, dataset_fingerprint=data_fingerprint)
    reordered = run(manifest, ranking=("t2", "t1", "t3"))
    with pytest.raises(ValueError, match="Baseline order"):
        verify_run(manifest, reordered)
    unknown = changed(run(manifest), candidate_id="unfrozen")
    with pytest.raises(ValueError, match="Unknown candidate"):
        verify_run(manifest, unknown)


def test_roundtrip_golden_and_fail_closed_versions(manifest):
    report = complete(manifest)
    golden = Path(__file__).parent / "fixtures/pilot-v1.json"
    assert report.to_json() + "\n" == golden.read_text()
    assert PilotReport.from_json(golden.read_bytes()) == report
    assert PilotManifest.from_json(manifest.to_json()) == manifest
    for version in (0, 2, 99, True, "1", None):
        payload = json.dumps({**json.loads(report.to_json()), "schema_version": version})
        with pytest.raises(ValueError):
            PilotReport.from_json(payload)
    with pytest.raises(ValueError):
        PilotReport.from_json('{"schema_version":1,"schema_version":1}')
    # A native legacy row remains its own contract; no implicit migration.
    with pytest.raises(ValueError):
        PilotReport.from_json('{"scenario":"ci","ttf":0,"detected":1}')


def test_metamorphic_order_precision_and_digests(manifest):
    build = manifest.builds[0]
    expected = complete(manifest)
    for permutation in itertools.permutations(build.tests):
        normalized = changed(build, tests=permutation)
        assert normalized == build
        for ranking in itertools.permutations(build.baseline_order):
            events = serial_events(normalized, ranking, manifest.budget)
            # Independent tiny oracle uses Python integers, never production metrics.
            elapsed = 0
            selected = []
            durations = {"t1": 1, "t2": 2, "t3": 4}
            for name in ranking:
                if elapsed + durations[name] > 3:
                    break
                elapsed += durations[name]
                selected.append(name)
            actual = replay_events(normalized, events, manifest.variable_price)
            assert actual.runner_work.value == elapsed
            assert actual.failing_test_recall == sum(name != "t1" for name in selected) / 2
    with localcontext() as context:
        context.prec = 3
        assert complete(manifest).to_json() == expected.to_json()
    assert changed(expected, runs=tuple(reversed(expected.runs))).fingerprint == expected.fingerprint
    assert changed(manifest, candidates=tuple(reversed(manifest.candidates))).fingerprint == manifest.fingerprint
    source = example.fixture()[1]
    assert manifest.source_sha256 == "sha256:" + hashlib.sha256(source.encode()).hexdigest()


def test_checkpoint_resume_idempotence_and_conflicting_retry(manifest, tmp_path):
    report = PilotReport(schema_version=1, manifest=manifest)
    assert report.status == "partial"
    first = run(manifest)
    partial = report.append(first)
    writer = ArtifactWriter(tmp_path)
    path = publish_pilot_report(partial, writer)
    resumed = PilotReport.from_json(path.read_bytes())
    assert resumed.append(first) == partial
    conflicting = changed(first, seed=43)
    with pytest.raises(ValueError, match="Conflicting"):
        resumed.append(conflicting)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: publish_pilot_report(partial, writer), range(8))) == [path] * 8
    for candidate, build_id in resumed.missing:
        build = next(build for build in manifest.builds if build.build_id == build_id)
        resumed = resumed.append(run(manifest, candidate, build))
    assert resumed == complete(manifest)
    assert resumed.status == "completed"
    assert all(finding.gate == "inconclusive" for finding in resumed.findings())
    assert all(finding.adoption == "insufficient_evidence" for finding in resumed.findings())
    assert "insufficient evidence" in render_pilot_report(resumed)
    assert "missing pairs: 8" in render_pilot_report(report)


def test_interrupted_publication_never_completes_artifact(manifest, tmp_path, monkeypatch):
    def interrupted(*_):
        raise OSError("injected before commit")

    monkeypatch.setattr(os, "link", interrupted)
    report, writer = complete(manifest), ArtifactWriter(tmp_path)
    with pytest.raises(OSError, match="injected"):
        publish_pilot_report(report, writer)
    assert list(tmp_path.rglob("*.json")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


def test_import_boundary():
    for path in (ROOT / "coleman/pilot").glob("*.py"):
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        assert not any(name.startswith(("coleman.cli", "coleman.transports", "mcp")) for name in modules)
    code = "import sys; import coleman.pilot; assert 'coleman.cli' not in sys.modules; assert 'mcp' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=30)


def test_observed_gate_remains_separate_from_adoption_and_fails_overhead(manifest):
    observed = changed(manifest, evidence_basis="observed_ci")
    report = complete(observed)
    assert all(finding.gate == "pass" for finding in report.findings())
    assert all(finding.adoption == "insufficient_evidence" for finding in report.findings())
    runs = tuple(
        changed(item, prioritization_overhead=q(2, OVERHEAD_SECONDS)) if item.candidate_id == "shortest-first" else item
        for item in report.runs
    )
    finding = next(item for item in changed(report, runs=runs).findings() if item.candidate_id == "shortest-first")
    assert finding.gate == "fail"
    assert "overhead_threshold_failure" in finding.reasons
    missing = PilotReport(schema_version=1, manifest=observed)
    assert all("missing_candidate_or_baseline" in finding.reasons for finding in missing.findings())
    assert all("too_few_distinct_builds" in finding.reasons for finding in missing.findings())


def test_price_precision_and_temporal_invariants(manifest):
    build = manifest.builds[0]
    with pytest.raises(ValueError, match="availability precedes"):
        changed(build, outcomes_available_at="2025-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="Duplicate build"):
        changed(manifest, builds=(build, build))
    with pytest.raises(ValueError, match="Unique candidates"):
        changed(manifest, candidates=("base-order", "base-order"))
    wall_time = q(0, WALL_SECONDS)
    with pytest.raises(ValueError, match="budget or variable-price"):
        changed(manifest, budget=wall_time)
    occupancy = q(0)
    with pytest.raises(ValueError, match="overhead threshold"):
        changed(manifest.thresholds, maximum_overhead=occupancy)
    existing_run = run(manifest)
    event = existing_run.events[0]
    with pytest.raises(ValueError, match="measured overhead"):
        changed(existing_run, prioritization_overhead=occupancy)
    with pytest.raises(ValueError, match="Duplicate logical"):
        changed(existing_run, events=(event, event))
    with pytest.raises(ValueError, match="elapsed seconds"):
        changed(event, started=occupancy)
    later = q(2, WALL_SECONDS)
    with pytest.raises(ValueError, match="before dispatch"):
        changed(event, started=later)
    builds = tuple(changed(item, decided_at="2026-01-01T00:00:00Z") for item in manifest.builds)
    data_fingerprint = fingerprint([item.model_dump(mode="json") for item in builds])
    with pytest.raises(ValueError, match="protected test period"):
        changed(manifest, builds=builds, dataset_fingerprint=data_fingerprint)
    tiny = changed(
        build,
        tests=(PilotTest(test_id="tiny", duration=q("0.000000000001"), outcome="failed"),),
        baseline_order=("tiny",),
    )
    tiny_events = serial_events(tiny, ("tiny",), q(1))
    tiny_price = q("0.000000000001", RATE)
    with pytest.raises(ValueError):
        replay_events(tiny, tiny_events, tiny_price)


def test_unavailable_and_zero_baseline_work_cannot_pass(manifest):
    builds = tuple(
        changed(build, tests=tuple(changed(test, outcome="unknown", duration=q(0)) for test in build.tests))
        for build in manifest.builds
    )
    unknown = changed(
        manifest,
        evidence_basis="observed_ci",
        builds=builds,
        dataset_fingerprint=fingerprint([build.model_dump(mode="json") for build in builds]),
    )
    report = complete(unknown)
    assert all(finding.gate == "inconclusive" for finding in report.findings())
    assert all("zero_baseline_work" in finding.reasons for finding in report.findings())
    assert all("unresolved_or_zero_failure_outcomes" in finding.reasons for finding in report.findings())


def test_example_recovery_merges_snapshots_and_rejects_conflicts(manifest, tmp_path):
    writer = ArtifactWriter(tmp_path)
    first = run(manifest)
    other = run(manifest, "shortest-first")
    a = PilotReport(schema_version=1, manifest=manifest).append(first)
    b = PilotReport(schema_version=1, manifest=manifest).append(other)
    publish_pilot_report(a, writer)
    publish_pilot_report(b, writer)
    assert example.restore_report(tmp_path, manifest) == a.append(other)
    conflict = PilotReport(schema_version=1, manifest=manifest).append(changed(first, seed=43))
    publish_pilot_report(conflict, writer)
    with pytest.raises(ValueError, match="Conflicting"):
        example.restore_report(tmp_path, manifest)
    different = changed(manifest, pilot_id="different")
    with pytest.raises(ValueError, match="Checkpoint manifest differs"):
        example.restore_report(tmp_path, different)


def test_replay_requires_first_dispatch_origin_zero(manifest):
    source = run(manifest).events
    shifted = tuple(
        changed(
            event,
            started=q(event.started.value + 5, WALL_SECONDS),
            completed=q(event.completed.value + 5, WALL_SECONDS),
        )
        for event in source
    )
    with pytest.raises(ValueError, match="first dispatch"):
        replay_events(manifest.builds[0], shifted, manifest.variable_price)


def native_params():
    from coleman.budget import BudgetMode
    from coleman.evaluation import NAPFDVerdictMetric
    from coleman.utils.monitor_params import CollectParams

    class FixtureScenario:
        name = "synthetic-ci"
        budget_mode = BudgetMode.FIXED_TIME
        budget_value = 3.0

    metric = NAPFDVerdictMetric()
    metric.update_budget("fixed_time", 3)
    metric.update_available_time(3)
    metric.evaluate(
        [
            {"Name": name, "Duration": duration, "Verdict": verdict}
            for name, duration, verdict in (("t1", 1, 0), ("t2", 2, 1), ("t3", 4, 1))
        ]
    )
    return CollectParams(
        scenario_provider=FixtureScenario(),
        available_time=3,
        experiment=1,
        t=2,
        policy="DeclaredOrder",
        reward_function="RNFail",
        metric=metric,
        total_build_duration=7,
        prioritization_time=0.001,
        rewards=1,
        prioritization_order=["t1", "t2", "t3"],
    )


@pytest.mark.parametrize("failure_at", ["audit", "checkpoint"])
def test_native_monitor_commits_audit_before_recoverable_checkpoint(manifest, tmp_path, monkeypatch, failure_at):
    from coleman.results.parquet_sink import ParquetSink

    sink = ParquetSink(str(tmp_path / "results"))
    monitor = example.PilotMonitor(sink, manifest, "base-order", tmp_path, "native-run")
    params = native_params()
    try:
        with monkeypatch.context() as fault:

            def interrupted(*_args, **_kwargs):
                raise OSError("injected commit failure")

            if failure_at == "audit":
                fault.setattr(sink, "write_domain_record", interrupted)
            else:
                fault.setattr(example, "publish_pilot_report", interrupted)
            with pytest.raises(OSError, match="injected"):
                monitor.collect(params)
        assert monitor.report.runs == ()
        assert list((tmp_path / "pilot-reports").glob("*.json")) == []
        prior_records = list(sink.domain_records())
        assert len(prior_records) == int(failure_at == "checkpoint")
        # A real retry may have a new measured overhead. The audit contains only
        # outcome metrics and must remain identical across this partial attempt.
        params.prioritization_time = 0.002
        resumed = example.PilotMonitor(sink, manifest, "base-order", tmp_path, "native-run")
        resumed.collect(params)
        records = list(sink.domain_records())
        assert len(records) == 1
        assert records[0].measurement("pilot_failing_test_recall").value == 0.5
        assert records[0].available_at == manifest.builds[0].outcomes_available_at
        if prior_records:
            assert records == prior_records
        restored = example.restore_report(tmp_path, manifest)
        assert len(restored.runs) == 1
        resumed.collect(params)
        assert list(sink.domain_records()) == records
        assert example.restore_report(tmp_path, manifest) == restored
    finally:
        sink.close()
