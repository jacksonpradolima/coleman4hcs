"""Actual native end-to-end matrix, immutable completion, relocation and observed failure retention."""

import json
import math
import shutil
from pathlib import Path

import pytest

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.baselines import read_baseline_matrix, run_baseline_matrix
from coleman.corpus.baseline_results import BaselineMatrixReceipt, BaselineMatrixReport
from tests.corpus.baseline_support import baseline_plan, change


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    """Run all twelve internal candidates through real process workers, metrics, sinks and checkpoints once."""
    root = tmp_path_factory.mktemp("real-baseline-matrix")
    original = baseline_plan()
    plan = change(original, seeds=(42,), budgets=(original.budgets[0],))
    report = run_baseline_matrix(plan, root)
    receipt = next(root.glob("baseline-matrices/*/*/complete.json"))
    return receipt.parent, report


def test_actual_complete_matrix_and_exact_retry(matrix):
    root, report = matrix
    assert len(report.cells) == 13
    assert sum(cell.state == "completed" for cell in report.cells) == 12
    assert sum(cell.state == "incompatible" for cell in report.cells) == 1
    assert all(len(cell.observations) == 3 for cell in report.cells if cell.state == "completed")
    assert all(
        len(cell.attempts) == 2 and cell.resource_evidence == "observed"
        for cell in report.cells
        if cell.state == "completed"
    )
    assert report.complete
    assert read_baseline_matrix(root) == report
    assert BaselineMatrixReport.from_json(report.to_json()) == report
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert run_baseline_matrix(report.plan, root.parents[2]) == report
    assert before == {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_fresh_native_matrices_keep_exact_source_budget_across_polars_threads(tmp_path, monkeypatch):
    original = baseline_plan()
    build = original.source.builds[0]
    tests = tuple(
        change(test, duration=change(test.duration, value=value))
        for test, value in zip(build.tests, (1e16, 1.0, 1.0), strict=True)
    )
    plan = change(
        original,
        source=change(original.source, builds=(change(build, tests=tests),)),
        methods=("seeded_random_v1",),
        seeds=(42,),
        budgets=(original.budgets[0],),
    )
    reports = []
    for threads in ("1", "2"):
        monkeypatch.setenv("POLARS_MAX_THREADS", threads)
        output = tmp_path / threads
        report = run_baseline_matrix(plan, output)
        root = next(output.glob("baseline-matrices/*/*/complete.json")).parent
        assert read_baseline_matrix(root) == report
        assert report.cells[0].state == "completed"
        path = next(root.glob("native/*/baseline-observations/*.json"))
        quantities = {m["name"]: m["value"] for m in json.loads(path.read_bytes())["record"]["measurements"]}
        assert quantities["source_full_runner_seconds"] == math.fsum((1e16, 1.0, 1.0))
        assert quantities["declared_native_work_budget_seconds"] == 5000000000000001.0
        reports.append(report)
    assert reports[0].mathematical_fingerprint == reports[1].mathematical_fingerprint


def test_completed_matrix_relocates_without_following_old_input_paths(matrix, tmp_path):
    root, report = matrix
    destination = tmp_path / "moved"
    shutil.copytree(root, destination)
    assert read_baseline_matrix(destination) == report


@pytest.mark.parametrize("target", ["raw", "parquet", "checkpoint", "attempt", "receipt"])
def test_changed_or_missing_actual_evidence_rejects(matrix, tmp_path, target):
    root, _ = matrix
    destination = tmp_path / "changed"
    shutil.copytree(root, destination)
    patterns = {
        "raw": "native/*/baseline-observations/*.json",
        "parquet": "native/*/results/**/*.parquet",
        "checkpoint": "native/*/checkpoints/**/*.pkl",
        "attempt": "attempts/*/*-terminal.json",
        "receipt": "complete.json",
    }
    path = next(destination.glob(patterns[target]))
    if target == "receipt":
        content = json.loads(path.read_bytes())
        content["schema_version"] = 2
        path.write_text(json.dumps(content))
    else:
        path.unlink()
    with pytest.raises((ValueError, OSError)):
        read_baseline_matrix(destination)


def test_lock_contention_never_marks_active_work_complete(tmp_path):
    path = tmp_path / "matrix.lock"
    with baseline_lock(path), pytest.raises(BaselineBusyError), baseline_lock(path):
        pytest.fail("contending lock must reject")
    with baseline_lock(path):
        assert path.exists()


def test_resource_failed_cell_is_retained_and_idempotent(tmp_path):
    original = baseline_plan()
    plan = change(
        original,
        methods=("seeded_random_v1",),
        seeds=(42,),
        budgets=(original.budgets[0],),
        limits=change(original.limits, maximum_elapsed_seconds=1e-12),
    )
    report = run_baseline_matrix(plan, tmp_path)
    assert report.complete
    cell = report.cells[0]
    assert cell.state == "failed"
    assert cell.reasons == ("native_simulation_elapsed_limit_exceeded",)
    assert not cell.observations
    assert len(cell.attempts) == 2
    assert run_baseline_matrix(plan, tmp_path) == report
    root = next(tmp_path.glob("baseline-matrices/*/*/complete.json")).parent
    assert read_baseline_matrix(root) == report
    receipt = BaselineMatrixReceipt.from_json((root / "complete.json").read_bytes())
    assert receipt.mathematical_fingerprint == report.mathematical_fingerprint


def test_real_process_interruption_resumes_checkpoint_without_inventing_cost(tmp_path):
    """Kill the real host/worker group after a committed build, then resume its unchanged logical cell."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from contextlib import suppress
    from datetime import timedelta

    original = baseline_plan()
    template = original.source.builds[0]
    builds = tuple(
        change(
            template,
            step=step,
            member_id=f"interrupt-build-{step}",
            decided_at=template.decided_at + timedelta(days=step),
            source_metadata=change(
                template.source_metadata,
                observation_id=f"interrupt-{step}",
                group_id=f"interrupt-{step}",
                event_at=template.source_metadata.event_at + timedelta(days=step),
                available_at=template.source_metadata.available_at + timedelta(days=step),
            ),
        )
        for step in range(1, 65)
    )
    plan = change(
        original,
        source=change(original.source, builds=builds),
        methods=("seeded_random_v1",),
        seeds=(42,),
        budgets=(original.budgets[0],),
    )
    plan_file = tmp_path / "source-plan.json"
    plan_file.write_text(plan.to_json())
    script = tmp_path / "execute.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from coleman.corpus.baseline_contracts import BaselinePlan\n"
        "from coleman.application.baselines import run_baseline_matrix\n"
        "if __name__ == '__main__':\n"
        "    run_baseline_matrix(BaselinePlan.from_json(Path(sys.argv[1]).read_bytes()), sys.argv[2])\n"
    )
    output = tmp_path / "execution"
    with (tmp_path / "worker.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), str(plan_file), str(output)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 60
            progress = None
            while time.monotonic() < deadline and process.poll() is None:
                paths = tuple(output.glob("baseline-matrices/*/*/native/*/checkpoints/*/progress_ex1.json"))
                if paths:
                    try:
                        frontier = json.loads(paths[0].read_bytes())["step_committed"]
                    except (OSError, ValueError, KeyError):
                        frontier = 0
                    if 2 <= frontier < 60:
                        progress = paths[0]
                        break
                time.sleep(0.01)
            assert progress is not None, (tmp_path / "worker.log").read_text()
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
    assert not tuple(output.glob("baseline-matrices/*/*/complete.json"))
    observations = tuple(output.glob("baseline-matrices/*/*/native/*/baseline-observations/*.json"))
    assert observations
    before = {path: path.read_bytes() for path in observations}
    report = run_baseline_matrix(plan, output)
    assert report.cells[0].state == "completed"
    assert len(report.cells[0].observations) == 64
    assert report.cells[0].unclosed_attempts == 1
    assert report.cells[0].resource_evidence == "unavailable"
    assert all(path.read_bytes() == payload for path, payload in before.items())
    root = next(output.glob("baseline-matrices/*/*/complete.json")).parent
    assert read_baseline_matrix(root) == report


def test_nested_observation_roundtrip_preserves_domain_artifact_bytes(matrix):
    """Regression: an omitted reward becoming explicit null must not break replay ahead of a checkpoint."""
    from coleman.corpus.baseline_results import BaselineObservation

    root, _ = matrix
    for path in root.glob("native/*/baseline-observations/*.json"):
        observation = BaselineObservation.from_json(path.read_bytes())
        again = BaselineObservation.from_json(observation.to_json())
        assert observation.record.to_json() == again.record.to_json()
        assert '"learning_reward":null' in again.record.to_json()
        domain = path.parents[1] / "results/domain-records" / f"{observation.record.identity_digest[7:]}.json"
        assert domain.read_text() == again.record.to_json() + "\n"


@pytest.mark.parametrize("mutation", ["resource", "failure", "raw_membership"])
def test_self_consistent_report_cannot_override_actual_native_evidence(matrix, tmp_path, mutation):
    """Rehashing an edited report/receipt cannot create observed costs, a failed attempt or different raw membership."""
    root, original = matrix
    destination = tmp_path / "forged"
    shutil.copytree(root, destination)
    cell = next(cell for cell in original.cells if cell.state == "completed")
    if mutation == "resource":
        altered = change(cell, resource_evidence="unavailable")
    elif mutation == "failure":
        altered = change(cell, state="failed", reasons=("invented_failure",))
    else:
        altered = change(cell, observations=("sha256:" + "f" * 64, *cell.observations[1:]))
    report = change(
        original, cells=tuple(altered if value.cell_id == cell.cell_id else value for value in original.cells)
    )
    data = (report.to_json() + "\n").encode()
    (destination / "reports" / f"{report.fingerprint[7:]}.json").write_bytes(data)
    receipt = BaselineMatrixReceipt(
        schema_version=1,
        plan_fingerprint=report.plan.fingerprint,
        runtime_fingerprint=report.producer_fingerprint,
        report_fingerprint=report.fingerprint,
        mathematical_fingerprint=report.mathematical_fingerprint,
        report_size_bytes=len(data),
    )
    (destination / "complete.json").write_text(receipt.to_json() + "\n")
    with pytest.raises(ValueError, match="resource completeness|failed physical attempt|raw observation membership"):
        read_baseline_matrix(destination)


@pytest.mark.parametrize(
    "mutation",
    [
        "ranking",
        "selection",
        "identity",
        "producer",
        "measurements",
        "unit",
        "censoring",
        "missing_work",
        "fractional_count",
    ],
)
def test_raw_observation_contract_rejects_ambiguous_evidence(matrix, mutation):
    """Raw results cannot hide scientific missingness, change metric definitions or detach their native identity."""
    from coleman.corpus.baseline_results import BaselineObservation

    root, _ = matrix
    path = next(root.glob("native/*/baseline-observations/*.json"))
    observation = BaselineObservation.from_json(path.read_bytes())
    record = observation.record
    if mutation == "ranking":
        updates = {"ranking": ("a", "a", "b")}
    elif mutation == "selection":
        updates = {"selected": tuple(reversed(observation.ranking))}
    elif mutation == "identity":
        updates = {"record": change(record, execution_id="wrong-logical-cell")}
    elif mutation == "producer":
        producer = change(record.provenance, producer="other-producer")
        updates = {"record": change(record, provenance=producer)}
    elif mutation == "measurements":
        updates = {"record": change(record, measurements=record.measurements[:-1])}
    elif mutation == "censoring":
        updates = {
            "first_failure_state": "right_censored" if observation.first_failure_state == "detected" else "detected"
        }
    else:
        selected_name = "missing_input_fallback_count" if mutation == "fractional_count" else "selected_runner_seconds"
        target = record.measurement(selected_name)
        if mutation == "unit":
            altered = change(target, unit="ms")
        elif mutation == "fractional_count":
            altered = change(target, value=0.5)
        else:
            altered = change(target, value=None, status="unavailable")
        measures = tuple(altered if value.name == selected_name else value for value in record.measurements)
        updates = {"record": change(record, measurements=measures)}
    with pytest.raises(ValueError):
        change(observation, **updates)


@pytest.mark.parametrize("stop_type", [KeyboardInterrupt, SystemExit])
def test_graceful_host_interrupt_audits_partial_grid_then_stops_caller(tmp_path, monkeypatch, stop_type):
    """Fault injection verifies a stop request remains a stop request after durable incomplete evidence."""
    from coleman.application import baselines
    from coleman.corpus.baseline_results import BaselineAttempt

    original = baseline_plan()
    plan = change(original, methods=("seeded_random_v1", "ucb_c1_v1"), seeds=(42,), budgets=(original.budgets[0],))

    def interrupted_native_call(spec, extension):
        """Inject only a host stop at the API boundary; real worker/CP tests cover native execution separately."""
        raise stop_type(23)

    monkeypatch.setattr(baselines, "run_with_extension", interrupted_native_call)
    with pytest.raises(stop_type) as stopped:
        run_baseline_matrix(plan, tmp_path)
    assert stopped.value.args == (23,)
    report_path = next(tmp_path.glob("baseline-matrices/*/*/reports/*.json"))
    report = BaselineMatrixReport.from_json(report_path.read_bytes())
    assert not report.complete
    assert {cell.state for cell in report.cells} == {"aborted", "unavailable"}
    assert not (report_path.parent.parent / "complete.json").exists()
    attempts = tuple(
        BaselineAttempt.from_json(path.read_bytes()) for path in report_path.parent.parent.glob("attempts/*/*.json")
    )
    assert len(attempts) == 2
    terminal = next(attempt for attempt in attempts if attempt.event == "terminal")
    assert terminal.state == "aborted"
    assert terminal.host_elapsed.value is not None
    assert terminal.worker_elapsed.value is None
    assert terminal.worker_peak.value is None


def test_actual_completed_native_corruption_is_not_reclassified_as_a_failed_run(tmp_path, monkeypatch):
    """Regression for PR334: the real native worker completes, then changed bytes must propagate an integrity error."""
    from coleman.application import baselines

    original = baseline_plan()
    plan = change(original, methods=("seeded_random_v1",), seeds=(42,), budgets=(original.budgets[0],))
    actual = baselines.run_with_extension

    def corrupt_after_native_completion(spec, extension):
        """Run the actual principal component; corrupt one byte inventory only after its receipt exists."""
        result = actual(spec, extension)
        assert result.artifacts_dir is not None
        native_root = Path(result.artifacts_dir)
        assert (native_root / "baseline-native-complete.json").exists()
        target = next((native_root / "results").rglob("*.parquet"))
        with target.open("ab") as stream:
            stream.write(b"owned-corruption")
        return result

    monkeypatch.setattr(baselines, "run_with_extension", corrupt_after_native_completion)
    with pytest.raises(ValueError, match="bytes or artifact membership changed"):
        run_baseline_matrix(plan, tmp_path)
    assert not tuple(tmp_path.glob("baseline-matrices/*/*/complete.json"))
    assert not tuple(tmp_path.glob("baseline-matrices/*/*/attempts/*/*-terminal.json"))
    with pytest.raises(ValueError, match="bytes or artifact membership changed"):
        run_baseline_matrix(plan, tmp_path)


def test_reader_rechecks_existing_native_completion_even_for_a_failed_cell(matrix, tmp_path):
    """A consistently rehashed failed inventory cannot hide corrupt bytes behind an existing native receipt."""
    from coleman.corpus.baseline_results import BaselineAttempt

    root, original = matrix
    destination = tmp_path / "failed-with-native-receipt"
    shutil.copytree(root, destination)
    cell = next(cell for cell in original.cells if cell.state == "completed")
    ledger = destination / "attempts" / cell.cell_id[7:]
    terminal_path = next(ledger.glob("*-terminal.json"))
    terminal = BaselineAttempt.from_json(terminal_path.read_bytes())
    failed = change(terminal, state="failed", reasons=("native_execution_failed",))
    terminal_path.write_text(failed.to_json() + "\n")
    events = tuple(BaselineAttempt.from_json(path.read_bytes()) for path in ledger.glob("*.json"))
    events = tuple(sorted(events, key=lambda event: (event.recorded_at, event.attempt_id, event.event)))
    changed_cell = change(
        cell, state="failed", reasons=failed.reasons, attempts=tuple(event.fingerprint for event in events)
    )
    cells = tuple(changed_cell if entry.cell_id == cell.cell_id else entry for entry in original.cells)
    report = change(original, cells=cells)
    data = (report.to_json() + "\n").encode()
    (destination / "reports" / f"{report.fingerprint[7:]}.json").write_bytes(data)
    receipt = BaselineMatrixReceipt(
        schema_version=1,
        plan_fingerprint=report.plan.fingerprint,
        runtime_fingerprint=report.producer_fingerprint,
        report_fingerprint=report.fingerprint,
        mathematical_fingerprint=report.mathematical_fingerprint,
        report_size_bytes=len(data),
    )
    (destination / "complete.json").write_text(receipt.to_json() + "\n")
    target = next((destination / "native" / cell.native_run_id / "results").rglob("*.parquet"))
    with target.open("ab") as stream:
        stream.write(b"owned-byte-corruption")
    with pytest.raises(ValueError, match="bytes or artifact membership changed"):
        read_baseline_matrix(destination)
