"""Shared CLI lifecycle, trusted registration and finite admission/compatibility boundaries."""

import argparse
import json
import time

import pytest

from coleman.application.workflow_cli import configure_parser, run_command
from coleman.application.workflow_registry import owned_fixture
from coleman.application.workflows import WorkflowService
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy


def invoke(root, operation, *arguments):
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    run_command(parser.parse_args(["--workspace", str(root), operation, *arguments]))


def test_cli_actual_count_budget_native_lifecycle_and_portable_evidence(tmp_path, capsys):
    invoke(tmp_path, "init-fixture")
    manifest = json.loads(capsys.readouterr().out)
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps(manifest))
    source = tmp_path / "datasets" / f"{manifest['sha256'][7:]}.parquet"
    invoke(tmp_path, "register", "--input", str(source), "--manifest", str(reviewed))
    assert json.loads(capsys.readouterr().out) == manifest
    request = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "policies": ["Random"],
        "seeds": [0],
        "budget": {"mode": "subset_size", "unit": "count", "values": [2.0]},
    }
    invoke(tmp_path, "plan", "--request", json.dumps(request))
    plan = json.loads(capsys.readouterr().out)
    invoke(
        tmp_path,
        "execute",
        "--request",
        json.dumps({"schema_version": 1, "request_id": "cli", "plan_id": plan["plan_id"]}),
    )
    job = json.loads(capsys.readouterr().out)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        invoke(tmp_path, "status", "--job", job["job_id"])
        job = json.loads(capsys.readouterr().out)
        if job["state"] == "completed" and not job["worker_active"]:
            break
        assert job["state"] in ("queued", "running", "completed"), job
        time.sleep(0.05)
    assert job["state"] == "completed"
    invoke(tmp_path, "inspect", "--job", job["job_id"], "--limit", "1")
    inspection = json.loads(capsys.readouterr().out)
    assert all(row["tests_ran"] == 2 for row in inspection["trials"][0]["rows"])
    invoke(tmp_path, "report", "--job", job["job_id"])
    report = json.loads(capsys.readouterr().out)
    assert len(report["trials"]) == 1
    assert len(report["physical_attempts"]) == 1
    for operation in ("cancel", "resume"):
        invoke(tmp_path, operation, "--job", job["job_id"])
        assert json.loads(capsys.readouterr().out)["state"] == "completed"
    invoke(tmp_path, "export", "--job", job["job_id"])
    bundle = capsys.readouterr().out
    portable = tmp_path / "portable.json"
    portable.write_text(bundle)
    invoke(tmp_path, "import", "--input", str(portable))
    assert json.loads(capsys.readouterr().out)["read_only"] is True
    invoke(tmp_path, "cleanup")
    assert json.loads(capsys.readouterr().out)["expired_jobs"] == []
    manifest["rows"] += 1
    reviewed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        invoke(tmp_path, "register", "--input", str(source), "--manifest", str(reviewed))


@pytest.mark.parametrize("operation", ["register", "register-karaf", "plan", "execute", "import", "status", "inspect"])
def test_missing_cli_configuration_is_never_guessed(tmp_path, operation):
    with pytest.raises(ValueError):
        invoke(tmp_path, operation)


def test_cli_rejects_oversized_portable_input_before_decode(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b"x" * 2097153)
    with pytest.raises(ValueError, match="2 MiB"):
        invoke(tmp_path, "import", "--input", str(path))
    with pytest.raises(ValueError, match="bounded"):
        invoke(tmp_path, "plan", "--request", "x" * 65537)


def test_queue_tombstone_registry_and_active_memory_caps(tmp_path, monkeypatch):
    policy = HostPolicy(schema_version=1, max_jobs=2, max_datasets=1, max_running=1, max_queued=0)
    service = WorkflowService(tmp_path, policy=policy)
    manifest = service.register(owned_fixture(tmp_path))
    another_manifest = manifest.model_copy(update={"dataset_id": "another-label"})
    with pytest.raises(ValueError, match="registry is full"):
        service.register(another_manifest)
    preview = service.plan(PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id))
    monkeypatch.setattr(service, "_start", lambda: None)
    job = service.execute(ExecutePlan(schema_version=1, request_id="one", plan_id=preview.plan_id))
    second_request = ExecutePlan(schema_version=1, request_id="two", plan_id=preview.plan_id)
    with pytest.raises(ValueError, match="queue capacity"):
        service.execute(second_request)
    assert service.cancel(job.job_id).state == "cancelled"
    second = service.execute(ExecutePlan(schema_version=1, request_id="two", plan_id=preview.plan_id))
    service.cancel(second.job_id)
    third_request = ExecutePlan(schema_version=1, request_id="three", plan_id=preview.plan_id)
    with pytest.raises(ValueError, match="tombstone"):
        service.execute(third_request)
    constrained = WorkflowService(
        tmp_path / "memory", policy=HostPolicy(schema_version=1, max_active_memory_bytes=67108864)
    )
    constrained.register(owned_fixture(constrained.store.root))
    memory_request = PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id)
    with pytest.raises(ValueError, match="aggregate active memory"):
        constrained.plan(memory_request)


def test_changed_runtime_and_native_thread_profile_reject_before_admission(tmp_path, monkeypatch):
    import coleman.application.workflows as module

    service = WorkflowService(tmp_path)
    manifest = service.register(owned_fixture(tmp_path))
    preview = service.plan(PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id))
    request = ExecutePlan(schema_version=1, request_id="compatible-only", plan_id=preview.plan_id)
    actual = InstalledRuntime.capture()
    incompatible = actual.model_copy(update={"package_fingerprint": "sha256:" + "0" * 64})
    with monkeypatch.context() as patch:
        patch.setattr(InstalledRuntime, "capture", lambda: incompatible)
        with pytest.raises(ValueError, match="incompatible"):
            service.execute(request)
    threads = module.pl.thread_pool_size()
    with monkeypatch.context() as patch:
        patch.setattr(module.pl, "thread_pool_size", lambda: threads + 1)
        with pytest.raises(ValueError, match="incompatible"):
            service.execute(request)
    with service.store.transaction() as connection:
        assert not service.store.jobs(connection)


@pytest.mark.parametrize(
    "limits",
    [
        WorkflowLimits(attempt_seconds=0.0001),
        WorkflowLimits(max_memory_bytes=67108864),
        WorkflowLimits(max_output_bytes=65536),
    ],
)
def test_real_worker_resource_exhaustion_never_certifies_completion(tmp_path, limits):
    from tests.workflows.test_execution import setup_workflow, wait

    service, preview = setup_workflow(tmp_path, limits=limits)
    job = service.execute(ExecutePlan(schema_version=1, request_id="finite-resource-budget", plan_id=preview.plan_id))
    view = wait(service, job.job_id, "failed")
    assert view.result_digest is None
    with pytest.raises(ValueError, match="completed report"):
        service.report(job.job_id)
