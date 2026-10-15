"""Real native application workflows, bounded admission and adversarial contracts."""

import base64
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from coleman.application import (
    ApplicationError,
    ApplicationService,
    Caller,
    Handle,
    JobView,
    Limits,
    Parameters,
    ResultBundle,
    ServerPolicy,
    Submit,
    default_registry,
    register_experiment,
)
from coleman.application.commands import ApplicationCommands
from coleman.application.contracts import PortableFile
from coleman.selection import CoverageResult
from coleman.spec.models import RunSpec

CALLER = Caller(subject="local")
OTHER = Caller(subject="other")
FIXTURES = Path(__file__).parents[1] / "fixtures/application"


def request(request_id="request-1", **overrides):
    return Submit.model_validate(
        {"schema_version": 1, "request_id": request_id, "spec_id": "weighted-overlap-3", **overrides}
    )


def completed(service, command=None):
    initial = service.submit(command or request(), caller=CALLER)
    view = service.wait(initial.job_id, caller=CALLER)
    assert view.state == "completed", view
    assert view.handle is not None
    return view, service.result(view.handle, caller=CALLER)


def expect_code(code, method, *args, **kwargs):
    with pytest.raises(ApplicationError) as error:
        method(*args, **kwargs)
    assert error.value.code == code


def test_native_fixture_roundtrip_golden_identity_and_portable_caller_rebinding(tmp_path):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        view, bundle = completed(service)
        assert isinstance(bundle.result, CoverageResult)
        assert bundle.result.selected_ids == ("A", "C")
        assert bundle.result.objective.value == 1
        assert bundle.result.cost.value == 3
        assert bundle.to_json() == ResultBundle.from_json(bundle.to_json()).to_json()
        assert bundle.to_json() + "\n" == (FIXTURES / "coverage-v1.json").read_text()
        assert bundle.fingerprint == json.loads((FIXTURES / "digests-v1.json").read_text())["coverage"]
        assert service.submit(request(), caller=CALLER) == view
        expect_code("invalid_input", service.submit, request(parameters={"budget": 2}), caller=CALLER)
        for method, argument in [
            (service.status, view.job_id),
            (service.cancel, view.job_id),
            (service.result, view.handle),
            (service.export_result, view.handle),
        ]:
            expect_code("forbidden", method, argument, caller=OTHER)
        exported = service.export_result(view.handle, caller=CALLER)
        imported = service.import_result(exported, caller=OTHER)
        assert imported.handle is not None
        assert view.handle is not None
        assert imported.handle.job_id != view.job_id
        assert imported.handle.digest == view.handle.digest
        assert service.result(imported.handle, caller=OTHER) == bundle
        expect_code("forbidden", service.result, imported.handle, caller=CALLER)
        assert service.cancel(view.job_id, caller=CALLER) == view
        assert len(list(tmp_path.rglob("result.json"))) == 2
    assert not list(tmp_path.iterdir())
    with ApplicationService(default_registry(), root=tmp_path) as restarted:
        expect_code("not_found", restarted.status, view.job_id, caller=CALLER)
        restored = restarted.import_result(exported, caller=CALLER)
        assert restored.handle is not None
        assert restarted.result(restored.handle, caller=CALLER) == bundle


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": None},
        {"spec_id": "../evil"},
        {"spec_id": "module:factory"},
        {"spec_id": "https://provider"},
        {"owner": "other"},
        {"caller": "other"},
        {"parameters": {"plugin": "x.y"}},
        {"parameters": {"budget": -1}},
        {"parameters": {"budget": "NaN"}},
        {"parameters": {"budget": "Infinity"}},
        {"parameters": {"budget": "0.0000000000001"}},
        {"limits": {"deadline_seconds": float("inf")}},
        {"limits": {"max_work": True}},
        {"limits": {"max_output_bytes": 999999999}},
        {"limits": {"shell": "echo bad"}},
    ],
)
def test_malicious_or_unknown_contract_never_admitted(tmp_path, overrides):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        commands = ApplicationCommands(service, caller=CALLER)
        raw = {"schema_version": 1, "request_id": "x", "spec_id": "weighted-overlap-3", **overrides}
        reply = json.loads(commands.invoke("submit", json.dumps(raw)))
        assert reply["error"] == "invalid_input"
        assert not list(tmp_path.rglob("request.json"))


def test_command_input_caps_duplicates_and_stable_errors(tmp_path):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        commands = ApplicationCommands(service, caller=CALLER)
        assert json.loads(commands.invoke("submit", " " * 16385))["error"] == "resource_limit"
        assert (
            json.loads(commands.invoke("submit", '{"schema_version":1,"schema_version":1}'))["error"] == "invalid_input"
        )
        assert json.loads(commands.invoke("submit", "[" * 2000))["error"] == "invalid_input"
        assert json.loads(commands.invoke("other", "{}"))["error"] == "invalid_input"  # ty:ignore[invalid-argument-type]
        expect_code("unknown_spec", service.submit, request(spec_id="unknown"), caller=CALLER)
        expect_code("not_found", service.status, "unknown", caller=CALLER)
        expect_code("invalid_input", service.submit, {}, caller=CALLER)
        with pytest.raises(ValueError):
            ApplicationService((), root=tmp_path)
        duplicated = default_registry() * 2
        with pytest.raises(ValueError):
            ApplicationService(duplicated, root=tmp_path)


def test_json_decimal_literals_are_validated_before_binary64_rounding(tmp_path):
    prefix = '{"schema_version":1,"request_id":"decimal","spec_id":"weighted-overlap-3","parameters":{"budget":'
    with pytest.raises(ValueError):
        Submit.from_json(prefix + "3.0000000000000001}}")
    exact = Submit.from_json(prefix + "0.000000000001}}")
    assert exact.parameters.budget == Decimal("0.000000000001")
    with ApplicationService(default_registry(), root=tmp_path) as service:
        commands = ApplicationCommands(service, caller=CALLER)
        reply = json.loads(commands.invoke("submit", prefix + "3.0000000000000001}}"))
        assert reply["error"] == "invalid_input"
        assert not list(tmp_path.rglob("request.json"))


def test_native_work_caps_zero_and_exact_decimal_override(tmp_path):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        _, zero_work = completed(service, request("work-zero", limits={"max_work": 0}))
        assert zero_work.result.work_used == 0
        assert zero_work.result.status != "optimal"
        _, zero_budget = completed(service, request("budget-zero", parameters={"budget": "0"}))
        assert zero_budget.result.cost.value == 0
        assert zero_budget.result.objective.value == 0
        assert Parameters(budget=Decimal("0.000000000001")).model_dump(mode="json")["budget"] == "0.000000000001"
        expect_code("resource_limit", service.submit, request("too-small", limits={"max_candidates": 2}), caller=CALLER)
        _, allocation = completed(service, request("allocation", spec_id="signed-ci-resource-3"))
        assert allocation.kind == "allocation"
        expect_code(
            "invalid_input",
            service.submit,
            request("allocation-budget", spec_id="signed-ci-resource-3", parameters={"budget": 1}),
            caller=CALLER,
        )
        expect_code(
            "resource_limit",
            service.submit,
            request("allocation-small", spec_id="signed-ci-resource-3", limits={"max_candidates": 1}),
            caller=CALLER,
        )


def test_effective_server_limits_and_concurrent_idempotency(tmp_path):
    policy = ServerPolicy(caps=Limits(max_work=20, max_candidates=5))
    with ApplicationService(default_registry(), root=tmp_path, policy=policy) as service:
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda _: service.submit(request(), caller=CALLER), range(16)))
        assert len({view.job_id for view in jobs}) == 1
        assert jobs[0].limits.max_work == 20
        assert jobs[0].limits.max_candidates == 5
        view = service.wait(jobs[0].job_id, caller=CALLER)
        assert view.state == "completed"
        assert len(list(tmp_path.rglob("result.json"))) == 1
        assert not list(tmp_path.rglob("request.json"))


def test_queue_saturation_and_cancel_before_start(tmp_path):
    policy = ServerPolicy(max_running=1, max_queued=1)
    with ApplicationService(default_registry(), root=tmp_path, policy=policy) as service:
        # Hold the lifecycle lock only to make queue admission deterministic.
        with service._condition:
            running = service.submit(request("running", spec_id="native-ci-smoke-v1"), caller=CALLER)
            view = service.submit(request(), caller=CALLER)
            assert view.state == "queued"
            expect_code("resource_limit", service.submit, request("extra"), caller=CALLER)
            cancelled = service.cancel(view.job_id, caller=CALLER)
            service.cancel(running.job_id, caller=CALLER)
        assert cancelled.state == "cancelled"
        assert service.cancel(view.job_id, caller=CALLER) == cancelled
        assert service.submit(request(), caller=CALLER) == cancelled
        assert not list(tmp_path.rglob("result.json"))


def test_cancel_running_reaps_worker_without_false_completed_artifact(tmp_path):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        with service._condition:
            view = service.submit(request(), caller=CALLER)
            job = service._jobs[view.job_id]
            assert job.view.state == "running"
            process = job.process
            assert process is not None
            cancelled = service.cancel(view.job_id, caller=CALLER)
        assert cancelled.state == "cancelled"
        assert process.poll() is not None
        expect_code(
            "unavailable_result",
            service.result,
            Handle(schema_version=1, job_id=view.job_id, digest="sha256:" + "0" * 64),
            caller=CALLER,
        )
        assert not list(tmp_path.rglob("result.json"))


def test_deadline_expiry_retention_and_bounded_tombstones(tmp_path):
    with ApplicationService(
        default_registry(), root=tmp_path, policy=ServerPolicy(max_jobs=2, retention_seconds=0.03)
    ) as service:
        early = service.submit(request("early", limits={"deadline_seconds": 0.000000001}), caller=CALLER)
        assert service.wait(early.job_id, caller=CALLER).state == "expired"
        view, _ = completed(service)
        time.sleep(0.04)
        assert service.status(view.job_id, caller=CALLER).state == "expired"
        expect_code("expired", service.result, view.handle, caller=CALLER)
        expect_code("resource_limit", service.submit, request("third"), caller=CALLER)
        assert service.submit(request(), caller=CALLER).state == "expired"
        assert not list(tmp_path.rglob("result.json"))


@pytest.mark.parametrize(
    "policy",
    [
        ServerPolicy(max_retained_bytes=1024),
        ServerPolicy(max_active_memory_bytes=1048576),
    ],
)
def test_aggregate_reservation_before_work(tmp_path, policy):
    with ApplicationService(default_registry(), root=tmp_path, policy=policy) as service:
        expect_code("resource_limit", service.submit, request(), caller=CALLER)
        assert not list(tmp_path.rglob("request.json"))


@pytest.mark.parametrize("limits", [Limits(max_output_bytes=1024), Limits(max_memory_bytes=1048576)])
def test_worker_resource_refusal_does_not_publish_completed_result(tmp_path, limits):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        view = service.submit(request(limits=limits), caller=CALLER)
        final = service.wait(view.job_id, caller=CALLER)
        assert final.state == "failed"
        assert final.error == "resource_limit"
        assert final.handle is None
        assert not list(tmp_path.rglob("result.json"))


def test_corruption_wrong_digest_and_portable_versions(tmp_path):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        view, bundle = completed(service)
        wrong = Handle(schema_version=1, job_id=view.job_id, digest="sha256:" + "0" * 64)
        expect_code("invalid_input", service.result, wrong, caller=CALLER)
        for version in [None, 0, 2, True, "1"]:
            payload = json.loads(bundle.to_json())
            payload["schema_version"] = version
            expect_code("invalid_input", service.import_result, json.dumps(payload), caller=CALLER)
        payload = json.loads(bundle.to_json())
        payload["result"]["objective"]["value"] = "99"
        expect_code("invalid_input", service.import_result, json.dumps(payload), caller=CALLER)
        expect_code("resource_limit", service.import_result, " " * 262145, caller=CALLER)
        path = next(tmp_path.rglob("result.json"))
        path.write_text("partial")
        expect_code("unavailable_result", service.result, view.handle, caller=CALLER)
        path.unlink()
        expect_code("unavailable_result", service.result, view.handle, caller=CALLER)
    service.close()
    expect_code("unavailable_result", service.submit, request(), caller=CALLER)


def test_portable_file_integrity_and_traversal():
    content = b"native legacy bytes"
    valid = {
        "name": "native.parquet",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "content_base64": base64.b64encode(content).decode(),
    }
    assert PortableFile.model_validate(valid).size_bytes == len(content)
    for change in [
        {"name": "../x"},
        {"name": "/tmp/x"},
        {"name": "a\\b"},
        {"name": "a//b"},
        {"name": ""},
        {"size_bytes": 0},
        {"sha256": "0" * 64},
        {"content_base64": "!!!"},
    ]:
        with pytest.raises(ValidationError):
            PortableFile.model_validate({**valid, **change})


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "completed"},
        {"state": "failed"},
        {"state": "expired"},
        {"state": "queued", "error": "unavailable_result"},
        {"state": "running", "handle": {"schema_version": 1, "job_id": "x", "digest": "sha256:" + "0" * 64}},
    ],
)
def test_job_receipts_reject_false_completion_and_contradictory_errors(changes):
    with pytest.raises(ValueError):
        JobView.model_validate(
            {
                "schema_version": 1,
                "job_id": "job",
                "request_id": "request",
                "spec_id": "weighted-overlap-3",
                "limits": {},
                **changes,
            }
        )


def native_registration(tmp_path):
    (tmp_path / "synthetic").mkdir()
    pl.DataFrame(
        {
            "BuildId": [1, 1, 1, 2, 2, 2],
            "Name": ["A", "B", "C"] * 2,
            "Duration": [2.0, 2.0, 1.0] * 2,
            "Verdict": [1, 0, 1] * 2,
            "CalcPrio": [0] * 6,
            "LastRun": ["2026-01-01"] * 3 + ["2026-01-02"] * 3,
        }
    ).write_parquet(tmp_path / "synthetic/features-engineered.parquet")
    spec = RunSpec.model_validate(
        {
            "execution": {"parallel_pool_size": 1, "independent_executions": 1, "seed": 42},
            "experiment": {
                "datasets_dir": str(tmp_path),
                "datasets": ["synthetic"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [3]},
            },
            "checkpoint": {"enabled": False},
        }
    )
    return spec, register_experiment("native-ci", spec)


def test_real_native_runspec_immutable_inputs_legacy_artifacts_and_portable_roundtrip(tmp_path):
    spec, configuration = native_registration(tmp_path)
    spec.execution.seed = 99
    (tmp_path / "synthetic/features-engineered.parquet").unlink()
    with ApplicationService((configuration,), root=tmp_path / "service") as service:
        expect_code(
            "resource_limit", service.submit, request(spec_id="native-ci", limits={"max_work": 1}), caller=CALLER
        )
        view, bundle = completed(service, request(spec_id="native-ci"))
        assert bundle.kind == "experiment"
        assert bundle.result.work_units == 2
        artifacts = [file for file in bundle.result.files if file.name.endswith(".parquet")]
        assert len(artifacts) == 1
        table = pl.read_parquet(base64.b64decode(artifacts[0].content_base64))
        assert table.height == 2
        assert "fitness" in table.columns
        resolved = next(file for file in bundle.result.files if file.name == "spec.resolved.json")
        assert json.loads(base64.b64decode(resolved.content_base64))["execution"]["seed"] == 42
        portable = service.export_result(view.handle, caller=CALLER)
        assert ResultBundle.from_json(portable) == bundle
        assert sorted(
            path.name for path in next(tmp_path.joinpath("service").glob("*/")).joinpath(view.job_id).iterdir()
        ) == ["result.json"]


def test_cli_library_equivalence_and_import_boundary(tmp_path):
    with ApplicationService(default_registry(), root=tmp_path) as service:
        _, expected = completed(service)
    output = subprocess.run(
        [sys.executable, "-m", "coleman.cli", "application", "--request", request().to_json()],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    actual = ResultBundle.model_validate(json.loads(output.stdout)["data"])
    assert actual == expected
    script = """
import sys
from coleman.application import ApplicationService
assert 'coleman.cli' not in sys.modules
assert not any(name == 'mcp' or name.startswith('mcp.') for name in sys.modules)
assert not any(name.startswith('coleman.transports') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


@pytest.mark.parametrize(
    "limits,error",
    [
        ({"deadline_seconds": 0.000000001}, "expired"),
        ({"max_output_bytes": 1024}, "resource_limit"),
    ],
)
def test_cli_terminal_errors_keep_the_versioned_response_envelope(limits, error):
    output = subprocess.run(
        [sys.executable, "-m", "coleman.cli", "application", "--request", request(limits=limits).to_json()],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert output.returncode == 2
    reply = json.loads(output.stdout)
    assert reply["schema_version"] == 1
    assert reply["ok"] is False
    assert reply["error"] == error
    assert reply["data"]["handle"] is None


@pytest.mark.parametrize("seed,repetitions", [(-1, 1), (2**64, 1), (2**64 - 1, 2)])
def test_native_rng_seed_and_every_repetition_are_validated_before_admission(tmp_path, seed, repetitions):
    spec, _ = native_registration(tmp_path)
    spec.execution.seed = seed
    spec.execution.independent_executions = repetitions
    with pytest.raises(ValueError, match="seed"):
        register_experiment("invalid-seed", spec)


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_valid_rng_endpoints_execute_through_the_real_native_runner(tmp_path, seed):
    spec, _ = native_registration(tmp_path)
    spec.execution.seed = seed
    configuration = register_experiment("rng-endpoint", spec)
    with ApplicationService((configuration,), root=tmp_path / "service") as service:
        _, bundle = completed(service, request(spec_id="rng-endpoint"))
        assert bundle.kind == "experiment"
