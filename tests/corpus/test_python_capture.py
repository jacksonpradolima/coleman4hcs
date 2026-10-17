"""Owned capture integration uses real cache/domain/artifacts; only external Docker execution is substituted."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.corpus.controlled import validate_controlled_dataset
from coleman.corpus.materialization import materialize_source
from coleman.results.domain import fingerprint
from coleman.transports import python as transport
from coleman.transports import python_capture as capture
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.producer_identity import installed_producer_identity
from coleman.transports.python_contracts import (
    PythonCapturePlan,
    PythonCaptureReceipt,
    PythonEnvironment,
    PythonRoleCapture,
)
from coleman.transports.python_source import owned_python_package, python_harness_bytes
from tests.corpus.test_junit_matrix import repinned_owned_forgery
from tests.corpus.test_python_contracts import owned_case, owned_header
from tests.corpus.test_python_protocol import owned_recipe, wire
from tests.corpus.test_python_wheels import owned_wheel
from tests.corpus.test_source_views import changed, digest


def prepared_inputs(tmp_path):
    """Materialize authored schema-only sources, real tiny wheel-shaped ZIP and typed runtime metadata."""
    wheel, payload = owned_wheel()
    spec = changed(owned_recipe(), wheels=(wheel,), harness_sha256=digest(python_harness_bytes()))
    cache = SourceCache(tmp_path / "cache")
    for view in (spec.buggy, spec.fixed, spec.fixture):
        for package in view.inputs:
            materialize_source(package, cache, lambda: io.BytesIO(b"Owned schema-only source fixture.\n"))
    materialize_source(wheel.package, cache, lambda: io.BytesIO(payload))
    environment = PythonEnvironment(
        schema_version=1,
        host_os="owned",
        host_architecture="aarch64",
        host_release="owned",
        producer_python_version="owned",
        producer_package_version="owned",
        docker_client_version="owned",
        docker_server_version="owned",
        docker_server_architecture="arm64",
        container_kernel="owned",
        image_os="linux",
        image_architecture="arm64",
        image_size_bytes=0,
        image_config_digest=digest(b"owned image metadata"),
    )
    receipt = owned_python_package(cache, "execution-environment.json", environment.to_json().encode(), "owned-v1")
    return spec, cache, receipt


def observed_role(*args):
    """Supply explicit synthetic process records; no actual runtime or foreign source claim is made."""
    plan = args[-2]
    header = changed(
        owned_header(),
        context_fingerprint=plan.context_fingerprint,
        trial_id=plan.trial_id,
        role=plan.role,
        architecture="aarch64",
        distributions=plan.distributions,
        plan_sha256=digest((plan.to_json() + "\n").encode()),
    )
    case = owned_case()
    if plan.role == "buggy":
        case = changed(case, outcome="assertion_failed", error_type="builtins.AssertionError")
    trailer = {"schema_version": 1, "record_type": "trailer", "cases": 1, "pytest_exit_code": int(plan.role == "buggy")}
    return ProcessStreams(
        wire([header.model_dump(mode="json"), case.model_dump(mode="json"), trailer]), b"", 0, 0, "exited"
    )


def substitute_runtime(monkeypatch, environment, run=observed_role):
    """Replace only Docker boundary calls, preserving actual preparation/protocol/domain/state paths."""

    @contextmanager
    def base():
        yield ("owned-docker",)

    monkeypatch.setattr(capture, "_docker_base", base)
    monkeypatch.setattr(capture, "python_runtime_environment", lambda *args: environment)
    monkeypatch.setattr(capture, "run_python_role", run)
    monkeypatch.setattr(capture, "_cleanup", lambda *args: None)


def test_real_cache_domain_roundtrip_completion_and_replay_without_docker(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    first = capture.capture_python_source(spec, cache, writer, "owned-trial")
    assert calls == ["buggy", "fixed"]
    assert first.receipt.status == "collected"
    assert first.dataset.pairs[0].classification == "expected"
    assert first.dataset.observations[0].elapsed.value == 0
    assert first.dataset.history == "constructed"
    assert first.dataset.admission == "not_evaluated"
    assert PythonCaptureReceipt.from_json((first.state_directory / "complete.json").read_bytes()) == first.receipt

    def unavailable():
        pytest.fail("Completed source replay must not access Docker")

    monkeypatch.setattr(capture, "_docker_base", unavailable)
    assert capture.capture_python_source(spec, cache, writer, "owned-trial") == first
    plan = PythonCapturePlan.from_json((first.state_directory / "plan.json").read_bytes())
    capture.validate_python_capture(first.receipt, plan, first.dataset, cache)
    with pytest.raises(ValueError, match="not historical CI"):
        first.dataset.require_temporal_evaluation()


def test_interruption_never_completes_and_recovers_without_reexecuting_acknowledged_role(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_python_source(spec, cache, writer, "interrupted")
    assert not list(writer.run_root.glob("python-captures/*/complete.json"))
    assert not list(writer.run_root.glob("python-captures/*/receipt.json"))
    resumed = capture.capture_python_source(spec, cache, writer, "interrupted")
    assert calls == ["buggy", "fixed"]
    assert [role.status for role in resumed.captures] == ["interrupted", "collected"]
    assert resumed.captures[0].producer_elapsed.value is None
    assert resumed.receipt.status == "incomplete"
    assert resumed.dataset.pairs[0].classification == "inconclusive"
    assert len(resumed.dataset.observations) == 1
    assert not (resumed.state_directory / "complete.json").exists()
    assert capture.capture_python_source(spec, cache, writer, "interrupted") == resumed


def test_atomic_receipt_interruption_replays_existing_roles_without_duplicate_execution(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    original = writer.write_immutable

    def interrupted(path, content):
        if path.name == "receipt.json":
            raise OSError("Owned interrupted publication")
        original(path, content)

    monkeypatch.setattr(writer, "write_immutable", interrupted)
    with pytest.raises(OSError, match="interrupted publication"):
        capture.capture_python_source(spec, cache, writer, "publication")
    assert not list(writer.run_root.glob("python-captures/*/complete.json"))
    monkeypatch.setattr(writer, "write_immutable", original)
    result = capture.capture_python_source(spec, cache, writer, "publication")
    assert result.receipt.status == "collected"
    assert calls == ["buggy", "fixed"]


def test_concurrent_same_trial_starts_exactly_one_physical_pair(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    entered, release = Event(), Event()
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        entered.set()
        if not release.wait(timeout=20):
            raise TimeoutError("Owned concurrency test deadline")
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(capture.capture_python_source, spec, cache, writer, "concurrent")
        try:
            assert entered.wait(timeout=20)
            with pytest.raises(ValueError, match="Another producer"):
                capture.capture_python_source(spec, cache, writer, "concurrent")
        finally:
            release.set()
        result = future.result(timeout=20)
    assert result.receipt.status == "collected"
    assert calls == ["buggy", "fixed"]


def test_runtime_drift_after_interruption_rejects_without_executing_another_role(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt

    substitute_runtime(monkeypatch, environment, interrupted)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_python_source(spec, cache, writer, "drift")
    other = owned_python_package(cache, "environment.json", b"Owned changed runtime facts", "owned-other-v1")
    monkeypatch.setattr(capture, "python_runtime_environment", lambda *args: other)
    with pytest.raises(ValueError, match="runtime changed"):
        capture.capture_python_source(spec, cache, writer, "drift")
    assert not list(writer.run_root.glob("python-captures/*/complete.json"))


def test_recomputed_metadata_does_not_authorize_rewritten_cases_or_missing_packets(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_python_source(spec, cache, writer, "tamper")
    raw = result.state_directory / "buggy.json"
    original = raw.read_bytes()
    role = PythonRoleCapture.from_json(original)
    altered = changed(role, cases=(changed(role.cases[0], diagnostics="rewritten"),))
    raw.write_text(altered.to_json())
    with pytest.raises(ValueError, match="disagree with raw"):
        capture.capture_python_source(spec, cache, writer, "tamper")
    raw.write_bytes(original)
    raw.unlink()
    with pytest.raises(ValueError, match="missing raw role"):
        capture.capture_python_source(spec, cache, writer, "tamper")


def test_launch_error_preserves_unavailable_time_and_incomplete_receipt(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def unavailable(*args):
        raise OSError("Owned child launch error")

    substitute_runtime(monkeypatch, environment, unavailable)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_python_source(spec, cache, writer, "launch-error")
    assert result.receipt.status == "incomplete"
    assert result.dataset.observations == ()
    assert result.dataset.pairs == ()
    assert all(role.producer_elapsed.value is None for role in result.captures)
    assert all(role.status == "process_error" for role in result.captures)
    assert not (result.state_directory / "complete.json").exists()


def test_unknown_producer_and_conflicting_completion_reject(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_python_source(spec, cache, writer, "producer")
    plan = PythonCapturePlan.from_json((result.state_directory / "plan.json").read_bytes())
    unknown = changed(plan, producer_sha256=digest(b"future implementation"))
    with pytest.raises(ValueError, match="Unknown Python capture producer"):
        capture.validate_python_capture(result.receipt, unknown, result.dataset, cache)
    completion = result.state_directory / "complete.json"
    completion.write_text(changed(result.receipt, dataset_fingerprint=digest(b"wrong dataset")).to_json())
    with pytest.raises(ValueError, match="Conflicting immutable"):
        capture.capture_python_source(spec, cache, writer, "producer")


@pytest.mark.parametrize("nested", ["state", "dataset"])
def test_nested_git_output_rejects_before_lock_or_runtime_access(tmp_path, monkeypatch, nested):
    spec, cache, _ = prepared_inputs(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    generation = fingerprint(
        ["python-capture-v1", spec.fingerprint, "nested-git", installed_producer_identity()]
    ).removeprefix("sha256:")
    forbidden = writer.run_root / ("python-captures/" + generation if nested == "state" else "corpus-controlled")
    (forbidden / ".git").mkdir(parents=True)

    def runtime():
        pytest.fail("Forbidden nested Git output must reject before runtime access")

    monkeypatch.setattr(capture, "_docker_base", runtime)
    with pytest.raises(ValueError, match="outside a repository"):
        capture.capture_python_source(spec, cache, writer, "nested-git")
    assert not list(writer.run_root.rglob("producer.lock"))
    assert not list(writer.run_root.rglob("plan.json"))


@pytest.mark.parametrize("kind", ["normalized_outcome", "context", "missing_packet", "packet_identity"])
def test_coherent_generic_repinning_cannot_rewrite_python_evidence(tmp_path, monkeypatch, kind):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_python_source(spec, cache, writer, "coherent-forgery")
    plan = PythonCapturePlan.from_json((result.state_directory / "plan.json").read_bytes())
    forged = repinned_owned_forgery(result.dataset, cache, kind)
    validate_controlled_dataset(forged, cache)
    index = changed(result.receipt, dataset_fingerprint=forged.fingerprint)
    with pytest.raises(ValueError, match="disagree|requires both|identity"):
        capture.validate_python_capture(index, plan, forged, cache)


def test_cli_uses_bounded_stdin_recipe_and_preserves_incomplete_accounting(tmp_path, monkeypatch, capsys):
    spec, cache, environment = prepared_inputs(tmp_path)

    def run(*args):
        if args[-2].role == "fixed":
            raise OSError("Owned fixed-role infrastructure error")
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, run)
    monkeypatch.setattr(transport.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(spec.to_json().encode())))
    args = ["--recipe-stdin", "--cache", str(cache.root), "--output", str(tmp_path / "output"), "--trial", "cli"]
    assert transport.main(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["pairs"] == {"inconclusive": 1}
    assert summary["planned_pairs"] == 1
    assert summary["unobserved_pairs"] == 0
    assert summary["role_statuses"] == {"buggy": "collected", "fixed": "process_error"}
    assert not list((tmp_path / "output").rglob("complete.json"))


def test_cli_default_preset_and_oversized_inputs_are_explicit(tmp_path, monkeypatch, capsys):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    monkeypatch.setattr(transport, "load_python_recipe", lambda preset: spec)
    assert (
        transport.main(["--cache", str(cache.root), "--output", str(tmp_path / "output"), "--trial", "cli-default"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "collected"
    monkeypatch.setattr(transport.PythonCaptureRecipe, "MAXIMUM_BYTES", 1)
    monkeypatch.setattr(transport.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))
    with pytest.raises(ValueError, match="stdin recipe exceeds"):
        transport.main(
            ["--recipe-stdin", "--cache", str(cache.root), "--output", str(tmp_path / "output"), "--trial", "oversized"]
        )


def test_local_client_configuration_is_empty_temporary_and_runtime_is_never_implicitly_pulled(monkeypatch):
    monkeypatch.setattr(capture.shutil, "which", lambda name: None)
    with pytest.raises(OSError, match="local Docker runtime"), capture._docker_base():
        pytest.fail("Unavailable local runtime entered")
    monkeypatch.setattr(capture.shutil, "which", lambda name: "/owned/docker")
    with capture._docker_base() as base:
        assert base[:2] == ("/owned/docker", "--config")
        root = Path(base[2])
        assert (root / "config.json").read_text() == "{}\n"
    assert not root.exists()


def test_unconfirmed_cleanup_preserves_ownership_for_recovery(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def failed_cleanup(*args):
        raise capture._CleanupError("Owned cleanup acknowledgment failed")

    substitute_runtime(monkeypatch, environment, failed_cleanup)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(capture._CleanupError):
        capture.capture_python_source(spec, cache, writer, "cleanup")
    assert list(writer.run_root.glob("python-captures/*/buggy.started.json"))
    assert not list(writer.run_root.glob("python-captures/*/buggy.json"))
    assert not list(writer.run_root.glob("python-captures/*/complete.json"))


def test_missing_plan_cannot_be_recreated_over_existing_evidence(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_python_source(spec, cache, writer, "missing-plan")
    (result.state_directory / "plan.json").unlink()
    monkeypatch.setattr(capture, "_docker_base", lambda: pytest.fail("Missing original plan must fail before Docker"))
    with pytest.raises(ValueError, match="original pre-execution plan"):
        capture.capture_python_source(spec, cache, writer, "missing-plan")
    assert not (result.state_directory / "plan.json").exists()


def test_future_existing_role_rejects_before_resuming_another_role(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def interrupted_fixed(*args):
        if args[-2].role == "fixed":
            raise KeyboardInterrupt
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, interrupted_fixed)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_python_source(spec, cache, writer, "future-packet")
    raw = next(writer.run_root.glob("python-captures/*/buggy.json"))
    payload = json.loads(raw.read_bytes())
    payload["schema_version"] = 2
    raw.write_text(json.dumps(payload))
    monkeypatch.setattr(capture, "_docker_base", lambda: pytest.fail("Unknown saved role must reject before Docker"))
    with pytest.raises(ValueError):
        capture.capture_python_source(spec, cache, writer, "future-packet")
    assert not list(writer.run_root.glob("python-captures/*/complete.json"))


def test_incomplete_index_cannot_replace_its_missing_raw_packet(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def unavailable(*args):
        raise OSError("Owned infrastructure error")

    substitute_runtime(monkeypatch, environment, unavailable)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_python_source(spec, cache, writer, "incomplete-index")
    assert result.receipt.status == "incomplete"
    (result.state_directory / "buggy.json").unlink()
    monkeypatch.setattr(
        capture, "_docker_base", lambda: pytest.fail("Incomplete index with lost packet must fail first")
    )
    with pytest.raises(ValueError, match="missing raw role"):
        capture.capture_python_source(spec, cache, writer, "incomplete-index")


def test_modified_container_ownership_never_cleans_an_unowned_name(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt

    substitute_runtime(monkeypatch, environment, interrupted)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_python_source(spec, cache, writer, "ownership")
    marker = next(writer.run_root.glob("python-captures/*/buggy.started.json"))
    data = json.loads(marker.read_bytes())
    data["container_name"] = "coleman-python-" + "f" * 24
    marker.write_text(json.dumps(data))
    cleanup = []
    monkeypatch.setattr(capture, "_cleanup", lambda *args: cleanup.append(args))
    with pytest.raises(ValueError, match="start marker identity"):
        capture.capture_python_source(spec, cache, writer, "ownership")
    assert cleanup == []


def test_maximum_escaped_logical_trial_label_remains_recoverable(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return observed_role(*args)

    substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    trial = "ci" + "\x01" * 1022
    with pytest.raises(KeyboardInterrupt):
        capture.capture_python_source(spec, cache, writer, trial)
    marker = next(writer.run_root.glob("python-captures/*/buggy.started.json"))
    assert len(marker.read_bytes()) > 4096
    result = capture.capture_python_source(spec, cache, writer, trial)
    assert result.receipt.status == "incomplete"
    assert [role.status for role in result.captures] == ["interrupted", "collected"]
    assert calls == ["buggy", "fixed"]


def test_multiple_cases_validate_the_existing_domain_canonical_identity_order(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    spec = changed(spec, test_ids=(*spec.test_ids, "tests/test_owned.py::test_second"))

    def multiple(*args):
        plan = args[-2]
        header = changed(
            owned_header(),
            context_fingerprint=plan.context_fingerprint,
            trial_id=plan.trial_id,
            role=plan.role,
            architecture="aarch64",
            distributions=plan.distributions,
            test_ids=plan.test_ids,
            plan_sha256=digest((plan.to_json() + "\n").encode()),
        )
        cases = [changed(owned_case(), test_id=name) for name in spec.test_ids]
        trailer = dict(schema_version=1, record_type="trailer", cases=2, pytest_exit_code=0)
        return ProcessStreams(
            wire([header.model_dump(mode="json"), *(c.model_dump(mode="json") for c in cases), trailer]),
            b"",
            0,
            0,
            "exited",
        )

    substitute_runtime(monkeypatch, environment, multiple)
    writer = ArtifactWriter(tmp_path / "multiple")
    result = capture.capture_python_source(spec, cache, writer, "multiple")
    assert len(result.dataset.observations) == 4
    assert result.receipt.status == "collected"
    assert list(result.dataset.observations) == sorted(result.dataset.observations, key=lambda o: o.logical_key)
    assert capture.capture_python_source(spec, cache, writer, "multiple") == result
