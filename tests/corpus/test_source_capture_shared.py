"""Common producer regressions run against both C and Python cache/protocol/artifact implementations."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.transports import c as c_transport
from coleman.transports import python as python_transport
from coleman.transports.c_contracts import CCaptureReceipt
from coleman.transports.python_contracts import PythonCaptureReceipt
from tests.corpus import test_c_capture, test_python_capture
from tests.corpus.test_source_views import changed, digest


@pytest.fixture(params=["c", "python"])
def provider(request):
    """Bind concrete producer contracts while retaining each provider's distinct process and oracle fixtures."""
    language = request.param
    module = test_c_capture if language == "c" else test_python_capture
    transport = c_transport if language == "c" else python_transport
    contracts = "C" if language == "c" else "Python"
    return SimpleNamespace(
        language=language,
        capture=module.capture,
        transport=transport,
        collect=getattr(module.capture, f"capture_{language}_source"),
        validate=getattr(module.capture, f"validate_{language}_capture"),
        load_recipe=f"load_{language}_recipe",
        recipe_type=getattr(transport, contracts + "CaptureRecipe"),
        plan_type=getattr(module, contracts + "CapturePlan"),
        receipt_type=CCaptureReceipt if language == "c" else PythonCaptureReceipt,
        unknown_producer_error=f"Unknown {contracts} capture",
        prepared_inputs=module.prepared_inputs,
        observed_role=module.observed_role,
        substitute_runtime=module.substitute_runtime,
    )


def test_real_cache_domain_roundtrip_completion_and_replay_without_docker(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    first = provider.collect(spec, cache, writer, "owned-trial")
    assert calls == ["buggy", "fixed"]
    assert first.receipt.status == "collected"
    assert first.dataset.pairs[0].classification == "expected"
    assert first.dataset.observations[0].elapsed.value == 0
    assert first.dataset.history == "constructed"
    assert first.dataset.admission == "not_evaluated"
    assert provider.receipt_type.from_json((first.state_directory / "complete.json").read_bytes()) == first.receipt

    def unavailable():
        pytest.fail("Completed source replay must not access Docker")

    monkeypatch.setattr(provider.capture, "_docker_base", unavailable)
    assert provider.collect(spec, cache, writer, "owned-trial") == first
    plan = provider.plan_type.from_json((first.state_directory / "plan.json").read_bytes())
    provider.validate(first.receipt, plan, first.dataset, cache)
    with pytest.raises(ValueError, match="not historical CI"):
        first.dataset.require_temporal_evaluation()


def test_interruption_never_completes_and_recovers_without_reexecuting_acknowledged_role(
    tmp_path, monkeypatch, provider
):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        provider.collect(spec, cache, writer, "interrupted")
    assert not list(writer.run_root.glob(f"{provider.language}-captures/*/complete.json"))
    assert not list(writer.run_root.glob(f"{provider.language}-captures/*/receipt.json"))
    resumed = provider.collect(spec, cache, writer, "interrupted")
    assert calls == ["buggy", "fixed"]
    assert [role.status for role in resumed.captures] == ["interrupted", "collected"]
    assert resumed.captures[0].producer_elapsed.value is None
    assert resumed.receipt.status == "incomplete"
    assert resumed.dataset.pairs[0].classification == "inconclusive"
    assert len(resumed.dataset.observations) == 1
    assert not (resumed.state_directory / "complete.json").exists()
    assert provider.collect(spec, cache, writer, "interrupted") == resumed


def test_atomic_receipt_interruption_replays_existing_roles_without_duplicate_execution(
    tmp_path, monkeypatch, provider
):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    original = writer.write_immutable

    def interrupted(path, content):
        if path.name == "receipt.json":
            raise OSError("Owned interrupted publication")
        original(path, content)

    monkeypatch.setattr(writer, "write_immutable", interrupted)
    with pytest.raises(OSError, match="interrupted publication"):
        provider.collect(spec, cache, writer, "publication")
    assert not list(writer.run_root.glob(f"{provider.language}-captures/*/complete.json"))
    monkeypatch.setattr(writer, "write_immutable", original)
    result = provider.collect(spec, cache, writer, "publication")
    assert result.receipt.status == "collected"
    assert calls == ["buggy", "fixed"]


def test_concurrent_same_trial_starts_exactly_one_physical_pair(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    entered, release = (Event(), Event())
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        entered.set()
        if not release.wait(timeout=20):
            raise TimeoutError("Owned concurrency test deadline")
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(provider.collect, spec, cache, writer, "concurrent")
        try:
            assert entered.wait(timeout=20)
            with pytest.raises(ValueError, match="Another producer"):
                provider.collect(spec, cache, writer, "concurrent")
        finally:
            release.set()
        result = future.result(timeout=20)
    assert result.receipt.status == "collected"
    assert calls == ["buggy", "fixed"]


def test_launch_error_preserves_unavailable_time_and_incomplete_receipt(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)

    def unavailable(*args):
        raise OSError("Owned child launch error")

    provider.substitute_runtime(monkeypatch, environment, unavailable)
    writer = ArtifactWriter(tmp_path / "output")
    result = provider.collect(spec, cache, writer, "launch-error")
    assert result.receipt.status == "incomplete"
    assert result.dataset.observations == ()
    assert result.dataset.pairs == ()
    assert all(role.producer_elapsed.value is None for role in result.captures)
    assert all(role.status == "process_error" for role in result.captures)
    assert not (result.state_directory / "complete.json").exists()


def test_unknown_producer_and_conflicting_completion_reject(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    provider.substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = provider.collect(spec, cache, writer, "producer")
    plan = provider.plan_type.from_json((result.state_directory / "plan.json").read_bytes())
    unknown = changed(plan, producer_sha256=digest(b"future implementation"))
    with pytest.raises(ValueError, match=provider.unknown_producer_error + " producer"):
        provider.validate(result.receipt, unknown, result.dataset, cache)
    completion = result.state_directory / "complete.json"
    completion.write_text(changed(result.receipt, dataset_fingerprint=digest(b"wrong dataset")).to_json())
    with pytest.raises(ValueError, match="Conflicting immutable"):
        provider.collect(spec, cache, writer, "producer")


def test_cli_uses_bounded_stdin_recipe_and_preserves_incomplete_accounting(tmp_path, monkeypatch, capsys, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)

    def run(*args):
        if args[-2].role == "fixed":
            raise OSError("Owned fixed-role infrastructure error")
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    monkeypatch.setattr(provider.transport.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(spec.to_json().encode())))
    args = ["--recipe-stdin", "--cache", str(cache.root), "--output", str(tmp_path / "output"), "--trial", "cli"]
    assert provider.transport.main(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["pairs"] == {"inconclusive": 1}
    assert summary["planned_pairs"] == 1
    assert summary["unobserved_pairs"] == 0
    assert summary["role_statuses"] == {"buggy": "collected", "fixed": "process_error"}
    assert not list((tmp_path / "output").rglob("complete.json"))


def test_cli_default_preset_and_oversized_inputs_are_explicit(tmp_path, monkeypatch, capsys, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    provider.substitute_runtime(monkeypatch, environment)
    monkeypatch.setattr(provider.transport, provider.load_recipe, lambda preset: spec)
    assert (
        provider.transport.main(
            ["--cache", str(cache.root), "--output", str(tmp_path / "output"), "--trial", "cli-default"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "collected"
    monkeypatch.setattr(provider.recipe_type, "MAXIMUM_BYTES", 1)
    monkeypatch.setattr(provider.transport.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))
    with pytest.raises(ValueError, match="stdin recipe exceeds"):
        provider.transport.main(
            ["--recipe-stdin", "--cache", str(cache.root), "--output", str(tmp_path / "output"), "--trial", "oversized"]
        )


def test_local_client_configuration_is_empty_temporary_and_runtime_is_never_implicitly_pulled(monkeypatch, provider):
    monkeypatch.setattr(provider.capture.shutil, "which", lambda name: None)
    with pytest.raises(OSError, match="local Docker runtime"), provider.capture._docker_base():
        pytest.fail("Unavailable local runtime entered")
    monkeypatch.setattr(provider.capture.shutil, "which", lambda name: "/owned/docker")
    with provider.capture._docker_base() as base:
        assert base[:2] == ("/owned/docker", "--config")
        root = Path(base[2])
        assert (root / "config.json").read_text() == "{}\n"
    assert not root.exists()


def test_unconfirmed_cleanup_preserves_ownership_for_recovery(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)

    def failed_cleanup(*args):
        raise provider.capture._CleanupError("Owned cleanup acknowledgment failed")

    provider.substitute_runtime(monkeypatch, environment, failed_cleanup)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(provider.capture._CleanupError):
        provider.collect(spec, cache, writer, "cleanup")
    assert list(writer.run_root.glob(f"{provider.language}-captures/*/buggy.started.json"))
    assert not list(writer.run_root.glob(f"{provider.language}-captures/*/buggy.json"))
    assert not list(writer.run_root.glob(f"{provider.language}-captures/*/complete.json"))


def test_missing_plan_cannot_be_recreated_over_existing_evidence(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    provider.substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = provider.collect(spec, cache, writer, "missing-plan")
    (result.state_directory / "plan.json").unlink()
    monkeypatch.setattr(
        provider.capture, "_docker_base", lambda: pytest.fail("Missing original plan must fail before Docker")
    )
    with pytest.raises(ValueError, match="original pre-execution plan"):
        provider.collect(spec, cache, writer, "missing-plan")
    assert not (result.state_directory / "plan.json").exists()


def test_future_existing_role_rejects_before_resuming_another_role(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)

    def interrupted_fixed(*args):
        if args[-2].role == "fixed":
            raise KeyboardInterrupt
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, interrupted_fixed)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        provider.collect(spec, cache, writer, "future-packet")
    raw = next(writer.run_root.glob(f"{provider.language}-captures/*/buggy.json"))
    payload = json.loads(raw.read_bytes())
    payload["schema_version"] = 2
    raw.write_text(json.dumps(payload))
    monkeypatch.setattr(
        provider.capture, "_docker_base", lambda: pytest.fail("Unknown saved role must reject before Docker")
    )
    with pytest.raises(ValueError):
        provider.collect(spec, cache, writer, "future-packet")
    assert not list(writer.run_root.glob(f"{provider.language}-captures/*/complete.json"))


def test_incomplete_index_cannot_replace_its_missing_raw_packet(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)

    def unavailable(*args):
        raise OSError("Owned infrastructure error")

    provider.substitute_runtime(monkeypatch, environment, unavailable)
    writer = ArtifactWriter(tmp_path / "output")
    result = provider.collect(spec, cache, writer, "incomplete-index")
    assert result.receipt.status == "incomplete"
    (result.state_directory / "buggy.json").unlink()
    monkeypatch.setattr(
        provider.capture, "_docker_base", lambda: pytest.fail("Incomplete index with lost packet must fail first")
    )
    with pytest.raises(ValueError, match="missing raw role"):
        provider.collect(spec, cache, writer, "incomplete-index")


def test_maximum_escaped_logical_trial_label_remains_recoverable(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2].role)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "output")
    trial = "ci" + "\x01" * 1022
    with pytest.raises(KeyboardInterrupt):
        provider.collect(spec, cache, writer, trial)
    marker = next(writer.run_root.glob(f"{provider.language}-captures/*/buggy.started.json"))
    assert len(marker.read_bytes()) > 4096
    result = provider.collect(spec, cache, writer, trial)
    assert result.receipt.status == "incomplete"
    assert [role.status for role in result.captures] == ["interrupted", "collected"]
    assert calls == ["buggy", "fixed"]


def test_separate_output_roots_have_independent_container_ownership(tmp_path, monkeypatch, provider):
    spec, cache, environment = provider.prepared_inputs(tmp_path)
    names = []

    def run(*args):
        names.append(args[-1])
        return provider.observed_role(*args)

    provider.substitute_runtime(monkeypatch, environment, run)
    first = provider.collect(spec, cache, ArtifactWriter(tmp_path / "first"), "same-trial")
    second = provider.collect(spec, cache, ArtifactWriter(tmp_path / "second"), "same-trial")
    assert len(names) == 4
    assert set(names[:2]).isdisjoint(names[2:]), (
        "Separate outputs must not clean or collide with another producer's containers"
    )
    assert first.dataset == second.dataset, "Physical ownership must not change logical observation identity"
