"""Owned capture integration uses real cache/domain/artifacts; only external Docker execution is substituted."""

import io
import json
from contextlib import contextmanager

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.corpus.controlled import validate_controlled_dataset
from coleman.corpus.materialization import materialize_source
from coleman.results.domain import fingerprint
from coleman.transports import c_capture as capture
from coleman.transports.c_contracts import (
    CCapturePlan,
    CEnvironment,
    CRoleCapture,
)
from coleman.transports.c_source import c_collector_bytes, c_harness_bytes, owned_c_package
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.producer_identity import installed_producer_identity
from tests.corpus.test_c_contracts import owned_case, owned_header
from tests.corpus.test_c_protocol import owned_recipe, wire
from tests.corpus.test_junit_matrix import repinned_owned_forgery
from tests.corpus.test_source_views import changed, digest


def prepared_inputs(tmp_path):
    """Materialize authored schema-only sources, typed runtime metadata and schema-only C/support views."""
    spec = changed(
        owned_recipe(), harness_sha256=digest(c_harness_bytes()), collector_sha256=digest(c_collector_bytes())
    )
    cache = SourceCache(tmp_path / "cache")
    for view in (spec.buggy, spec.fixed, spec.fixture):
        for package in view.inputs:
            materialize_source(package, cache, lambda: io.BytesIO(b"Owned schema-only source fixture.\n"))
    environment = CEnvironment(
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
    receipt = owned_c_package(cache, "execution-environment.json", environment.to_json().encode(), "owned-v1")
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
        plan_sha256=digest((plan.to_json() + "\n").encode()),
    )
    case = owned_case()
    if plan.role == "buggy":
        case = changed(case, outcome="assertion_failed", stdout="0\n")
    trailer = {"schema_version": 1, "record_type": "trailer", "cases": 1, "oracle_exit_code": int(plan.role == "buggy")}
    return ProcessStreams(
        wire([header.model_dump(mode="json"), case.model_dump(mode="json"), trailer]), b"", 0, 0, "exited"
    )


def substitute_runtime(monkeypatch, environment, run=observed_role):
    """Replace only Docker boundary calls, preserving actual preparation/protocol/domain/state paths."""

    @contextmanager
    def base():
        yield ("owned-docker",)

    monkeypatch.setattr(capture, "_docker_base", base)
    monkeypatch.setattr(capture, "c_runtime_environment", lambda *args: environment)
    monkeypatch.setattr(capture, "run_c_role", run)
    monkeypatch.setattr(capture, "_cleanup", lambda *args: None)


def test_runtime_drift_after_interruption_rejects_without_executing_another_role(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt

    substitute_runtime(monkeypatch, environment, interrupted)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_c_source(spec, cache, writer, "drift")
    other = owned_c_package(cache, "environment.json", b"Owned changed runtime facts", "owned-other-v1")
    monkeypatch.setattr(capture, "c_runtime_environment", lambda *args: other)
    with pytest.raises(ValueError, match="runtime changed"):
        capture.capture_c_source(spec, cache, writer, "drift")
    assert not list(writer.run_root.glob("c-captures/*/complete.json"))


def test_recomputed_metadata_does_not_authorize_rewritten_cases_or_missing_packets(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_c_source(spec, cache, writer, "tamper")
    raw = result.state_directory / "buggy.json"
    original = raw.read_bytes()
    role = CRoleCapture.from_json(original)
    altered = changed(role, cases=(changed(role.cases[0], stderr="rewritten"),))
    raw.write_text(altered.to_json())
    with pytest.raises(ValueError, match="disagree with raw"):
        capture.capture_c_source(spec, cache, writer, "tamper")
    raw.write_bytes(original)
    raw.unlink()
    with pytest.raises(ValueError, match="missing raw role"):
        capture.capture_c_source(spec, cache, writer, "tamper")


@pytest.mark.parametrize("nested", ["state", "dataset"])
def test_nested_git_output_rejects_before_lock_or_runtime_access(tmp_path, monkeypatch, nested):
    spec, cache, _ = prepared_inputs(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    generation = fingerprint(
        ["c-capture-v1", spec.fingerprint, "nested-git", installed_producer_identity()]
    ).removeprefix("sha256:")
    forbidden = writer.run_root / ("c-captures/" + generation if nested == "state" else "corpus-controlled")
    (forbidden / ".git").mkdir(parents=True)

    def runtime():
        pytest.fail("Forbidden nested Git output must reject before runtime access")

    monkeypatch.setattr(capture, "_docker_base", runtime)
    with pytest.raises(ValueError, match="outside a repository"):
        capture.capture_c_source(spec, cache, writer, "nested-git")
    assert not list(writer.run_root.rglob("producer.lock"))
    assert not list(writer.run_root.rglob("plan.json"))


@pytest.mark.parametrize("kind", ["normalized_outcome", "context", "missing_packet", "packet_identity"])
def test_coherent_generic_repinning_cannot_rewrite_c_evidence(tmp_path, monkeypatch, kind):
    spec, cache, environment = prepared_inputs(tmp_path)
    substitute_runtime(monkeypatch, environment)
    writer = ArtifactWriter(tmp_path / "output")
    result = capture.capture_c_source(spec, cache, writer, "coherent-forgery")
    plan = CCapturePlan.from_json((result.state_directory / "plan.json").read_bytes())
    forged = repinned_owned_forgery(result.dataset, cache, kind)
    validate_controlled_dataset(forged, cache)
    index = changed(result.receipt, dataset_fingerprint=forged.fingerprint)
    with pytest.raises(ValueError, match="disagree|requires both|identity"):
        capture.validate_c_capture(index, plan, forged, cache)


def test_modified_container_ownership_never_cleans_an_unowned_name(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt

    substitute_runtime(monkeypatch, environment, interrupted)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_c_source(spec, cache, writer, "ownership")
    marker = next(writer.run_root.glob("c-captures/*/buggy.started.json"))
    data = json.loads(marker.read_bytes())
    data["container_name"] = "coleman-c-" + "f" * 24
    marker.write_text(json.dumps(data))
    cleanup = []
    monkeypatch.setattr(capture, "_cleanup", lambda *args: cleanup.append(args))
    with pytest.raises(ValueError, match="start marker identity"):
        capture.capture_c_source(spec, cache, writer, "ownership")
    assert cleanup == []


def test_multiple_cases_validate_the_existing_domain_canonical_identity_order(tmp_path, monkeypatch):
    from coleman.transports.c_contracts import CInputCase

    spec, cache, environment = prepared_inputs(tmp_path)
    spec = changed(spec, cases=(*spec.cases, CInputCase(test_id="owned-second", value=1)))

    def multiple(*args):
        plan = args[-2]
        header = changed(
            owned_header(),
            context_fingerprint=plan.context_fingerprint,
            trial_id=plan.trial_id,
            role=plan.role,
            test_ids=spec.test_ids,
            plan_sha256=digest((plan.to_json() + "\n").encode()),
        )
        cases = [changed(owned_case(), test_id=c.test_id, stdout=c.expected_stdout) for c in spec.cases]
        trailer = dict(schema_version=1, record_type="trailer", cases=2, oracle_exit_code=0)
        return ProcessStreams(
            wire([header.model_dump(mode="json"), *(c.model_dump(mode="json") for c in cases), trailer]),
            b"",
            0,
            0,
            "exited",
        )

    substitute_runtime(monkeypatch, environment, multiple)
    writer = ArtifactWriter(tmp_path / "multiple")
    result = capture.capture_c_source(spec, cache, writer, "multiple")
    assert len(result.dataset.observations) == 4
    assert result.receipt.status == "collected"
    assert list(result.dataset.observations) == sorted(result.dataset.observations, key=lambda o: o.logical_key)
    assert capture.capture_c_source(spec, cache, writer, "multiple") == result
