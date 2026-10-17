"""Shared filtered Docker metadata checks exercise each concrete producer contract."""

from types import SimpleNamespace

import pytest

from coleman.transports import c_process, python_process
from coleman.transports.c_contracts import CEnvironment
from coleman.transports.python_contracts import PythonEnvironment
from tests.corpus.test_c_capture import prepared_inputs as c_inputs
from tests.corpus.test_python_capture import prepared_inputs as python_inputs
from tests.corpus.test_source_views import changed, digest


@pytest.fixture(params=["c", "python"])
def provider(request):
    """Choose concrete source preparation and runtime contract, sharing only invariant assertions."""
    if request.param == "c":
        return SimpleNamespace(
            process=c_process,
            inputs=c_inputs,
            environment=CEnvironment,
            runtime=c_process.c_runtime_environment,
            kind="filtered-local-c-runtime-v1",
        )
    return SimpleNamespace(
        process=python_process,
        inputs=python_inputs,
        environment=PythonEnvironment,
        runtime=python_process.python_runtime_environment,
        kind="filtered-local-python-runtime-v1",
    )


def metadata(monkeypatch, provider, *, info=None, versions=None, machine="aarch64"):
    """Supply only the public filtered Docker facts; no image is pulled or foreign process executed."""
    responses = iter(
        [
            info if info is not None else ["linux", "arm64", 0, digest(b"owned config")],
            versions if versions is not None else ["owned-client", "owned-server", "arm64"],
            "owned-kernel",
        ]
    )
    monkeypatch.setattr(provider.process, "_docker_query", lambda *args: next(responses))
    monkeypatch.setattr(provider.process.platform, "machine", lambda: machine)


def test_filtered_runtime_records_only_declared_public_facts_and_does_not_pull(tmp_path, monkeypatch, provider):
    spec, cache, _ = provider.inputs(tmp_path)
    metadata(monkeypatch, provider)
    receipt = provider.runtime(("owned-docker",), spec, cache)
    actual = provider.environment.from_json(cache.member_path(receipt, receipt.spec.members[0].name).read_bytes())
    assert actual.kind == provider.kind
    assert actual.image_size_bytes == 0
    assert actual.host_architecture == "aarch64"
    assert actual.container_kernel == "owned-kernel"


@pytest.mark.parametrize(
    "options",
    [
        {"info": []},
        {"info": ["windows", "arm64", 0, digest(b"owned")]},
        {"info": ["linux", "amd64", 0, digest(b"owned")]},
        {"versions": []},
        {"versions": ["owned", 1, "arm64"]},
        {"machine": "unrecognized"},
        {"versions": ["owned", "owned", "unrecognized"]},
    ],
)
def test_unknown_inconsistent_or_non_scalar_runtime_metadata_rejects(tmp_path, monkeypatch, provider, options):
    spec, cache, _ = provider.inputs(tmp_path)
    metadata(monkeypatch, provider, **options)
    with pytest.raises(ValueError):
        provider.runtime(("owned-docker",), spec, cache)


def test_visible_architecture_translation_requires_explicit_emulation(tmp_path, monkeypatch, provider):
    spec, cache, _ = provider.inputs(tmp_path)
    metadata(monkeypatch, provider, versions=["owned", "owned", "amd64"])
    with pytest.raises(ValueError, match="translation"):
        provider.runtime(("owned-docker",), spec, cache)
    declared = changed(spec, runtime=changed(spec.runtime, emulated=True))
    metadata(monkeypatch, provider, versions=["owned", "owned", "amd64"])
    receipt = provider.runtime(("owned-docker",), declared, cache)
    assert receipt.spec.archive == "raw"
