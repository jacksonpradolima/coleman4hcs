"""Owned Docker-boundary tests inspect actual staging; installed tests separately execute the real subject."""

import csv
import io
from pathlib import Path

import pytest

from coleman.transports import python_process as process
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.python_contracts import PythonEnvironment, PythonHarnessPlan
from coleman.transports.python_source import prepare_python_inputs, python_harness_plan
from tests.corpus.test_python_capture import prepared_inputs
from tests.corpus.test_source_views import changed, digest


def metadata(monkeypatch, *, info=None, versions=None, machine="aarch64"):
    """Supply only the public filtered Docker facts; no image is pulled or foreign process executed."""
    responses = iter(
        [
            info if info is not None else ["linux", "arm64", 0, digest(b"owned config")],
            versions if versions is not None else ["owned-client", "owned-server", "arm64"],
            "owned-kernel",
        ]
    )
    monkeypatch.setattr(process, "_docker_query", lambda *args: next(responses))
    monkeypatch.setattr(process.platform, "machine", lambda: machine)


def test_filtered_runtime_records_only_declared_public_facts_and_does_not_pull(tmp_path, monkeypatch):
    spec, cache, _ = prepared_inputs(tmp_path)
    metadata(monkeypatch)
    receipt = process.python_runtime_environment(("owned-docker",), spec, cache)
    actual = PythonEnvironment.from_json(cache.member_path(receipt, receipt.spec.members[0].name).read_bytes())
    assert actual.kind == "filtered-local-python-runtime-v1"
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
def test_unknown_inconsistent_or_non_scalar_runtime_metadata_rejects(tmp_path, monkeypatch, options):
    spec, cache, _ = prepared_inputs(tmp_path)
    metadata(monkeypatch, **options)
    with pytest.raises(ValueError):
        process.python_runtime_environment(("owned-docker",), spec, cache)


def test_visible_architecture_translation_requires_explicit_emulation(tmp_path, monkeypatch):
    spec, cache, _ = prepared_inputs(tmp_path)
    metadata(monkeypatch, versions=["owned", "owned", "amd64"])
    with pytest.raises(ValueError, match="translation"):
        process.python_runtime_environment(("owned-docker",), spec, cache)
    declared = changed(spec, runtime=changed(spec.runtime, emulated=True))
    metadata(monkeypatch, versions=["owned", "owned", "amd64"])
    receipt = process.python_runtime_environment(("owned-docker",), declared, cache)
    assert receipt.spec.archive == "raw"


@pytest.mark.parametrize("failure", [None, OSError, KeyboardInterrupt])
def test_staging_uses_exact_original_bytes_fixed_guards_and_owned_cleanup(tmp_path, monkeypatch, failure):
    spec, cache, environment = prepared_inputs(tmp_path)
    prepared = prepare_python_inputs(spec, cache, environment)
    plan = python_harness_plan(spec, prepared.context.fingerprint, "owned-process", "buggy")
    container = "coleman-python-" + "a" * 24
    cleanup = []
    staged = []

    def observe(args, *, timeout_seconds, maximum_output_bytes):
        assert timeout_seconds == 60
        assert maximum_output_bytes == 1024**2
        assert "--pull=never" in args
        assert args[args.index("--network") + 1] == "none"
        assert args[args.index("--name") + 1] == container
        assert "--read-only" in args
        assert "no-new-privileges" in args
        assert "PYTHONCOERCECLOCALE=0" in args
        assert "PYTHONUTF8=0" in args
        assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in args
        assert "-B" in args
        assert "-s" in args
        mounts = {}
        for index, argument in enumerate(args):
            if argument == "--mount":
                parts = next(csv.reader(io.StringIO(args[index + 1])))
                assert "readonly" in parts
                options = dict(part.split("=", 1) for part in parts if "=" in part)
                mounts[options["target"]] = Path(options["source"])
        assert set(mounts) == {"/source", "/fixture", "/wheels", "/harness"}
        staged.extend(mounts.values())
        assert (mounts["/source"] / "owned/__init__.py").read_bytes() == b"Owned schema-only source fixture.\n"
        assert (mounts["/harness"] / "ColemanPytestCapture.py").read_bytes() == prepared.harness
        assert PythonHarnessPlan.from_json((mounts["/harness"] / "plan.json").read_bytes()) == plan
        wheel_name = spec.wheels[0].package.members[0].name
        original = cache.member_path(prepared.wheels[0], wheel_name).read_bytes()
        assert (mounts["/wheels"] / wheel_name).read_bytes() == original
        assert (mounts["/wheels"] / wheel_name).stat().st_mode & 0o777 == 0o644
        if failure is not None:
            raise failure("Owned process boundary failure")
        return ProcessStreams(b"owned-observer-stream", b"", 0, 0, "exited")

    monkeypatch.setattr(process, "collect_process", observe)
    monkeypatch.setattr(process, "_cleanup", lambda base, name: cleanup.append(name))
    if failure is not None:
        with pytest.raises(failure):
            process.run_python_role(("owned-docker",), spec, prepared, cache, plan, container)
    else:
        actual = process.run_python_role(("owned-docker",), spec, prepared, cache, plan, container)
        assert actual.stdout == b"owned-observer-stream"
    assert cleanup == [container]
    assert staged
    assert all(not path.exists() for path in staged)
