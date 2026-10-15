"""Reject empty release inventories and preserve prior bytes on failed generation."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("release_sbom", Path(__file__).parents[1] / "scripts/release_sbom.py")
assert SPEC is not None
assert SPEC.loader is not None
sbom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sbom)
VERSION = "4.0.0"
WHEEL = "coleman-4.0.0-py3-none-any.whl"


def inventory():
    return {
        "bomFormat": "CycloneDX",
        "metadata": {"component": {"name": WHEEL, "version": VERSION}},
        "components": [{"name": "coleman", "version": VERSION}],
    }


@pytest.mark.parametrize("change", ["empty", "other", "version", "root", "format", "extra"])
def test_rejects_incomplete_or_unrelated_inventory(change):
    payload = inventory()
    if change == "empty":
        del payload["components"]
    elif change == "other":
        payload["components"][0]["name"] = "unrelated"
    elif change == "version":
        payload["components"][0]["version"] = "3.6.1"
    elif change == "root":
        payload["metadata"]["component"]["name"] = "dist"
    elif change == "format":
        payload["bomFormat"] = "unknown"
    else:
        payload["components"].append({"name": "build-host-package", "version": "1"})
    with pytest.raises(ValueError):
        sbom.validate_inventory(payload, VERSION, WHEEL)


@pytest.mark.parametrize("metadata", [b"Name: other\nVersion: 4.0.0\n", b"Name: coleman\nVersion: 3.6.1\n"])
def test_wrong_wheel_identity_never_replaces_prior_inventory(tmp_path, metadata):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("coleman-4.0.0.dist-info/METADATA", metadata)
    output = tmp_path / "sbom.json"
    output.write_text("previous complete inventory")
    with pytest.raises(ValueError, match="Wheel metadata"):
        sbom._build_sbom(wheel, VERSION, output)
    assert output.read_text() == "previous complete inventory"


def test_invalid_tool_output_does_not_publish_or_replace_bytes(tmp_path, monkeypatch):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("coleman-4.0.0.dist-info/METADATA", "Name: coleman\nVersion: 4.0.0\n")
    output = tmp_path / "sbom.json"
    output.write_text("previous complete inventory")

    def empty_inventory(command, **_kwargs):
        raw = Path(command[-1].split("=", 1)[1])
        raw.write_text(json.dumps({"bomFormat": "CycloneDX"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sbom.subprocess, "run", empty_inventory)
    with pytest.raises(ValueError):
        sbom._build_sbom(wheel, VERSION, output)
    assert output.read_text() == "previous complete inventory"


@pytest.mark.skipif(shutil.which("syft") is None, reason="Syft is release-only tooling")
def test_real_syft_catalogs_wheel_metadata_without_host_paths(tmp_path):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("coleman-4.0.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: coleman\nVersion: 4.0.0\n")
        archive.writestr("../../outside", "not extracted")
    output = tmp_path / "sbom.json"
    sbom._build_sbom(wheel, VERSION, output)
    payload = json.loads(output.read_text())
    sbom.validate_inventory(payload, VERSION, WHEEL)
    assert payload["metadata"]["component"]["hashes"] == [
        {"alg": "SHA-256", "content": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    ]
    assert payload["components"][0]["purl"] == "pkg:github/jacksonpradolima/coleman@v4.0.0"
    assert "pkg:pypi/coleman" not in output.read_text()
    assert str(tmp_path) not in output.read_text()
    assert not (tmp_path.parent / "outside").exists()


def test_interrupted_atomic_publication_preserves_prior_bytes_and_cleans_temporary_file(tmp_path, monkeypatch):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("coleman-4.0.0.dist-info/METADATA", "Name: coleman\nVersion: 4.0.0\n")
    output = tmp_path / "sbom.json"
    output.write_text("previous complete inventory")

    def valid_inventory(command, **_kwargs):
        Path(command[-1].split("=", 1)[1]).write_text(json.dumps(inventory()))
        return subprocess.CompletedProcess(command, 0)

    def interrupted(*_args):
        raise OSError("interrupted publication")

    monkeypatch.setattr(sbom.subprocess, "run", valid_inventory)
    monkeypatch.setattr(sbom.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted publication"):
        sbom._build_sbom(wheel, VERSION, output)
    assert output.read_text() == "previous complete inventory"
    assert {path.name for path in tmp_path.iterdir()} == {WHEEL, "sbom.json"}


def test_source_binding_keeps_dependency_references_coherent(tmp_path, monkeypatch):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("coleman-4.0.0.dist-info/METADATA", "Name: coleman\nVersion: 4.0.0\n")
    output = tmp_path / "sbom.json"
    payload = inventory()
    previous = "pkg:pypi/coleman@4.0.0"
    payload["components"][0].update({"bom-ref": previous, "purl": previous, "cpe": "guessed-cpe"})
    payload["dependencies"] = [{"ref": "wheel", "dependsOn": [previous]}, {"ref": previous, "dependsOn": []}]

    def catalog(command, **_kwargs):
        Path(command[-1].split("=", 1)[1]).write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sbom.subprocess, "run", catalog)
    sbom._build_sbom(wheel, VERSION, output)
    result = json.loads(output.read_text())
    references = {item["bom-ref"] for item in result["components"]} | {"wheel"}
    assert all(edge["ref"] in references for edge in result["dependencies"])
    assert all(target in references for edge in result["dependencies"] for target in edge["dependsOn"])
    assert previous not in output.read_text()
    assert "guessed-cpe" not in output.read_text()


def test_release_command_refuses_caller_supplied_output_paths(tmp_path):
    script = Path(__file__).parents[1] / "scripts/release_sbom.py"
    outside = tmp_path / "caller-selected.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(outside)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 2
    assert "unrecognized arguments" in completed.stderr
    assert not outside.exists()
