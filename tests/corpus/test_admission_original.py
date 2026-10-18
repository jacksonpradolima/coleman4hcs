"""Original-verifier pin, package and binding failures stop before any foreign process or source execution."""

import hashlib
import json
import zipfile
from dataclasses import replace

import pytest

from coleman.application.admission import OriginalCaptureVerifier
from coleman.corpus.controlled_contracts import ControlledDataset
from tests.corpus.admission_support import controlled_source


def controlled_payload(path):
    """Keep the deliberately inert controlled fixture's typed payload/cache explicit."""
    evidence = controlled_source(path)
    assert isinstance(evidence.payload, ControlledDataset)
    assert evidence.cache is not None
    return evidence.payload, evidence.cache


def owned_wheel(path, *, metadata=b"Name: coleman\nVersion: owned-v1\n\n", sdk=True, duplicate=False):
    """Create an inert owned metadata ZIP, never imported as a Python package."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        if metadata is not None:
            archive.writestr("coleman-owned.dist-info/METADATA", metadata)
        if sdk:
            archive.writestr("coleman/__init__.py", b"# Owned inert metadata fixture; never imported.\n")
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("coleman/__init__.py", b"# Duplicate owned member\n")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verifier(tmp_path, wheel, digest):
    """A nonexistent interpreter proves all declared preflight failures occur before process creation."""
    return OriginalCaptureVerifier(
        "python",
        tmp_path / "must-not-start-python",
        wheel,
        digest,
        "urn:coleman:owned-inert-wheel",
        tmp_path / "must-not-read-receipt.json",
        plan=tmp_path / "must-not-read-plan.json",
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"family": "unknown-family"}, "family"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"timeout_seconds": 901}, "timeout"),
        ({"timeout_seconds": True}, "timeout"),
        ({"timeout_seconds": 1.5}, "timeout"),
        ({"plan": None}, "separate original plan"),
        ({"family": "junit_matrix", "plan": None}, "output tree"),
    ],
)
def test_invalid_family_timeout_and_plan_bindings_never_start_a_process(tmp_path, mutation, reason):
    payload, cache = controlled_payload(tmp_path / "source")
    wheel = tmp_path / "wheel.whl"
    digest = owned_wheel(wheel)
    item = replace(verifier(tmp_path, wheel, digest), **json.loads(json.dumps(mutation)))
    with pytest.raises(ValueError, match=reason):
        item.verify(payload, cache)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("wrong-pin", "SHA-256"),
        ("empty", "32-MiB"),
        ("no-metadata", "metadata/decoded"),
        ("wrong-project", "COLEMAN version"),
        ("missing-version", "COLEMAN version"),
        ("no-sdk", "no authored SDK"),
        ("duplicate", "duplicate members"),
    ],
)
def test_inert_wheel_metadata_cannot_pass_pin_or_package_checks(tmp_path, kind, reason):
    payload, cache = controlled_payload(tmp_path / "source")
    wheel = tmp_path / "wheel.whl"
    metadata = {
        "no-metadata": None,
        "wrong-project": b"Name: unrelated\nVersion: 1\n\n",
        "missing-version": b"Name: coleman\n\n",
    }.get(kind, b"Name: coleman\nVersion: owned-v1\n\n")
    digest = owned_wheel(wheel, metadata=metadata, sdk=kind != "no-sdk", duplicate=kind == "duplicate")
    if kind == "wrong-pin":
        digest = "sha256:" + "0" * 64
    if kind == "empty":
        wheel.write_bytes(b"")
    item = verifier(tmp_path, wheel, digest)
    with pytest.raises(ValueError, match=reason):
        item.verify(payload, cache)


def test_oversized_declared_decoded_archive_is_rejected_without_extracting(tmp_path):
    payload, cache = controlled_payload(tmp_path / "source")
    wheel = tmp_path / "wheel.whl"
    digest = owned_wheel(wheel)
    # Change the central directory's uncompressed size only; no 64-MiB file
    # or untrusted extraction is needed to exercise the explicit decoded bound.
    value = bytearray(wheel.read_bytes())
    offset = value.index(b"PK\x01\x02")
    value[offset + 24 : offset + 28] = (64 * 1024**2 + 1).to_bytes(4, "little")
    wheel.write_bytes(value)
    digest = "sha256:" + hashlib.sha256(value).hexdigest()
    item = verifier(tmp_path, wheel, digest)
    with pytest.raises(ValueError, match="decoded bytes"):
        item.verify(payload, cache)
