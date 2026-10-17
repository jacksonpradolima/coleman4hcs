"""Versioned JUnit start markers preserve cleanup ownership without inventing completed execution."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coleman.results.domain import fingerprint
from coleman.transports import junit_capture as capture
from coleman.transports.junit_contracts import JunitRoleStart, JunitRoleStartV2


def owned_start(version):
    """Construct schema-only ownership with a fixed UTC instant; no source or container is executed."""
    fields = {
        "schema_version": version,
        "context_fingerprint": fingerprint("owned-junit-context"),
        "trial_id": "owned-trial",
        "role": "buggy",
        "capture_started_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    if version == 1:
        return JunitRoleStart.model_validate(fields)
    namespace = fingerprint("owned-output-namespace")
    ownership = fingerprint([fields["context_fingerprint"], fields["trial_id"], namespace]).removeprefix("sha256:")
    return JunitRoleStartV2.model_validate(
        {**fields, "output_namespace": namespace, "container_name": "coleman-junit-" + ownership[:24] + "-buggy"}
    )


@pytest.mark.parametrize("version", [1, 2])
def test_start_goldens_roundtrip_and_digest_are_stable(version):
    record = owned_start(version)
    encoded = record.to_json()
    reordered = json.dumps(dict(reversed(list(json.loads(encoded).items()))))
    restored = type(record).from_json(reordered)
    assert restored == record
    assert restored.fingerprint == record.fingerprint
    path = Path(__file__).with_name("fixtures") / f"junit-start-owned-v{version}.json"
    assert path.read_text() == encoded + "\n"


@pytest.mark.parametrize("version", [None, 0, 3, True, "2", 2.0])
def test_unknown_or_coerced_start_version_rejects_before_cleanup(tmp_path, monkeypatch, version):
    data = owned_start(2).model_dump(mode="json")
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    marker = tmp_path / "buggy.started.json"
    marker.write_text(json.dumps(data))
    monkeypatch.setattr(capture, "_cleanup", lambda *args: pytest.fail("Invalid version must not clean a container"))
    with pytest.raises(ValueError):
        capture._recover_started_role(
            ("owned-docker",), marker, tmp_path, (data["context_fingerprint"], "owned-trial", "buggy")
        )


@pytest.mark.parametrize(
    "update",
    [
        {"container_name": "another-container"},
        {"container_name": "coleman-junit-" + "0" * 24 + "-buggy"},
        {"output_namespace": fingerprint("changed namespace")},
        {"role": "fixed"},
    ],
)
def test_inconsistent_container_ownership_rejects(update):
    data = owned_start(2).model_dump(mode="json")
    data.update(update)
    with pytest.raises(ValueError):
        JunitRoleStartV2.model_validate(data)


def test_legacy_model_remains_strict_and_does_not_relabel_v2_as_v1():
    payload = owned_start(2).to_json()
    with pytest.raises(ValueError):
        JunitRoleStart.from_json(payload)
    legacy = owned_start(1)
    assert set(json.loads(legacy.to_json())) == {
        "schema_version",
        "phase",
        "context_fingerprint",
        "trial_id",
        "role",
        "capture_started_at",
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_oversized_markers_reject_before_cleanup(tmp_path, monkeypatch, legacy):
    record = owned_start(1 if legacy else 2)
    marker = tmp_path / "buggy.started.json"
    limit = 4096 if legacy else JunitRoleStartV2.MAXIMUM_BYTES
    marker.write_text(record.to_json() + " " * limit)
    monkeypatch.setattr(capture, "_cleanup", lambda *args: pytest.fail("Oversized marker must not clean a container"))
    with pytest.raises(ValueError, match="exceeds"):
        capture._recover_started_role(
            ("owned-docker",), marker, tmp_path, (record.context_fingerprint, "owned-trial", "buggy")
        )
