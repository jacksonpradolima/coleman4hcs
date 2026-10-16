"""Owned source views: immutable raw acquisition, explicit provenance and deterministic assembly."""

import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coleman.corpus import MaterializationSpec, SourceCache, load_registry, validate_materialization
from coleman.corpus.archives import deterministic_zip_bytes
from coleman.corpus.source_views import SourceViewSpec, materialize_source_view, source_view_bytes


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def source_spec(name, payload, archive="raw", members=None):
    if members is None:
        members = [{"name": name, "sha256": digest(payload), "size_bytes": len(payload)}]
    return MaterializationSpec.model_validate(
        {
            "schema_version": 1,
            "package_id": name,
            "archive": archive,
            "source": {
                "uri": "urn:coleman:owned-source-view:" + name,
                "version": "owned-v1",
                "citation": "Owned synthetic source-view fixture, not foreign code",
                "sha256": digest(payload),
                "size_bytes": len(payload),
                "availability": "not_verified",
            },
            "license": load_registry().card("ci-serial-boundary-v1").manifest.license.model_dump(),
            "members": members,
        }
    )


def recipe():
    entries = (("src/owned.txt", b"Owned CI source.\n"), ("LICENSE", b"Owned schema fixture only.\n"))
    inputs = tuple(source_spec(name, payload) for name, payload in entries)
    payload = source_view_bytes(entries)
    output = source_spec(
        "owned-source-view-v1",
        payload,
        "zip",
        [p.members[0].model_dump() for p in sorted(inputs, key=lambda p: p.members[0].name)],
    )
    return SourceViewSpec(schema_version=1, inputs=inputs, output=output), dict(entries)


def test_roundtrip_golden_real_cache_offline_recovery_and_concurrent_replay(tmp_path):
    spec, data = recipe()
    cache = SourceCache(tmp_path / "cache")
    calls = []

    def reader_for(package):
        name = package.members[0].name

        def factory():
            calls.append(name)
            return io.BytesIO(data[name])

        return factory

    receipt = materialize_source_view(spec, cache, reader_for)
    assert set(calls) == set(data)
    assert SourceViewSpec.from_json(spec.to_json()) == spec
    golden = Path(__file__).with_name("fixtures") / "source-view-owned-v1.json"
    assert spec.to_json() + "\n" == golden.read_text()
    validate_materialization(receipt, cache)
    assert materialize_source_view(spec, cache) == receipt
    materialize_source_view(spec, cache, reader_for)
    assert len(calls) == 2
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: materialize_source_view(spec, cache), range(8)))
    assert all(result == receipt for result in results)
    assert cache.member_path(receipt, "src/owned.txt").read_bytes() == data["src/owned.txt"]


def test_zip_metadata_order_metamorphism_empty_file_and_exact_pins():
    entries = (("z.txt", b"Z"), ("a.txt", b""))
    first = source_view_bytes(entries)
    assert source_view_bytes(tuple(reversed(entries))) == first
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["a.txt", "z.txt"]
        for member in archive.infolist():
            assert member.date_time == (1980, 1, 1, 0, 0, 0)
            assert member.compress_type == zipfile.ZIP_STORED
            assert member.create_system == 3
            assert member.external_attr >> 16 == 0o100644
        assert archive.read("a.txt") == b""
    spec, _ = recipe()
    assert changed(spec, inputs=tuple(reversed(spec.inputs))) == spec


@pytest.mark.parametrize("entries", [(), (("a", b"a"), ("a", b"b")), (("../escape", b"a"),), (("a", bytearray(b"x")),)])
def test_invalid_entries_reject_before_writing(entries):
    with pytest.raises((ValueError, TypeError)):
        source_view_bytes(entries)


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_source_view_versions_fail_closed(version):
    spec, _ = recipe()
    payload = spec.model_dump(mode="json")
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    encoded = json.dumps(payload)
    with pytest.raises(ValueError):
        SourceViewSpec.from_json(encoded)


def test_member_names_package_ids_licence_and_versions_are_bound():
    spec, _ = recipe()
    for updates in ({"recipe": "future"}, {"inputs": (spec.inputs[0],) * 2}, {"output": spec.inputs[0]}):
        with pytest.raises(ValueError):
            changed(spec, **updates)
    duplicated_id = changed(spec.inputs[0], package_id=spec.inputs[1].package_id)
    with pytest.raises(ValueError, match="package identity"):
        changed(spec, inputs=(duplicated_id, spec.inputs[1]))
    altered_terms = changed(spec.inputs[0].license, redistribution="recipe_only")
    input_terms = changed(spec.inputs[0], license=altered_terms)
    with pytest.raises(ValueError, match="terms must match"):
        changed(spec, inputs=(input_terms, spec.inputs[1]))
    input_version = changed(spec.inputs[0], source=changed(spec.inputs[0].source, version="other"))
    with pytest.raises(ValueError, match="versions disagree"):
        changed(spec, inputs=(input_version, spec.inputs[1]))
    missing = changed(spec.output, members=spec.output.members[:-1])
    with pytest.raises(ValueError, match="preserve every input"):
        changed(spec, output=missing)


def test_interrupted_input_retry_output_pins_and_byte_tampering(tmp_path):
    spec, data = recipe()
    cache = SourceCache(tmp_path / "cache")
    seen = []

    def interrupted(package):
        name = package.members[0].name
        seen.append(name)
        if len(seen) == 2:

            def fail():
                raise OSError("interrupted source acquisition")

            return fail
        return lambda: io.BytesIO(data[name])

    with pytest.raises(OSError, match="interrupted"):
        materialize_source_view(spec, cache, interrupted)
    assert len(list((cache.root / "blobs").iterdir())) == 1
    receipt = materialize_source_view(spec, cache, lambda p: lambda: io.BytesIO(data[p.members[0].name]))
    wrong_output = changed(spec.output, source=changed(spec.output.source, sha256=digest(b"wrong archive")))
    wrong = changed(spec, output=wrong_output)
    with pytest.raises(ValueError):
        materialize_source_view(wrong, cache)
    path = cache.member_path(receipt, "src/owned.txt")
    path.write_bytes(b"changed cached input")
    with pytest.raises(ValueError):
        materialize_source_view(spec, cache)


@pytest.mark.parametrize("member,total", [(0, 16), (True, 16), (-1, 16), (16, 0), (16, True), (16, 64 * 1024**2 + 1)])
def test_shared_zip_encoder_rejects_invalid_explicit_byte_budgets(member, total):
    entries = (("owned.txt", b"owned"),)
    with pytest.raises(ValueError, match="byte budgets"):
        deterministic_zip_bytes(entries, maximum_member_bytes=member, maximum_total_bytes=total)


def test_shared_zip_encoder_exact_bounds_order_and_source_view_byte_compatibility():
    entries = (("b.txt", b"bb"), ("a.txt", b"aa"))
    encoded = deterministic_zip_bytes(entries, maximum_member_bytes=2, maximum_total_bytes=4)
    assert source_view_bytes(entries) == encoded
    assert deterministic_zip_bytes(tuple(reversed(entries)), maximum_member_bytes=2, maximum_total_bytes=4) == encoded
    with pytest.raises(ValueError, match="decoded-byte bounds"):
        deterministic_zip_bytes(entries, maximum_member_bytes=1, maximum_total_bytes=4)
    with pytest.raises(ValueError, match="decoded-byte bounds"):
        deterministic_zip_bytes(entries, maximum_member_bytes=2, maximum_total_bytes=3)


def test_diagnostic_profile_does_not_widen_source_view_limits():
    payload = b"x" * (8 * 1024**2 + 1)
    entries = (("diagnostic.json", payload),)
    with pytest.raises(ValueError, match="decoded-byte bounds"):
        source_view_bytes(entries)
    encoded = deterministic_zip_bytes(entries, maximum_member_bytes=16 * 1024**2, maximum_total_bytes=40 * 1024**2)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert archive.read("diagnostic.json") == payload
