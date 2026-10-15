"""Owned tiny packages: licensing, byte integrity, safe members, concurrency and offline recovery."""

import hashlib
import io
import json
import os
import stat
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    MaterializationReceipt,
    MaterializationSpec,
    PackageMember,
    SourceCache,
    bundled_fixture_bytes,
    load_registry,
    materialize_source,
    publish_materialization,
    validate_fixture_source,
    validate_materialization,
)
from coleman.manifest import generate_manifest

ROOT = Path(__file__).parents[2]
ID = "ci-serial-boundary-v1"


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@contextmanager
def reader(payload):
    with io.BytesIO(payload) as stream:
        yield stream


def zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = mode << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def recipe(payload=None, *, archive="raw", members=None):
    card = load_registry().card(ID)
    data = bundled_fixture_bytes(ID) if payload is None else payload
    if members is None:
        members = ({"name": "fixture.json", "sha256": digest(data), "size_bytes": len(data)},)
    source = changed(
        card.manifest.source,
        uri="urn:coleman:owned-source-package:v1",
        version="owned-package-v1",
        citation="Owned known-answer CI correctness package",
        sha256=digest(data),
        size_bytes=len(data),
    )
    spec = MaterializationSpec(
        schema_version=1,
        package_id="owned-ci-package-v1",
        source=source,
        license=card.manifest.license,
        archive=archive,
        members=tuple(PackageMember.model_validate(member) for member in members),
    )
    return spec, data


def zipped(entries=None):
    fixture = bundled_fixture_bytes(ID)
    if entries is None:
        entries = [("fixture.json", fixture, stat.S_IFREG | 0o644)]
    payload = zip_bytes(entries)
    members = ({"name": "fixture.json", "sha256": digest(fixture), "size_bytes": len(fixture)},)
    return recipe(payload, archive="zip", members=members)


def test_raw_zip_roundtrip_golden_and_real_native_correctness(tmp_path):
    for kind, make in (("raw", recipe), ("zip", zipped)):
        spec, payload = make()
        cache = SourceCache(tmp_path / kind)
        receipt = materialize_source(spec, cache, lambda payload=payload: reader(payload))
        assert receipt.admission == "not_evaluated"
        assert MaterializationReceipt.from_json(receipt.to_json()) == receipt
        assert MaterializationSpec.from_json(spec.to_json()) == spec
        validate_materialization(receipt, cache)
        path = cache.member_path(receipt, "fixture.json")
        manifest = load_registry().card(ID).manifest
        report = validate_fixture_source(manifest, path.read_bytes())
        assert len(report.observations) == 2
        assert materialize_source(spec, cache) == receipt
        published = publish_materialization(receipt, cache, ArtifactWriter(tmp_path / "artifacts"))
        assert published.read_text() == receipt.to_json() + "\n"
        assert publish_materialization(receipt, cache, ArtifactWriter(tmp_path / "artifacts")) == published
    golden = Path(__file__).parent / "fixtures/corpus-materialization-receipt-v1.json"
    assert golden.read_text() == materialize_source(*_raw_cached(tmp_path / "golden")).to_json() + "\n"


def _raw_cached(root):
    spec, payload = recipe()
    cache = SourceCache(root)
    materialize_source(spec, cache, lambda payload=payload: reader(payload))
    return spec, cache


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_unknown_and_legacy_versions_fail_closed(version):
    spec, _ = recipe()
    payload = json.loads(MaterializationReceipt(schema_version=1, spec=spec).to_json())
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    serialized = json.dumps(payload)
    with pytest.raises(ValueError):
        MaterializationReceipt.from_json(serialized)


def test_terms_pins_budgets_and_raw_aliases_reject_before_reader_access(tmp_path):
    spec, _ = recipe()
    for license_data in ({"status": "pending"}, {"redistribution": "prohibited"}, {"status": "prohibited"}):
        invalid = changed(spec.license, **license_data)
        with pytest.raises(ValueError, match="terms"):
            changed(spec, license=invalid)
    for source_data in ({"sha256": None}, {"size_bytes": None}):
        invalid = changed(spec.source, **source_data)
        with pytest.raises(ValueError, match="digest"):
            changed(spec, source=invalid)
    invalid = changed(spec.limits, maximum_source_bytes=1)
    with pytest.raises(ValueError, match="source/disk"):
        changed(spec, limits=invalid)
    invalid = changed(spec.limits, maximum_member_bytes=1)
    with pytest.raises(ValueError, match="member"):
        changed(spec, limits=invalid)
    invalid = changed(spec.limits, maximum_extracted_bytes=1)
    with pytest.raises(ValueError, match="total decoded"):
        changed(spec, limits=invalid)
    wrong = changed(spec.members[0], sha256="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="unchanged"):
        changed(spec, members=(wrong,))
    with pytest.raises(ValueError, match="Duplicate"):
        changed(spec, members=(*spec.members, *spec.members))
    called = []
    invalid = spec.model_copy(update={"license": changed(spec.license, status="pending")})
    cache = SourceCache(tmp_path)
    with pytest.raises(ValueError, match="terms"):
        materialize_source(invalid, cache, lambda: called.append(True))
    assert called == []


@pytest.mark.parametrize(
    "name", ["../outside", "/outside", "a/../b", "a//b", "./a", "C:/a", "a\\b", "a\x00b", "a\nb", " a", "a."]
)
def test_member_paths_reject_before_any_cache_write(name):
    spec, _ = recipe()
    with pytest.raises(ValueError):
        changed(spec.members[0], name=name)


@pytest.mark.parametrize("payload", [b"changed", b"", b"too many source bytes"], ids=["changed", "short", "long"])
def test_checksum_size_failure_has_no_completed_blob_or_receipt(tmp_path, payload):
    spec, _ = recipe()
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_source(spec, cache, lambda payload=payload: reader(payload))
    assert not list(cache.root.glob("blobs/*"))
    assert not list(cache.root.rglob("*.tmp"))
    receipt = MaterializationReceipt(schema_version=1, spec=spec)
    writer = ArtifactWriter(tmp_path / "artifacts")
    with pytest.raises(FileNotFoundError):
        publish_materialization(receipt, cache, writer)
    assert not list((tmp_path / "artifacts").rglob("*.json"))


def test_actual_checksum_is_checked_even_when_length_matches(tmp_path):
    spec, payload = recipe()
    cache = SourceCache(tmp_path)
    changed_payload = bytes([payload[0] ^ 1]) + payload[1:]
    with pytest.raises(ValueError, match="checksum"):
        materialize_source(spec, cache, lambda: reader(changed_payload))
    assert not list(cache.root.glob("blobs/*"))


def test_cache_hit_is_lazy_refresh_checks_explicit_source_and_corruption_rejects(tmp_path):
    spec, payload = recipe()
    cache = SourceCache(tmp_path)
    receipt = materialize_source(spec, cache, lambda payload=payload: reader(payload))
    called = []
    assert materialize_source(spec, cache, lambda: called.append(True)) == receipt
    assert called == []
    with pytest.raises(ValueError):
        materialize_source(spec, cache, lambda: reader(b"bad"), refresh_source=True)
    path = cache.member_path(receipt, "fixture.json")
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        validate_materialization(receipt, cache)
    with pytest.raises(ValueError):
        materialize_source(spec, cache, lambda payload=payload: reader(payload))


def test_cache_format_root_symlinks_and_legacy_marker_fail_closed(tmp_path):
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        SourceCache(repository / "cache")
    assert not (repository / "cache").exists()
    root = tmp_path / "cache"
    cache = SourceCache(root)
    marker = cache.root / "cache-format.json"
    marker.write_text('{"schema_version":2,"kind":"coleman-source-cache-v1"}')
    with pytest.raises(ValueError):
        SourceCache(root)
    marker.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"sentinel bytes")
    marker.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        SourceCache(root)
    assert outside.read_bytes() == b"sentinel bytes"


@pytest.mark.parametrize("entry", ["../outside", "/outside", "a\\b", "./fixture.json"])
def test_frozen_malicious_archive_names_cannot_escape_cache(tmp_path, entry):
    spec, payload = zipped([(entry, bundled_fixture_bytes(ID), stat.S_IFREG | 0o644)])
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_source(spec, cache, lambda payload=payload: reader(payload))
    assert not (tmp_path / "outside").exists()
    assert len(list(cache.root.glob("blobs/*"))) == 1  # Verified source, never a completed package.


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644, stat.S_IFCHR | 0o644])
def test_archive_links_and_special_files_reject(tmp_path, mode):
    spec, payload = zipped([("fixture.json", bundled_fixture_bytes(ID), mode)])
    cache = SourceCache(tmp_path)
    with pytest.raises(ValueError, match="links and special"):
        materialize_source(spec, cache, lambda payload=payload: reader(payload))


def test_archive_duplicate_extra_missing_size_crc_and_metadata_failures(tmp_path):
    fixture = bundled_fixture_bytes(ID)
    cases = [
        [("fixture.json", fixture, stat.S_IFREG), ("fixture.json", fixture, stat.S_IFREG)],
        [("extra.json", fixture, stat.S_IFREG)],
        [("fixture.json", b"short", stat.S_IFREG)],
    ]
    for index, entries in enumerate(cases):
        with pytest.warns(UserWarning) if index == 0 else _no_warning():
            spec, payload = zipped(entries)
        cache = SourceCache(tmp_path / str(index))
        with pytest.raises(ValueError):
            materialize_source(spec, cache, lambda payload=payload: reader(payload))
    spec, payload = zipped()
    damaged = bytearray(payload)
    start = payload.index(fixture)
    damaged[start] ^= 1
    spec = changed(spec, source=changed(spec.source, sha256=digest(damaged)))
    cache = SourceCache(tmp_path / "crc")
    with pytest.raises(ValueError, match="ZIP"):
        materialize_source(spec, cache, lambda: reader(bytes(damaged)))
    for label, broken in (("missing-footer", b"not a zip"), ("trailing", payload + b"trailing")):
        source = changed(spec.source, sha256=digest(broken), size_bytes=len(broken))
        altered = changed(spec, source=source)
        cache = SourceCache(tmp_path / label)
        with pytest.raises(ValueError, match="ZIP"):
            materialize_source(altered, cache, lambda broken=broken: reader(broken))


@contextmanager
def _no_warning():
    yield


def test_interrupted_concurrent_package_replay_and_native_manifest(tmp_path, monkeypatch):
    spec, payload = zipped()
    cache = SourceCache(tmp_path / "cache")
    original = os.link
    count = 0

    def interrupted(source, target):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("interrupted member publication")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            materialize_source(spec, cache, lambda payload=payload: reader(payload))
    assert len(list(cache.root.glob("blobs/*"))) == 1
    assert not list(cache.root.rglob("*.tmp"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: materialize_source(spec, cache), range(8)))
    assert len({receipt.fingerprint for receipt in receipts}) == 1
    writer = ArtifactWriter(tmp_path / "native")
    path = publish_materialization(receipts[0], cache, writer)
    manifest = json.loads(generate_manifest(writer.run_root, run_id="owned-cache-ci").read_bytes())
    assert any(
        entry["type"] == "corpus_materialization"
        and entry["relative_path"] == path.relative_to(writer.run_root).as_posix()
        for entry in manifest["artifacts"]
    )


def test_member_order_canonicalizes_and_changed_source_bytes_remain_distinct(tmp_path):
    first = b"first owned CI bytes"
    second = b"second owned CI bytes"
    entries = [("a.txt", first, stat.S_IFREG), ("b.txt", second, stat.S_IFREG)]
    members = (
        {"name": "a.txt", "sha256": digest(first), "size_bytes": len(first)},
        {"name": "b.txt", "sha256": digest(second), "size_bytes": len(second)},
    )
    package, payload = recipe(zip_bytes(entries), archive="zip", members=members)
    reordered = changed(package, members=tuple(reversed(package.members)))
    assert reordered.fingerprint == package.fingerprint
    cache = SourceCache(tmp_path)
    one = materialize_source(package, cache, lambda: reader(payload))
    assert materialize_source(reordered, cache) == one
    changed_bytes = zip_bytes(list(reversed(entries)))
    changed_source = changed(package.source, sha256=digest(changed_bytes), size_bytes=len(changed_bytes))
    other = changed(package, source=changed_source)
    two = materialize_source(other, cache, lambda: reader(changed_bytes))
    assert one.fingerprint != two.fingerprint
    assert cache.member_path(one, "a.txt") == cache.member_path(two, "a.txt")


def test_unrelated_cached_member_cannot_certify_archive_membership(tmp_path):
    spec, payload = zipped()
    actual = bundled_fixture_bytes(ID)
    unrelated = bytes([actual[0] ^ 1]) + actual[1:]
    expected = changed(spec.members[0], sha256=digest(unrelated))
    spec = changed(spec, members=(expected,))
    cache = SourceCache(tmp_path)
    cache.store_blob(expected.sha256, expected.size_bytes, io.BytesIO(unrelated))
    with pytest.raises(ValueError, match="checksum"):
        materialize_source(spec, cache, lambda: reader(payload))
    receipt = MaterializationReceipt(schema_version=1, spec=spec)
    with pytest.raises(ValueError, match="checksum"):
        validate_materialization(receipt, cache)


def test_source_cache_and_export_tree_must_not_overlap(tmp_path):
    spec, cache = _raw_cached(tmp_path / "cache")
    receipt = MaterializationReceipt(schema_version=1, spec=spec)
    for output in (tmp_path, cache.root, cache.root / "artifacts"):
        writer = ArtifactWriter(output)
        with pytest.raises(ValueError, match="separate"):
            publish_materialization(receipt, cache, writer)
    assert not list(tmp_path.rglob("corpus-materializations/*.json"))
    with pytest.raises(AttributeError):
        cache.root = tmp_path / "different"


def test_empty_raw_entity_and_recipe_only_terms_are_valid_integrity_inputs(tmp_path):
    spec, payload = recipe(b"")
    license_data = changed(spec.license, redistribution="recipe_only")
    spec = changed(spec, license=license_data)
    cache = SourceCache(tmp_path / "cache")
    receipt = materialize_source(spec, cache, lambda: reader(payload))
    assert cache.member_path(receipt, "fixture.json").read_bytes() == b""
    assert receipt.admission == "not_evaluated"
    path = publish_materialization(receipt, cache, ArtifactWriter(tmp_path / "exports"))
    assert path.exists()
    assert "source-integrity-only" in path.read_text()


def test_archive_directories_and_budget_preflight(tmp_path):
    fixture = bundled_fixture_bytes(ID)
    members = ({"name": "owned/fixture.json", "sha256": digest(fixture), "size_bytes": len(fixture)},)
    payload = zip_bytes([("owned/", b"", stat.S_IFDIR), ("owned/fixture.json", fixture, stat.S_IFREG)])
    spec, _ = recipe(payload, archive="zip", members=members)
    cache = SourceCache(tmp_path / "valid")
    receipt = materialize_source(spec, cache, lambda: reader(payload))
    assert cache.member_path(receipt, "owned/fixture.json").read_bytes() == fixture
    assert not (cache.root / "owned").exists()
    limits = changed(spec.limits, maximum_archive_entries=1)
    one_entry = changed(spec, limits=limits)
    small = SourceCache(tmp_path / "entry-bound")
    with pytest.raises(ValueError, match="resource limits"):
        materialize_source(one_entry, small, lambda: reader(payload))
    limits = changed(spec.limits, maximum_zip_metadata_bytes=22)
    metadata_bound = changed(spec, limits=limits)
    small = SourceCache(tmp_path / "metadata-bound")
    with pytest.raises(ValueError, match="resource limits"):
        materialize_source(metadata_bound, small, lambda: reader(payload))


@pytest.mark.parametrize(
    "kind", ["multidisk", "zip64-footer", "offset", "encrypted", "zip64-extra", "extra-tail", "version"]
)
def test_unsupported_or_inconsistent_zip_structures_reject(tmp_path, kind):
    fixture = bundled_fixture_bytes(ID)
    info = zipfile.ZipInfo("fixture.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = stat.S_IFREG << 16
    if kind == "zip64-extra":
        info.extra = struct.pack("<HH", 1, 0)
    elif kind == "extra-tail":
        info.extra = b"xx"
    elif kind == "version":
        info.extract_version = 45
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, fixture)
    payload = bytearray(buffer.getvalue())
    end = payload.rfind(b"PK\x05\x06")
    if kind == "multidisk":
        struct.pack_into("<H", payload, end + 4, 1)
    elif kind == "zip64-footer":
        struct.pack_into("<H", payload, end + 10, 65535)
    elif kind == "offset":
        struct.pack_into("<I", payload, end + 16, 0)
    elif kind == "encrypted":
        struct.pack_into("<H", payload, 6, 1)
        central = payload.index(b"PK\x01\x02")
        struct.pack_into("<H", payload, central + 8, 1)
    members = ({"name": "fixture.json", "sha256": digest(fixture), "size_bytes": len(fixture)},)
    spec, _ = recipe(bytes(payload), archive="zip", members=members)
    cache = SourceCache(tmp_path)
    with pytest.raises(ValueError, match="ZIP"):
        materialize_source(spec, cache, lambda: reader(bytes(payload)))
    assert len(list(cache.root.glob("blobs/*"))) == 1


def test_cache_low_level_bounds_missing_inputs_and_bad_reader_contract(tmp_path):
    cache = SourceCache(tmp_path)
    for size in (-1, True, 8 * 1024**3 + 1):
        with pytest.raises(ValueError, match="byte count"):
            cache.verify_blob("sha256:" + "0" * 64, size)
        with pytest.raises(ValueError, match="byte count"):
            cache.store_blob("sha256:" + "0" * 64, size, io.BytesIO())
    with pytest.raises(ValueError, match="SHA-256"):
        cache.verify_blob("../outside", 0)
    spec, _ = recipe()
    with pytest.raises(ValueError, match="not cached"):
        materialize_source(spec, cache)
    invalid_refresh = cast(bool, 1)
    with pytest.raises(ValueError, match="boolean"):
        materialize_source(spec, cache, refresh_source=invalid_refresh)

    class BadReader:
        def read(self, maximum, /):
            return "not binary"

    with pytest.raises(TypeError, match="return bytes"):
        cache.store_blob("sha256:" + "0" * 64, 3, BadReader())
    assert not list(cache.root.glob("blobs/*"))


def test_unknown_cache_marker_and_missing_member_do_not_get_silently_repaired(tmp_path):
    root = tmp_path / "cache"
    cache = SourceCache(root)
    marker = root / "cache-format.json"
    marker.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError, match="metadata limit"):
        SourceCache(root)
    marker.unlink()
    cache = SourceCache(root)
    spec, payload = zipped()
    receipt = materialize_source(spec, cache, lambda: reader(payload))
    with pytest.raises(KeyError):
        cache.member_path(receipt, "unknown")
    member = cache.member_path(receipt, "fixture.json")
    member.unlink()
    with pytest.raises(FileNotFoundError):
        validate_materialization(receipt, cache)
    assert not member.exists()
    assert materialize_source(spec, cache) == receipt
    assert member.exists()


def test_fixture_license_and_source_limits_fail_before_publication(tmp_path):
    spec, payload = zipped()
    extra = changed(spec.members[0], name="other.json")
    limits = changed(spec.limits, maximum_archive_entries=1)
    with pytest.raises(ValueError, match="entry limit"):
        changed(spec, members=(*spec.members, extra), limits=limits)
    cache = SourceCache(tmp_path / "cache")
    receipt = materialize_source(spec, cache, lambda: reader(payload))
    blocked = changed(spec.license, status="prohibited")
    invalid_spec = spec.model_copy(update={"license": blocked})
    forged = receipt.model_copy(update={"spec": invalid_spec})
    writer = ArtifactWriter(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="terms"):
        publish_materialization(forged, cache, writer)
    assert not list((tmp_path / "artifacts").rglob("*.json"))


def test_reader_cannot_bypass_fixed_chunk_bound(tmp_path):
    cache = SourceCache(tmp_path)

    class OversizedReader:
        def read(self, maximum, /):
            return b"x" * (maximum + 1)

    with pytest.raises(ValueError, match="chunk bound"):
        cache.store_blob("sha256:" + "0" * 64, 1_000_000, OversizedReader())
    assert not list(cache.root.glob("blobs/*"))
    assert not list(cache.root.rglob("*.tmp"))


def test_source_byte_limit_rejects_a_real_overlong_entity(tmp_path):
    spec, payload = recipe()
    cache = SourceCache(tmp_path)
    excessive = payload + b"unexpected tail"
    with pytest.raises(ValueError, match="exceeds frozen"):
        materialize_source(spec, cache, lambda: reader(excessive))
    assert not list(cache.root.glob("blobs/*"))


@pytest.mark.parametrize(
    "kind", ["zip64-count", "zip64-size", "unsupported-compression", "directory-data", "lying-count", "valid-extra"]
)
def test_additional_real_zip_metadata_boundaries(tmp_path, kind):
    fixture = bundled_fixture_bytes(ID)
    info = zipfile.ZipInfo("fixture.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = stat.S_IFREG << 16
    if kind == "unsupported-compression":
        info.compress_type = zipfile.ZIP_BZIP2
    elif kind == "directory-data":
        info.filename = "directory/"
        info.external_attr = stat.S_IFDIR << 16
    elif kind == "valid-extra":
        info.extra = struct.pack("<HHBI", 0x5455, 5, 1, 0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, fixture)
        if kind == "lying-count":
            archive.writestr("directory/", b"")
    payload = bytearray(buffer.getvalue())
    end = payload.rfind(b"PK\x05\x06")
    if kind == "zip64-count":
        struct.pack_into("<HH", payload, end + 8, 65535, 65535)
    elif kind == "zip64-size":
        struct.pack_into("<I", payload, end + 12, 0xFFFFFFFF)
    elif kind == "lying-count":
        struct.pack_into("<HH", payload, end + 8, 0, 0)
    members = ({"name": "fixture.json", "sha256": digest(fixture), "size_bytes": len(fixture)},)
    spec, _ = recipe(bytes(payload), archive="zip", members=members)
    if kind == "lying-count":
        spec = changed(spec, limits=changed(spec.limits, maximum_archive_entries=1))
    cache = SourceCache(tmp_path)
    if kind == "valid-extra":
        receipt = materialize_source(spec, cache, lambda: reader(bytes(payload)))
        validate_materialization(receipt, cache)
    else:
        with pytest.raises(ValueError, match="ZIP"):
            materialize_source(spec, cache, lambda: reader(bytes(payload)))
