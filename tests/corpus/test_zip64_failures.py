"""ZIP64 adversarial metadata, exact byte identity and interrupted-cache recovery."""

import io
import struct
import zipfile

import pytest

from coleman.corpus.materialization import SourceCache, materialize_source
from coleman.corpus.zip64_contracts import Zip64Member, Zip64SelectionSpec
from coleman.corpus.zip64_records import _sizes
from coleman.corpus.zip64_selection import materialize_zip64_files, validate_zip64_files
from tests.corpus.test_source_views import digest, source_spec
from tests.corpus.test_zip64_selection import mutated, owned_zip64, recipe


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True, 1.5])
def test_byte_count_never_conflates_null_negative_or_nonfinite_with_zero(value):
    with pytest.raises(ValueError):
        Zip64Member(name="empty.csv", sha256=digest(b""), size_bytes=value)


def test_real_empty_selected_file_roundtrip(tmp_path):
    raw = owned_zip64((("owner@project/results.csv", b""),))
    spec = recipe(raw, b"")
    cache = SourceCache(tmp_path / "cache")
    (receipt,) = materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    assert cache.member_path(receipt, "results.csv").read_bytes() == b""
    validate_zip64_files(spec, cache)


@pytest.mark.parametrize("extra", [b"x", b"\x01\x00\x10\x00", struct.pack("<HHQ", 1, 8, 42) * 2])
def test_malformed_or_duplicate_zip64_extra_fields_reject(extra):
    with pytest.raises(ValueError):
        _sizes(extra, (0xFFFFFFFF,))


def test_only_sentinel_required_values_are_resolved_in_order():
    for sentinels in ((True, False, False), (False, False, True), (True, True, True)):
        values = tuple(0xFFFFFFFF if wide else 7 for wide in sentinels)
        expected = tuple(2**32 + index if wide else 7 for index, wide in enumerate(sentinels))
        resolved = tuple(expected[index] for index, wide in enumerate(sentinels) if wide)
        extra = struct.pack("<HH" + "Q" * len(resolved), 1, 8 * len(resolved), *resolved)
        assert _sizes(extra, values) == expected
        with pytest.raises(ValueError):
            _sizes(extra + b"\x01\x00\x00\x00", values)


@pytest.mark.parametrize("descriptor", [False, True])
def test_zip64_container_accepts_legacy_local_data_headers(tmp_path, descriptor):
    payload = owned_zip64(descriptor=descriptor, force_zip64=False)
    spec = recipe(payload)
    cache = SourceCache(tmp_path / "cache")
    assert materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload))
    validate_zip64_files(spec, cache)


def test_stored_entry_retains_inert_producer_preference_bit(tmp_path):
    raw = owned_zip64(compression=zipfile.ZIP_STORED)
    raw = mutated(raw, 6, "<H", 2)
    raw = mutated(raw, raw.index(b"PK\x01\x02") + 8, "<H", 2)
    spec = recipe(raw)
    cache = SourceCache(tmp_path / "cache")
    assert materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    validate_zip64_files(spec, cache)


@pytest.mark.parametrize("flags", [1, 4, 6, 16, 64, 8192])
def test_other_stored_flags_remain_unsupported(tmp_path, flags):
    raw = owned_zip64(compression=zipfile.ZIP_STORED)
    raw = mutated(raw, 6, "<H", flags)
    raw = mutated(raw, raw.index(b"PK\x01\x02") + 8, "<H", flags)
    with pytest.raises(ValueError, match="flags"):
        materialize_zip64_files(recipe(raw), SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


def test_legacy_materializer_still_rejects_stored_preference_bit(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("results.csv", b"owned")
    raw = stream.getvalue()
    raw = mutated(raw, 6, "<H", 2)
    raw = mutated(raw, raw.index(b"PK\x01\x02") + 8, "<H", 2)
    spec = source_spec(
        "legacy-stored", raw, "zip", [{"name": "results.csv", "sha256": digest(b"owned"), "size_bytes": 5}]
    )
    with pytest.raises(ValueError, match="unsupported member metadata"):
        materialize_source(spec, SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


@pytest.mark.parametrize("field", [20, 24])
def test_only_offset_sentinels_can_retain_legacy_central_versions(tmp_path, field):
    raw = owned_zip64()
    central = raw.index(b"PK\x01\x02")
    raw = mutated(raw, central + 6, "<H", 20)
    raw = mutated(raw, central + field, "<I", 0xFFFFFFFF)
    with pytest.raises(ValueError, match="size sentinel"):
        materialize_zip64_files(recipe(raw), SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


@pytest.mark.parametrize(
    "relative,offset,fmt,value",
    [
        ("local", 4, "<H", 46),
        ("local", 6, "<H", 1),
        ("local", 8, "<H", 0),
        ("local", 14, "<I", 0),
        ("local", 26, "<H", 65535),
        ("local", 28, "<H", 65535),
        ("local", 30, "<B", ord("X")),
        ("central", 6, "<H", 46),
        ("central", 8, "<H", 1),
        ("central", 10, "<H", 99),
        ("central", 28, "<H", 1025),
        ("central", 30, "<H", 65535),
        ("central", 34, "<H", 1),
        ("central", 38, "<I", 0o120777 << 16),
        ("central", 42, "<I", 1),
    ],
)
def test_contradictory_headers_never_publish_children(tmp_path, relative, offset, fmt, value):
    raw = owned_zip64()
    start = 0 if relative == "local" else raw.index(b"PK\x01\x02")
    raw = mutated(raw, start + offset, fmt, value)
    spec = recipe(raw)
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    assert len(list((cache.root / "blobs").iterdir())) == 1


@pytest.mark.parametrize("offset,value", [(0, 2), (2, 15), (4, 1), (12, 1)])
def test_local_zip64_extra_metadata_is_bound_to_central_sizes(tmp_path, offset, value):
    raw = owned_zip64()
    start = 30 + struct.unpack_from("<H", raw, 26)[0]
    raw = mutated(raw, start + offset, "<H" if offset < 4 else "<Q", value)
    spec = recipe(raw)
    with pytest.raises(ValueError):
        materialize_zip64_files(spec, SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


def test_descriptor_crc_disagreement_rejects(tmp_path):
    raw = owned_zip64(descriptor=True)
    raw = mutated(raw, raw.index(b"PK\x07\x08") + 4, "<I", 0)
    with pytest.raises(ValueError, match="descriptor"):
        materialize_zip64_files(recipe(raw), SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


@pytest.mark.parametrize("force_zip64", [False, True])
def test_unsigned_descriptor_is_accepted_in_both_widths(tmp_path, force_zip64):
    raw = owned_zip64(descriptor=True, force_zip64=force_zip64)
    signature = raw.index(b"PK\x07\x08")
    raw = raw[:signature] + raw[signature + 4 :]
    for offset in (-50, -34):
        original = struct.unpack_from("<Q", raw, len(raw) + offset)[0]
        raw = mutated(raw, len(raw) + offset, "<Q", original - 4)
    spec = recipe(raw)
    cache = SourceCache(tmp_path / "cache")
    assert materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    validate_zip64_files(spec, cache)


def test_local_zip64_extra_requires_both_size_fields(tmp_path):
    raw = owned_zip64()
    extra = 30 + struct.unpack_from("<H", raw, 26)[0]
    compressed = struct.unpack_from("<Q", raw, extra + 12)[0]
    raw = mutated(raw, 18, "<I", compressed)
    raw = mutated(raw, 28, "<H", 12)
    raw = mutated(raw, extra + 2, "<H", 8)
    raw = raw[: extra + 12] + raw[extra + 20 :]
    for offset in (-50, -34):
        original = struct.unpack_from("<Q", raw, len(raw) + offset)[0]
        raw = mutated(raw, len(raw) + offset, "<Q", original - 8)
    with pytest.raises(ValueError, match="both"):
        materialize_zip64_files(recipe(raw), SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


def test_offset_only_central_version_does_not_allow_deflate_with_local_version10(tmp_path):
    raw = owned_zip64(force_zip64=False)
    raw = mutated(raw, 4, "<H", 10)
    raw = mutated(raw, raw.index(b"PK\x01\x02") + 6, "<H", 45)
    with pytest.raises(ValueError, match="version"):
        materialize_zip64_files(recipe(raw), SourceCache(tmp_path / "cache"), lambda: io.BytesIO(raw))


def test_source_digest_failure_precedes_any_structural_inspection(tmp_path):
    raw = owned_zip64()
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_zip64_files(recipe(raw), cache, lambda: io.BytesIO(b"X" + raw[1:]))
    assert not list((cache.root / "blobs").iterdir())


def test_selected_digest_is_independent_of_container_crc(tmp_path):
    raw = owned_zip64()
    spec = recipe(raw, b"other CI records\n")
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    assert len(list((cache.root / "blobs").iterdir())) == 1


def two_file_recipe(raw):
    value = recipe(raw).model_dump(mode="json")
    child = source_spec("second.csv", b"second\n")
    value["selections"].append(
        {
            "member": {"name": "z@project/second.csv", "sha256": digest(b"second\n"), "size_bytes": 7},
            "package": child,
        }
    )
    return Zip64SelectionSpec.model_validate(value)


def test_interrupted_source_has_no_false_completion_and_retry_converges(tmp_path):
    raw = owned_zip64()
    spec = recipe(raw)
    cache = SourceCache(tmp_path / "cache")

    class Interrupted(io.BytesIO):
        def read(self, maximum=-1):
            if self.tell():
                raise OSError("Owned interrupted source")
            return super().read(12)

    with pytest.raises(OSError, match="interrupted"):
        materialize_zip64_files(spec, cache, lambda: Interrupted(raw))
    assert not list((cache.root / "blobs").iterdir())
    assert materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    validate_zip64_files(spec, cache)


def test_interrupted_child_retains_only_verified_blobs_and_retry_converges(tmp_path, monkeypatch):
    raw = owned_zip64((("owner@project/results.csv", b"owned CI records\n"), ("z@project/second.csv", b"second\n")))
    spec = two_file_recipe(raw)
    cache = SourceCache(tmp_path / "cache")
    original = SourceCache.store_blob

    def interrupted(self, fingerprint, size, reader):
        if fingerprint == digest(b"second\n"):
            raise OSError("Owned interrupted child publication")
        return original(self, fingerprint, size, reader)

    with monkeypatch.context() as patch:
        patch.setattr(SourceCache, "store_blob", interrupted)
        with pytest.raises(OSError, match="interrupted child"):
            materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    assert len(list((cache.root / "blobs").iterdir())) == 2
    receipts = materialize_zip64_files(spec, cache)
    assert len(receipts) == 2
    validate_zip64_files(spec, cache)
    assert len(list((cache.root / "blobs").iterdir())) == 3


def test_selection_order_is_canonical_but_container_identity_is_not_erased(tmp_path):
    entries = (("owner@project/results.csv", b"owned CI records\n"), ("z@project/second.csv", b"second\n"))
    raw = owned_zip64(entries)
    spec = two_file_recipe(raw)
    value = spec.model_dump(mode="json")
    value["selections"].reverse()
    assert Zip64SelectionSpec.model_validate(value).to_json() == spec.to_json()
    other_raw = owned_zip64(tuple(reversed(entries)))
    other = two_file_recipe(other_raw)
    assert other.fingerprint != spec.fingerprint
    cache = SourceCache(tmp_path / "cache")
    assert materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw)) == materialize_zip64_files(
        other, cache, lambda: io.BytesIO(other_raw)
    )
