"""Owned ZIP64 framing fixtures; no test downloads external archives or executes subject code."""

import io
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coleman.corpus.materialization import SourceCache
from coleman.corpus.materialization_contracts import PackageMember
from coleman.corpus.zip64_contracts import Zip64Member, Zip64SelectionSpec
from coleman.corpus.zip64_selection import consume_zip64_files, materialize_zip64_files, validate_zip64_files
from tests.corpus.test_source_views import changed, digest, source_spec


def owned_zip64(
    entries=(("owner@project/results.csv", b"owned CI records\n"),),
    *,
    descriptor=False,
    compression=zipfile.ZIP_DEFLATED,
    force_zip64=True,
):
    """Produce actual ZIP64 local sizes and end records with tiny deterministic owned data."""

    class Unseekable(io.BytesIO):
        def seekable(self):
            return False

        def seek(self, *args):
            raise io.UnsupportedOperation("Owned streaming fixture")

    stream = Unseekable() if descriptor else io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            with archive.open(info, "w", force_zip64=force_zip64) as member:
                member.write(payload)
    raw = stream.getvalue()
    footer = struct.unpack("<4s4H2IH", raw[-22:])
    directory_size, directory_offset = footer[5:7]
    record_offset = len(raw) - 22
    end64 = struct.pack(
        "<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, len(entries), len(entries), directory_size, directory_offset
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, record_offset, 1)
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return raw[:-22] + end64 + locator + end


def recipe(payload, content=b"owned CI records\n"):
    source = source_spec("container.zip", payload)
    child = source_spec("results.csv", content)
    return Zip64SelectionSpec.model_validate(
        {
            "schema_version": 1,
            "source": source,
            "selections": [
                {
                    "member": {
                        "name": "owner@project/results.csv",
                        "sha256": digest(content),
                        "size_bytes": len(content),
                    },
                    "package": child,
                }
            ],
        }
    )


def test_actual_zip64_materialization_and_offline_validation(tmp_path):
    payload = owned_zip64()
    spec = recipe(payload)
    cache = SourceCache(tmp_path / "cache")
    receipts = materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload))
    assert len(receipts) == 1
    assert cache.member_path(receipts[0], "results.csv").read_bytes() == b"owned CI records\n"
    validate_zip64_files(spec, cache)
    assert materialize_zip64_files(spec, cache) == receipts


@pytest.mark.parametrize("descriptor", [False, True])
@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_both_methods_and_descriptor_widths_are_real(tmp_path, descriptor, compression):
    payload = owned_zip64(descriptor=descriptor, compression=compression)
    spec = recipe(payload)
    cache = SourceCache(tmp_path / "cache")
    results = materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload))
    validate_zip64_files(spec, cache)
    assert results[0].spec == spec.selections[0].package


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "x/../escape",
        "x//y",
        "x/./y",
        "x\\y",
        "x./y",
        "x /y",
        "x/ y",
        "x/",
        "",
        "x:drive",
        "x/é",
    ],
)
def test_new_name_grammar_rejects_unsafe_portable_names(name):
    with pytest.raises(ValueError):
        Zip64Member(name=name, sha256=digest(b""), size_bytes=0)


def test_new_grammar_never_changes_old_package_contract():
    with pytest.raises(ValueError):
        PackageMember(name="owner@project/results.csv", sha256=digest(b""), size_bytes=0)
    assert Zip64Member(name="owner@project/results.csv", sha256=digest(b""), size_bytes=0).size_bytes == 0


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_unknown_recipe_versions_fail_closed(version):
    value = recipe(owned_zip64()).model_dump(mode="json")
    if version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = version
    with pytest.raises(ValueError):
        Zip64SelectionSpec.model_validate(value)


def test_recipe_roundtrip_and_versioned_owned_golden():
    spec = recipe(owned_zip64())
    assert Zip64SelectionSpec.from_json(spec.to_json()) == spec
    fixture = Path(__file__).with_name("fixtures") / "zip64-selection-owned-v1.json"
    assert fixture.read_text() == spec.to_json() + "\n"


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_entries", True),
        ("maximum_entries", 4097),
        ("maximum_entries", 0),
        ("maximum_metadata_bytes", 21),
        ("maximum_metadata_bytes", 2 * 1024**2 + 1),
    ],
)
def test_declared_limits_reject_before_io(field, value):
    spec = recipe(owned_zip64())
    with pytest.raises(ValueError):
        changed(spec, **{field: value})


@pytest.mark.parametrize(
    "case", ["duplicate", "source_zip", "child_zip", "child_digest", "terms", "version", "byte_total", "entry_budget"]
)
def test_binding_contradictions_cannot_cross_the_domain(case):
    spec = recipe(owned_zip64())
    value = spec.model_dump(mode="json")
    if case == "duplicate":
        value["selections"].append(value["selections"][0])
    elif case == "source_zip":
        value["source"]["archive"] = "zip"
    elif case == "child_zip":
        value["selections"][0]["package"]["archive"] = "zip"
    elif case == "child_digest":
        value["selections"][0]["member"]["sha256"] = digest(b"other")
    elif case == "terms":
        value["selections"][0]["package"]["license"]["reviewed_at"] = "2026-01-01T00:00:00Z"
    elif case == "version":
        value["selections"][0]["package"]["source"]["version"] = "different"
    elif case == "entry_budget":
        value["selections"].append(
            {**value["selections"][0], "member": {**value["selections"][0]["member"], "name": "other.csv"}}
        )
        value["maximum_entries"] = 1
    else:
        for name in ("a.csv", "b.csv"):
            child = source_spec(name, b"x").model_dump(mode="json")
            child["source"]["size_bytes"] = 40 * 1024**2
            child["members"][0]["size_bytes"] = 40 * 1024**2
            value["selections"].append(
                {
                    "member": {"name": name, "sha256": child["source"]["sha256"], "size_bytes": 40 * 1024**2},
                    "package": child,
                }
            )
    with pytest.raises(ValueError):
        Zip64SelectionSpec.model_validate(value)


def mutated(payload, offset, fmt, value):
    out = bytearray(payload)
    struct.pack_into(fmt, out, offset, value)
    return bytes(out)


@pytest.mark.parametrize(
    "offset,fmt,value",
    [
        (-94, "<Q", 45),
        (-84, "<H", 46),
        (-82, "<I", 1),
        (-78, "<I", 1),
        (-74, "<Q", 2),
        (-66, "<Q", 4097),
        (-58, "<Q", 2 * 1024**2 + 1),
        (-50, "<Q", 1),
        (-38, "<I", 1),
        (-34, "<Q", 0),
        (-26, "<I", 2),
        (-18, "<H", 1),
        (-16, "<H", 1),
        (-14, "<H", 2),
        (-10, "<I", 1),
        (-6, "<I", 1),
    ],
)
def test_invalid_end_records_reject_before_standard_directory_allocation(tmp_path, monkeypatch, offset, fmt, value):
    payload = owned_zip64()
    payload = mutated(payload, len(payload) + offset, fmt, value)
    spec = recipe(payload)
    path = tmp_path / "invalid.zip"
    path.write_bytes(payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("Standard reader allocated before metadata gate")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError):
        consume_zip64_files(path, spec, lambda member, stream: stream.read())


@pytest.mark.parametrize("case", ["truncated", "trailing", "missing_locator", "wrong_record_signature", "nonzip64"])
def test_unsupported_or_incomplete_footers_reject(tmp_path, case):
    raw = owned_zip64()
    if case == "truncated":
        raw = raw[:-8]
    elif case == "trailing":
        raw += b"unclaimed"
    elif case == "missing_locator":
        raw = raw[:-42] + b"BAD!" + raw[-38:]
    elif case == "wrong_record_signature":
        raw = raw[:-98] + b"BAD!" + raw[-94:]
    else:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("owner@project/results.csv", b"owned CI records\n")
        raw = stream.getvalue()
    spec = recipe(raw)
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_zip64_files(spec, cache, lambda: io.BytesIO(raw))
    child = spec.selections[0].member
    with pytest.raises(FileNotFoundError):
        cache.verify_blob(child.sha256, child.size_bytes)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "disk\\path", "foo/../bar"])
def test_unselected_unsafe_names_reject_before_any_child_publication(tmp_path, name):
    payload = owned_zip64((("owner@project/results.csv", b"owned CI records\n"), (name, b"unselected")))
    spec = recipe(payload)
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload))
    with pytest.raises(FileNotFoundError):
        cache.verify_blob(spec.selections[0].member.sha256, spec.selections[0].member.size_bytes)


def test_missing_cached_child_rejects_without_repair(tmp_path):
    payload = owned_zip64()
    spec = recipe(payload)
    cache = SourceCache(tmp_path / "cache")
    materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload))
    member = spec.selections[0].member
    path = cache.verify_blob(member.sha256, member.size_bytes)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_zip64_files(spec, cache)
    assert not path.exists()


def test_reproducible_concurrent_replay_returns_the_existing_raw_receipts(tmp_path):
    payload = owned_zip64()
    spec = recipe(payload)
    cache = SourceCache(tmp_path / "cache")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload)), range(3)))
    assert results[0] == results[1] == results[2]
    assert len(list((cache.root / "blobs").iterdir())) == 2
    assert Zip64SelectionSpec.from_json(spec.to_json()).fingerprint == spec.fingerprint


def test_legacy_v1_recipes_still_reject_actual_zip64(tmp_path):
    payload = owned_zip64((("results.csv", b"owned"),))
    old = source_spec("legacy", payload, "zip", [{"name": "results.csv", "sha256": digest(b"owned"), "size_bytes": 5}])
    from coleman.corpus.materialization import materialize_source

    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="ZIP64"):
        materialize_source(old, cache, lambda: io.BytesIO(payload))
