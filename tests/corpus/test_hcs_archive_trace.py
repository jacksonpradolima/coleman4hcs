"""Owned original-archive profiles, byte membership and complete-last source corroboration."""

import hashlib
import io
import json
import shutil
import stat
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import ArchiveSelectionSpec, SelectedPackage
from coleman.corpus.hcs_archive_trace import (
    HCSArchiveLogSpec,
    HCSArchiveTraceReceipt,
    load_hcs_archive_log_spec,
    publish_hcs_archive_trace,
    read_hcs_archive_log,
    validate_hcs_archive_trace,
)
from coleman.corpus.materialization import materialize_source
from coleman.corpus.materialization_contracts import MaterializationReceipt, PackageMember
from coleman.corpus.zip64_contracts import Zip64Member
from tests.corpus.hcs_support import prepare_hcs
from tests.corpus.test_hcs_trace import OWNED, changed
from tests.corpus.test_source_views import source_spec

NAME = "owned@logs/101_" + "a" * 40 + "_1.log"


def archive_bytes(entries, *, compression=zipfile.ZIP_STORED, mode=stat.S_IFREG | 0o644, force_zip64=False):
    """Write deterministic authored file records; no foreign payload or filesystem extraction is involved."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, payload in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = mode << 16
            info.compress_type = compression
            with archive.open(info, "w", force_zip64=force_zip64) as stream:
                stream.write(payload)
    return output.getvalue()


def owned_archive(tmp_path, *, entries=None, raw=None, **kwargs):
    """Use the real cache/source reader and a byte-pinned original-member spec."""
    recipe, cache = prepare_hcs(tmp_path)
    payload = archive_bytes([(NAME, OWNED)] if entries is None else entries, **kwargs) if raw is None else raw
    package = source_spec("owned-logs.zip", payload)
    materialize_source(package, cache, lambda: io.BytesIO(payload))
    spec = HCSArchiveLogSpec(
        schema_version=1,
        job_id=1,
        container=package,
        member=Zip64Member(name=NAME, sha256="sha256:" + hashlib.sha256(OWNED).hexdigest(), size_bytes=len(OWNED)),
    )
    return recipe, cache, spec, payload


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_whole_archive_and_original_metadata_bind_a_license_verified_complete_receipt(tmp_path, compression):
    recipe, cache, spec, _ = owned_archive(tmp_path, compression=compression)
    assert read_hcs_archive_log(spec, cache) == OWNED
    writer = ArtifactWriter(tmp_path / "output")
    path = publish_hcs_archive_trace(recipe, spec, cache, writer)
    receipt = validate_hcs_archive_trace(path, recipe, cache)
    assert HCSArchiveTraceReceipt.from_json(receipt.to_json()) == receipt
    assert receipt.corroboration.log_spec.terms == spec.container.license
    assert receipt.corroboration.log_spec.source.uri == spec.container.source.uri + "#" + NAME
    assert receipt.corroboration.files[0].rows == 3
    assert receipt.admission == "not_granted"
    assert receipt.corroboration.admission == "not_granted"
    assert not list(writer.run_root.rglob("*.log"))
    before = {p: p.read_bytes() for p in writer.run_root.rglob("*") if p.is_file()}
    assert publish_hcs_archive_trace(recipe, spec, cache, writer) == path
    assert {p: p.read_bytes() for p in writer.run_root.rglob("*") if p.is_file()} == before


def test_explicit_larger_profile_does_not_relax_old_selected_archive_limits(tmp_path):
    entries = [(NAME, OWNED), *((f"unselected/{i}.log", b"") for i in range(4100))]
    _, cache, spec, _ = owned_archive(tmp_path, entries=entries)
    assert read_hcs_archive_log(spec, cache) == OWNED
    child = source_spec("unselected.log", b"")
    selected = SelectedPackage(
        member=PackageMember(name="unselected/0.log", sha256=child.source.sha256, size_bytes=0),
        package=child,
    )
    with pytest.raises(ValueError, match="maximum_entries"):
        ArchiveSelectionSpec(schema_version=1, source=spec.container, selections=(selected,), maximum_entries=4101)


def test_versioned_owned_source_golden_and_original_public_preset_remain_explicit(tmp_path):
    _, _, spec, _ = owned_archive(tmp_path)
    golden = Path(__file__).with_name("fixtures") / "hcs-archive-log-owned-v1.json"
    assert spec.to_json() + "\n" == golden.read_text()
    assert HCSArchiveLogSpec.from_json(golden.read_bytes()) == spec
    public = load_hcs_archive_log_spec()
    assert public.job_id == 63565028
    assert public.container.source.size_bytes == 97363308
    assert public.container.license.license_id == "CC0-1.0"
    assert public.member.size_bytes == 32780


@pytest.mark.parametrize("kind", ["digest", "size", "absent", "job", "pending_terms", "future"])
def test_changed_member_metadata_and_unreviewed_or_unknown_contracts_reject(tmp_path, kind):
    recipe, cache, spec, _ = owned_archive(tmp_path)
    if kind == "pending_terms":
        terms = changed(spec.container.license, status="pending")
        with pytest.raises(ValueError, match="verified data terms"):
            changed(spec.container, license=terms)
        return
    if kind == "future":
        wire = json.loads(spec.to_json())
        wire["schema_version"] = 2
        encoded = json.dumps(wire)
        with pytest.raises(ValueError):
            HCSArchiveLogSpec.from_json(encoded)
        return
    if kind == "digest":
        spec = changed(spec, member=changed(spec.member, sha256="sha256:" + "0" * 64))
    elif kind == "size":
        spec = changed(spec, member=changed(spec.member, size_bytes=len(OWNED) - 1))
    elif kind == "absent":
        spec = changed(spec, member=changed(spec.member, name="absent.log"))
    else:
        spec = changed(spec, job_id=2)
    if kind != "job":
        with pytest.raises(ValueError):
            read_hcs_archive_log(spec, cache)
    root = tmp_path / "output"
    writer = ArtifactWriter(root)
    with pytest.raises(ValueError):
        publish_hcs_archive_trace(recipe, spec, cache, writer)
    assert not list(root.rglob("complete.json"))


@pytest.mark.parametrize("kind", ["traversal", "symlink", "alias", "local_crc", "zip64"])
def test_re_pinned_malicious_container_still_requires_real_structure_and_safe_unselected_headers(tmp_path, kind):
    kwargs = {}
    entries = [(NAME, OWNED)]
    if kind == "traversal":
        entries.append(("../escape", b"owned"))
    elif kind == "symlink":
        kwargs["mode"] = stat.S_IFLNK | 0o777
    elif kind == "alias":
        entries.append((NAME, OWNED))
    elif kind == "zip64":
        kwargs["force_zip64"] = True
    if kind == "alias":
        with pytest.warns(UserWarning, match="Duplicate name"):
            payload = archive_bytes(entries)
    else:
        payload = archive_bytes(entries, **kwargs)
    if kind == "local_crc":
        mutable = bytearray(payload)
        struct.pack_into("<I", mutable, 14, 1)
        payload = bytes(mutable)
    _, cache, spec, _ = owned_archive(tmp_path, raw=payload)
    with pytest.raises(ValueError):
        read_hcs_archive_log(spec, cache)
    assert not (tmp_path / "escape").exists()


def test_bounded_central_metadata_is_rejected_before_standard_reader_allocation(tmp_path, monkeypatch):
    entries = [(NAME, OWNED), *((f"unselected/{i}.log", b"") for i in range(16384))]
    _, cache, spec, _ = owned_archive(tmp_path, entries=entries)

    def forbidden(*args, **kwargs):
        raise AssertionError("Standard ZIP reader allocated metadata before the entry bound")

    monkeypatch.setattr("coleman.corpus.hcs_archive_trace.zipfile.ZipFile", forbidden)
    with pytest.raises(ValueError, match="metadata"):
        read_hcs_archive_log(spec, cache)


def test_source_corruption_and_copied_or_forged_wrapper_never_validate(tmp_path):
    recipe, cache, spec, _ = owned_archive(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    path = publish_hcs_archive_trace(recipe, spec, cache, writer)
    copied = tmp_path / "copy" / "hcs-archive-trace" / ("0" * 64) / "complete.json"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(path, copied)
    with pytest.raises(ValueError, match="namespace"):
        validate_hcs_archive_trace(copied, recipe, cache)
    wire = json.loads(path.read_bytes())
    wire["producer_sha256"] = "sha256:" + "0" * 64
    original = path.read_bytes()
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match="producer"):
        validate_hcs_archive_trace(path, recipe, cache)
    path.write_bytes(original)
    blob = cache.member_path(MaterializationReceipt(schema_version=1, spec=spec.container), "owned-logs.zip")
    blob.write_bytes(b"corrupt original archive")
    with pytest.raises(ValueError):
        validate_hcs_archive_trace(path, recipe, cache)


def test_interrupted_archive_wrapper_retains_child_and_concurrent_retry_converges(tmp_path):
    recipe, cache, spec, _ = owned_archive(tmp_path)

    class Interrupted(ArtifactWriter):
        """Stop after the direct-log child finishes, before its enclosing-archive receipt is published."""

        def write_immutable(self, path, content):
            if path.parent.parent.name == "hcs-archive-trace":
                raise OSError("owned archive-wrapper interruption")
            return super().write_immutable(path, content)

    root = tmp_path / "output"
    interrupted = Interrupted(root)
    with pytest.raises(OSError, match="archive-wrapper"):
        publish_hcs_archive_trace(recipe, spec, cache, interrupted)
    child = next((root / "hcs-trace").rglob("complete.json"))
    before = child.read_bytes()
    assert not list((root / "hcs-archive-trace").rglob("complete.json"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: publish_hcs_archive_trace(recipe, spec, cache, ArtifactWriter(root)), range(3)))
    assert len(set(paths)) == 1
    assert child.read_bytes() == before
    validate_hcs_archive_trace(paths[0], recipe, cache)


@pytest.mark.parametrize("kind", ["selected_archive", "fragment", "oversized_log"])
def test_archive_profile_rejects_wrong_container_and_member_bounds_before_io(tmp_path, kind):
    _, _, spec, _ = owned_archive(tmp_path)
    wire = json.loads(spec.to_json())
    if kind == "selected_archive":
        wire["container"]["archive"] = "zip"
    elif kind == "fragment":
        wire["container"]["source"]["uri"] += "#selected.log"
    else:
        wire["member"]["size_bytes"] = 2 * 1024**2 + 1
    encoded = json.dumps(wire)
    with pytest.raises(ValueError):
        HCSArchiveLogSpec.from_json(encoded)


@pytest.mark.parametrize("kind", ["symlink", "oversized", "future", "changed_child"])
def test_wrapper_requires_bounded_current_metadata_and_the_actual_unchanged_child(tmp_path, kind):
    recipe, cache, spec, _ = owned_archive(tmp_path)
    path = publish_hcs_archive_trace(recipe, spec, cache, ArtifactWriter(tmp_path / "output"))
    original = path.read_bytes()
    if kind == "symlink":
        target = tmp_path / "original-receipt.json"
        path.rename(target)
        path.symlink_to(target)
    elif kind == "oversized":
        path.write_bytes(b" " * (2 * 1024**2 + 1))
    else:
        wire = json.loads(original)
        if kind == "future":
            wire["schema_version"] = 2
        else:
            wire["corroboration"]["log_spec"]["source"]["citation"] += " changed"
        path.write_text(json.dumps(wire))
    with pytest.raises(ValueError):
        validate_hcs_archive_trace(path, recipe, cache)
