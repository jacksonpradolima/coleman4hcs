"""Explicit single-disk ZIP64 file selection through the existing verified source cache."""

from __future__ import annotations

import zipfile
import zlib
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from coleman.corpus.archives import _validate_deflate
from coleman.corpus.materialization import ReaderFactory, SourceCache, _verified_chunks, materialize_source
from coleman.corpus.materialization_contracts import ByteReader, MaterializationReceipt
from coleman.corpus.zip64_contracts import Zip64Member, Zip64SelectionSpec
from coleman.corpus.zip64_records import Zip64Record, read_zip64_records, zip64_data_offset
from coleman.corpus.zip_metadata import entry_boundaries


def _bind_metadata(infos: tuple[zipfile.ZipInfo, ...], records: tuple[Zip64Record, ...]) -> None:
    """Require standard-reader metadata to exactly match the independently bounded central records."""
    if len(infos) != len(records):
        raise ValueError("ZIP64 standard-reader entry count disagrees")
    for info, record in zip(infos, records, strict=True):
        actual = (
            info.filename,
            info.extract_version | (info.reserved << 8),
            info.flag_bits,
            info.compress_type,
            info.CRC,
            info.compress_size,
            info.file_size,
            info.header_offset,
        )
        if info.orig_filename != info.filename or actual != record:
            raise ValueError("ZIP64 standard-reader metadata disagrees with actual central bytes")


def consume_zip64_files(
    path: Path,
    spec: Zip64SelectionSpec,
    consume: Callable[[Zip64Member, ByteReader], None],
) -> None:
    """Check all central/local headers and decode selected files through a caller-owned full checksum consumer.

    Callers MUST verify the complete enclosing source digest first. This lower
    reader itself never publishes or certifies completion. The callback must
    consume and checksum the entire selected stream before publishing its bytes;
    materialize_zip64_files/validate_zip64_files enforce that protocol.
    Metadata is bounded before ZipFile allocation. At most64MiB selected decoded
    bytes are allowed; unselected payloads are neither decoded nor certified.
    Stored/deflated single-disk forms only; no filesystem extraction or network.
    Format/CRC/deflate errors raise ValueError, I/O propagates, all handles close.
    Trusted immutable local files/parents are required, not a hostile-FS sandbox.
    """
    spec = Zip64SelectionSpec.model_validate(spec)
    directory = read_zip64_records(
        path, maximum_entries=spec.maximum_entries, maximum_metadata_bytes=spec.maximum_metadata_bytes
    )
    expected = {item.member.name: item.member for item in spec.selections}
    records = {record.name: record for record in directory.records}
    if expected.keys() - records.keys() or any(
        records[name].decoded != member.size_bytes for name, member in expected.items()
    ):
        raise ValueError("ZIP64 selected member names/sizes disagree with frozen expectation")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = tuple(archive.infolist())
            _bind_metadata(infos, directory.records)
            boundaries = entry_boundaries(infos, directory.offset)
            if min(boundaries) != 0:
                raise ValueError("ZIP64 prefixed/self-extracting archives are unsupported")
            offsets = {
                record.name: zip64_data_offset(path, record, boundaries[record.offset]) for record in directory.records
            }
            by_name = {info.filename: info for info in infos}
            for name, member in sorted(expected.items()):
                info = by_name[name]
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    _validate_deflate(path, info, offsets[name])
                with archive.open(info) as stream:
                    consume(member, stream)
    except (zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ValueError("Invalid or unsupported selected ZIP64 source") from exc


def materialize_zip64_files(
    spec: Zip64SelectionSpec, cache: SourceCache, source: ReaderFactory | None = None
) -> tuple[MaterializationReceipt, ...]:
    """Acquire/reverify a full pinned ZIP64 source and its selected raw files; offline by default.

    Reuses the existing raw package materializer, cache and immutable writes.
    The complete source digest is verified before structural inspection or child
    publication. Existing children are still verified against actual membership.
    Partial failures may retain individually verified cache blobs but never return
    a completed selection. Only after every selection succeeds are all receipts
    returned, in canonical member order. No learning checkpoint or group-complete
    file is created; callers persist the full selection recipe with these receipts.
    Callback/validation/I/O errors propagate. No URL is opened implicitly.
    """
    spec = Zip64SelectionSpec.model_validate(spec)
    parent = materialize_source(spec.source, cache, source)
    path = cache.member_path(parent, spec.source.members[0].name)
    packages = {item.member.name: item.package for item in spec.selections}
    receipts: dict[str, MaterializationReceipt] = {}

    def consume(member: Zip64Member, stream: ByteReader) -> None:
        """Force the bounded actual member stream through the existing raw-file hash/size/cache verifier."""

        @contextmanager
        def reader():
            """Borrow the archive-owned stream; the enclosing archive context closes it on all paths."""
            yield stream

        receipts[member.name] = materialize_source(packages[member.name], cache, reader, refresh_source=True)

    consume_zip64_files(path, spec, consume)
    return tuple(receipts[item.member.name] for item in spec.selections)


def validate_zip64_files(spec: Zip64SelectionSpec, cache: SourceCache) -> None:
    """Reverify full source/member membership and existing child bytes without network, repair or admission.

    Reads may scan the entire large source to verify its SHA-256. Individual child
    receipts alone do not establish enclosing membership; retain the full recipe.
    Missing/corrupt cache bytes, malformed archives or mismatched members reject.
    Unselected payloads remain uncertified; no subject code or transport is loaded.
    """
    spec = Zip64SelectionSpec.model_validate(spec)
    path = cache.verify_blob(cast(str, spec.source.source.sha256), cast(int, spec.source.source.size_bytes))

    def verify(member: Zip64Member, stream: ByteReader) -> None:
        """Verify actual selected bytes and their independent cached copy, without recreating missing data."""
        deque(_verified_chunks(stream, member.sha256, member.size_bytes), maxlen=0)
        cache.verify_blob(member.sha256, member.size_bytes)

    consume_zip64_files(path, spec, verify)
