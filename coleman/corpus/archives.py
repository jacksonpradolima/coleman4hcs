"""Strict bounded ZIP inspection; logical member names never become output paths."""

from __future__ import annotations

import io
import stat
import struct
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path

from coleman.corpus.materialization_contracts import ByteReader, MaterializationSpec, PackageMember, safe_member_name
from coleman.corpus.zip_metadata import entry_boundaries, local_data_offset, validate_directory, validate_extra_records

_UTF8_DESCRIPTOR_FLAGS = 0x0808
_DEFLATE_LEVEL_FLAGS = 0x0006


def deterministic_zip_bytes(
    entries: tuple[tuple[str, bytes], ...], *, maximum_member_bytes: int, maximum_total_bytes: int
) -> bytes:
    """Encode deterministic regular-file ZIP bytes under explicit decoded-byte budgets.

    Budgets are strict positive integers at most 64 MiB; at most 128 unique safe
    names are supported. Stored bytes, fixed DOS epoch/Unix permissions and empty
    metadata remove host/time/compressor variability. ZIP64 remains disabled.
    Encoded size includes bounded ZIP headers beyond the decoded total. No files
    or network are accessed. Invalid names/budgets/duplicates/sizes raise ValueError;
    non-bytes raise TypeError. Callers retain their own narrower format limits.
    """
    if any(
        type(value) is not int or not 1 <= value <= 64 * 1024**2
        for value in (maximum_member_bytes, maximum_total_bytes)
    ):
        raise ValueError("Deterministic ZIP requires positive byte budgets at most 64 MiB")
    if not 1 <= len(entries) <= 128 or len({name for name, _ in entries}) != len(entries):
        raise ValueError("Deterministic ZIP requires 1..128 uniquely named files")
    total = 0
    for name, payload in entries:
        safe_member_name(name)
        if not isinstance(payload, bytes):
            raise TypeError("Deterministic ZIP payload must be bytes")
        total += len(payload)
        if len(payload) > maximum_member_bytes or total > maximum_total_bytes:
            raise ValueError("Deterministic ZIP payload exceeds decoded-byte bounds")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=False) as archive:
        for name, payload in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def _zip_footer(path: Path, maximum_entries: int, maximum_metadata_bytes: int) -> int:
    """Bound central-directory bytes/entries before ZipFile allocates its metadata structures.

    V1 rejects multi-disk, ZIP64, missing/trailing EOCD and metadata beyond declared
    limits. This is a pinned-format integrity gate, not an executable-code sandbox.
    """
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65557))
        tail = stream.read(65557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < 22:
        raise ValueError("ZIP has no complete end-of-central-directory record")
    _, disk, directory_disk, count_here, count, metadata_size, metadata_offset, comment = struct.unpack_from(
        "<4s4H2IH", tail, offset
    )
    if disk or directory_disk or count_here != count:
        raise ValueError("Multi-disk ZIP is unsupported by this recipe")
    if count == 65535 or metadata_size == 0xFFFFFFFF or metadata_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 is unsupported by this recipe")
    if offset + 22 + comment != len(tail):
        raise ValueError("ZIP has inconsistent end record or trailing bytes")
    if count > maximum_entries or metadata_size > maximum_metadata_bytes:
        raise ValueError("ZIP metadata exceeds declared resource limits")
    if metadata_offset + metadata_size != size - len(tail) + offset:
        raise ValueError("ZIP central-directory offsets are inconsistent")
    validate_directory(path, metadata_offset, metadata_size, count)
    return metadata_offset


def _entry_extra(info: zipfile.ZipInfo) -> None:
    """Reject ZIP64 entry extensions and malformed bounded extra-field records before decoding."""
    needed_version = info.extract_version | (info.reserved << 8)
    if needed_version > 20 or 0xFFFFFFFF in (info.compress_size, info.file_size):
        raise ValueError("ZIP entry version or ZIP64 sizes are unsupported by this recipe")
    validate_extra_records(info.extra)


def _entry_name(info: zipfile.ZipInfo) -> str:
    """Reject encrypted, aliased, linked/special or unsafe entries before opening member bytes."""
    _entry_extra(info)
    allowed_flags = _UTF8_DESCRIPTOR_FLAGS | (_DEFLATE_LEVEL_FLAGS if info.compress_type == zipfile.ZIP_DEFLATED else 0)
    if info.orig_filename != info.filename or info.flag_bits & ~allowed_flags:
        raise ValueError("ZIP contains aliased, encrypted or unsupported member metadata")
    name = info.filename[:-1] if info.is_dir() else info.filename
    safe_member_name(name)
    mode = stat.S_IFMT(info.external_attr >> 16)
    allowed = {0, stat.S_IFDIR} if info.is_dir() else {0, stat.S_IFREG}
    if mode not in allowed:
        raise ValueError("ZIP links and special file entries are prohibited")
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ValueError("ZIP compression method is unsupported by this recipe")
    if info.compress_type == zipfile.ZIP_STORED and info.compress_size != info.file_size:
        raise ValueError("ZIP stored member compressed and decoded sizes must agree")
    if info.is_dir() and info.file_size:
        raise ValueError("ZIP directory entries cannot contain data")
    return name


def _zip_entries(archive: zipfile.ZipFile, spec: MaterializationSpec) -> tuple[zipfile.ZipInfo, ...]:
    """Require exactly declared regular files and bounded canonical metadata, including directories."""
    entries = archive.infolist()
    if len(entries) > spec.limits.maximum_archive_entries:
        raise ValueError("ZIP entry count exceeds declared limit")
    names = [_entry_name(info) for info in entries]
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate or aliased member identities")
    files = {info.filename: info for info in entries if not info.is_dir()}
    expected = {member.name: member for member in spec.members}
    if set(files) != set(expected):
        raise ValueError("ZIP regular members disagree with the frozen recipe")
    for name, info in files.items():
        if info.file_size != expected[name].size_bytes:
            raise ValueError("ZIP member decoded size disagrees with frozen expectation")
    return tuple(sorted(entries, key=lambda info: info.filename))


class _DeflateVerifier:
    """Track bounded raw-deflate completeness; CRC and frozen member hashes remain the caller's checks."""

    def __init__(self, expected_bytes: int) -> None:
        """Initialize one decoder with the already validated expected uncompressed byte count."""
        self.decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        self.expected_bytes = expected_bytes
        self.total = 0

    def consume(self, chunk: bytes) -> None:
        """Consume compressed input and drain buffered output in chunks of at most 64 KiB.

        A full output chunk may leave buffered output without unconsumed input.
        Empty-input calls drain that state until EOF or an incomplete input block;
        zero progress returns so a truncated stream cannot loop indefinitely.
        Byte overflow, input after EOF and unused tails raise ValueError.
        """
        if self.decoder.eof:
            raise ValueError("ZIP deflate compressed size is inconsistent")
        while True:
            limit = min(65536, self.expected_bytes - self.total + 1)
            decoded = self.decoder.decompress(chunk, limit)
            self.total += len(decoded)
            if self.total > self.expected_bytes:
                raise ValueError("ZIP deflate decoded size exceeds frozen expectation")
            if self.decoder.unused_data:
                raise ValueError("ZIP deflate contains unused compressed tail bytes")
            chunk = self.decoder.unconsumed_tail
            if not chunk and (self.decoder.eof or len(decoded) < limit):
                return

    def finish(self) -> None:
        """Require a complete stream and exact decoded bytes; truncated/mis-sized streams raise ValueError."""
        if not self.decoder.eof or self.total != self.expected_bytes:
            raise ValueError("ZIP deflate stream is truncated or has inconsistent decoded size")


def _validate_deflate(path: Path, info: zipfile.ZipInfo, data_offset: int) -> None:
    """Verify that one raw deflate stream consumes every declared compressed byte with bounded memory.

    ZipExtFile verifies CRC/decoded data later but can ignore compressed tails.
    This independent first pass rejects tails, concatenated/truncated streams
    and decoded-size disagreement. Both passes limit each decoded chunk to 64 KiB;
    no complete member is retained. ZipFile.open validates the local header and
    overlap before this helper runs. The trusted-local-filesystem contract applies.
    """
    verifier = _DeflateVerifier(info.file_size)
    with path.open("rb") as stream:
        stream.seek(data_offset)
        remaining = info.compress_size
        while remaining:
            chunk = stream.read(min(65536, remaining))
            if not chunk:
                raise ValueError("ZIP deflate compressed size is inconsistent")
            remaining -= len(chunk)
            verifier.consume(chunk)
    verifier.finish()


def consume_zip_members(
    path: Path,
    spec: MaterializationSpec,
    consume: Callable[[PackageMember, ByteReader], None],
) -> None:
    """Read and verify each selected stream through the caller's bounded checksum consumer.

    All names, types and metadata bounds are checked before member decoding.
    Every regular file must be declared; directories never create output paths.
    Zip/CRC/I/O failures propagate as ValueError/OSError. All archive/member input
    handles close on successful consumption or any callback/decoder exception.
    """
    directory_offset = _zip_footer(path, spec.limits.maximum_archive_entries, spec.limits.maximum_zip_metadata_bytes)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = _zip_entries(archive, spec)
            boundaries = entry_boundaries(entries, directory_offset)
            data_offsets = {
                info.filename: local_data_offset(path, info, boundaries[info.header_offset]) for info in entries
            }
            members = {member.name: member for member in spec.members}
            for info in entries:
                with archive.open(info) as stream:
                    if info.compress_type == zipfile.ZIP_DEFLATED:
                        _validate_deflate(path, info, data_offsets[info.filename])
                    if info.is_dir():
                        if stream.read(1):
                            raise ValueError("ZIP directory entries cannot contain data")
                    else:
                        consume(members[info.filename], stream)
    except (zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ValueError("Invalid or unsupported ZIP source package") from exc


def _selection_profile(members: tuple[PackageMember, ...], maximum_entries: int, maximum_metadata_bytes: int) -> None:
    """Check independent selection ceilings before any metadata allocation or byte decoding."""
    if (
        type(maximum_entries) is not int
        or not 1 <= maximum_entries <= 4096
        or type(maximum_metadata_bytes) is not int
        or not 22 <= maximum_metadata_bytes <= 2 * 1024**2
        or not 1 <= len(members) <= 128
        or len({member.name for member in members}) != len(members)
        or sum(member.size_bytes for member in members) > 64 * 1024**2
    ):
        raise ValueError("Invalid bounded ZIP selection profile")


def _selection_entries(
    archive: zipfile.ZipFile, expected: dict[str, PackageMember], maximum_entries: int
) -> tuple[zipfile.ZipInfo, ...]:
    """Inspect every entry and bind the selected subset without certifying unselected payloads."""
    entries = tuple(archive.infolist())
    if len(entries) > maximum_entries:
        raise ValueError("ZIP selection exceeds entry limit")
    names = [_entry_name(info) for info in entries]
    if len(names) != len(set(names)):
        raise ValueError("ZIP selection has duplicate or aliased names")
    files = {info.filename: info for info in entries if not info.is_dir()}
    if not expected.keys() <= files.keys() or any(
        files[name].file_size != member.size_bytes for name, member in expected.items()
    ):
        raise ValueError("Selected ZIP members disagree with frozen names/sizes")
    return entries


def consume_selected_zip_members(
    path: Path,
    members: tuple[PackageMember, ...],
    consume: Callable[[PackageMember, ByteReader], None],
    *,
    maximum_entries: int,
    maximum_metadata_bytes: int,
) -> None:
    """Inspect a bounded non-ZIP64 container and decode only explicitly pinned regular members.

    This separate selection profile supports up to 4096 directory entries and
    2 MiB metadata, with at most 128 selected members totaling 64 MiB. Existing
    complete-package v1 limits/semantics remain unchanged. All names, types and
    local headers are validated, including unselected entries; their payloads
    are not decoded or certified. The caller must verify the complete enclosing
    source digest first and checksum each selected stream before publication.
    No archive path becomes a filesystem output. Invalid bounds, aliases,
    traversal, ZIP64, links, CRC/deflate errors raise ValueError; I/O propagates.
    """
    _selection_profile(members, maximum_entries, maximum_metadata_bytes)
    directory = _zip_footer(path, maximum_entries, maximum_metadata_bytes)
    expected = {member.name: member for member in members}
    try:
        with zipfile.ZipFile(path) as archive:
            entries = _selection_entries(archive, expected, maximum_entries)
            files = {info.filename: info for info in entries if not info.is_dir()}
            boundaries = entry_boundaries(entries, directory)
            offsets = {info.filename: local_data_offset(path, info, boundaries[info.header_offset]) for info in entries}
            for name, member in sorted(expected.items()):
                info = files[name]
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    _validate_deflate(path, info, offsets[name])
                with archive.open(info) as stream:
                    consume(member, stream)
    except (zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ValueError("Invalid or unsupported selected ZIP source") from exc
