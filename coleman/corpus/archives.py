"""Strict bounded ZIP inspection; logical member names never become output paths."""

from __future__ import annotations

import stat
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path

from coleman.corpus.materialization_contracts import ByteReader, MaterializationSpec, PackageMember, safe_member_name


def _zip_footer(path: Path, spec: MaterializationSpec) -> None:
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
    if count > spec.limits.maximum_archive_entries or metadata_size > spec.limits.maximum_zip_metadata_bytes:
        raise ValueError("ZIP metadata exceeds declared resource limits")
    if metadata_offset + metadata_size != size - len(tail) + offset:
        raise ValueError("ZIP central-directory offsets are inconsistent")


def _entry_extra(info: zipfile.ZipInfo) -> None:
    """Reject ZIP64 entry extensions and malformed bounded extra-field records before decoding."""
    if info.extract_version > 20:
        raise ValueError("ZIP entry version is unsupported by this recipe")
    offset = 0
    while offset < len(info.extra):
        if len(info.extra) - offset < 4:
            raise ValueError("ZIP extra metadata is truncated")
        tag, size = struct.unpack_from("<HH", info.extra, offset)
        if tag == 1:
            raise ValueError("ZIP64 member metadata is unsupported by this recipe")
        offset += 4 + size
        if offset > len(info.extra):
            raise ValueError("ZIP extra metadata size is inconsistent")


def _entry_name(info: zipfile.ZipInfo) -> str:
    """Reject encrypted, aliased, linked/special or unsafe entries before opening member bytes."""
    _entry_extra(info)
    if info.orig_filename != info.filename or info.flag_bits & 1:
        raise ValueError("ZIP contains aliased or encrypted member metadata")
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
    return tuple(files[name] for name in sorted(files))


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
    _zip_footer(path, spec)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = _zip_entries(archive, spec)
            members = {member.name: member for member in spec.members}
            for info in entries:
                with archive.open(info) as stream:
                    consume(members[info.filename], stream)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("Invalid or unsupported ZIP source package") from exc
