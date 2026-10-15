"""Bounded internal ZIP local-record checks; central metadata alone is not an integrity proof."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import BinaryIO, NamedTuple, Self


class _LocalHeader(NamedTuple):
    """Fixed-width local metadata; byte lengths/sizes are unsigned integers, CRC is an unsigned CRC-32."""

    version: int
    flags: int
    method: int
    crc: int
    compressed_bytes: int
    decoded_bytes: int
    name_bytes: int
    extra_bytes: int

    @classmethod
    def read(cls, stream: BinaryIO) -> Self:
        """Read exactly the 30-byte header; reject incomplete/signature-invalid records without decoding data."""
        raw = stream.read(30)
        if len(raw) != 30 or raw[:4] != b"PK\x03\x04":
            raise ValueError("ZIP local header is incomplete or has an invalid signature")
        _, version, flags, method, _, _, crc, compressed, decoded, name, extra = struct.unpack("<4s5H3I2H", raw)
        return cls(version, flags, method, crc, compressed, decoded, name, extra)


def validate_extra_records(extra: bytes) -> None:
    """Check bounded local/central TLV framing and reject ZIP64; inert vendor extras may legitimately differ."""
    offset = 0
    while offset < len(extra):
        if len(extra) - offset < 4:
            raise ValueError("ZIP extra metadata is truncated")
        tag, size = struct.unpack_from("<HH", extra, offset)
        if tag == 1:
            raise ValueError("ZIP64 member metadata is unsupported by this recipe")
        offset += 4 + size
        if offset > len(extra):
            raise ValueError("ZIP extra metadata size is inconsistent")


def validate_directory(path: Path, offset: int, size: int, count: int) -> None:
    """Verify real central record counts/framing before ZipFile allocates entry objects.

    The caller first bounds declared count/size and physical offsets. Read one
    fixed 46-byte record at a time, skipping bounded variable fields. Reject
    count/length/signature inconsistencies and per-entry nonzero disk references.
    No names, extra sections or complete metadata directory are allocated here.
    """
    consumed = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        for _ in range(count):
            if size - consumed < 46:
                raise ValueError("ZIP central directory count exceeds its complete records")
            header = stream.read(46)
            if len(header) != 46 or header[:4] != b"PK\x01\x02":
                raise ValueError("ZIP central directory record is incomplete or has an invalid signature")
            name, extra, comment, disk = struct.unpack_from("<4H", header, 28)
            if disk:
                raise ValueError("ZIP central entry refers to an unsupported additional disk")
            record_bytes = 46 + name + extra + comment
            consumed += record_bytes
            if consumed > size:
                raise ValueError("ZIP central record length exceeds the declared directory")
            stream.seek(record_bytes - 46, 1)
    if consumed != size:
        raise ValueError("ZIP central directory count disagrees with actual metadata records")


def entry_boundaries(entries: tuple[zipfile.ZipInfo, ...], directory_offset: int) -> dict[int, int]:
    """Return each physical entry's exclusive end; reject shared, negative or out-of-directory offsets."""
    offsets = sorted(info.header_offset for info in entries)
    if len(offsets) != len(set(offsets)) or any(offset < 0 or offset >= directory_offset for offset in offsets):
        raise ValueError("ZIP local entry offsets are aliased or inconsistent")
    return dict(zip(offsets, [*offsets[1:], directory_offset], strict=True))


def _validate_header_values(header: _LocalHeader, info: zipfile.ZipInfo) -> None:
    """Require matching flags/method/version and exact local CRC/sizes, or explicit descriptor placeholders."""
    needed_version = info.extract_version | (info.reserved << 8)
    if (header.version, header.flags, header.method) != (needed_version, info.flag_bits, info.compress_type):
        raise ValueError("ZIP local version, flags or compression method disagree with central metadata")
    actual = (header.crc, header.compressed_bytes, header.decoded_bytes)
    expected = (info.CRC, info.compress_size, info.file_size)
    if header.flags & 8:
        if any(value not in (0, frozen) for value, frozen in zip(actual, expected, strict=True)):
            raise ValueError("ZIP local descriptor placeholders disagree with central CRC/sizes")
    elif actual != expected:
        raise ValueError("ZIP local CRC/sizes disagree with central metadata")


def _validate_descriptor(stream: BinaryIO, info: zipfile.ZipInfo, boundary: int) -> None:
    """Verify signed/unsigned non-ZIP64 descriptors inside this entry, without consuming a following header."""
    available = boundary - stream.tell()
    if available < 12:
        raise ValueError("ZIP data descriptor is truncated or overlaps another record")
    raw = stream.read(min(16, available))
    expected = struct.pack("<3I", info.CRC, info.compress_size, info.file_size)
    if raw[:12] == expected or (raw[:4] == b"PK\x07\x08" and raw[4:] == expected):
        return
    raise ValueError("ZIP data descriptor CRC/sizes disagree with central metadata")


def local_data_offset(path: Path, info: zipfile.ZipInfo, boundary: int) -> int:
    """Validate a local record and return its physical compressed-data offset, without publishing/decoding.

    Reads at most one fixed header, a 65,535-byte name, a 65,535-byte extra
    section and a 16-byte descriptor. Local names must equal the already-validated
    ASCII central name. Local/central extra fields are independently framed and
    non-ZIP64; legitimate vendor differences are retained as inert metadata.
    Data/descriptor overlap, mismatched metadata and malformed extras raise
    ValueError. Trusted local parents and the caller's source-hash check apply.
    """
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = _LocalHeader.read(stream)
        _validate_header_values(header, info)
        if stream.read(header.name_bytes) != info.filename.encode("ascii"):
            raise ValueError("ZIP local and central member names disagree")
        extra = stream.read(header.extra_bytes)
        if len(extra) != header.extra_bytes:
            raise ValueError("ZIP local extra metadata is truncated")
        validate_extra_records(extra)
        offset = stream.tell()
        data_end = offset + info.compress_size
        if data_end > boundary:
            raise ValueError("ZIP local member data overlaps another record")
        if header.flags & 8:
            stream.seek(data_end)
            _validate_descriptor(stream, info, boundary)
    return offset
