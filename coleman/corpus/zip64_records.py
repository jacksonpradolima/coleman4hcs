"""Bounded ZIP64 structural validation; no decompression, network or artifact publication."""

from __future__ import annotations

import stat
import struct
from pathlib import Path
from typing import BinaryIO, NamedTuple

from coleman.corpus.zip64_contracts import zip64_member_name
from coleman.corpus.zip_metadata import _LocalHeader

_SENTINEL = 0xFFFFFFFF


class Zip64Record(NamedTuple):
    """Validated central metadata; sizes/offset are unsigned byte counts and CRC is CRC-32."""

    name: str
    version: int
    flags: int
    method: int
    crc: int
    compressed: int
    decoded: int
    offset: int


class Zip64Directory(NamedTuple):
    """Validated central-directory offset plus bounded metadata records; all offsets are in bytes."""

    offset: int
    records: tuple[Zip64Record, ...]


def _extras(extra: bytes) -> bytes | None:
    """Validate bounded TLV records and return the single ZIP64 payload; inert vendor extras remain opaque."""
    cursor = 0
    zip64 = None
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise ValueError("ZIP64 extra field is truncated")
        tag, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if size > len(extra) - cursor:
            raise ValueError("ZIP64 extra length is inconsistent")
        if tag == 1:
            if zip64 is not None:
                raise ValueError("Duplicate ZIP64 extra field")
            zip64 = extra[cursor : cursor + size]
        cursor += size
    return zip64


def _sizes(extra: bytes, values: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve only sentinel-required 64-bit fields in format order; missing/surplus values reject."""
    payload = _extras(extra)
    required = sum(value == _SENTINEL for value in values)
    if payload is None:
        if required:
            raise ValueError("ZIP64 sentinel lacks required extra metadata")
        return values
    if len(payload) != required * 8 or required == 0:
        raise ValueError("ZIP64 extra payload disagrees with sentinel fields")
    decoded = iter(struct.unpack("<" + "Q" * required, payload))
    return tuple(next(decoded) if value == _SENTINEL else value for value in values)


def _footer_bounds(raw: bytes, size: int, maximum_entries: int, maximum_metadata_bytes: int) -> tuple[int, int, int]:
    """Bind fixed single-disk ZIP64/legacy end records before central-directory allocation."""
    end = raw.rfind(b"PK\x05\x06")
    if end < 76 or len(raw) - end < 22:
        raise ValueError("ZIP64 requires complete end records")
    _, disk, directory_disk, count_here, count, metadata_size, metadata_offset, comment = struct.unpack_from(
        "<4s4H2IH", raw, end
    )
    if disk or directory_disk or end + 22 + comment != len(raw):
        raise ValueError("ZIP64 legacy end record is multi-disk or inconsistent")
    signature, locator_disk, record_offset, disks = struct.unpack_from("<4sIQI", raw, end - 20)
    if signature != b"PK\x06\x07" or locator_disk or disks != 1:
        raise ValueError("ZIP64 requires a single-disk end locator")
    position = end - 76
    record = struct.unpack_from("<4sQ2H2I4Q", raw, position)
    if record[:2] != (b"PK\x06\x06", 44) or record[3:6] != (45, 0, 0):
        raise ValueError("Unsupported ZIP64 end-record form")
    _, _, _, _, _, _, actual_here, actual_count, actual_size, actual_offset = record
    if record_offset != size - len(raw) + position or actual_offset + actual_size != record_offset:
        raise ValueError("ZIP64 central/end physical offsets disagree")
    if actual_here != actual_count or not 1 <= actual_count <= maximum_entries or actual_size > maximum_metadata_bytes:
        raise ValueError("ZIP64 central metadata exceeds declared limits or spans disks")
    _legacy_agreement(
        (count_here, count, metadata_size, metadata_offset), (actual_count, actual_count, actual_size, actual_offset)
    )
    return actual_offset, actual_size, actual_count


def _legacy_agreement(legacy: tuple[int, ...], actual: tuple[int, ...]) -> None:
    """Legacy end fields must equal resolved metadata or their precise width-specific sentinel."""
    sentinels = (65535, 65535, _SENTINEL, _SENTINEL)
    if any(value not in (frozen, sentinel) for value, frozen, sentinel in zip(legacy, actual, sentinels, strict=True)):
        raise ValueError("ZIP64 legacy directory fields disagree")


def _central_record(stream: BinaryIO, available: int) -> tuple[Zip64Record, int]:
    """Read one bounded central record, resolve ZIP64 fields and validate names/types before ZipFile parses it."""
    raw = stream.read(46)
    if len(raw) != 46 or raw[:4] != b"PK\x01\x02":
        raise ValueError("ZIP64 central record is incomplete or signature-invalid")
    version, flags, method = struct.unpack_from("<3H", raw, 6)
    crc, compressed, decoded = struct.unpack_from("<3I", raw, 16)
    name_bytes, extra_bytes, comment_bytes, disk = struct.unpack_from("<4H", raw, 28)
    attributes, offset = struct.unpack_from("<2I", raw, 38)
    consumed = 46 + name_bytes + extra_bytes + comment_bytes
    if consumed > available or name_bytes > 1024 or disk:
        raise ValueError("ZIP64 central lengths/disk are inconsistent")
    try:
        name = stream.read(name_bytes).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("ZIP64 profile requires portable ASCII member names") from exc
    extra = stream.read(extra_bytes)
    if len(extra) != extra_bytes:
        raise ValueError("ZIP64 central extra metadata is truncated")
    if _SENTINEL in (decoded, compressed) and version != 45:
        raise ValueError("ZIP64 central size sentinel requires extraction version45")
    decoded, compressed, offset = _sizes(extra, (decoded, compressed, offset))
    stream.seek(comment_bytes, 1)
    record = Zip64Record(name, version, flags, method, crc, compressed, decoded, offset)
    _entry_policy(record, attributes)
    return record, consumed


def _entry_policy(record: Zip64Record, attributes: int) -> None:
    """Restrict compression/flags/types; selected file size limits are separately frozen by the caller."""
    directory = record.name.endswith("/")
    zip64_member_name(record.name[:-1] if directory else record.name)
    # Reviewed Info-ZIP forms retain bit1 on stored files and legacy extraction
    # versions for offset-only ZIP64. Neither changes the stored payload format.
    allowed_flags = 0x0808 | (0x0006 if record.method == 8 else 0x0002)
    if record.version not in (10, 20, 45) or record.flags & ~allowed_flags or record.method not in (0, 8):
        raise ValueError("Unsupported ZIP64 entry version, flags or compression")
    if record.method == 8 and record.version < 20:
        raise ValueError("ZIP64 deflate requires extraction version20 or later")
    mode = stat.S_IFMT(attributes >> 16)
    if mode not in ({0, stat.S_IFDIR} if directory else {0, stat.S_IFREG}):
        raise ValueError("ZIP64 links/special entries are prohibited")
    if (record.method == 0 and record.compressed != record.decoded) or (directory and record.decoded):
        raise ValueError("ZIP64 stored size or directory payload is inconsistent")


def read_zip64_records(path: Path, *, maximum_entries: int, maximum_metadata_bytes: int) -> Zip64Directory:
    """Validate bounded fixed ZIP64 footer/directory before allocating ZipFile objects.

    Accepts only one disk, fixed 44-byte ZIP64 end payload and safe stored/deflated
    members. Metadata/count/source bounds reject before directory allocation.
    Every central record is checked, including unselected entries. Payloads and
    the full source digest are not certified here; callers verify those separately.
    Invalid framing, unsafe names/links, duplicates and unsupported forms raise
    ValueError. OSError propagates; trusted local parents/immutability apply.
    """
    if (
        type(maximum_entries) is not int
        or not 1 <= maximum_entries <= 4096
        or type(maximum_metadata_bytes) is not int
        or not 22 <= maximum_metadata_bytes <= 2 * 1024**2
    ):
        raise ValueError("Invalid ZIP64 metadata resource limits")
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if size > 8 * 1024**3:
            raise ValueError("ZIP64 source exceeds 8 GiB")
        stream.seek(max(0, size - 65633))
        tail = stream.read(65633)
        offset, metadata_bytes, count = _footer_bounds(tail, size, maximum_entries, maximum_metadata_bytes)
        stream.seek(offset)
        consumed = 0
        records = []
        for _ in range(count):
            if metadata_bytes - consumed < 46:
                raise ValueError("ZIP64 entry count exceeds complete central records")
            record, amount = _central_record(stream, metadata_bytes - consumed)
            records.append(record)
            consumed += amount
        if consumed != metadata_bytes:
            raise ValueError("ZIP64 central count/length disagree")
    names = [record.name.removesuffix("/") for record in records]
    if len(names) != len(set(names)):
        raise ValueError("ZIP64 contains duplicate/aliased member identities")
    return Zip64Directory(offset, tuple(records))


def _descriptor(stream: BinaryIO, record: Zip64Record, boundary: int, wide: bool) -> None:
    """Require a matching signed/unsigned 32/64-bit descriptor, without crossing another record."""
    expected = struct.pack("<IQQ" if wide else "<III", record.crc, record.compressed, record.decoded)
    available = boundary - stream.tell()
    if available < len(expected):
        raise ValueError("ZIP64 descriptor overlaps another record")
    raw = stream.read(min(len(expected) + 4, available))
    if raw[: len(expected)] != expected and raw != b"PK\x07\x08" + expected:
        raise ValueError("ZIP64 descriptor CRC/sizes disagree")


def _local_identity(header: _LocalHeader, record: Zip64Record) -> None:
    """Require supported local compression/flags/version; offset-only ZIP64 can elevate central version."""
    if header.version not in (10, 20, 45) or (header.flags, header.method) != (record.flags, record.method):
        raise ValueError("ZIP64 local flags/method/version disagree")
    if header.version > record.version or (record.version != 45 and header.version != record.version):
        raise ValueError("ZIP64 local extraction version disagrees")
    if header.method == 8 and header.version < 20:
        raise ValueError("ZIP64 local deflate requires extraction version20 or later")


def _local_sizes(header: _LocalHeader, extra: bytes, record: Zip64Record) -> bool:
    """Validate exact local CRC/sizes or declared descriptor placeholders and return descriptor width."""
    if (header.decoded_bytes == _SENTINEL) != (header.compressed_bytes == _SENTINEL):
        raise ValueError("ZIP64 local extra must include both original and compressed sizes")
    decoded, compressed = _sizes(extra, (header.decoded_bytes, header.compressed_bytes))
    wide = _SENTINEL in (header.decoded_bytes, header.compressed_bytes)
    if not wide and max(record.compressed, record.decoded) >= _SENTINEL:
        raise ValueError("ZIP64 large sizes require wide local metadata")
    if wide and header.version != 45:
        raise ValueError("ZIP64 local sizes require extraction version45")
    actual = (header.crc, compressed, decoded)
    expected = (record.crc, record.compressed, record.decoded)
    if header.flags & 8:
        if any(value not in (0, frozen) for value, frozen in zip(actual, expected, strict=True)):
            raise ValueError("ZIP64 local descriptor placeholders disagree")
    elif actual != expected:
        raise ValueError("ZIP64 local CRC/sizes disagree")
    return wide


def zip64_data_offset(path: Path, record: Zip64Record, boundary: int) -> int:
    """Bind local flags/method/name/sizes to central metadata and return the compressed-data byte offset.

    Offset-only ZIP64 may elevate the central extraction version while retaining
    a legacy local data header. Local sentinel sizes require version45. Data and
    descriptor bytes must fit their physical entry boundary. No payload is decoded.
    """
    with path.open("rb") as stream:
        stream.seek(record.offset)
        if boundary - record.offset < 30:
            raise ValueError("ZIP64 local header overlaps another record")
        header = _LocalHeader.read(stream)
        if record.offset + 30 + header.name_bytes + header.extra_bytes > boundary:
            raise ValueError("ZIP64 local metadata overlaps another record")
        _local_identity(header, record)
        if stream.read(header.name_bytes) != record.name.encode("ascii"):
            raise ValueError("ZIP64 local and central names disagree")
        extra = stream.read(header.extra_bytes)
        if len(extra) != header.extra_bytes:
            raise ValueError("ZIP64 local extra metadata is truncated")
        wide = _local_sizes(header, extra, record)
        offset = stream.tell()
        data_end = offset + record.compressed
        if data_end > boundary:
            raise ValueError("ZIP64 compressed data overlaps another record")
        if header.flags & 8:
            stream.seek(data_end)
            _descriptor(stream, record, boundary, wide)
    return offset
