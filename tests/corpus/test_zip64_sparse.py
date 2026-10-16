"""Sparse owned ZIP64 format test: large offsets, never a full-source integrity assertion."""

import struct
import zlib

import pytest

from coleman.corpus.zip64_contracts import Zip64SelectionSpec
from coleman.corpus.zip64_selection import consume_zip64_files
from tests.corpus.test_zip64_selection import owned_zip64, recipe


def local(name, payload_size, *, crc=0, wide=False):
    extra = struct.pack("<HHQQ", 1, 16, payload_size, payload_size) if wide else b""
    size = 0xFFFFFFFF if wide else payload_size
    return (
        struct.pack("<4s5H3I2H", b"PK\x03\x04", 45 if wide else 20, 0, 0, 0, 0, crc, size, size, len(name), len(extra))
        + name
        + extra
    )


def central(name, size, offset, *, crc=0, large_sizes=False, large_offset=False, version=45):
    values = ((size, size) if large_sizes else ()) + ((offset,) if large_offset else ())
    extra = struct.pack("<HH" + "Q" * len(values), 1, 8 * len(values), *values) if values else b""
    stored_size = 0xFFFFFFFF if large_sizes else size
    return (
        struct.pack(
            "<4s6H3I5H2I",
            b"PK\x01\x02",
            45,
            version,
            0,
            0,
            0,
            0,
            crc,
            stored_size,
            stored_size,
            len(name),
            len(extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF if large_offset else offset,
        )
        + name
        + extra
    )


@pytest.mark.parametrize("central_version", [20, 45])
def test_selected_offset_above_4gib_skips_uncertified_unselected_payload(tmp_path, central_version):
    """The lower format reader does not certify the enclosing sparse bytes or the skipped CRC.

    Public materialization tests separately prove the full-source SHA gate; this
    sparse case intentionally bypasses that gate to test 64-bit seeks cheaply.
    """
    name = b"owner@project/results.csv"
    content = b"owned CI records\n"
    unused_name = b"unselected.bin"
    huge = 2**32
    first = local(unused_name, huge, wide=True)
    selected_offset = len(first) + huge
    second = local(name, len(content), crc=zlib.crc32(content))
    directory_offset = selected_offset + len(second) + len(content)
    metadata = central(unused_name, huge, 0, large_sizes=True) + central(
        name, len(content), selected_offset, crc=zlib.crc32(content), large_offset=True, version=central_version
    )
    footer_offset = directory_offset + len(metadata)
    footer = (
        struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 2, 2, len(metadata), directory_offset)
        + struct.pack("<4sIQI", b"PK\x06\x07", 0, footer_offset, 1)
        + struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    )
    path = tmp_path / "format-only.zip"
    with path.open("wb") as stream:
        stream.write(first)
        stream.seek(selected_offset)
        stream.write(second + content + metadata + footer)
    value = recipe(owned_zip64()).model_dump(mode="json")
    source = value["source"]
    source["source"]["size_bytes"] = path.stat().st_size
    source["members"][0]["size_bytes"] = path.stat().st_size
    for limit in ("maximum_source_bytes", "maximum_member_bytes", "maximum_extracted_bytes"):
        source["limits"][limit] = 8 * 1024**3
    spec = Zip64SelectionSpec.model_validate(value)
    consumed = []
    consume_zip64_files(path, spec, lambda member, stream: consumed.append((member.name, stream.read())))
    assert consumed == [(name.decode("ascii"), content)]
