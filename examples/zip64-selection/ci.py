"""Select an owned CI correctness fixture from real ZIP64 framing, outside the checkout.

Usage: python ci.py OUTPUT_DIRECTORY EXTERNAL_CACHE_DIRECTORY
Requires the source revision containing #301; this is not a foreign benchmark.
"""

import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache, bundled_fixture_bytes, load_registry, validate_cache_separation
from coleman.corpus.zip64_contracts import Zip64SelectionSpec
from coleman.corpus.zip64_selection import materialize_zip64_files, validate_zip64_files


def digest(payload):
    """Return the canonical SHA-256 identity of complete bytes, not a provider checksum."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def owned_bundle():
    """Build a deterministic tiny ZIP64 container and explicit source/selected-file recipe."""
    content = bundled_fixture_bytes("ci-serial-boundary-v1")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        entry = zipfile.ZipInfo("owned@ci/serial.json", date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        with archive.open(entry, "w", force_zip64=True) as output:
            output.write(content)
    raw = stream.getvalue()
    end = struct.unpack("<4s4H2IH", raw[-22:])
    payload = (
        raw[:-22]
        + struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, end[5], end[6])
        + struct.pack("<4sIQI", b"PK\x06\x07", 0, len(raw) - 22, 1)
        + struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    )
    terms = load_registry().card("ci-serial-boundary-v1").manifest.license.model_dump(mode="json")

    def raw_package(name, data):
        """Retain explicit reviewed data terms and complete source identity for each raw object."""
        return {
            "schema_version": 1,
            "package_id": name,
            "archive": "raw",
            "source": {
                "uri": "urn:coleman:owned-zip64:" + name,
                "version": "owned-zip64-v1",
                "citation": "Repository-authored CI correctness fixture, not an independent benchmark",
                "sha256": digest(data),
                "size_bytes": len(data),
                "availability": "not_verified",
            },
            "license": terms,
            "members": [{"name": name, "sha256": digest(data), "size_bytes": len(data)}],
        }

    spec = Zip64SelectionSpec.model_validate(
        {
            "schema_version": 1,
            "source": raw_package("owned-ci.zip", payload),
            "selections": [
                {
                    "member": {"name": "owned@ci/serial.json", "sha256": digest(content), "size_bytes": len(content)},
                    "package": raw_package("serial.json", content),
                }
            ],
        }
    )
    return spec, payload


def main(output, cache_root):
    """Acquire, validate offline and persist the recipe with raw receipts after every member succeeds.

    An individual raw receipt does not certify enclosing membership. No new group
    completion marker is introduced; retain selection.json with all child receipts
    and call validate_zip64_files when reopening. Concurrent identical writes are
    idempotent under the existing trusted-local ArtifactWriter durability contract.
    """
    root = Path(output).resolve()
    validate_cache_separation(cache_root, root)
    spec, payload = owned_bundle()
    cache = SourceCache(cache_root)
    receipts = materialize_zip64_files(spec, cache, lambda: io.BytesIO(payload))
    validate_zip64_files(spec, cache)
    writer = ArtifactWriter(root)
    writer.write_immutable(root / "selection.json", spec.to_json() + "\n")
    for index, receipt in enumerate(receipts):
        writer.write_immutable(root / f"raw-{index}.json", receipt.to_json() + "\n")
    print(
        json.dumps(
            {
                "selection": str(root / "selection.json"),
                "raw_receipt": str(root / "raw-0.json"),
                "selected_file": str(cache.member_path(receipts[0], "serial.json")),
                "integrity": "verified-selected-membership",
                "admission": "not_evaluated",
                "generalization": "insufficient_evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
