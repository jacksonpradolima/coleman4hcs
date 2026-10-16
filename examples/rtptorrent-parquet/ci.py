"""Owned complete RTPTorrent-format ZIP64 → analytical/native Parquet → TCPCI → checkpoint continuation.

Usage outside checkout: python ci.py OUTPUT_DIRECTORY EXTERNAL_CACHE_DIRECTORY
These constructed source jobs establish functional correctness, not independent observations.
"""

import hashlib
import io
import struct
import sys
import zipfile
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import load_registry, validate_cache_separation
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLRecipe
from coleman.transports.rtptorrent_etl import main as convert_and_run


def digest(data):
    """Hash complete owned bytes; no provider checksum or external source is involved."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def owned_recipe():
    """Build actual tiny ZIP64 framing and a full-history recipe with zero/null/repeated-class observations."""
    content = (
        b"travisJobId,testName,index,duration,count,failures,errors,skipped\n"
        b"101,owned.Repeated,0,0,1,0,0,0\n"
        b"101,owned.Repeated,1,2.5,1,1,0,0\n"
        b"101,owned.Unknown,2,,,,,\n"
        b"101,owned.Skipped,3,0,2,0,0,2\n"
        b"101,owned.NoDuration,4,,1,0,0,0\n"
        b"102,owned.Repeated,10,1,1,0,0,0\n"
        b"102,owned.Repeated,11,3,1,1,0,0\n"
        b"999,owned.Outside,0,0,1,0,0,0\n"
    )
    logical = "owned@ci/class-results.csv"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        entry = zipfile.ZipInfo(logical, date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        with archive.open(entry, "w", force_zip64=True) as stream:
            stream.write(content)
    raw = buffer.getvalue()
    footer = struct.unpack("<4s4H2IH", raw[-22:])
    source = (
        raw[:-22]
        + struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, footer[5], footer[6])
        + struct.pack("<4sIQI", b"PK\x06\x07", 0, len(raw) - 22, 1)
        + struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    )
    terms = load_registry().card("ci-serial-boundary-v1").manifest.license.model_dump(mode="json")

    def package(name, data):
        """Retain explicit owned terms, format version and raw byte identity for each object."""
        return {
            "schema_version": 1,
            "package_id": name,
            "archive": "raw",
            "license": terms,
            "source": {
                "uri": "urn:coleman:owned-rtptorrent-parquet:" + name,
                "version": "1.1",
                "citation": "Repository-authored RTPTorrent-format correctness fixture; no foreign data",
                "sha256": digest(data),
                "size_bytes": len(data),
                "availability": "not_verified",
            },
            "members": [{"name": name, "sha256": digest(data), "size_bytes": len(data)}],
        }

    child = package("class-results.csv", content)
    recipe = RTPTorrentETLRecipe.model_validate(
        {
            "schema_version": 1,
            "source": {
                "schema_version": 1,
                "source": package("owned-history.zip", source),
                "selections": [
                    {
                        "member": {"name": logical, "sha256": digest(content), "size_bytes": len(content)},
                        "package": child,
                    }
                ],
            },
            "conversion": {
                "schema_version": 1,
                "materialization": {"schema_version": 1, "spec": child},
                "member": "class-results.csv",
                "project": "owned/ci",
                "history": "constructed",
            },
            "dataset_id": "owned-rtptorrent",
            "replay_job_ids": [101, 102],
            "feedback": "budget-censored-native-RNFail-zero-nondetection",
        }
    )
    return recipe, source


def main(output, cache):
    """Persist immutable owned input/recipe and reuse the installed transport/core/native API without mocks."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Owned example output must remain inside its workspace")
    validate_cache_separation(cache, root)
    recipe, source = owned_recipe()
    writer = ArtifactWriter(root)
    writer.write_immutable(root / "recipe.json", recipe.to_json() + "\n")
    writer.write_immutable_stream(root / "owned-history.zip", (source,), maximum_bytes=len(source))
    return convert_and_run(
        [
            "--recipe",
            str(root / "recipe.json"),
            "--archive",
            str(root / "owned-history.zip"),
            "--cache",
            cache,
            "--output",
            str(root / "etl"),
            "--run",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
