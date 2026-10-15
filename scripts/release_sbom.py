#!/usr/bin/env python3
"""Catalog this release's wheel metadata with Syft; never publish an empty SBOM.

This inventories the distribution itself, not unbundled dependencies or the build
host. Dependency advisories use the complete uv.lock through audit_lock.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path


def validate_inventory(payload: dict, version: str, wheel_name: str) -> None:
    """Require CycloneDX, logical wheel identity and exactly the expected distribution.

    Empty, unrelated and mismatched inventories raise ValueError. This validates
    the release inventory's scope, not all CycloneDX fields or a signature.
    """
    root = payload.get("metadata", {}).get("component", {})
    packages = payload.get("components", [])
    if payload.get("bomFormat") != "CycloneDX" or root.get("name") != wheel_name or root.get("version") != version:
        raise ValueError("SBOM must identify its actual release wheel")
    if len(packages) != 1 or (packages[0].get("name"), packages[0].get("version")) != ("coleman", version):
        raise ValueError("SBOM must contain the expected COLEMAN distribution")


def build_sbom(wheel: Path, version: str, output: Path) -> None:
    """Run installed Syft against bounded wheel METADATA and atomically publish JSON.

    Only a fixed metadata member is read: no archive paths are extracted and no
    wheel code executes. Missing/duplicate/oversized metadata, version mismatches,
    malformed Syft output or tool failure abort without replacing existing output.
    The output parent must exist. The root SHA-256 binds complete wheel bytes.
    Syft timestamps and serial numbers remain observations; identical SBOM bytes
    are not promised.
    """
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise ValueError("Expected a stable semantic release version")
    if wheel.name != f"coleman-{version}-py3-none-any.whl":
        raise ValueError("Expected this release's COLEMAN wheel")
    member = f"coleman-{version}.dist-info/METADATA"
    with zipfile.ZipFile(wheel) as archive:
        matches = [item for item in archive.infolist() if item.filename == member]
        if len(matches) != 1 or matches[0].file_size > 1048576:
            raise ValueError("Expected one bounded wheel METADATA member")
        metadata = archive.read(matches[0])
    parsed = BytesParser(policy=policy.default).parsebytes(metadata)
    if parsed.get("Name") != "coleman" or parsed.get("Version") != version:
        raise ValueError("Wheel metadata differs from its release identity")
    output = output.resolve()
    with tempfile.TemporaryDirectory(prefix="coleman-sbom-") as temporary:
        work = Path(temporary)
        directory = work / f"coleman-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_bytes(metadata)
        raw = work / "inventory.json"
        subprocess.run(
            [
                "syft",
                "scan",
                f"dir:{work}",
                "--source-name",
                wheel.name,
                "--source-version",
                version,
                "-o",
                f"cyclonedx-json={raw}",
            ],
            env={**os.environ, "SYFT_FILE_METADATA_SELECTION": "none"},
            check=True,
            timeout=120,
        )
        payload = json.loads(raw.read_text())
        validate_inventory(payload, version, wheel.name)
        # The PyPI name belongs to another project. Bind the known GitHub source
        # explicitly instead of publishing Syft's inferred PyPI/CPE coordinates.
        component = payload["components"][0]
        previous = component.get("bom-ref")
        source = f"pkg:github/jacksonpradolima/coleman@v{version}"
        component["bom-ref"] = component["purl"] = source
        component.pop("cpe", None)
        component["properties"] = [item for item in component.get("properties", []) if item["name"] != "syft:cpe23"]
        for edge in payload.get("dependencies", []):
            if edge.get("ref") == previous:
                edge["ref"] = source
            edge["dependsOn"] = [source if ref == previous else ref for ref in edge.get("dependsOn", [])]
        payload["metadata"]["component"]["hashes"] = [
            {"alg": "SHA-256", "content": hashlib.sha256(wheel.read_bytes()).hexdigest()}
        ]
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    # Keep temporary output on the destination filesystem so replace is atomic.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as pending:
        path = Path(pending.name)
        try:
            pending.write(content)
            pending.flush()
            os.fsync(pending.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    try:
        os.replace(path, output)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_sbom(args.wheel, args.version, args.output)
