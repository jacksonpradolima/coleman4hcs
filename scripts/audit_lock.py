#!/usr/bin/env python3
"""Export every registry package in uv.lock for a platform-independent advisory audit.

Unlike environment-filtered exports, this includes optional extras and packages
for other operating systems. It does not install or execute dependencies. Only
the root COLEMAN editable entry is omitted. Unknown sources fail closed instead
of silently escaping the audit. Advisory results are time-dependent and cannot
prove absence of undisclosed vulnerabilities.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def audit_requirements(path: Path) -> str:
    """Read a uv v1 lock and return exact registry pins across all extras/platforms.

    Malformed names/versions, unsupported lock versions and non-PyPI sources raise
    ValueError. This is advisory lookup input, not an installation requirements
    file; wheel hashes and platform markers remain authoritative in uv.lock.
    """
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    schema_version = lock.get("version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ValueError("Unsupported uv lock schema")
    requirements = set()
    root_seen = False
    for package in lock["package"]:
        name, version, source = package["name"], package["version"], package["source"]
        if name == "coleman" and source == {"editable": "."}:
            if root_seen:
                raise ValueError("Duplicate root lock entry")
            root_seen = True
            continue
        requirements.add(_registry_requirement(name, version, source))
    if not root_seen or not requirements:
        raise ValueError("Lock lacks root or registry dependencies")
    return "\n".join(sorted(requirements)) + "\n"


def _registry_requirement(name: str, version: str, source: dict[str, str]) -> str:
    """Reject unsupported sources and unsafe requirement syntax before advisory lookup."""
    if source != {"registry": "https://pypi.org/simple"}:
        raise ValueError(f"Cannot advisory-audit unsupported package source: {name}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version
    ):
        raise ValueError("Invalid registry name/version in lock")
    return f"{name}=={version}"


if __name__ == "__main__":
    # The CLI audits this repository's committed lock; it accepts no caller-supplied path.
    argparse.ArgumentParser(description="Audit this checkout's committed uv.lock only").parse_args()
    print(audit_requirements(Path(__file__).resolve().parents[1] / "uv.lock"), end="")
