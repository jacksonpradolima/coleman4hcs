"""Shared implementation identity for optional source producers, with no resource imports or execution."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from importlib.resources import files
from importlib.resources.abc import Traversable

from coleman.results.domain import canonical_json


def package_pins(root: Traversable, prefix: str = "coleman") -> dict[str, str]:
    """Hash all regular package resources, excluding hidden files, bytecode and interpreter caches.

    This conservative closure includes unrelated modules and future resource
    formats. It never imports modules or modifies resources. I/O failures
    propagate, rather than silently omitting an unreadable implementation input.
    """
    pins = {}
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(".") or entry.name == "__pycache__" or entry.name.endswith((".pyc", ".pyo")):
            continue
        name = prefix + "/" + entry.name
        if entry.is_dir():
            pins.update(package_pins(entry, name))
        elif entry.is_file():
            pins[name] = "sha256:" + hashlib.sha256(entry.read_bytes()).hexdigest()
    return pins


def installed_producer_identity() -> str:
    """Bind installed COLEMAN version and all package bytes; a changed closure requires a new generation."""
    payload = canonical_json({"package_version": version("coleman"), "modules": package_pins(files("coleman"))})
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
