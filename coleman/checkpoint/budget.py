"""Isolate native checkpoints for distinct budgets without reinterpreting older unscoped state."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from coleman.budget import BudgetMode


def _scoped_marker(base: Path, marker: Path) -> bool:
    """Recognize new scope structure and original run ID without loading checkpoint pickle.

    A legacy run ID may itself start with ``budget-v1/<digest>``. Directory
    spelling alone therefore cannot establish a scoped checkpoint. Read at most
    1 MiB of progress JSON and require its run ID to match the path below the
    digest. Missing/malformed/oversized metadata remains ambiguous; I/O errors
    propagate. This does not validate payload state or rewrite legacy metadata.
    """
    parts = marker.relative_to(base).parts
    if len(parts) < 3 or parts[0] != "budget-v1" or re.fullmatch(r"[0-9a-f]{64}", parts[1]) is None:
        return False
    with marker.open("rb") as stream:
        payload = stream.read(1024**2 + 1)
    if len(payload) > 1024**2:
        return False
    try:
        metadata = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
    return isinstance(run_id, str) and Path(run_id).parts == parts[2:-1]


def budget_checkpoint_config(
    config: dict[str, Any],
    points: list[tuple[BudgetMode, float]],
    mode: BudgetMode,
    value: float,
) -> dict[str, Any]:
    """Scope enabled multi-budget runs by exact binary64 mode/value, preserving single-budget paths.

    Native checkpoint payloads identify the dataset/execution but historically
    omitted budget. Sharing them across budget points skips unevaluated rounds.
    New multi-budget runs use ``budget-v1/<digest>`` under the existing base.
    Full float.hex values distinguish close budgets that round to the same
    percent. Single-budget runs and existing payload versions are unchanged.
    Any existing unscoped progress marker under this base is ambiguous for a
    multi-budget run, including nested contextual/extension scenario names. Reject
    before loading pickle or modifying it. Retain the original
    package/output, or start explicitly in a fresh output tree; no automatic
    migration guesses which budget produced old results. Scoped markers require
    bounded progress metadata matching their original run ID; a namespace-like
    legacy name cannot bypass detection. This function reads directories/metadata,
    creates nothing, and never unpickles state. Invalid scoped run-ID metadata
    blocks multi-budget continuation rather than silently starting again.
    Trusted local checkpoint parents and existing store durability limits apply.
    """
    resolved = dict(config)
    if not config.get("enabled", False) or len(set(points)) <= 1:
        return resolved
    base = Path(config.get("base_dir", "checkpoints"))
    if any(not _scoped_marker(base, marker) for marker in base.rglob("progress_ex*.json") if marker.is_file()):
        raise ValueError("Ambiguous legacy multi-budget checkpoint; retain old output and use a fresh output tree")
    identity = hashlib.sha256((mode.value + "|" + float(value).hex()).encode("ascii")).hexdigest()
    resolved["base_dir"] = str(base / "budget-v1" / identity)
    return resolved
