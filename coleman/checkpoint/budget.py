"""Isolate native checkpoints for distinct budgets without reinterpreting older unscoped state."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from coleman.budget import BudgetMode


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
    multi-budget run, including contextual/extension scenario names. Reject
    before loading pickle or modifying it. Retain the original
    package/output, or start explicitly in a fresh output tree; no automatic
    migration guesses which budget produced old results. This function only
    reads directory membership, creates nothing, and never unpickles state.
    Trusted local checkpoint parents and existing store durability limits apply.
    """
    resolved = dict(config)
    if not config.get("enabled", False) or len(set(points)) <= 1:
        return resolved
    base = Path(config.get("base_dir", "checkpoints"))
    if next(base.glob("*/progress_ex*.json"), None) is not None:
        raise ValueError("Ambiguous legacy multi-budget checkpoint; retain old output and use a fresh output tree")
    identity = hashlib.sha256((mode.value + "|" + float(value).hex()).encode("ascii")).hexdigest()
    resolved["base_dir"] = str(base / "budget-v1" / identity)
    return resolved
