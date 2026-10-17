"""Shared presentation for optional source producers; controlled domain code never imports this module."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from coleman.corpus.controlled_contracts import ControlledDataset


def source_arguments(description: str | None, presets: tuple[str, ...], argv: list[str] | None) -> argparse.Namespace:
    """Parse the common source CLI without acquiring inputs, loading recipes or starting a runtime.

    Empty presets raise ValueError. Argparse reports invalid/missing arguments using its
    usual SystemExit behavior; recipe byte limits remain the producer's contract.
    """
    if not presets:
        raise ValueError("Source CLI requires at least one declared preset")
    parser = argparse.ArgumentParser(description=description)
    recipes = parser.add_mutually_exclusive_group()
    recipes.add_argument("--preset", choices=presets, default=presets[0])
    recipes.add_argument("--recipe-stdin", action="store_true")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--fetch", action="store_true")
    return parser.parse_args(argv)


def source_summary(
    dataset: ControlledDataset,
    dataset_path: Path,
    state: Path,
    status: str,
    role_statuses: Mapping[str, str],
    planned_pairs: int,
) -> str:
    """Render deterministic CLI JSON from already validated evidence, without writing artifacts.

    Pair counts describe the finite trial; absent pairs stay unobserved. This
    presentation preserves constructed-history and admission labels verbatim and
    never turns process timing into CI cost or asserts statistical independence.
    """
    return json.dumps(
        {
            "dataset": str(dataset_path),
            "state": str(state),
            "receipt": str(state / "receipt.json"),
            "status": status,
            "role_statuses": dict(role_statuses),
            "planned_pairs": planned_pairs,
            "unobserved_pairs": planned_pairs - len(dataset.pairs),
            "pairs": dict(sorted(Counter(pair.classification for pair in dataset.pairs).items())),
            "history": dataset.history,
            "admission": dataset.admission,
            "generalization": dataset.generalization,
        },
        sort_keys=True,
    )
