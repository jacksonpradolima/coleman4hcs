"""Verified HCS WTS/VTS execution through the existing native API, result sinks and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from coleman.api import RunResult
from coleman.application.corpus import _payload, _run_verified, _VerifiedReplay
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.hcs_etl import validate_hcs_etl
from coleman.corpus.materialization import SourceCache


def run_hcs_replay(
    completion: Path,
    cache: SourceCache,
    output: Path,
    *,
    strategy: Literal["wts", "vts"],
    policies: tuple[str, ...] = ("Random", "UCB"),
    budgets: tuple[float, ...] = (0.5, 1.0),
    seed: int = 42,
) -> RunResult:
    """Verify original bytes and run/resume one declared HCS strategy without a second experiment engine.

    Costs explicitly round to the native bandit's binary32 precision before
    budgets; original decimal seconds remain in analytical evidence. WTS learns
    on the explicitly modeled max-cost/any-nonpass union and projects
    its ranking to each current variant. VTS learns separately in each retained
    variant dataset. Both use budget-censored native RNFail; nondetection zero
    is not a passing label. Entirely excluded cells/variants stay in analytical
    evidence and are not fabricated into native runs. Full orders up to the
    converter's 1,000-test cap are recorded for independent accounting checks.
    The receipt participates in run identity and is persisted in the manifest.
    No source admission, probability, statistical independence, observed setup
    cost or protected temporal evaluation is inferred. Identical arguments resume
    existing native checkpoints without duplicate results. Invalid strategy,
    empty input, source/file/path errors raise ValueError/OSError; runner failures
    propagate. Output must be outside Git and separate from cache/export.
    """
    if strategy not in {"wts", "vts"}:
        raise ValueError("HCS strategy must be wts or vts")
    receipt = HCSReceipt.from_json(_payload(completion))
    validate_hcs_etl(receipt, cache, completion.parent)
    if receipt.accounting.replay_rows == 0:
        raise ValueError("HCS export has no compatible native executions")
    counts = {f.path: f.rows for f in receipt.files}
    datasets = tuple(v.alias for v in receipt.variants if counts[f"vts/{v.alias}/features-engineered.parquet"])
    return _run_verified(
        _VerifiedReplay(
            completion.parent,
            receipt.recipe.dataset_id,
            receipt,
            receipt.scope,
            "hcs_etl",
            strategy,
            datasets if strategy == "vts" else (),
        ),
        cache,
        output,
        policies,
        budgets,
        seed,
    )
