"""Native TCPCI execution of explicitly modeled, verified LRTS Parquet exports."""

from __future__ import annotations

from pathlib import Path

from coleman.api import RunResult, run
from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_etl import validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.manifest import generate_manifest
from coleman.spec.models import RunSpec


def run_lrts_replay(
    completion: Path,
    cache: SourceCache,
    output: Path,
    *,
    policies: tuple[str, ...] = ("Random", "UCB"),
    budgets: tuple[float, ...] = (0.5,),
    seed: int = 42,
) -> RunResult:
    """Validate real ETL files and execute/resume through the existing native API/checkpoint.

    This is a post-hoc serial simulation: scheduled-label order, additive reported
    method seconds, full feedback after each modeled round and assumed test-label
    correspondence. It grants no temporal/corpus admission or policy promotion.
    The exact ETL receipt is persisted inside the native run and participates in
    run identity. Empty projections reject rather than silently omitting a run.
    Output is external to Git and separate from input/cache. Native exceptions
    propagate and existing checkpoint/durability behavior is unchanged. Repeating
    identical arguments resumes the same native run; seeds are not independent
    projects/faults. Policy/budget validity uses RunSpec's existing validation.
    """
    with completion.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("ETL completion receipt exceeds 2 MiB")
    receipt = LRTSETLReceipt.from_json(payload)
    root = completion.parent
    validate_lrts_etl(receipt, cache, root)
    if receipt.native_builds == 0:
        raise ValueError("ETL has no compatible native replay cases")
    validate_source_storage(output)
    validate_cache_separation(cache.root, output)
    validate_cache_separation(root, output)
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": seed, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "datasets"),
                "datasets": [receipt.recipe.project],
                "policies": list(policies),
                "rewards": ["RNFail"],
                "budget": {"mode": "ratio", "values": list(budgets)},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
            "results": {"out_dir": str(output), "top_k_prioritization": 10},
            "extensions": {"lrts_etl": {"receipt": receipt.fingerprint, "evidence": receipt.evidence}},
        }
    )
    result = run(spec)
    if result.artifacts_dir is None:
        raise ValueError("Native replay returned no artifact directory")
    directory = Path(result.artifacts_dir)
    target = directory / "corpus-etl-receipts" / (receipt.fingerprint.removeprefix("sha256:") + ".json")
    validate_source_storage(target)
    if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Native ETL provenance path escapes its output")
    ArtifactWriter(directory).write_immutable(target, receipt.to_json() + "\n")
    generate_manifest(directory, run_id=result.run_id)
    return result
