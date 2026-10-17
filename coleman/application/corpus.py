"""Native TCPCI execution of explicitly modeled, verified corpus Parquet exports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from coleman.api import RunResult, run
from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_etl import validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.lrts_stage_etl import validate_lrts_stage_etl
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.rtptorrent_etl import validate_rtptorrent_etl
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.manifest import generate_manifest
from coleman.results.versioned import VersionedContract
from coleman.spec.models import RunSpec


class _VerifiedReplay(NamedTuple):
    """Caller-verified ETL input; source-specific assumptions remain in the complete versioned receipt."""

    root: Path
    dataset_id: str
    receipt: VersionedContract
    evidence: str
    extension: Literal["lrts_etl", "rtptorrent_etl", "lrts_stage_etl", "hcs_etl"]
    strategy: Literal["serial", "wts", "vts"] = "serial"
    dataset_ids: tuple[str, ...] = ()


def _payload(completion: Path) -> bytes:
    """Bound completion metadata before parsing; I/O failures propagate and no native run starts."""
    with completion.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("ETL completion receipt exceeds 2 MiB")
    return payload


def _run_verified(
    source: _VerifiedReplay,
    cache: SourceCache,
    output: Path,
    policies: tuple[str, ...],
    budgets: tuple[float, ...],
    seed: int,
) -> RunResult:
    """Reuse the existing native RunSpec/API/checkpoint and manifest publication for a verified projection.

    Source-specific wrappers validate complete source/file correspondence and
    eligibility first. Existing policy/budget validation, immutable provenance,
    output separation, run identity and checkpoint semantics remain unchanged.
    """
    validate_source_storage(output)
    validate_cache_separation(cache.root, output)
    validate_cache_separation(source.root, output)
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": seed, "verbose": False},
            "experiment": {
                "datasets_dir": str(source.root / ("vts" if source.strategy == "vts" else "datasets")),
                "datasets": list(source.dataset_ids or (source.dataset_id,)),
                "policies": list(policies),
                "rewards": ["RNFail"],
                "budget": {"mode": "ratio", "values": list(budgets)},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
            "results": {"out_dir": str(output), "top_k_prioritization": 1000 if source.strategy != "serial" else 10},
            "hcs_configuration": {"wts_strategy": source.strategy == "wts"},
            "extensions": {source.extension: {"receipt": source.receipt.fingerprint, "evidence": source.evidence}},
        }
    )
    result = run(spec)
    if result.artifacts_dir is None:
        raise ValueError("Native replay returned no artifact directory")
    directory = Path(result.artifacts_dir)
    target = directory / "corpus-etl-receipts" / (source.receipt.fingerprint.removeprefix("sha256:") + ".json")
    validate_source_storage(target)
    if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Native ETL provenance path escapes its output")
    ArtifactWriter(directory).write_immutable(target, source.receipt.to_json() + "\n")
    generate_manifest(directory, run_id=result.run_id)
    return result


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
    method seconds, budget-censored native RNFail detection rewards and assumed
    test-label correspondence. Unscheduled cases receive zero nondetection reward,
    not an observed pass. Legacy full-label receipts reject before execution.
    It grants no temporal/corpus admission or policy promotion.
    The exact ETL receipt is persisted inside the native run and participates in
    run identity. Empty projections reject rather than silently omitting a run.
    Output is external to Git and separate from input/cache. Native exceptions
    propagate and existing checkpoint/durability behavior is unchanged. Repeating
    identical arguments resumes the same native run; seeds are not independent
    projects/faults. Policy/budget validity uses RunSpec's existing validation.
    """
    receipt = LRTSETLReceipt.from_json(_payload(completion))
    receipt.recipe.require_native_feedback()
    root = completion.parent
    validate_lrts_etl(receipt, cache, root)
    if receipt.native_builds == 0:
        raise ValueError("ETL has no compatible native replay cases")
    return _run_verified(
        _VerifiedReplay(root, receipt.recipe.project, receipt, receipt.evidence, "lrts_etl"),
        cache,
        output,
        policies,
        budgets,
        seed,
    )


def run_rtptorrent_replay(
    completion: Path,
    cache: SourceCache,
    output: Path,
    *,
    policies: tuple[str, ...] = ("Random", "UCB"),
    budgets: tuple[float, ...] = (0.5,),
    seed: int = 42,
) -> RunResult:
    """Verify RTPTorrent source/Parquet and execute/resume an explicitly modeled native source-job history.

    Native BuildId represents a local modeled round, not an observed logical CI
    build. Numeric source-job order, additive reported class seconds and assumed
    class-label/within-job-occurrence correspondence are declared in the receipt.
    Native RNFail provides budget-censored detection/zero-nondetection rewards;
    zero for an unscheduled class is not a pass or full-label feedback. No clock,
    temporal/corpus admission, independent-observation count or policy promotion
    is inferred. Empty projections reject. Existing runner/checkpoint behavior
    and immutable receipt/manifest publication are reused; no new engine starts.
    Caller output is separate from cache/ETL and outside Git. Invalid/missing
    source/metadata/bytes raise ValueError/OSError; native exceptions propagate.
    Identical arguments resume the same run; seeds do not create independent jobs.
    """
    receipt = RTPTorrentETLReceipt.from_json(_payload(completion))
    root = completion.parent
    validate_rtptorrent_etl(receipt, cache, root)
    if receipt.native_rounds == 0:
        raise ValueError("RTPTorrent ETL has no compatible native source-job rounds")
    return _run_verified(
        _VerifiedReplay(root, receipt.recipe.dataset_id, receipt, receipt.evidence, "rtptorrent_etl"),
        cache,
        output,
        policies,
        budgets,
        seed,
    )


def run_lrts_stage_replay(
    completion: Path,
    cache: SourceCache,
    output: Path,
    *,
    policies: tuple[str, ...] = ("Random", "UCB"),
    budgets: tuple[float, ...] = (0.5,),
    seed: int = 42,
) -> RunResult:
    """Verify one explicit LRTS stage view and execute/resume through the existing API/checkpoint.

    Every raw observation and processed stage remains in the analytical export;
    the native input selects the recipe's declared stage. Stage views share
    source project/PR/build groups, never independent projects or unique faults.
    Scheduled-label rounds, additive reported method seconds and label/occurrence
    continuity remain simulation assumptions. Native RNFail is budget-censored:
    zero nondetection reward for unscheduled cases never means an observed pass.
    No executor/outcome/feature clock or temporal admission is inferred.
    Exact receipt provenance participates in run identity and the manifest.
    Empty projections reject; source/path/byte errors raise ValueError/OSError;
    native errors propagate. Output is separate from cache/ETL and outside Git.
    Identical arguments resume the same checkpoint without duplicating results.
    """
    receipt = LRTSStageReceipt.from_json(_payload(completion))
    root = completion.parent
    validate_lrts_stage_etl(receipt, cache, root)
    if receipt.native_builds == 0:
        raise ValueError("LRTS stage ETL has no compatible native cases")
    return _run_verified(
        _VerifiedReplay(root, receipt.recipe.dataset_id, receipt, receipt.evidence, "lrts_stage_etl"),
        cache,
        output,
        policies,
        budgets,
        seed,
    )
