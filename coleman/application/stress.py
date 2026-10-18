"""Eligible synthetic CI execution through the existing native API, sinks and checkpoints."""

from pathlib import Path

from coleman.api import RunResult, run
from coleman.application.corpus import _payload
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import validate_cache_separation, validate_source_storage
from coleman.corpus.stress_artifacts import validate_stress
from coleman.corpus.stress_contracts import StressReceipt
from coleman.manifest import generate_manifest
from coleman.spec import RunSpec


def run_stress_replay(completion: Path, output: Path, *, policies: tuple[str, ...] = ("Random", "UCB")) -> RunResult:
    """Verify one generated case and run/resume its eligible immediate-reward projection.

    All variant executions share one synthetic suite and policy state here;
    variant copies are dependent, not independent faults or VTS evidence. The
    native executor uses serial additive occupancy and native RNFail detection
    rewards. Four-worker cases additionally support explicit modeled post-hoc
    list replay; they do not change the executor. Current exact durations are
    scoring truth, not automatically known to a predecision policy.
    Delayed/missing/corrected feedback and full-label under partial budget reject
    before output creation. Policies must be Random/UCB; further eligibility and
    comparable baseline protocols belong to #311, not this mechanism slice.
    Case/producer/receipt participate in stable run identity. Existing native
    attempt serialization, recovery and checkpoint durability limits apply.
    Run different cases sequentially or in separate processes: existing native
    RNGs are process-global, not thread-isolated. No second experiment engine or
    implicit migration is introduced.
    """
    receipt = StressReceipt.from_json(_payload(completion))
    receipt.case.require_native()
    if not policies or len(set(policies)) != len(policies) or any(p not in {"Random", "UCB"} for p in policies):
        raise ValueError("Stress native qualification supports unique Random/UCB candidates only")
    validate_source_storage(output)
    validate_cache_separation(completion.parent, output)
    validate_stress(receipt, completion.parent)
    ratio = 1.0 if receipt.case.factors.budget == "full" else 0.0625
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": receipt.case.seed},
            "experiment": {
                "datasets_dir": str(completion.parent / "datasets"),
                "datasets": ["stress"],
                "policies": list(policies),
                "rewards": ["RNFail"],
                "budget": {"mode": "ratio", "values": [ratio]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
            "results": {"out_dir": str(output), "top_k_prioritization": 150000},
            "extensions": {
                "synthetic_stress": {
                    "receipt": receipt.fingerprint,
                    "evidence": "synthetic-immediate-native-RNFail-serial-replay",
                }
            },
        }
    )
    result = run(spec)
    if result.artifacts_dir is None:
        raise ValueError("Native stress replay returned no artifact directory")
    root = Path(result.artifacts_dir)
    path = root / "corpus-etl-receipts" / (receipt.fingerprint.removeprefix("sha256:") + ".json")
    validate_source_storage(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Stress native provenance escapes output")
    ArtifactWriter(root).write_immutable(path, receipt.to_json() + "\n")
    generate_manifest(root, run_id=result.run_id)
    return result
