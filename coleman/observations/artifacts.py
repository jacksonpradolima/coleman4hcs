"""Publication through existing immutable artifacts; no second state or checkpoint store."""

from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.observations.contracts import DecisionSignal
from coleman.observations.mean import QuantityMeanModel


def publish_model(model: QuantityMeanModel, writer: ArtifactWriter) -> Path:
    """Revalidate and atomically publish a frozen model including its eligible training evidence.

    Identical retries/concurrent writes are idempotent. Conflicts and I/O failures
    propagate. Existing local hard-link/fsync and process-interruption durability
    limits apply; callers own the trusted writer root and any retention policy.
    """
    model = QuantityMeanModel.model_validate(model)
    path = writer.run_root / "observation-models" / f"{model.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, model.to_json() + "\n")
    return path


def publish_signal(signal: DecisionSignal, writer: ArtifactWriter) -> Path:
    """Publish a validated signal without changing native results or advancing a checkpoint.

    Native callers persist the matching domain sidecar before their existing
    checkpoint advances. Interrupted temporary files are never completed signals;
    same-content retries are idempotent. The stable logical ID addresses the file;
    changed content for that ID conflicts instead of silently creating a second
    decision. Content digests remain separate deterministic integrity identifiers.
    """
    signal = DecisionSignal.model_validate(signal)
    path = writer.run_root / "decision-signals" / f"{signal.logical_id.removeprefix('sha256:')}.json"
    writer.write_immutable(path, signal.to_json() + "\n")
    return path
