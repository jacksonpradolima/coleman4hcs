"""Conservative source-exposure accounting before a protected comparison is frozen."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Annotated, Literal

from pydantic import Field

from coleman.application.baseline_lock import baseline_lock
from coleman.application.baseline_storage import read_baseline_artifact
from coleman.artifacts import ArtifactWriter
from coleman.evaluation.protected import AccessReceipt, EvaluationAudit, FrozenEvaluation
from coleman.results.domain import Fingerprint, Finite, fingerprint
from coleman.workflows.contracts import ExperimentPlan, WorkflowContract, WorkflowError, WorkflowId
from coleman.workflows.evidence import FrozenWorkflowConfiguration


class SourceExposure(WorkflowContract):
    """Attempted descriptive/imported outcome access, retained independently of job expiry.

    Parameters
    ----------
    identity : str
        Stable host-issued job/import ID, never a caller path.
    invocation : int
        Physical worker start or first immutable import reservation.
    kind : str
        Descriptive native replay or read-only evidence import; neither is confirmation.
    source_fingerprint, plan_fingerprint : str
        Stable upstream scope and complete actual execution/evidence context.
    recorded_epoch_seconds : float
        Actual operational timestamp. Later audit integration never backdates a
        protected access receipt to this earlier time or invents a hypothesis.
    """

    identity: WorkflowId
    invocation: Annotated[int, Field(strict=True, ge=1, le=8)]
    kind: Literal["descriptive_replay", "read_only_import"]
    source_fingerprint: Fingerprint
    plan_fingerprint: Fingerprint
    recorded_epoch_seconds: Finite


def source_lock(root: Path, source_fingerprint: str):
    """Return the existing retained kernel lock for one stable upstream source scope.

    Parameters
    ----------
    root : pathlib.Path
        Trusted private workspace root on supported local POSIX storage.
    source_fingerprint : str
        Validated upstream identity; projections/encodings must not reset exposure.

    Returns
    -------
    contextlib.AbstractContextManager
        Nonblocking existing lock. Contention rejects deterministically; no PID
        guess, automatic key recycling or distributed locking is implied.
    """
    return baseline_lock(root / "locks" / f"source-{fingerprint([source_fingerprint])[7:]}.lock")


def mark_exposure(
    root: Path,
    plan: ExperimentPlan,
    identity: str,
    invocation: int,
    kind: Literal["descriptive_replay", "read_only_import"],
) -> None:
    """Retain an attempted full-history access without inventing a frozen comparison.

    Parameters
    ----------
    root : pathlib.Path
        Trusted workspace; markers survive terminal artifact cleanup.
    plan : ExperimentPlan
        Exact source/context. The ledger scope uses upstream source identity,
        independently of storage encoding, runtime, protocol and candidate choice.
    identity : str
        Host-issued job/import identity.
    invocation : int
        Bounded physical worker/import reservation ordinal.
    kind : str
        Descriptive replay or read-only evidence import.

    Raises
    ------
    ValueError, OSError, BaselineBusyError
        Corrupt/conflicting event, finite exposure capacity, storage or live source
        contention. Failed attempts remain conservative exposures; no test access
        is inferred from merely planning or reading an already completed report.
    """
    source = plan.dataset.source_fingerprint
    scope = root / "source-exposures" / fingerprint([source])[7:]
    event = SourceExposure(
        schema_version=1,
        identity=identity,
        invocation=invocation,
        kind=kind,
        source_fingerprint=source,
        plan_fingerprint=plan.fingerprint,
        recorded_epoch_seconds=time(),
    )
    target = scope / f"{identity}-{invocation:02d}.json"
    with source_lock(root, source):
        if target.exists():
            old = read_baseline_artifact(target, SourceExposure)
            if old.model_dump(exclude={"recorded_epoch_seconds"}) != event.model_dump(
                exclude={"recorded_epoch_seconds"}
            ):
                raise WorkflowError("Source exposure identity conflicts with its original context")
            return
        if len(list(scope.glob("*.json"))) >= 1024:
            raise WorkflowError("Source exposure slots exhausted")
        ArtifactWriter(root).write_immutable(target, event.to_json())


def synchronize_exposures(root: Path, frozen: FrozenEvaluation) -> None:
    """Integrate earlier source exposure into the existing audit before protected loading.

    Parameters
    ----------
    root : pathlib.Path
        Trusted workspace. Caller must hold the retained source lock through the
        protected evaluator, serializing source exposures and comparison access.
    frozen : FrozenEvaluation
        Actual candidates/protocol, saved before this operation. Earlier exposures
        are recorded as development accesses, not invented confirmatory hypotheses.

    Raises
    ------
    ValueError, OSError
        Missing source binding, malformed events/maps or bounded audit exhaustion.
        A crash between reservation and mapping may conservatively overcount access;
        it can never erase previous exposure or make a reused holdout appear virgin.
    """
    from coleman.results.domain import decode_json

    configuration = FrozenWorkflowConfiguration.model_validate(decode_json(frozen.candidates[0].configuration_json))
    source = configuration.source_scope
    scope = fingerprint([source])[7:]
    paths = sorted((root / "source-exposures" / scope).glob("*.json"))
    if len(paths) > 1024:
        raise WorkflowError("Source exposure metadata exceeds its finite bound")
    audit = EvaluationAudit(root / "audit")
    history = audit.history(frozen.protocol.holdout_id)
    for path in paths:
        event = read_baseline_artifact(path, SourceExposure)
        if event.source_fingerprint != source:
            raise WorkflowError("Source exposure belongs to another upstream scope")
        target = root / "exposure-audit-maps" / scope / f"{event.fingerprint[7:]}.json"
        if target.exists():
            receipt = read_baseline_artifact(target, AccessReceipt)
            if receipt.sequence > len(history) or history[receipt.sequence - 1] != receipt:
                raise WorkflowError("Source exposure map is absent from the actual protected audit")
        else:
            if len(audit.history(frozen.protocol.holdout_id)) >= 512:
                raise WorkflowError("Protected scope access slots exhausted")
            receipt = audit.access(frozen, partition="test", purpose="development", actor="prior-source-exposure")
            ArtifactWriter(root).write_immutable(target, receipt.to_json())
