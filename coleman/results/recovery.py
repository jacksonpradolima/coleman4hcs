"""Checkpoint reconciliation of native Parquet rows after a process interruption."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from coleman.artifacts import ArtifactWriter
from coleman.budget import BudgetMode
from coleman.results.domain import Fingerprint, fingerprint
from coleman.results.versioned import VersionedContract

_OWNER_METADATA_KEY = b"coleman.native-results-owner-v1"


class _CheckpointOwner(VersionedContract):
    """Persistent operational pairing token, distinct from reproducible mathematical identities."""

    owner_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{32}$")]


class _CheckpointBinding(_CheckpointOwner):
    """Version-one ownership of one native dataset/execution/exact-budget coordinate."""

    scope_id: Fingerprint


@dataclass(frozen=True)
class ResultCheckpoint:
    """One native dataset/execution/budget's last durable processed step.

    ``step`` is a nonnegative local round count, not a source BuildId. Zero
    means no checkpoint committed. Budget value uses the exact native binary64
    value in the indicated mode. Policies and HCS variants commit together.
    This key intentionally excludes the new attempt's random execution ID.
    A persisted result/checkpoint ownership binding scopes it to its original run.
    """

    scenario: str
    experiment: int
    mode: BudgetMode
    value: float
    step: int

    def __post_init__(self) -> None:
        """Reject invalid progress before any result file can be changed."""
        if type(self.step) is not int or self.step < 0:
            raise ValueError("Result checkpoint step must be a nonnegative integer")
        if type(self.experiment) is not int or self.experiment < 0 or not self.scenario:
            raise ValueError("Result checkpoint requires a scenario and nonnegative execution number")
        if not isinstance(self.mode, BudgetMode) or isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("Result checkpoint requires an explicit budget mode and finite value")

    @property
    def scope_id(self) -> Fingerprint:
        """Identify the coordinate independently of the attempt and committed round count."""
        return fingerprint(
            [
                "native-parquet-checkpoint-scope-v1",
                self.scenario,
                self.experiment,
                self.mode.value,
                float(self.value).hex(),
            ]
        )


def _metadata(path: Path) -> bytes:
    """Bound private ownership metadata before strict versioned parsing; missing/corrupt state blocks recovery."""
    with path.open("rb") as stream:
        payload = stream.read(4097)
    if len(payload) > 4096:
        raise ValueError("Native checkpoint ownership metadata exceeds 4 KiB")
    return payload


def _owner(checkpoint_root: Path) -> _CheckpointOwner:
    """Create once or read the checkpoint directory's token; concurrent creators adopt the atomic winner."""
    path = checkpoint_root / ".native-results-owner-v1.json"
    if not path.exists():
        candidate = _CheckpointOwner(schema_version=1, owner_id=uuid4().hex)
        try:
            ArtifactWriter(checkpoint_root).write_immutable(path, candidate.to_json() + "\n")
        except ValueError:
            # A conflicting creator can only publish complete metadata. Parse the
            # winner strictly rather than replacing or silently resetting it.
            return _CheckpointOwner.from_json(_metadata(path))
    return _CheckpointOwner.from_json(_metadata(path))


def bind_checkpoint_results(
    root: Path, checkpoint: ResultCheckpoint, checkpoint_root: Path, checkpoint_loaded: bool, schema: pa.Schema
) -> _CheckpointOwner:
    """Pair result coordinates with their originating checkpoint directory before trimming any rows.

    A fresh/emptied checkpoint directory receives a new persistent token. It may
    not reclaim an existing coordinate bound to another token, even if dataset,
    experiment and budget happen to agree. New empty coordinates bind before
    their first result write. Legacy unbound results can be adopted only with a
    successfully loaded checkpoint and no rows beyond that committed step;
    unpaired/ambiguous tails reject unchanged. No old execution identity is guessed.
    Tokens are operational ownership, not secrets or mathematical artifact IDs.
    Immutable publication converges for concurrent distinct-coordinate executions.
    Unknown/corrupt versions and I/O failures propagate before result deletion.
    """
    owner = _owner(checkpoint_root)
    path = root / ".native-checkpoint-bindings-v1" / (checkpoint.scope_id.removeprefix("sha256:") + ".json")
    binding = _CheckpointBinding(schema_version=1, owner_id=owner.owner_id, scope_id=checkpoint.scope_id)
    if path.exists():
        if _CheckpointBinding.from_json(_metadata(path)) != binding:
            raise ValueError(
                "Parquet results and checkpoint directory are not paired; retain output and use fresh paths"
            )
        return owner
    _validate_legacy_pairing(root, checkpoint, checkpoint_loaded, schema)
    ArtifactWriter(root).write_immutable(path, binding.to_json() + "\n")
    return owner


def _validate_legacy_pairing(
    root: Path, checkpoint: ResultCheckpoint, checkpoint_loaded: bool, schema: pa.Schema
) -> None:
    """Reject ambiguous old tails without modifying data; complete committed legacy coordinates remain readable."""
    for path in root.rglob("*.parquet"):
        try:
            table = pq.read_table(path, schema=schema, partitioning="hive")
        except FileNotFoundError:
            continue
        columns = (
            table[name].to_pylist() for name in ("scenario", "experiment", "budget_mode", "budget_value", "step")
        )
        for scenario, experiment, mode, value, step in zip(*columns, strict=True):
            matches = (scenario, experiment, mode, value) == (
                checkpoint.scenario,
                checkpoint.experiment,
                checkpoint.mode.value,
                checkpoint.value,
            )
            if matches and (not checkpoint_loaded or step is None or step > checkpoint.step):
                raise ValueError("Unpaired legacy Parquet results; retain output and start a fresh paired run")


def _recover_parquet_results(
    root: Path, checkpoint: ResultCheckpoint, schema: pa.Schema, owner: _CheckpointOwner
) -> None:
    """Remove only this coordinate's rows beyond its committed native checkpoint.

    Caller must validate persistent ownership using bind_checkpoint_results,
    hold its sink lock with an empty buffer and have exclusive
    ownership of this run's recovery; concurrent independent executions write
    distinct files. Concurrent attempts of the same run are unsupported. The
    checkpoint must already have been loaded successfully from trusted local
    storage. Absent checkpoints use step zero. Other coordinates, domain
    records and committed rows are retained; unchanged files are byte-identical.
    Physical partition columns and schema remain unchanged. No deduplication or
    reinterpretation of previously committed observations occurs.

    Each affected file is atomically replaced or unlinked. Retrying an interrupted
    reconciliation converges before simulation resumes; temporary files never
    have a Parquet suffix. Files removed by another independent recovery after
    enumeration are skipped; other I/O/Arrow errors block execution. This
    handles process interruption, not power-loss durability or corrupt/missing
    result files predating the checkpoint. No CLI/MCP or checkpoint pickle is read.
    """
    for path in sorted(root.rglob("*.parquet")):
        try:
            with pq.ParquetFile(path) as shard:
                physical = shard.schema_arrow
            table = pq.read_table(path, schema=schema, partitioning="hive")
        except FileNotFoundError:
            # Another independent execution may remove its own uncommitted
            # shard after enumeration. Never recreate a disappeared file.
            continue
        keep = [
            not (
                scenario == checkpoint.scenario
                and experiment == checkpoint.experiment
                and mode == checkpoint.mode.value
                and value == checkpoint.value
                and step is not None
                and step > checkpoint.step
            )
            for scenario, experiment, mode, value, step in zip(
                *(
                    table[name].to_pylist()
                    for name in ("scenario", "experiment", "budget_mode", "budget_value", "step")
                ),
                strict=True,
            )
        ]
        if not all(keep):
            recorded = (physical.metadata or {}).get(_OWNER_METADATA_KEY)
            if recorded is None or len(recorded) > 4096 or _CheckpointOwner.from_json(recorded) != owner:
                raise ValueError("Unowned Parquet tail; retain independent results and use fresh paired paths")
            retained = table.filter(pa.array(keep, type=pa.bool_())).select(physical.names).cast(physical)
            _replace_retained(path, retained)


def _replace_retained(path: Path, table: pa.Table) -> None:
    """Publish a complete retained shard atomically; errors preserve its original bytes."""
    if table.num_rows == 0:
        path.unlink()
        return
    fd, name = tempfile.mkstemp(prefix=".recovery-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            pq.write_table(table, stream, compression="zstd")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
