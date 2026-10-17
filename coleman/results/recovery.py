"""Checkpoint reconciliation of native Parquet rows after a process interruption."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from coleman.budget import BudgetMode


@dataclass(frozen=True)
class ResultCheckpoint:
    """One native dataset/execution/budget's last durable processed step.

    ``step`` is a nonnegative local round count, not a source BuildId. Zero
    means no checkpoint committed. Budget value uses the exact native binary64
    value in the indicated mode. Policies and HCS variants commit together.
    This key intentionally excludes the new attempt's random execution ID.
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


def recover_parquet_results(root: Path, checkpoint: ResultCheckpoint, schema: pa.Schema) -> None:
    """Remove only this coordinate's rows beyond its committed native checkpoint.

    Caller must hold its sink lock with an empty buffer and have exclusive
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
