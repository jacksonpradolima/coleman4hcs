"""Bounded transactional metadata for a trusted persistent local workflow workspace."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

from pydantic import Field

from coleman.corpus.materialization import validate_source_storage
from coleman.results.domain import Fingerprint, Finite
from coleman.workflows.contracts import DatasetManifest, ExperimentPlan, WorkflowContract, WorkflowId
from coleman.workflows.lifecycle import HostPolicy, Stage, State


class JobRecord(WorkflowContract):
    """Private durable job metadata, revalidated at every storage boundary.

    Values are bounded, typed operational state rather than mathematical evidence.
    Epoch seconds record actual lifecycle times; they are never inferred source
    availability. Ownership/request/plan columns are checked against this record.

    Parameters
    ----------
    job_id, owner, request_id, request_digest, plan_id : str
        Stable host-issued identity, trusted principal, retained idempotency key,
        exact admitted payload and immutable executable plan binding.
    state, stage : str
        Known lifecycle and descriptive/tuning/evaluation stage. Native ownership
        is separately established by kernel locks, never these labels or a PID.
    created, changed : float
        Actual operational epoch seconds, separate from source/evidence clocks.
    current_trial : str or None
        Current coordinate, or unavailable/no active trial.
    checkpoint_step, completed_trials, total_trials : int
        Real committed native frontier and stage counts, never inference units.
    attempts, invocations : int
        Physical per-trial reservations and separately bounded worker starts.
    error : str or None
        Stable bounded safe public failure, without host exception tracebacks.
    result_digest : str or None
        Complete-last report identity, published with a serialized completion
        transition. An artifact directory alone never establishes completion.
    """

    job_id: WorkflowId
    owner: WorkflowId
    request_id: WorkflowId
    request_digest: Fingerprint
    plan_id: WorkflowId
    state: State
    stage: Stage
    created: Finite
    changed: Finite
    current_trial: Fingerprint | None = None
    checkpoint_step: Annotated[int, Field(strict=True, ge=0, le=512)] = 0
    completed_trials: Annotated[int, Field(strict=True, ge=0, le=128)] = 0
    total_trials: Annotated[int, Field(strict=True, ge=1, le=128)]
    attempts: Annotated[int, Field(strict=True, ge=0, le=2048)] = 0
    invocations: Annotated[int, Field(strict=True, ge=0, le=8)] = 0
    error: Annotated[str, Field(strict=True, max_length=128)] | None = None
    result_digest: Fingerprint | None = None


class WorkflowStore:
    """Versioned SQLite metadata; kernel ownership/artifact publication remain separate.

    Parameters
    ----------
    root : pathlib.Path
        Trusted local workspace; never accepted from a workflow tool request.
    policy : HostPolicy or None
        Policy for a new workspace. Reopening an existing workspace reuses its
        immutable policy; explicit conflicting policy rejects before admission.

    Notes
    -----
    Opens private local files, enables WAL/FULL synchronization and uses bounded
    one-second SQLite lock waits. Transactions serialize admissions/transitions
    across threads and processes. This is not distributed storage, a hostile-path
    sandbox or a guarantee against arbitrary disk/power loss. Unknown database
    versions fail without migration. Operational I/O/SQLite errors propagate.
    """

    def __init__(self, root: Path, policy: HostPolicy | None = None):
        """Create/reopen the explicitly owned local metadata store; class parameters apply."""
        self.root = root.absolute()
        validate_source_storage(self.root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.root / "workspace.sqlite3"
        validate_source_storage(self.path)
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            expected = {"settings", "datasets", "plans", "jobs", "imports"}
            if version not in (0, 1) or (version == 0 and tables) or (version == 1 and tables != expected):
                raise ValueError("Unknown/corrupt workflow workspace schema; no implicit migration")
            connection.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS plans (id TEXT NOT NULL, owner TEXT NOT NULL, "
                "payload TEXT NOT NULL, PRIMARY KEY(id,owner))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                "request TEXT NOT NULL, plan TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(owner,request))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS imports (id TEXT PRIMARY KEY, owner TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            existing = connection.execute("SELECT payload FROM settings WHERE name='policy'").fetchone()
            if existing is None:
                if version == 1:
                    raise ValueError("Workspace is missing its immutable policy")
                self.policy = HostPolicy.model_validate(policy or HostPolicy(schema_version=1))
                connection.execute("INSERT INTO settings VALUES ('policy',?)", (self.policy.to_json(),))
            else:
                self.policy = HostPolicy.from_json(existing[0])
                if policy is not None and self.policy != HostPolicy.model_validate(policy):
                    raise ValueError("Existing workspace policy cannot be silently replaced")
            connection.execute("PRAGMA user_version=1")
            connection.commit()
        os.chmod(self.path, 0o600)

    def connect(self) -> sqlite3.Connection:
        """Open a short-lived private metadata connection; no native checkpoint is loaded.

        Returns
        -------
        sqlite3.Connection
            Caller-owned connection with a one-second busy timeout and FULL sync.
        """
        validate_source_storage(self.path)
        connection = sqlite3.connect(self.path, timeout=1, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize one bounded metadata operation and roll back on any failure.

        Yields
        ------
        sqlite3.Connection
            Exclusive write transaction; filesystem/native execution stays outside
            it unless needed for atomic admission/publication membership.

        Raises
        ------
        sqlite3.Error, OSError, ValueError
            Lock, storage or validation failure; no partial transition is committed.
        """
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def datasets(self, connection: sqlite3.Connection) -> tuple[DatasetManifest, ...]:
        """Return the bounded typed host registry, rejecting key/payload mismatches.

        Parameters
        ----------
        connection : sqlite3.Connection
            Caller-owned transaction/read connection.

        Returns
        -------
        tuple[DatasetManifest, ...]
            Sorted exact-byte manifests; no input outcome bytes are returned.
        """
        rows = connection.execute("SELECT id,payload FROM datasets ORDER BY id LIMIT 33").fetchall()
        if len(rows) > self.policy.max_datasets:
            raise ValueError("Workspace registry exceeds its policy")
        values = tuple(DatasetManifest.from_json(row["payload"]) for row in rows)
        if any(value.dataset_id != row["id"] for value, row in zip(values, rows, strict=True)):
            raise ValueError("Dataset registry identity mismatch")
        return values

    def plan(self, connection: sqlite3.Connection, plan_id: str, owner: str) -> ExperimentPlan:
        """Load only a caller-owned saved plan and verify its content-derived key.

        Parameters
        ----------
        connection : sqlite3.Connection
            Caller-owned metadata connection.
        plan_id, owner : str
            Validated plan key and trusted host principal.

        Returns
        -------
        ExperimentPlan
            Revalidated complete plan, without running any configuration.

        Raises
        ------
        KeyError, ValueError
            Missing/nonowned identity or corrupt/version-incompatible content.
        """
        row = connection.execute("SELECT payload FROM plans WHERE id=? AND owner=?", (plan_id, owner)).fetchone()
        if row is None:
            raise KeyError("No owned workflow plan")
        plan = ExperimentPlan.from_json(row["payload"])
        if plan.fingerprint[7:] != plan_id:
            raise ValueError("Saved plan identity mismatch")
        return plan

    def jobs(self, connection: sqlite3.Connection) -> tuple[JobRecord, ...]:
        """Load bounded durable records, checking indexed ownership/idempotency identity.

        Parameters
        ----------
        connection : sqlite3.Connection
            Caller-owned transaction/read connection.

        Returns
        -------
        tuple[JobRecord, ...]
            Creation-ordered operational records, including expired tombstones.
        """
        rows = connection.execute("SELECT * FROM jobs ORDER BY rowid LIMIT 129").fetchall()
        if len(rows) > self.policy.max_jobs:
            raise ValueError("Workspace job metadata exceeds its policy")
        values = tuple(JobRecord.from_json(row["payload"]) for row in rows)
        for value, row in zip(values, rows, strict=True):
            if (value.job_id, value.owner, value.request_id, value.plan_id) != (
                row["id"],
                row["owner"],
                row["request"],
                row["plan"],
            ):
                raise ValueError("Job metadata identity mismatch")
        return values

    @staticmethod
    def save_job(connection: sqlite3.Connection, job: JobRecord, *, create: bool = False) -> None:
        """Commit one validated operational record inside the caller's transaction.

        Parameters
        ----------
        connection : sqlite3.Connection
            Active transaction; callers serialize compare-and-transition updates.
        job : JobRecord
            Typed full state; identity remains immutable for an existing job.
        create : bool
            Insert a new bounded admission instead of updating an existing identity.

        Raises
        ------
        ValueError, sqlite3.Error
            Missing identity or conflicting caller/request; transaction rolls back.
        """
        job = JobRecord.model_validate(job)
        if create:
            connection.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?)",
                (job.job_id, job.owner, job.request_id, job.plan_id, job.to_json()),
            )
        else:
            cursor = connection.execute(
                "UPDATE jobs SET payload=? WHERE id=? AND owner=? AND request=? AND plan=?",
                (job.to_json(), job.job_id, job.owner, job.request_id, job.plan_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Cannot replace workflow job identity")
