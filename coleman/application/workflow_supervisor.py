"""Fixed detached local coordinator; kernel locks own workers across host restarts."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.workflow_ownership import worker_active
from coleman.application.workflow_store import JobRecord, WorkflowStore


def _spawn(root: Path, role: str, descriptor: int, job_id: str | None = None) -> None:
    """Spawn only a fixed installed module, explicitly inheriting one ownership descriptor."""
    arguments = [sys.executable, "-m", "coleman.application.workflow_supervisor", role, str(root), str(descriptor)]
    if job_id is not None:
        arguments.append(job_id)
    subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root,
        pass_fds=(descriptor,),
        start_new_session=True,
    )


def ensure_supervisor(root: Path) -> None:
    """Ensure one workspace coordinator without taking execution ownership from live workers.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent POSIX workspace. A fixed detached Python process inherits
        its lock; no transport lifetime, PID guess or arbitrary command is involved.

    Raises
    ------
    OSError, ValueError
        Unsupported/unsafe storage or spawn failure. Existing queued records remain
        recoverable by reopening the workspace and retrying the same request.
    """
    try:
        with baseline_lock(root / "supervisor.lock", handoff=True) as descriptor:
            _spawn(root, "supervise", descriptor)
    except BaselineBusyError:
        pass


def _schedule(store: WorkflowStore, supervisor_descriptor: int) -> bool:
    """Reconcile dead ownership and claim bounded queued work atomically under native locks."""
    with store.transaction() as connection:
        jobs = store.jobs(connection)
        active = [job for job in jobs if worker_active(store.root, job.job_id)]
        _reconcile_ownership(store, connection, jobs, active)
        memory = sum(store.plan(connection, job.plan_id, job.owner).request.limits.max_memory_bytes for job in active)
        for job in jobs:
            if job.state != "queued" or len(active) >= store.policy.max_running:
                continue
            plan = store.plan(connection, job.plan_id, job.owner)
            if memory + plan.request.limits.max_memory_bytes > store.policy.max_active_memory_bytes:
                continue
            try:
                with baseline_lock(store.root / "locks" / f"{job.job_id}.lock", handoff=True) as worker_descriptor:
                    if job.invocations >= plan.request.limits.max_attempts:
                        store.save_job(
                            connection,
                            JobRecord.model_validate(
                                {
                                    **job.model_dump(),
                                    "state": "failed",
                                    "changed": time.time(),
                                    "error": "invocation_limit",
                                }
                            ),
                        )
                        continue
                    updated = JobRecord.model_validate(
                        {
                            **job.model_dump(),
                            "state": "running",
                            "changed": time.time(),
                            "invocations": job.invocations + 1,
                            "error": None,
                        }
                    )
                    store.save_job(connection, updated)
                    # The child waits for this transaction; inherited ownership is
                    # already continuous before running becomes visible to clients.
                    _spawn(store.root, "work", worker_descriptor, job.job_id)
                    active.append(updated)
                    memory += plan.request.limits.max_memory_bytes
            except BaselineBusyError:
                continue
        pending = bool(active) or any(job.state == "queued" for job in jobs)
        if not pending:
            # Release ownership while admission is still serialized. A later
            # admission cannot observe a retiring owner and strand queued work.
            os.close(supervisor_descriptor)
        return pending


def _reconcile_ownership(store, connection, jobs, active) -> None:
    """Record lost kernel ownership inside the caller's transaction without restarting native work."""
    active_ids = {job.job_id for job in active}
    for job in jobs:
        if job.state == "running" and job.job_id not in active_ids:
            store.save_job(
                connection,
                JobRecord.model_validate(
                    {
                        **job.model_dump(),
                        "state": "interrupted",
                        "changed": time.time(),
                        "error": "worker_ownership_lost",
                    }
                ),
            )


def _supervise(root: Path, descriptor: int) -> None:
    """Poll bounded metadata while work exists; exiting is safe because workers own separate locks."""
    store = WorkflowStore(root)
    while _schedule(store, descriptor):
        time.sleep(0.05)


def main() -> None:
    """Private fixed subprocess entry point; public clients cannot select these arguments."""
    role, directory, descriptor, *identity = sys.argv[1:]
    root = Path(directory)
    lock = root / ("supervisor.lock" if role == "supervise" else f"locks/{identity[0]}.lock")
    _verify_inherited_lock(lock, int(descriptor))
    if role == "supervise" and not identity:
        _supervise(root, int(descriptor))
    elif role == "work" and len(identity) == 1:
        from coleman.application.workflow_worker import run_job

        run_job(root, identity[0])
    else:
        raise ValueError("Unknown fixed workflow subprocess role")


def _verify_inherited_lock(path: Path, descriptor: int) -> None:
    """Verify the actual retained inode and exclusive inherited ownership, without a PID assumption."""
    import fcntl

    from coleman.corpus.materialization import validate_source_storage

    validate_source_storage(path)
    inherited, current = os.fstat(descriptor), path.stat()
    if (inherited.st_dev, inherited.st_ino) != (current.st_dev, current.st_ino):
        raise ValueError("Fixed workflow worker did not inherit its actual retained lock")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    main()
