"""Bounded local command service with trusted identity and atomic native result receipts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from coleman.application.contracts import (
    ApplicationError,
    Caller,
    Capability,
    Discovery,
    Handle,
    JobView,
    Limits,
    ResultBundle,
    SafeId,
    ServerPolicy,
    Submit,
)
from coleman.application.registry import Configuration
from coleman.application.worker import WorkItem
from coleman.artifacts import ArtifactWriter
from coleman.results.domain import fingerprint
from coleman.selection import CoverageProblem

TERMINAL = frozenset({"completed", "failed", "cancelled", "expired"})


@dataclass
class _Job:
    """Private mutable lifecycle state, accessed only while holding the service condition."""

    owner: str
    request_digest: str
    view: JobView
    directory: Path
    admitted: float
    configuration: Configuration | None = None
    process: subprocess.Popen | None = None
    finished: float | None = None


class ApplicationService:
    """One finite local session; public operations are synchronized and caller-isolated.

    Only host-approved immutable configurations execute. A fixed Python module
    calls native services in a supervised process: no shell or request-selected
    imports. Deadline includes queue/startup; termination may take OS scheduling
    time. Memory/disk are sampled guards with documented overshoot, not a sandbox.

    Result publication and cancellation race under one lock: first terminal state
    wins. There are no partial application results. Expiry removes local bytes,
    retaining bounded identity tombstones. Close cancels work and removes session
    files; restart never resumes jobs. Explicit portable exports survive restart.
    Crashed session directories require trusted operator cleanup; never auto-resume
    an orphan result. Use this object as a context manager to guarantee cleanup.
    """

    def __init__(
        self, configurations: tuple[Configuration, ...], *, root: Path, policy: ServerPolicy | None = None
    ) -> None:
        """Snapshot at most 32 unique approved configurations and start one bounded supervisor.

        The trusted local root is host configuration, never a request parameter.
        Construction creates a private unique directory; filesystem errors propagate.
        """
        if not 1 <= len(configurations) <= 32:
            raise ValueError("Expected one to 32 registered configurations")
        approved = tuple(Configuration.model_validate(item) for item in configurations)
        self._registry = {item.spec_id: item for item in approved}
        if len(self._registry) != len(approved):
            raise ValueError("Duplicate approved spec_id")
        self.policy = ServerPolicy.model_validate(policy or ServerPolicy())
        root.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix="coleman-session-", dir=root))
        self._jobs: dict[str, _Job] = {}
        self._requests: dict[tuple[str, str], str] = {}
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._thread = threading.Thread(target=self._supervise, name="coleman-supervisor", daemon=True)
        self._thread.start()

    def __enter__(self) -> ApplicationService:
        """Return this owned local service."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Cancel remaining workers and remove all session-owned files."""
        self.close()

    def close(self) -> None:
        """Idempotently stop admission and reap all children before deleting local bytes."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for job in self._jobs.values():
                if job.view.state not in TERMINAL:
                    self._finish(job, "cancelled")
            self._condition.notify_all()
        self._thread.join()
        shutil.rmtree(self._directory)

    def capabilities(self) -> Discovery:
        """Return bounded public discovery; no native paths, imports or configuration secrets."""
        return Discovery(
            schema_version=1,
            specifications=tuple(
                Capability(
                    spec_id=item.spec_id,
                    kind=item.kind,
                    configuration_fingerprint=item.fingerprint,
                    parameters=("budget",) if item.kind == "coverage" else (),
                    budget_semantics=(
                        item.problem.constraints.budget.semantics if isinstance(item.problem, CoverageProblem) else None
                    ),
                )
                for item in self._registry.values()
            ),
            policy=self.policy,
        )

    def _open(self) -> None:
        """Fail closed after shutdown, including duplicate submissions."""
        if self._closed:
            raise ApplicationError("unavailable_result", "Application session is closed")

    def _limits(self, requested: Limits) -> Limits:
        """Compute effective componentwise caps before request identity and admission."""
        return Limits.model_validate(
            {key: min(value, getattr(self.policy.caps, key)) for key, value in requested.model_dump().items()}
        )

    def _admission(self, limits: Limits) -> None:
        """Reserve worst-case retained output and bounded queue slots before side effects."""
        live = [job for job in self._jobs.values() if job.view.state != "expired"]
        active = [job for job in live if job.view.state not in TERMINAL]
        if (
            len(self._jobs) >= self.policy.max_jobs
            or len(active) >= self.policy.max_running + self.policy.max_queued
            or sum(job.view.limits.max_output_bytes for job in live) + limits.max_output_bytes
            > self.policy.max_retained_bytes
            or limits.max_memory_bytes > self.policy.max_active_memory_bytes
        ):
            raise ApplicationError("resource_limit", "Session job, queue, output or memory reservation exhausted")

    def _publish(self, job_id: str, name: str, content: str) -> Path:
        """Create an owned job directory and publish once, cleaning failed admission side effects."""
        directory = self._directory / job_id
        try:
            directory.mkdir(mode=0o700)
            ArtifactWriter(directory).write_immutable(directory / name, content)
        except (OSError, ValueError) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise ApplicationError("unavailable_result", "Unable to publish local job artifact") from exc
        return directory

    def submit(self, request: Submit, *, caller: Caller) -> JobView:
        """Validate and admit once; identical retries reuse the job, changed requests fail.

        No job starts before full input and native size validation. Limits are
        capped by policy. Duplicate requests remain tombstoned after expiry;
        starting new work then requires a new request_id or a new local session.
        """
        try:
            request = Submit.model_validate(request)
            caller = Caller.model_validate(caller)
            if len(request.to_json().encode("utf-8")) > 16384:
                raise ValueError("Command exceeds 16 KiB")
        except (ValueError, TypeError) as exc:
            raise ApplicationError("invalid_input", "Invalid submit contract") from exc
        with self._condition:
            self._open()
            self._expire()
            configuration = self._registry.get(request.spec_id)
            if configuration is None:
                raise ApplicationError("unknown_spec", "No approved configuration has that spec_id")
            limits = self._limits(request.limits)
            try:
                configuration = configuration.resolve(request.parameters, limits)
            except (ValidationError, ValueError) as exc:
                raise ApplicationError("invalid_input", "Invalid approved parameter override") from exc
            digest = fingerprint(
                {"request": request.model_dump(mode="json"), "configuration": configuration.fingerprint}
            )
            key = (caller.subject, request.request_id)
            if key in self._requests:
                job = self._jobs[self._requests[key]]
                if job.request_digest != digest:
                    raise ApplicationError("invalid_input", "request_id already binds a different command")
                return job.view
            self._admission(limits)
            job_id = uuid.uuid4().hex
            item = WorkItem(configuration=configuration, limits=limits)
            directory = self._publish(job_id, "request.json", item.model_dump_json())
            view = JobView(
                schema_version=1,
                job_id=job_id,
                request_id=request.request_id,
                spec_id=request.spec_id,
                state="queued",
                limits=limits,
            )
            self._jobs[job_id] = _Job(caller.subject, digest, view, directory, time.monotonic(), configuration)
            self._requests[key] = job_id
            self._condition.notify_all()
            return view

    def _access(self, job_id: str, caller: Caller) -> _Job:
        """Authenticate using host context before exposing lifecycle or checking digests."""
        self._open()
        try:
            TypeAdapter(SafeId).validate_python(job_id)
        except ValueError as exc:
            raise ApplicationError("invalid_input", "Invalid job identity") from exc
        caller = Caller.model_validate(caller)
        job = self._jobs.get(job_id)
        if job is None:
            raise ApplicationError("not_found", "Unknown job in this session")
        if job.owner != caller.subject:
            raise ApplicationError("forbidden", "Job belongs to a different caller")
        self._expire()
        return job

    def status(self, job_id: str, *, caller: Caller) -> JobView:
        """Inspect only the caller-owned job, including retained expiry tombstones."""
        with self._condition:
            return self._access(job_id, caller).view

    def cancel(self, job_id: str, *, caller: Caller) -> JobView:
        """Cancel queued/running work; terminal retries preserve the first winning state."""
        with self._condition:
            job = self._access(job_id, caller)
            if job.view.state not in TERMINAL:
                self._finish(job, "cancelled")
                self._condition.notify_all()
            return job.view

    def result(self, handle: Handle, *, caller: Caller) -> ResultBundle:
        """Read schema-validated canonical bytes only for the owning caller and exact receipt."""
        try:
            handle = Handle.model_validate(handle)
        except ValueError as exc:
            raise ApplicationError("invalid_input", "Invalid result handle") from exc
        with self._condition:
            job = self._access(handle.job_id, caller)
            if job.view.state == "expired":
                raise ApplicationError("expired", "Result retention expired")
            if job.view.state != "completed":
                raise ApplicationError("unavailable_result", "Job has no completed result")
            if job.view.handle != handle:
                raise ApplicationError("invalid_input", "Result digest does not match the issued handle")
            try:
                content = self._read(job)
                bundle = ResultBundle.from_json(content)
                if bundle.fingerprint != handle.digest:
                    raise ValueError("Result integrity mismatch")
                return bundle
            except (ValueError, OSError, RecursionError) as exc:
                raise ApplicationError("unavailable_result", "Stored result is unavailable or corrupted") from exc

    def export_result(self, handle: Handle, *, caller: Caller) -> str:
        """Export canonical portable JSON; no server filesystem path is returned."""
        return self.result(handle, caller=caller).to_json()

    def import_result(self, payload: str, *, caller: Caller) -> JobView:
        """Validate a bounded portable bundle and rebind a fresh receipt to this caller.

        This imports evidence, never executes or trusts its producer. Each import
        is explicit and allocates a new job; use the returned handle for retries.
        Future versions, tampered proofs/files and oversized inputs fail closed.
        """
        caller = Caller.model_validate(caller)
        if not isinstance(payload, str):
            raise ApplicationError("invalid_input", "Portable input must be JSON text")
        if len(payload.encode("utf-8")) > self.policy.caps.max_output_bytes:
            raise ApplicationError("resource_limit", "Portable input exceeds server output cap")
        try:
            bundle = ResultBundle.from_json(payload)
            content = bundle.to_json()
        except (ValueError, TypeError, RecursionError) as exc:
            raise ApplicationError("invalid_input", "Invalid portable result contract") from exc
        with self._condition:
            self._open()
            self._expire()
            limits = self.policy.caps
            if len(content.encode("utf-8")) > limits.max_output_bytes:
                raise ApplicationError("resource_limit", "Canonical portable input exceeds output cap")
            self._admission(limits)
            job_id = uuid.uuid4().hex
            directory = self._publish(job_id, "result.json", content)
            handle = Handle(schema_version=1, job_id=job_id, digest=bundle.fingerprint)
            view = JobView(
                schema_version=1,
                job_id=job_id,
                request_id=job_id,
                spec_id=bundle.spec_id,
                state="completed",
                limits=limits,
                handle=handle,
            )
            now = time.monotonic()
            self._jobs[job_id] = _Job(caller.subject, bundle.fingerprint, view, directory, now, finished=now)
            return view

    def wait(self, job_id: str, *, caller: Caller) -> JobView:
        """Wait locally until terminal, bounded by the job's already admitted deadline."""
        with self._condition:
            while True:
                job = self._access(job_id, caller)
                if job.view.state in TERMINAL:
                    return job.view
                self._condition.wait(timeout=0.05)

    def _read(self, job: _Job) -> bytes:
        """Read at most the admitted result bytes before schema/proof validation."""
        with (job.directory / "result.json").open("rb") as stream:
            content = stream.read(job.view.limits.max_output_bytes + 1)
        if len(content) > job.view.limits.max_output_bytes:
            raise ValueError("Worker output exceeds admitted bytes")
        return content

    def _finish(self, job: _Job, state: str, error: str | None = None) -> None:
        """Reap children before committing a terminal state; uncommitted bytes remain inaccessible."""
        if job.process is not None:
            if job.process.poll() is None:
                job.process.terminate()
                try:
                    job.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    job.process.kill()
            job.process.wait()
            job.process = None
        job.view = JobView.model_validate({**job.view.model_dump(), "state": state, "error": error})
        job.finished = time.monotonic()
        if state != "completed":
            shutil.rmtree(job.directory, ignore_errors=True)

    def _expire(self) -> None:
        """Delete retained bytes while preserving bounded request/job tombstones."""
        now = time.monotonic()
        for job in self._jobs.values():
            if (
                job.finished is not None
                and job.view.state != "expired"
                and now - job.finished >= self.policy.retention_seconds
            ):
                self._finish(job, "expired", "expired")

    def _collect(self, job: _Job, code: int) -> None:
        """Validate a stopped worker against its approved source before issuing a receipt."""
        if code != 0:
            self._finish(job, "failed", "resource_limit" if code == 74 else "unavailable_result")
            return
        try:
            bundle = ResultBundle.from_json(self._read(job))
            if job.configuration is None or not job.configuration.matches_result(bundle):
                raise ValueError("Worker configuration mismatch")
            if time.monotonic() - job.admitted >= job.view.limits.deadline_seconds:
                self._finish(job, "expired", "expired")
                return
            handle = Handle(schema_version=1, job_id=job.view.job_id, digest=bundle.fingerprint)
            job.view = JobView.model_validate(
                {**job.view.model_dump(), "state": "completed", "handle": handle.model_dump()}
            )
            self._finish(job, "completed")
        except (ValueError, OSError, RecursionError):
            self._finish(job, "failed", "unavailable_result")

    def _start_ready(self) -> None:
        """Start only admitted jobs for which concurrency and aggregate memory slots exist."""
        running = [job for job in self._jobs.values() if job.view.state == "running"]
        for job in self._jobs.values():
            if job.view.state != "queued" or len(running) >= self.policy.max_running:
                continue
            if sum(item.view.limits.max_memory_bytes for item in running) + job.view.limits.max_memory_bytes > (
                self.policy.max_active_memory_bytes
            ):
                continue
            environment = {**os.environ, "POLARS_MAX_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            try:
                job.process = subprocess.Popen(
                    [sys.executable, "-m", "coleman.application.worker"],
                    cwd=job.directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                job.view = JobView.model_validate({**job.view.model_dump(), "state": "running"})
                running.append(job)
            except OSError:
                self._finish(job, "failed", "unavailable_result")

    def _advance(self) -> None:
        """Perform one bounded supervisor iteration; deadline/cancel win before publication."""
        self._expire()
        for job in self._jobs.values():
            if job.view.state in TERMINAL:
                continue
            if time.monotonic() - job.admitted >= job.view.limits.deadline_seconds:
                self._finish(job, "expired", "expired")
            elif job.process is not None and (code := job.process.poll()) is not None:
                self._collect(job, code)
        self._start_ready()
        self._condition.notify_all()

    def _supervise(self) -> None:
        """Keep a single bounded manager thread; never create threads for pending jobs."""
        with self._condition:
            while not self._closed:
                self._advance()
                self._condition.wait(timeout=0.02)
