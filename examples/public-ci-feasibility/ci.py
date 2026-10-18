"""Bounded public Actions metadata study using the installed COLEMAN 4.17 SDK.

This reference collector is a feasibility example, not a production provider or
forward TCPCI study. It never triggers CI, fetches logs/artifact bodies, loads
checkpoints or derives test verdicts/runner occupancy from job conclusions.
Raw public responses can contain personal identifiers; keep outputs outside Git.
Only explicit ``--fetch`` performs anonymous HTTPS to the fixed GitHub API origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import Field, model_validator

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission_contracts import FactAxis, RealityFact
from coleman.corpus.materialization import validate_source_storage
from coleman.evaluation.protected.temporal import TemporalObservation, Timestamp
from coleman.observations import ObservationHistory, ObservationRevision
from coleman.results.domain import Fingerprint, Measurement, canonical_json, decode_json, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.spec.runtime_identity import InstalledRuntime

API = "https://api.github.com"
API_VERSION = "2022-11-28"
MAXIMUM_BYTES = 2 * 1024**2
MAXIMUM_JOBS = 100
MAXIMUM_STEPS = 256
Label = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
Counter = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
Slug = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9-]+/[A-Za-z0-9_.-]+$", max_length=160)]
Token = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]


class Request(VersionedContract):
    """One explicitly named poll; its identity is fixed before fetching future data.

    A new poll needs a new capture_id. Repeating a completed request reuses its
    first immutable receipts. The same ID is never a new independent CI build.
    Limits and API version are part of the frozen v1 profile, not hidden defaults.
    """

    repository: Slug
    run_id: Counter
    capture_id: Token
    api_version: Literal["2022-11-28"] = API_VERSION
    maximum_response_bytes: Literal[2097152] = MAXIMUM_BYTES
    maximum_jobs: Literal[100] = MAXIMUM_JOBS
    basis: Literal["public_api_metadata", "owned_fixture"] = "public_api_metadata"

    @model_validator(mode="after")
    def _repository(self) -> Self:
        """Canonicalize case-insensitive slugs and reject traversal-looking repository segments."""
        if self.repository.split("/")[1] in {".", ".."}:
            raise ValueError("A concrete public repository slug is required")
        object.__setattr__(self, "repository", self.repository.lower())
        return self


class Repository(Immutable):
    """Only provider-reported repository identity/privacy; other wire fields stay in original bytes."""

    id: Counter
    full_name: Slug
    private: Annotated[bool, Field(strict=True)]


class Run(Immutable):
    """A selected projection of run metadata; updated_at is not a field-availability clock."""

    id: Counter
    run_attempt: Counter
    head_sha: Label
    head_branch: Label | None
    event: Label
    status: Label
    conclusion: Label | None
    created_at: Timestamp
    run_started_at: Timestamp | None
    updated_at: Timestamp
    repository: Repository
    head_repository: Repository | None


class Step(Immutable):
    """Reported ordinal and clocks, without interpreting names as test/shard identities."""

    number: Annotated[int, Field(strict=True, ge=1, le=65536)]
    name: Label
    status: Label
    conclusion: Label | None
    started_at: Timestamp | None
    completed_at: Timestamp | None


class Job(Immutable):
    """Provider job/attempt identity; its elapsed envelope is distinct from occupied runner work."""

    id: Counter
    run_id: Counter
    run_attempt: Counter
    head_sha: Label
    name: Label
    status: Label
    conclusion: Label | None
    started_at: Timestamp | None
    completed_at: Timestamp | None
    labels: Annotated[tuple[Label, ...], Field(max_length=32)]
    steps: Annotated[tuple[Step, ...], Field(max_length=MAXIMUM_STEPS)]

    @model_validator(mode="after")
    def _ordering(self) -> Self:
        """Reject ambiguous ordinals/backward reported intervals and canonicalize unordered metadata."""
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("Reported job interval is reversed")
        numbers = [step.number for step in self.steps]
        if len(set(numbers)) != len(numbers) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Duplicate reported step or runner label")
        for step in self.steps:
            if step.started_at is not None and step.completed_at is not None and step.completed_at < step.started_at:
                raise ValueError("Reported step interval is reversed")
        object.__setattr__(self, "steps", tuple(sorted(self.steps, key=lambda step: step.number)))
        object.__setattr__(self, "labels", tuple(sorted(self.labels)))
        return self


class Response(VersionedContract):
    """One complete bounded response plus its actual local receipt, atomically committed together.

    body is the original UTF-8 JSON text, including whitespace/key order; its
    SHA-256 is independent of the normalized projection. First provider-side
    availability is not reported by these endpoints and is never inferred.
    """

    part: Literal["run", "jobs"]
    url: Label
    received_at: Timestamp
    selected_api_version: Label | None
    body: Annotated[str, Field(max_length=MAXIMUM_BYTES)]
    body_sha256: Fingerprint

    @model_validator(mode="after")
    def _bytes(self) -> Self:
        """Verify exact response bytes, strict JSON and the supported selected API version."""
        payload = self.body.encode("utf-8")
        if len(payload) > MAXIMUM_BYTES or "sha256:" + hashlib.sha256(payload).hexdigest() != self.body_sha256:
            raise ValueError("Captured response byte bound or digest differs")
        if self.selected_api_version not in {None, API_VERSION}:
            raise ValueError("Unsupported selected provider API version")
        canonical_json(decode_json(payload))
        return self


class Snapshot(VersionedContract):
    """Materialized public metadata and an explicitly blocked TCPCI feasibility view.

    All raw references, runtime and example bytes are provenance, not proof of
    authentic provider execution. created_at is this snapshot producer's assembly
    instant, not upstream availability or a pre-outcome policy decision seal.
    Unknown schema versions reject; old native artifacts are never migrated.
    """

    request: Request
    runtime: InstalledRuntime
    example_sha256: Fingerprint
    created_at: Timestamp
    responses: Annotated[tuple[Fingerprint, ...], Field(min_length=2, max_length=2)]
    run: Run
    jobs: Annotated[tuple[Job, ...], Field(max_length=MAXIMUM_JOBS)]
    measurements: Annotated[tuple[Measurement, ...], Field(max_length=128)]
    reality: Annotated[tuple[RealityFact, ...], Field(min_length=35, max_length=35)]
    upstream_first_available_at: None = None
    decision_sealed_at: None = None
    scope: Literal["public_metadata_feasibility_only"] = "public_metadata_feasibility_only"
    conclusion: Literal["insufficient_evidence_for_forward_tcpci"] = "insufficient_evidence_for_forward_tcpci"

    @model_validator(mode="after")
    def _correspondence(self) -> Self:
        """Rederive quantities/facts; refuse private sources, unrelated jobs and fabricated decision clocks."""
        _check_public(self.request, self.run)
        ids = [job.id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate provider job identity")
        jobs = tuple(sorted(self.jobs, key=lambda job: job.id))
        for job in jobs:
            if (job.run_id, job.run_attempt, job.head_sha) != (self.run.id, self.run.run_attempt, self.run.head_sha):
                raise ValueError("Reported job belongs to another run, attempt or source revision")
            instants = (job.started_at, job.completed_at)
            if any(value is not None and value > self.created_at for value in instants):
                raise ValueError("Provider and local UTC clocks are not comparable")
        if self.measurements != measurements(jobs) or self.reality != reality(self.request, self.run, jobs):
            raise ValueError("Snapshot quantities or reality facts differ from their declared source projection")
        object.__setattr__(self, "jobs", jobs)
        return self

    @property
    def logical_id(self) -> str:
        """Identify the source run/attempt across polls; captures/retries do not create independent builds."""
        return fingerprint(["github-actions", self.run.repository.id, self.run.id, self.run.run_attempt])

    @property
    def semantic_fingerprint(self) -> str:
        """Hash normalized source facts/quantities without receipt, runtime or acquisition noise."""
        return fingerprint(
            {
                "run": self.run.model_dump(mode="json"),
                "jobs": [job.model_dump(mode="json") for job in self.jobs],
                "measurements": [item.model_dump(mode="json") for item in self.measurements],
            }
        )

    def history(self) -> ObservationHistory:
        """Project only observed job intervals through #258, using this materialized snapshot as producer.

        Source identity is fixed by the pre-data capture request. Availability and
        first receipt refer to this local snapshot's assembly, never an invented
        upstream availability. In-progress/unknown intervals produce no interval
        observation. This is a bounded offline projection, not an appendable
        production receiver or an actual #260 shadow admission.
        """
        rows = []
        for job in self.jobs:
            measure = job_measurement(job)
            if measure.status != "observed" or job.started_at is None or job.completed_at is None:
                continue
            logical = f"repo-{self.run.repository.id}:run-{self.run.id}:attempt-{self.run.run_attempt}:job-{job.id}"
            rows.append(
                ObservationRevision(
                    schema_version=1,
                    entity_id=f"job-{job.id}",
                    received_at=self.created_at,
                    supersedes=None,
                    metadata=TemporalObservation(
                        observation_id=logical,
                        revision_id="snapshot-v1",
                        group_id=f"github-repository-{self.run.repository.id}",
                        event_at=job.started_at,
                        target_end=job.completed_at,
                        available_at=self.created_at,
                    ),
                    measurement=measure,
                )
            )
        return ObservationHistory(
            schema_version=1,
            source_id="snapshot-" + self.request.fingerprint[7:],
            source_version="job-envelope-projection-v1",
            revisions=tuple(rows),
        )


class Completion(VersionedContract):
    """Publish last: complete means captured response inventory, not a completed CI run or admitted corpus."""

    kind: Literal["public_metadata_snapshot"] = "public_metadata_snapshot"
    request_fingerprint: Fingerprint
    snapshot_fingerprint: Fingerprint
    snapshot_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)]
    responses: Annotated[tuple[Fingerprint, ...], Field(min_length=2, max_length=2)]


def _check_public(request: Request, run: Run) -> None:
    """Refuse private/unverifiable source repositories before persisting their raw response."""
    if run.id != request.run_id or run.repository.full_name.lower() != request.repository:
        raise ValueError("Provider returned another repository or run")
    if run.repository.private or run.head_repository is None or run.head_repository.private:
        raise ValueError("Only verifiably public base and head repositories are supported")


def job_measurement(job: Job) -> Measurement:
    """Return binary64 seconds for a reported completed interval; unknown/unavailable is never zero work."""
    value = None
    status = "unknown" if job.status == "completed" else "unavailable"
    if job.status == "completed" and job.started_at is not None and job.completed_at is not None:
        value = (job.completed_at - job.started_at).total_seconds()
        status = "observed"
    return Measurement(
        name=f"job_{job.id}_recorded_envelope_seconds",
        value=value,
        unit="s",
        meaning="provider-recorded-job-envelope-seconds-v1",
        direction="minimize",
        status=status,
        minimum=0.0,
    )


def measurements(jobs: tuple[Job, ...]) -> tuple[Measurement, ...]:
    """Keep observed metadata counts, uncollected physical quantities and absent decisions distinct."""
    rows = [job_measurement(job) for job in jobs]
    rows.append(
        Measurement(
            name="returned_job_metadata_count",
            value=float(len(jobs)),
            unit="1",
            meaning="provider-returned-job-metadata-count-v1",
            direction="none",
            status="observed",
            minimum=0.0,
        )
    )
    for name, unit, meaning in (
        ("expected_test_count", "1", "expected-required-test-universe-size-v1"),
        ("occupied_runner_seconds", "s", "physical-whole-runner-occupied-seconds-v1"),
        ("queue_elapsed_seconds", "s", "ci-queue-elapsed-seconds-v1"),
        ("shared_setup_seconds", "s", "ci-shared-setup-elapsed-seconds-v1"),
    ):
        rows.append(
            Measurement(
                name=name, value=None, unit=unit, meaning=meaning, direction="none", status="unavailable", minimum=0.0
            )
        )
    rows.append(
        Measurement(
            name="collector_policy_inference_seconds",
            value=None,
            unit="s",
            meaning="observer-policy-inference-elapsed-seconds-v1",
            direction="minimize",
            status="not_applicable",
            minimum=0.0,
        )
    )
    return tuple(sorted(rows, key=lambda row: row.name))


def reality(request: Request, run: Run, jobs: tuple[Job, ...]) -> tuple[RealityFact, ...]:
    """Feed the existing 35-axis vocabulary without turning metadata labels into source admission."""
    result = {
        axis: RealityFact(
            axis=axis,
            status="missing",
            values=None,
            locator="not_collected",
            limitation="This metadata snapshot does not establish this axis.",
        )
        for axis in get_args(FactAxis)
    }
    status = "synthetic" if request.basis == "owned_fixture" else "observed"
    declarations: tuple[tuple[FactAxis, tuple[str, ...], str, str], ...] = (
        (
            "source_execution_model",
            ("GitHub-Actions-reported-run-job-attempt",),
            "run.id/jobs[].id",
            "Reported scheduler identities, not test membership, physical topology or independent evidence.",
        ),
        ("triggers", (run.event,), "run.event", "Reported event label; it does not establish branch protection."),
        (
            "reruns",
            (f"reported-workflow-attempt:{run.run_attempt}",),
            "run.run_attempt",
            "Current workflow attempt only; test retries and future reruns remain unverified.",
        ),
    )
    for axis, values, locator, limitation in declarations:
        result[axis] = RealityFact(axis=axis, status=status, values=values, locator=locator, limitation=limitation)
    labels = tuple(sorted({label for job in jobs for label in job.labels}))
    if labels and len(labels) <= 32:
        result["runner_type"] = RealityFact(
            axis="runner_type",
            status=status,
            values=labels,
            locator="jobs[].labels",
            limitation="Provider runner labels, not hardware, billing or physical occupancy.",
        )
    partial = tuple(sorted({job.status for job in jobs if job.status != "completed" or job.conclusion is None}))
    if partial and len(partial) <= 32:
        result["partial_outcomes"] = RealityFact(
            axis="partial_outcomes",
            status=status,
            values=partial,
            locator="jobs[].status/conclusion",
            limitation="Incomplete job metadata, not collected partial test outcomes.",
        )
    conclusions: tuple[tuple[FactAxis, str], ...] = (("cancellation", "cancelled"), ("timeouts", "timed_out"))
    for axis, conclusion in conclusions:
        if any(job.conclusion == conclusion for job in jobs):
            result[axis] = RealityFact(
                axis=axis,
                status=status,
                values=("provider-reported-" + conclusion,),
                locator="jobs[].conclusion",
                limitation="Reported job conclusion; test-level cause/accounting is unknown.",
            )
    return tuple(result[key] for key in sorted(result))


def _fields(model: type[Immutable], value: Any) -> dict[str, Any]:
    """Select documented wire fields only at the HTTP boundary; preserve all other fields in original text."""
    if not isinstance(value, dict):
        raise ValueError("Provider JSON requires an object")
    return {name: value[name] for name in model.model_fields if name in value}


def parse_run(response: Response, request: Request) -> Run:
    """Normalize a bounded source response and verify public scope before any publication."""
    data = _fields(Run, decode_json(response.body))
    data["repository"] = Repository.model_validate(_fields(Repository, data.get("repository")))
    if data.get("head_repository") is not None:
        data["head_repository"] = Repository.model_validate(_fields(Repository, data["head_repository"]))
    run = Run.model_validate(data)
    _check_public(request, run)
    return run


def parse_jobs(response: Response, run: Run) -> tuple[Job, ...]:
    """Require the complete bounded returned page for the explicit attempt; never infer expected test membership."""
    payload = decode_json(response.body)
    if not isinstance(payload, dict) or type(payload.get("total_count")) is not int:
        raise ValueError("Provider jobs response requires its explicit total_count")
    count, raw = payload["total_count"], payload.get("jobs")
    if not isinstance(raw, list) or count != len(raw) or not 0 <= count <= MAXIMUM_JOBS:
        raise ValueError("Truncated, oversized or invalid provider job inventory")
    jobs = []
    for value in raw:
        data = _fields(Job, value)
        steps = data.get("steps")
        if not isinstance(steps, list) or len(steps) > MAXIMUM_STEPS:
            raise ValueError("Provider step inventory exceeds its bounded profile")
        data["steps"] = tuple(Step.model_validate(_fields(Step, item)) for item in steps)
        job = Job.model_validate(data)
        if (job.run_id, job.run_attempt, job.head_sha) != (run.id, run.run_attempt, run.head_sha):
            raise ValueError("Provider returned a job from another run, attempt or revision")
        jobs.append(job)
    return tuple(sorted(jobs, key=lambda item: item.id))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Require an explicitly confirmed canonical public slug rather than following metadata redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Refuse redirected requests, including a different origin; no credentials are ever supplied."""
        return None


def fetch_response(url: str, part: Literal["run", "jobs"]) -> Response:
    """Explicit anonymous public HTTPS, at most 2 MiB, 10-second socket waits and a cooperative 30-second body window.

    Only caller-derived fixed-origin endpoints are passed by capture(). Redirects,
    non-JSON responses, byte overruns and unknown API selections raise. OS/DNS and
    socket operations are not a hard wall-time sandbox. read1 bounds individual
    body reads; a socket wait can overshoot the cooperative body window. No logs,
    artifact downloads, token lookup, gh CLI, mutation or background polling occurs.
    """
    if not url.startswith(API + "/repos/"):
        raise ValueError("Metadata acquisition requires the fixed public API origin")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "Accept-Encoding": "identity",
            "User-Agent": "COLEMAN-public-feasibility-v1",
        },
    )
    started, blocks, size = time.monotonic(), [], 0
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=10) as response:
        if response.status != 200 or not response.headers.get("Content-Type", "").startswith("application/json"):
            raise ValueError("Expected a successful JSON metadata response")
        while True:
            if time.monotonic() - started > 30:
                raise TimeoutError("Public metadata body window exceeded")
            block = response.read1(min(65536, MAXIMUM_BYTES + 1 - size))
            if not block:
                break
            blocks.append(block)
            size += len(block)
            if size > MAXIMUM_BYTES:
                raise ValueError("Public metadata response exceeds 2 MiB")
        selected = response.headers.get("X-GitHub-Api-Version-Selected")
    payload = b"".join(blocks)
    return Response(
        schema_version=1,
        part=part,
        url=url,
        received_at=datetime.now(UTC),
        selected_api_version=selected,
        body=payload.decode("utf-8"),
        body_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def read_contract[Contract: VersionedContract](path: Path, contract: type[Contract]) -> Contract:
    """Read one trusted local strict JSON artifact with a 16 MiB cap; never repair, fetch or unpickle."""
    validate_source_storage(path)
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("Feasibility artifact exceeds 16 MiB")
    return contract.from_json(payload)


def _response(root: Path, part: Literal["run", "jobs"], url: str, fetch: bool) -> Response:
    """Resume a complete response/receipt checkpoint without restamping its first committed local receipt."""
    path = root / (part + ".json")
    if path.exists():
        result = read_contract(path, Response)
    else:
        if not fetch:
            raise FileNotFoundError("Missing response; explicit --fetch is required for public network acquisition")
        result = fetch_response(url, part)
    if result.part != part or result.url != url:
        raise ValueError("Captured response endpoint differs from its frozen request")
    return result


def _urls(request: Request, run: Run | None = None) -> str:
    """Build endpoints from validated slug/numeric identity, with no upstream URL execution."""
    base = f"{API}/repos/{request.repository}/actions/runs/{request.run_id}"
    return base if run is None else f"{base}/attempts/{run.run_attempt}/jobs?per_page=100"


def _validate_responses(
    request: Request, run_response: Response, jobs_response: Response
) -> tuple[Run, tuple[Job, ...]]:
    """Keep response-local receipts and source clocks separate; reject incompatible UTC instead of shifting it."""
    if (run_response.part, jobs_response.part) != ("run", "jobs"):
        raise ValueError("Captured response roles differ from the declared inventory")
    run = parse_run(run_response, request)
    if (run_response.url, jobs_response.url) != (_urls(request), _urls(request, run)):
        raise ValueError("Captured endpoints differ from the selected source attempt")
    jobs = parse_jobs(jobs_response, run)
    if jobs_response.received_at < run_response.received_at:
        raise ValueError("Local response receipts reverse")
    if max(run.created_at, run.updated_at, run.run_started_at or run.created_at) > run_response.received_at:
        raise ValueError("Provider run and local UTC clocks are not comparable")
    for job in jobs:
        instants = [job.started_at, job.completed_at]
        for step in job.steps:
            if step.started_at is not None and step.completed_at is not None and step.completed_at < step.started_at:
                raise ValueError("Reported step interval is reversed")
            instants.extend((step.started_at, step.completed_at))
        if any(value is not None and value > jobs_response.received_at for value in instants):
            raise ValueError("Provider job/step and local UTC clocks are not comparable")
    return run, jobs


def read_capture(root: Path) -> Snapshot:
    """Verify a completed local capture, original bytes and the declared v1 projection without network I/O.

    Completion certifies this bounded captured response inventory, not that CI
    completed or that all expected tests/jobs were observed. Unknown versions,
    missing/changed content, private sources or incompatible clocks fail closed.
    The recorded example/runtime are required for reproduction; metadata/digests
    alone never prove authentic upstream execution or legal reuse rights.
    """
    receipt = read_contract(root / "complete.json", Completion)
    snapshot = read_contract(root / "snapshot.json", Snapshot)
    request = read_contract(root / "request.json", Request)
    responses = tuple(read_contract(root / (part + ".json"), Response) for part in ("run", "jobs"))
    run, jobs = _validate_responses(request, *responses)
    if (snapshot.request, snapshot.run, snapshot.jobs) != (request, run, jobs):
        raise ValueError("Snapshot projection differs from its actual source bytes")
    pins = tuple(response.fingerprint for response in responses)
    if (
        (receipt.request_fingerprint, receipt.snapshot_fingerprint, receipt.responses)
        != (request.fingerprint, snapshot.fingerprint, pins)
        or snapshot.responses != pins
        or (root / "snapshot.json").stat().st_size != receipt.snapshot_bytes
    ):
        raise ValueError("Capture completion differs from its actual immutable inventory")
    if snapshot.created_at < max(response.received_at for response in responses):
        raise ValueError("Snapshot assembly precedes its actual source receipts")
    return snapshot


def capture(request: Request, output: Path, *, fetch: bool = False) -> tuple[Path, Snapshot]:
    """Capture/replay one bounded public poll through existing immutable artifacts and local process locking.

    Writes only below output/public-ci-feasibility/<request>/<runtime>/<example>.
    A completed retry reads/verifies original bytes without fetching or restamping.
    Interrupted invocations reuse committed response envelopes as checkpoints;
    complete.json publishes last. Concurrent ownership rejects, and kernel locks
    release after process death. This reuses the existing trusted-local POSIX
    lock; callers serialize their own threads. No native experiment/CI job is
    started or altered. Incompatible runtime/example bytes create another scope.
    """
    request = Request.model_validate(request)
    runtime = InstalledRuntime.capture()
    example = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root = output.resolve() / "public-ci-feasibility" / request.fingerprint[7:] / runtime.fingerprint[7:] / example[7:]
    validate_source_storage(root)
    try:
        with baseline_lock(root / "capture.lock"):
            return _capture_locked(request, runtime, example, root, fetch)
    except BaselineBusyError as error:
        raise RuntimeError("This public metadata capture is already active; retry after its owner finishes") from error


def _capture_locked(request: Request, runtime: InstalledRuntime, example: str, root: Path, fetch: bool):
    """Publish staged responses/snapshot before completion while holding the existing local ownership lock."""
    if (root / "complete.json").exists():
        snapshot = read_capture(root)
        if (snapshot.request, snapshot.runtime, snapshot.example_sha256) != (request, runtime, example):
            raise ValueError("Completed capture differs from its requested runtime or example scope")
        return root, snapshot
    writer = ArtifactWriter(root)
    writer.write_immutable(root / "request.json", request.to_json() + "\n")
    run_response = _response(root, "run", _urls(request), fetch)
    run = parse_run(run_response, request)
    writer.write_immutable(root / "run.json", run_response.to_json() + "\n")
    jobs_response = _response(root, "jobs", _urls(request, run), fetch)
    run, jobs = _validate_responses(request, run_response, jobs_response)
    writer.write_immutable(root / "jobs.json", jobs_response.to_json() + "\n")
    pins = (run_response.fingerprint, jobs_response.fingerprint)
    path = root / "snapshot.json"
    if path.exists():
        snapshot = read_contract(path, Snapshot)
        if (snapshot.request, snapshot.runtime, snapshot.example_sha256, snapshot.responses) != (
            request,
            runtime,
            example,
            pins,
        ):
            raise ValueError("Interrupted snapshot differs from its frozen capture")
    else:
        snapshot = Snapshot(
            schema_version=1,
            request=request,
            runtime=runtime,
            example_sha256=example,
            created_at=datetime.now(UTC),
            responses=pins,
            run=run,
            jobs=jobs,
            measurements=measurements(jobs),
            reality=reality(request, run, jobs),
        )
        writer.write_immutable(path, snapshot.to_json() + "\n")
    if (snapshot.run, snapshot.jobs) != (run, jobs) or snapshot.created_at < jobs_response.received_at:
        raise ValueError("Interrupted snapshot differs from its actual source projection or receipts")
    receipt = Completion(
        schema_version=1,
        request_fingerprint=request.fingerprint,
        snapshot_fingerprint=snapshot.fingerprint,
        snapshot_bytes=path.stat().st_size,
        responses=pins,
    )
    writer.write_immutable(root / "complete.json", receipt.to_json() + "\n")
    return root, read_capture(root)


def main() -> None:
    """Expose explicit fetch or offline verification; printing never uploads artifacts or creates decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Explicit anonymous read-only public metadata requests")
    args = parser.parse_args()
    request = Request(schema_version=1, repository=args.repository, run_id=args.run_id, capture_id=args.capture_id)
    root, snapshot = capture(request, args.output, fetch=args.fetch)
    print(
        json.dumps(
            {
                "capture": str(root),
                "logical_run_attempt": snapshot.logical_id,
                "semantic_fingerprint": snapshot.semantic_fingerprint,
                "reported_jobs": len(snapshot.jobs),
                "conclusion": snapshot.conclusion,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
