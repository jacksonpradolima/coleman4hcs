"""Versioned benchmark observations, independent replay and deterministic mathematical identities."""

from __future__ import annotations

import platform
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from importlib.metadata import version
from typing import Annotated, Literal, Protocol, Self

from pydantic import BeforeValidator, Field, model_validator

from coleman.allocation import AllocationResult
from coleman.benchmarks.contracts import BenchmarkSuite, CoverageCase
from coleman.benchmarks.gates import GateSummary, check_allocation, check_coverage, same_mathematics
from coleman.results.domain import Fingerprint, Identifier, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection import CoverageResult
from coleman.selection.contracts import Immutable


def _cell(value) -> str:
    """Escape report text without changing source identity, units or mathematical values."""
    text = re.sub(r"([\\`*_{}\[\]()#+.!|])", r"\\\1", escape(str(value)))
    return text.replace("\n", " ").replace("\r", " ")


def _availability(value):
    """Require explicit timezone-aware ISO text/datetimes; epoch units are never inferred."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Benchmark availability requires an explicit timezone-aware datetime")
    return value.astimezone(UTC)


ExplicitDatetime = Annotated[datetime, BeforeValidator(_availability)]


def work_unit(method: str) -> Literal["allocation_action_attempt", "coverage_candidate_examination_or_node_pop"]:
    """Name the declared counter unit; equal counts are not equal CPU budgets across methods."""
    return (
        "allocation_action_attempt"
        if method == "enumerate_allocation_v1"
        else "coverage_candidate_examination_or_node_pop"
    )


class BenchmarkEnvironment(Immutable):
    """Recorded software/platform identity; unknown source/dirty state remains explicitly null.

    This is reproducibility metadata, not hardware calibration or authentication.
    Package versions are collected from the installed distribution, never inferred
    from a source directory. Source commit must be supplied by the caller.
    """

    python: str
    platform: str
    machine: str | None
    processor: str | None
    coleman_version: str
    pydantic_version: str
    source_commit: Annotated[str, Field(pattern="^[0-9a-f]{40}$")] | None = None
    source_dirty: Annotated[bool, Field(strict=True)] | None = None


def collect_environment(*, source_commit: str | None = None, source_dirty: bool | None = None) -> BenchmarkEnvironment:
    """Collect local metadata without subprocess, network, directory assumptions or solver work."""
    return BenchmarkEnvironment(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine() or None,
        processor=platform.processor() or None,
        coleman_version=version("coleman"),
        pydantic_version=version("pydantic"),
        source_commit=source_commit,
        source_dirty=source_dirty,
    )


class BenchmarkInvocation(Immutable):
    """One declared case/method/seed occurrence; repeated labels remain separate invocations."""

    case_id: Identifier
    method: Literal["greedy_density_v1", "branch_bound_v1", "enumerate_allocation_v1"]
    seed_index: Annotated[int, Field(strict=True, ge=0)]
    seed: Annotated[int, Field(strict=True, ge=0)]


class BenchmarkObservation(BenchmarkInvocation):
    """One complete invocation with observational runtime and Python allocator peak.

    Runtime includes production validation/result construction, excludes independent
    reference gates and serialization. Peak is process-wide tracemalloc bytes,
    not RSS/total native memory or an isolated per-thread allocation count. Complete
    canonical mathematics retains proofs, forecasts, source, work and statuses.
    """

    work_used: Annotated[int, Field(strict=True, ge=0)]
    work_unit: Literal["allocation_action_attempt", "coverage_candidate_examination_or_node_pop"]
    elapsed_ns: Annotated[int, Field(strict=True, ge=0)]
    python_peak_bytes: Annotated[int, Field(strict=True, ge=0)]
    mathematics: Annotated[str, Field(max_length=4194304)]
    mathematics_fingerprint: Fingerprint
    summary: GateSummary


class BenchmarkFailure(BenchmarkInvocation):
    """A failed or aborted attempt, not an omitted run or a verified mathematical observation.

    Exception messages/tracebacks are not persisted; the raised error retains its
    original cause locally. Instrumentation failures have unavailable measurements,
    distinct from measured zero. Failed verification never creates a successful row.
    """

    attempt_index: Annotated[int, Field(strict=True, ge=0)]
    finished_at: ExplicitDatetime
    work_used: Annotated[int, Field(strict=True, ge=0)] | None
    work_unit: Literal["allocation_action_attempt", "coverage_candidate_examination_or_node_pop"]
    phase: Literal["solver", "verification", "instrumentation", "interrupted"]
    error_type: Identifier
    elapsed_ns: Annotated[int, Field(strict=True, ge=0)] | None
    python_peak_bytes: Annotated[int, Field(strict=True, ge=0)] | None

    @model_validator(mode="after")
    def _measurements(self) -> Self:
        """Keep missing instrumentation measurements paired rather than inventing zeros."""
        if (self.elapsed_ns is None) != (self.python_peak_bytes is None):
            raise ValueError("Failure measurements must both be present or unavailable")
        if self.phase == "instrumentation" and (self.work_used != 0 or self.elapsed_ns is not None):
            raise ValueError("Rejected instrumentation must retain zero solver work and unavailable measurements")
        return self


def verify_observation(suite: BenchmarkSuite, observation: BenchmarkObservation) -> None:
    """Rebind a completed observation to its case/seed/method and independently check all math."""
    cases = {case.case_id: case for case in suite.cases}
    case = cases.get(observation.case_id)
    if case is None or observation.method not in case.methods:
        raise ValueError("Benchmark observation has unknown case or method")
    if observation.seed_index >= len(suite.seeds) or suite.seeds[observation.seed_index] != observation.seed:
        raise ValueError("Benchmark observation changed its declared seed occurrence")
    result: CoverageResult | AllocationResult
    if isinstance(case, CoverageCase):
        result = CoverageResult.from_json(observation.mathematics)
        if result.method != observation.method:
            raise ValueError("Coverage observation method differs from its mathematical artifact")
        summary = check_coverage(case, result)
    else:
        result = AllocationResult.from_json(observation.mathematics)
        summary = check_allocation(case, result)
    if observation.mathematics != result.to_json() or observation.mathematics_fingerprint != result.fingerprint:
        raise ValueError("Benchmark mathematical artifact is noncanonical or has a different digest")
    if observation.summary != summary:
        raise ValueError("Benchmark summary differs from independently verified mathematics")
    if observation.work_used != result.work_used or observation.work_unit != work_unit(observation.method):
        raise ValueError("Benchmark counter or work unit differs from native mathematical evidence")


def verify_observations(suite: BenchmarkSuite, observations: tuple[BenchmarkObservation, ...]) -> None:
    """Check an ordered complete prefix and repeated labels; never silently drop missing invocations."""
    plan = [
        (case.case_id, method, index)
        for case in suite.cases
        for method in case.methods
        for index in range(len(suite.seeds))
    ]
    if len(observations) > len(plan):
        raise ValueError("Benchmark contains excess or duplicate invocations")
    repeats = {}
    for observation, expected in zip(observations, plan, strict=False):
        if (observation.case_id, observation.method, observation.seed_index) != expected:
            raise ValueError("Benchmark progress is not the declared invocation prefix")
        verify_observation(suite, observation)
        key = (observation.case_id, observation.method, observation.seed)
        previous = repeats.get(key)
        if previous is not None:
            same_mathematics(previous, observation.mathematics)
        repeats[key] = observation.mathematics


def verify_failures(suite, observations, failures) -> None:
    """Bind ordered failed attempts to already reached slots without erasing retry overhead."""
    plan = [
        (case.case_id, method, index)
        for case in suite.cases
        for method in case.methods
        for index in range(len(suite.seeds))
    ]
    previous = -1
    for index, failure in enumerate(failures):
        key = (failure.case_id, failure.method, failure.seed_index)
        if key not in plan or failure.attempt_index != index:
            raise ValueError("Failure history has unknown or duplicate attempt identity")
        position = plan.index(key)
        if position < previous or position > len(observations):
            raise ValueError("Failure history skipped or reversed the reached invocation prefix")
        if failure.seed != suite.seeds[failure.seed_index] or failure.work_unit != work_unit(failure.method):
            raise ValueError("Failure history changed its seed or work unit")
        previous = position


class BenchmarkCheckpoint(VersionedContract):
    """Complete observation prefix in native CheckpointPayload.extra, validated on resume.

    No partially timed invocation is retained. Cached measurements are historical
    observations, not new independent samples, and are replay-checked before use.
    Source/protocol changes reject the checkpoint before any solver starts.
    """

    suite_fingerprint: Fingerprint
    environment: BenchmarkEnvironment
    prior_failures: Annotated[tuple[BenchmarkFailure, ...], Field(max_length=128)] = ()
    observations: Annotated[tuple[BenchmarkObservation, ...], Field(max_length=128)]


class BenchmarkReport(VersionedContract):
    """Self-contained correctness/performance evidence for one frozen finite suite.

    Complete means every declared invocation is present, not statistically conclusive.
    Interrupted reports retain a verified prefix with status=interrupted. Repeated
    seeds establish deterministic artifact replay only, never independent units,
    empirical efficacy or calibrated uncertainty. Metadata/timing are excluded from
    mathematical_identity but included in the content fingerprint. Unknown schema
    versions fail closed; no native artifact is rewritten or migrated.
    """

    suite: BenchmarkSuite
    environment: BenchmarkEnvironment
    observations: Annotated[tuple[BenchmarkObservation, ...], Field(max_length=128)]
    status: Literal["complete", "interrupted", "failed"]
    prior_failures: Annotated[tuple[BenchmarkFailure, ...], Field(max_length=128)] = ()
    failure: BenchmarkFailure | None = None
    completed_at: ExplicitDatetime
    reproducibility: Literal["verified_repeated_labels", "untested"]

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Recompute prefix integrity, mathematical gates and claimed completeness/reproducibility."""
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("Benchmark availability time must include a timezone")
        verify_observations(self.suite, self.observations)
        verify_failures(self.suite, self.observations, self.prior_failures)
        count = sum(len(case.methods) for case in self.suite.cases) * len(self.suite.seeds)
        if self.status == "complete" and len(self.observations) != count:
            raise ValueError("Benchmark falsely claims complete execution")
        if self.status == "failed" and self.failure is None:
            raise ValueError("Failed benchmark must retain an explicit failed invocation")
        if self.failure is not None:
            self._failure_position()
        attempts = (*self.prior_failures, *((self.failure,) if self.failure else ()))
        if any(attempt.finished_at > self.completed_at for attempt in attempts):
            raise ValueError("Report availability predates a failed/aborted attempt")
        expected = _reproducibility(self.suite, self.observations)
        if self.reproducibility != expected:
            raise ValueError("Benchmark reproducibility requires completed repeated labels")
        return self

    def _failure_position(self) -> None:
        """Associate a failed/aborted attempt with the next declared invocation, never a skipped slot."""
        plan = [
            (case.case_id, method, index)
            for case in self.suite.cases
            for method in case.methods
            for index in range(len(self.suite.seeds))
        ]
        failed = self.failure
        if failed is None or self.status == "complete" or len(self.observations) >= len(plan):
            raise ValueError("Completed benchmark cannot contain an extra failed invocation")
        if (failed.case_id, failed.method, failed.seed_index) != plan[len(self.observations)]:
            raise ValueError("Failed invocation does not follow the complete observation prefix")
        if len(self.prior_failures) >= 128:
            raise ValueError("Current failure exceeds the attempt history limit")
        if failed.attempt_index != len(self.prior_failures) or failed.work_unit != work_unit(failed.method):
            raise ValueError("Current failure changed its attempt identity or work unit")
        if failed.seed != self.suite.seeds[failed.seed_index]:
            raise ValueError("Failed invocation differs from its declared seed")
        if (failed.phase == "interrupted") != (self.status == "interrupted"):
            raise ValueError("Failure phase and benchmark termination status disagree")

    @property
    def mathematical_identity(self) -> str:
        """Hash the protocol and complete mathematical stream, excluding all measurements/metadata."""
        return fingerprint(
            {
                "suite": self.suite.fingerprint,
                "artifacts": [row.mathematics_fingerprint for row in self.observations],
            }
        )

    def checkpoint(self) -> BenchmarkCheckpoint:
        """Export verified complete progress into the existing checkpoint contract, without I/O."""
        checked = type(self).model_validate(self)
        return BenchmarkCheckpoint(
            schema_version=1,
            suite_fingerprint=checked.suite.fingerprint,
            environment=checked.environment,
            prior_failures=(*checked.prior_failures, *((checked.failure,) if checked.failure else ())),
            observations=checked.observations,
        )

    def markdown(self) -> str:
        """Render concise observed results; no timing threshold, inferential test or winner promotion."""
        checked = type(self).model_validate(self)
        lines = [
            f"# Decision benchmark: {_cell(checked.suite.benchmark_id)}",
            "",
            f"Status: {checked.status}; reproducibility: {checked.reproducibility}.",
            f"Mathematical identity: `{checked.mathematical_identity}`.",
            f"Source commit: `{checked.environment.source_commit or 'unknown'}`; "
            f"COLEMAN {_cell(checked.environment.coleman_version)}; Python {_cell(checked.environment.python)}.",
            "",
            "| Case | Method | Seed occurrence | Status / evidence | Selected IDs | Objective | Regret | Work | "
            "Runtime ns | Python peak bytes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in checked.observations:
            summary = row.summary
            ids = "unavailable" if summary.selected_ids is None else _cell(list(summary.selected_ids))
            objective = "unavailable" if summary.objective is None else str(summary.objective)
            regret = "unavailable" if summary.absolute_regret is None else str(summary.absolute_regret)
            lines.append(
                f"| {_cell(row.case_id)} | {row.method} | {row.seed_index}: {row.seed} | "
                f"{summary.status} / {summary.evidence} | {ids} | "
                f"{objective} {_cell(summary.objective_semantics.unit)} | "
                f"{regret} {_cell(summary.objective_semantics.unit)} | "
                f"{row.work_used} {_cell(row.work_unit)} | {row.elapsed_ns} | {row.python_peak_bytes} |"
            )
        for failure in (*checked.prior_failures, *((checked.failure,) if checked.failure else ())):
            lines.extend(
                [
                    "",
                    f"Failed/aborted attempt: {_cell(failure.case_id)}, {failure.method}, seed occurrence "
                    f"{failure.seed_index}: {failure.seed}; phase={failure.phase}; error={_cell(failure.error_type)}; "
                    f"work={failure.work_used} {_cell(failure.work_unit)}; "
                    f"runtime_ns={failure.elapsed_ns}; python_peak_bytes={failure.python_peak_bytes}.",
                ]
            )
        lines.extend(
            [
                "",
                "Exact rational gates use zero tolerance. Current solvers use no RNG; seeds are labels, "
                "not independent observations. Timing is observational; Python peak excludes native allocations. "
                "Equal work counters do not imply equal CPU budgets. Model optima are not evidence of CI efficacy.",
                "",
            ]
        )
        return "\n".join(lines)


def _reproducibility(suite, observations):
    """Require the entire repeated-label protocol, retaining already verified complete cached evidence."""
    count = sum(len(case.methods) for case in suite.cases) * len(suite.seeds)
    repeated = len(suite.seeds) != len(set(suite.seeds))
    return "verified_repeated_labels" if repeated and len(observations) == count else "untested"


class _Progress(Protocol):
    """Read-only typed view of invocation-local progress, without transport or storage coupling."""

    @property
    def observations(self) -> Sequence[BenchmarkObservation]:
        """Return completed immutable mathematical observations in protocol order."""
        ...

    @property
    def prior_failures(self) -> Sequence[BenchmarkFailure]:
        """Return all earlier failed/aborted attempts, retaining unknown resource costs."""
        ...


def completed_report(
    suite: BenchmarkSuite,
    environment: BenchmarkEnvironment,
    progress: _Progress,
    status: Literal["complete", "interrupted", "failed"],
    failure: BenchmarkFailure | None = None,
) -> BenchmarkReport:
    """Stamp actual post-run availability and freeze progress; time never enters mathematical equality."""
    return BenchmarkReport(
        schema_version=1,
        suite=suite,
        environment=environment,
        observations=tuple(progress.observations),
        status=status,
        failure=failure,
        prior_failures=tuple(progress.prior_failures),
        completed_at=datetime.now(UTC),
        reproducibility=_reproducibility(suite, progress.observations),
    )
