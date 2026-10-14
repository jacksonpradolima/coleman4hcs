"""Bounded sequential benchmark execution through native decision services, without transport or I/O."""

from __future__ import annotations

import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, NoReturn, cast, overload

from pydantic import Field

from coleman.allocation import AllocationResult, select_allocation
from coleman.benchmarks.contracts import AllocationCase, BenchmarkSuite, CoverageCase, verify_termination
from coleman.benchmarks.gates import check_allocation, check_coverage, reference
from coleman.benchmarks.report import (
    BenchmarkCheckpoint,
    BenchmarkEnvironment,
    BenchmarkFailure,
    BenchmarkInvocation,
    BenchmarkObservation,
    BenchmarkReport,
    collect_environment,
    completed_report,
    verify_failures,
    verify_observations,
    work_unit,
)
from coleman.selection import CoverageResult, solve_coverage
from coleman.selection.contracts import Immutable
from coleman.selection.resources import Method

NativeResult = CoverageResult | AllocationResult
Case = CoverageCase | AllocationCase
_TRACE_LOCK = threading.Lock()


# Positional callable injection is for local adapters/tests; no import names or transport dispatch.
type SolverAdapter = Callable[[Case, str, int], NativeResult]


@overload
def execute_native(case: CoverageCase, method: str, seed: int) -> CoverageResult:  # noqa: D418
    """Return native coverage evidence; see the implementation for execution invariants."""


@overload
def execute_native(case: AllocationCase, method: str, seed: int) -> AllocationResult:  # noqa: D418
    """Return native allocation evidence; see the implementation for execution invariants."""


@overload
def execute_native(case: Case, method: str, seed: int) -> NativeResult:  # noqa: D418
    """Preserve the native result kind when the input kind is known only at runtime."""


def execute_native(case: Case, method: str, seed: int) -> NativeResult:
    """Call native bounded decision services; current deterministic solvers consume no RNG.

    Seed is retained by the benchmark invocation, not converted into an observation
    or used to change the finite problem. Unknown methods fail before solver work.
    """
    if method not in case.methods or seed < 0:
        raise ValueError("Undeclared benchmark method or invalid seed label")
    if isinstance(case, CoverageCase):
        return solve_coverage(case.problem, method=cast(Method, method), limits=case.limits)
    return select_allocation(case.problem, limits=case.limits)


@dataclass(frozen=True)
class _Measured:
    """Complete measurement boundary, retaining exceptions for explicit fail-fast reporting."""

    value: NativeResult | None
    error: Exception | KeyboardInterrupt | None
    elapsed_ns: int
    peak_bytes: int


def _measure(call: Callable[[], NativeResult]) -> _Measured:
    """Time one native call and restore tracing on every exit, rejecting overlapping instrumentation.

    A process-global lock protects the tracing lifecycle. Unrelated threads can
    still allocate while measured; results disclose process-wide Python allocator
    scope. Existing tracing is never stopped/reset by this API. Exceptions are
    captured only to record failed attempts and are re-raised with their cause by
    the caller; no failed invocation can become a successful mathematical row.
    """
    # A blocking context manager would wait without an admission deadline.
    if not _TRACE_LOCK.acquire(blocking=False):  # pylint: disable=consider-using-with
        raise RuntimeError("Another benchmark measurement is active")
    owned = False
    try:
        if tracemalloc.is_tracing():
            raise RuntimeError("Benchmark requires tracing to be initially inactive")
        tracemalloc.start()
        owned = True
        started = time.perf_counter_ns()
        value = None
        error = None
        try:
            value = call()
        # Arbitrary adapter errors must retain measured failure evidence and are re-raised below.
        # pylint: disable-next=broad-exception-caught
        except (Exception, KeyboardInterrupt) as exc:
            error = exc
        elapsed = time.perf_counter_ns() - started
        _, peak = tracemalloc.get_traced_memory()
        return _Measured(value, error, elapsed, peak)
    finally:
        if owned:
            tracemalloc.stop()
        _TRACE_LOCK.release()


class BenchmarkRunError(RuntimeError):
    """Fail-fast execution/gate error carrying an auditable failed report and resumable prefix.

    The original exception is chained, not swallowed. Persist report/checkpoint
    through the existing artifact/checkpoint infrastructure at the application
    boundary; core execution performs no filesystem or network I/O.
    """

    def __init__(self, report: BenchmarkReport) -> None:
        """Retain typed failed evidence; never represent a broken solver as a completed suite."""
        super().__init__("Benchmark failed; inspect the typed report and original exception cause")
        self.report = report


def _observation(case: Case, invocation: BenchmarkInvocation, measured: _Measured) -> BenchmarkObservation:
    """Run independent mathematical gates before admitting a complete native artifact."""
    result = measured.value
    if isinstance(case, CoverageCase) and isinstance(result, CoverageResult):
        if result.method != invocation.method:
            raise ValueError("Adapter changed the requested coverage method")
        summary = check_coverage(case, result)
    elif isinstance(case, AllocationCase) and isinstance(result, AllocationResult):
        summary = check_allocation(case, result)
    else:
        raise TypeError("Adapter result kind differs from its benchmark case")
    return BenchmarkObservation(
        **invocation.model_dump(),
        work_used=result.work_used,
        work_unit=work_unit(invocation.method),
        elapsed_ns=measured.elapsed_ns,
        python_peak_bytes=measured.peak_bytes,
        mathematics=result.to_json(),
        mathematics_fingerprint=result.fingerprint,
        summary=summary,
    )


def _failure(invocation, phase, error, measured=None, *, attempt_index=0) -> BenchmarkFailure:
    """Retain failure identity and actual measurements without copying exception text into artifacts."""
    return BenchmarkFailure(
        **invocation.model_dump(),
        phase=phase,
        attempt_index=attempt_index,
        finished_at=datetime.now(UTC),
        work_unit=work_unit(invocation.method),
        work_used=0 if phase == "instrumentation" else None,
        error_type=type(error).__name__,
        elapsed_ns=None if measured is None else measured.elapsed_ns,
        python_peak_bytes=None if measured is None else measured.peak_bytes,
    )


class BenchmarkExecution(Immutable):
    """Invocation chunk limit and explicit environment metadata, independent of solver work limits.

    Environment defaults to local installed metadata at API entry. A chunk size of
    one permits durable native checkpoints between complete observations. Changing
    chunk size does not change a frozen suite's mathematical protocol or work caps.
    """

    max_new_invocations: Annotated[int, Field(strict=True, ge=0, le=128)] = 128
    environment: BenchmarkEnvironment | None = None


def _raise_failure(report: BenchmarkReport, cause: Exception) -> NoReturn:
    """Raise typed failed evidence with the exact ordinary exception as its explicit cause."""
    raise BenchmarkRunError(report) from cause


@dataclass
class _Session:
    """Invocation-local mutable progress over immutable source, environment and adapter."""

    suite: BenchmarkSuite
    environment: BenchmarkEnvironment
    adapter: SolverAdapter
    observations: list[BenchmarkObservation]
    prior_failures: tuple[BenchmarkFailure, ...]

    def report(self, status, failure=None) -> BenchmarkReport:
        """Keep all previous failed attempts when emitting a new complete or partial report."""
        return completed_report(self.suite, self.environment, self, status, failure)


def _one(case: Case, invocation: BenchmarkInvocation, session: _Session):
    """Fail fast on solver/instrumentation/math errors, preserving the verified prefix and failed slot."""
    try:
        measured = _measure(lambda: session.adapter(case, invocation.method, invocation.seed))
    except RuntimeError as exc:
        failed = _failure(invocation, "instrumentation", exc, attempt_index=len(session.prior_failures))
        _raise_failure(session.report("failed", failed), exc)
    error = measured.error
    if isinstance(error, KeyboardInterrupt):
        failed = _failure(invocation, "interrupted", error, measured, attempt_index=len(session.prior_failures))
        return session.report("interrupted", failed)
    if isinstance(error, Exception):
        failed = _failure(invocation, "solver", error, measured, attempt_index=len(session.prior_failures))
        _raise_failure(session.report("failed", failed), error)
    try:
        interrupted = _interrupted(case, invocation, measured, len(session.prior_failures))
        if interrupted is not None:
            return session.report("interrupted", interrupted)
        row = _observation(case, invocation, measured)
        verify_observations(session.suite, (*session.observations, row))
    except (ValueError, TypeError) as exc:
        failed = _failure(invocation, "verification", exc, measured, attempt_index=len(session.prior_failures))
        _raise_failure(session.report("failed", failed), exc)
    return row


def _interrupted(case: Case, invocation: BenchmarkInvocation, measured: _Measured, attempt: int):
    """Retain native abort counters only after binding their kind/source/limits and replaying validation."""
    native = measured.value
    if not isinstance(native, (CoverageResult, AllocationResult)):
        return None
    verify_termination(native.termination_reason)
    if native.termination_reason not in {"interrupted", "cancelled"}:
        return None
    if isinstance(case, CoverageCase) != isinstance(native, CoverageResult):
        raise TypeError("Interrupted adapter result kind differs from its case")
    if native.problem != case.problem or native.limits != case.limits:
        raise ValueError("Interrupted adapter changed the frozen problem or limits")
    if isinstance(native, CoverageResult) and native.method != invocation.method:
        raise ValueError("Interrupted adapter changed its requested method")
    if isinstance(native, AllocationResult) and native.checkpoint_input is not None:
        raise ValueError("Interrupted allocation reused outside cached work")
    type(native).model_validate(native)
    return BenchmarkFailure(
        **invocation.model_dump(),
        phase="interrupted",
        attempt_index=attempt,
        finished_at=datetime.now(UTC),
        work_used=native.work_used,
        work_unit=work_unit(invocation.method),
        error_type="native_" + native.termination_reason,
        elapsed_ns=measured.elapsed_ns,
        python_peak_bytes=measured.peak_bytes,
    )


def _invocations(suite: BenchmarkSuite):
    """Yield the fixed ordered plan; seed occurrences are never collapsed into one run."""
    for case in suite.cases:
        for method in case.methods:
            for seed_index, seed in enumerate(suite.seeds):
                yield case, BenchmarkInvocation(case_id=case.case_id, method=method, seed_index=seed_index, seed=seed)


def _session(suite, execution, checkpoint, adapter) -> _Session:
    """Validate full references and checkpoint integrity before any new timed solver call."""
    environment = BenchmarkEnvironment.model_validate(execution.environment or collect_environment())
    for case in suite.cases:
        reference(case)
    observations = []
    failures = ()
    if checkpoint is not None:
        checkpoint = BenchmarkCheckpoint.model_validate(checkpoint)
        if checkpoint.suite_fingerprint != suite.fingerprint or checkpoint.environment != environment:
            raise ValueError("Benchmark checkpoint differs from frozen suite/environment")
        verify_observations(suite, checkpoint.observations)
        verify_failures(suite, checkpoint.observations, checkpoint.prior_failures)
        if any(failure.finished_at > datetime.now(UTC) for failure in checkpoint.prior_failures):
            raise ValueError("Benchmark checkpoint contains an unavailable future failed attempt")
        observations.extend(checkpoint.observations)
        failures = checkpoint.prior_failures
    return _Session(suite, environment, adapter, observations, failures)


def run_benchmarks(
    suite: BenchmarkSuite,
    *,
    execution: BenchmarkExecution | None = None,
    checkpoint: BenchmarkCheckpoint | None = None,
    cancelled: Callable[[], bool] | None = None,
    adapter: SolverAdapter = execute_native,
) -> BenchmarkReport:
    """Run bounded frozen cases, retaining exact replay and explicit failure/interruption.

    References/checkpoints are verified before solver work. At most 128 calls run;
    no correctness gate uses wall-clock thresholds. A chunk size of one permits
    saving report.checkpoint() through existing CheckpointStore between calls.
    A killed attempt contributes no completed row. Resume requires identical source
    and environment metadata; cached timings are never counted as new observations.
    Solver/gate/instrumentation errors raise BenchmarkRunError with failed evidence
    and the original cause. Cancellation is polled between calls, including cached
    rows; native solver work caps apply inside a call. No transport or I/O occurs.
    """
    suite = BenchmarkSuite.model_validate(suite)
    execution = BenchmarkExecution.model_validate(execution or BenchmarkExecution())
    session = _session(suite, execution, checkpoint, adapter)
    cached = len(session.observations)
    for index, (case, invocation) in enumerate(_invocations(suite)):
        try:
            requested = cancelled is not None and cancelled()
        except KeyboardInterrupt:
            return session.report("interrupted")
        if requested:
            return session.report("interrupted")
        if index < cached:
            continue
        if len(session.observations) - cached >= execution.max_new_invocations:
            return session.report("interrupted")
        if len(session.prior_failures) >= 128:
            raise ValueError("Benchmark failed-attempt history exhausted before another solver call")
        row = _one(case, invocation, session)
        if isinstance(row, BenchmarkReport):
            return row
        session.observations.append(row)
    return session.report("complete")
