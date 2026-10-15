"""Private bounded process adapter invoking existing native services, never a second runner."""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import sys
import threading
from pathlib import Path

from coleman.application.contracts import ExperimentResult, Limits, PortableFile, ResultBundle
from coleman.application.registry import Configuration, NativeConfiguration
from coleman.artifacts import ArtifactWriter
from coleman.selection.contracts import Immutable


class WorkItem(Immutable):
    """Private validated host-to-worker snapshot; not accepted by public transports."""

    configuration: Configuration
    limits: Limits


def _check_resources(limits: Limits) -> None:
    """Refuse unknown/over-limit peak RSS or worker artifact bytes synchronously."""
    from coleman.telemetry.resources import ProcessResourceTracker

    peak = ProcessResourceTracker().sample().peak_rss_mib
    try:
        refused = peak is None or peak * 1048576 > limits.max_memory_bytes or _output_exceeds(limits.max_output_bytes)
    except OSError:
        refused = True
    if refused:
        os._exit(74)


def _output_exceeds(maximum_bytes: int) -> bool:
    """Bound sampled file metadata without retaining an unbounded directory listing.

    Disappearing atomic temporary files are normal; other I/O failures propagate
    to the fail-closed guard. No symlink is followed while measuring file bytes.
    """
    total = 0
    for count, path in enumerate(Path("run").rglob("*"), start=1):
        if count > 128:
            return True
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            return True
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
        if total > maximum_bytes:
            return True
    return False


def _guard(limits: Limits, stopped: threading.Event) -> None:
    """Sample own peak RSS and output bytes; exit without a completed receipt on exhaustion.

    The parent separately enforces wall-clock deadline and cancellation. Sampling
    is a resource guard for trusted native code, with possible transient overshoot.
    Unknown peak RSS fails closed rather than silently disabling the memory cap.
    """
    while not stopped.is_set():
        _check_resources(limits)
        stopped.wait(0.02)


def _experiment(configuration: NativeConfiguration, limits: Limits) -> ExperimentResult:
    """Run approved native RunSpec unchanged and package bounded original artifacts."""
    from coleman.api import run
    from coleman.spec.models import RunSpec

    for item in configuration.inputs:
        path = Path("inputs") / item.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(item.content_base64))
    native = run(RunSpec.model_validate_json(configuration.spec_json))
    root = Path(str(native.artifacts_dir))
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                content = stream.read(limits.max_output_bytes + 1)
            total += len(content)
            if len(files) >= 64 or total > limits.max_output_bytes:
                sys.exit(74)
            files.append(
                PortableFile(
                    name=path.relative_to(root).as_posix(),
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                    content_base64=base64.b64encode(content).decode("ascii"),
                )
            )
    return ExperimentResult(
        schema_version=1, run_id=native.run_id, work_units=configuration.work_units, files=tuple(files)
    )


def execute(item: WorkItem) -> ResultBundle:
    """Invoke the native solver/runner with admitted work caps and no transport imports."""
    from coleman.allocation import AllocationLimits, AllocationProblem, select_allocation
    from coleman.selection import CoverageProblem, SearchLimits, solve_coverage

    configuration, limits = item.configuration, item.limits
    problem = configuration.problem
    if isinstance(problem, CoverageProblem):
        result = solve_coverage(
            problem,
            method="branch_bound_v1",
            limits=SearchLimits(
                max_work=limits.max_work,
                deadline_seconds=None,
                max_candidates=limits.max_candidates,
                max_working_bytes=min(1073741824, limits.max_memory_bytes),
            ),
        )
    elif isinstance(problem, AllocationProblem):
        result = select_allocation(
            problem,
            limits=AllocationLimits(
                max_work=min(64, limits.max_work),
                deadline_seconds=None,
                max_working_bytes=min(1073741824, limits.max_memory_bytes),
            ),
        )
    else:
        result = _experiment(problem, limits)
    return ResultBundle(
        schema_version=1,
        spec_id=configuration.spec_id,
        configuration_fingerprint=configuration.fingerprint,
        kind=configuration.kind,
        result=result,
    )


def main() -> None:
    """Read the private fixed request file and publish JSON last; exceptions cannot complete a job."""
    item = WorkItem.model_validate_json(Path("request.json").read_bytes())
    _check_resources(item.limits)
    stopped = threading.Event()
    guard = threading.Thread(target=_guard, args=(item.limits, stopped), daemon=True)
    guard.start()
    try:
        bundle = execute(item)
        _check_resources(item.limits)
        content = bundle.to_json()
        if len(content.encode("utf-8")) > item.limits.max_output_bytes:
            sys.exit(74)
        ArtifactWriter(Path.cwd()).write_immutable(Path("result.json"), content)
    finally:
        stopped.set()
        guard.join(timeout=1)


if __name__ == "__main__":
    main()
