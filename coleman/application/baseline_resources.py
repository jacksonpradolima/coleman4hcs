"""Cooperative native matrix limits reusing existing own-executable RSS/output measurement."""

from __future__ import annotations

import time
from pathlib import Path

from coleman.application.worker import _output_exceeds, _worker_peak_rss_mib
from coleman.corpus.baseline_contracts import BaselineLimits
from coleman.telemetry.resources import ProcessResourceTracker


class BaselineLimitError(RuntimeError):
    """A frozen native-cell resource limit was reached; partial evidence remains a failed result."""

    def __init__(self, reason: str, peak_mib: float | None, elapsed_seconds: float):
        """Preserve observed resource evidence through the existing multiprocessing exception transport."""
        super().__init__(reason, peak_mib, elapsed_seconds)
        self.reason, self.peak_mib, self.elapsed_seconds = reason, peak_mib, elapsed_seconds


class BaselineResourceGuard:
    """Check one native cell at cooperative phase boundaries, never silently disabling a missing measurement.

    Native policy/metric/checkpoint work runs through the existing runner; this is
    neither another executor nor a preemptive sandbox. The same frozen limits are
    used for every candidate. Output walking is bounded and rejects symlinks;
    resource/I/O errors propagate rather than certifying successful completion.
    """

    def __init__(self, limits: BaselineLimits, root: Path):
        """Start a monotonic simulation-loop deadline; no output or process is created."""
        self.limits, self.root = limits, root
        self.started = time.perf_counter()
        self.tracker = ProcessResourceTracker()

    def check(self) -> None:
        """Raise a stable failure code for elapsed/memory/output exhaustion or unavailable peak RSS."""
        elapsed = time.perf_counter() - self.started
        peak = _worker_peak_rss_mib()
        if elapsed > self.limits.maximum_elapsed_seconds:
            raise BaselineLimitError("native_simulation_elapsed_limit_exceeded", peak, elapsed)
        if peak is None:
            raise BaselineLimitError("native_worker_peak_memory_unavailable", peak, elapsed)
        if peak > self.limits.maximum_peak_memory_mib:
            raise BaselineLimitError("native_worker_peak_memory_limit_exceeded", peak, elapsed)
        if _output_exceeds(
            self.limits.maximum_output_bytes, root=self.root, maximum_entries=self.limits.maximum_output_entries
        ):
            raise BaselineLimitError("native_output_limit_exceeded", peak, elapsed)

    def peak_mib(self) -> float | None:
        """Return the existing executable-specific sampler, excluding inherited Linux getrusage peaks."""
        return _worker_peak_rss_mib()

    def check_publication(self, additional_bytes: int) -> None:
        """Reserve the final receipt's completed bytes/entry before its atomic publication."""
        self.check()
        if additional_bytes > self.limits.maximum_output_bytes or _output_exceeds(
            self.limits.maximum_output_bytes - additional_bytes,
            root=self.root,
            maximum_entries=self.limits.maximum_output_entries - 1,
        ):
            raise BaselineLimitError(
                "native_output_limit_exceeded", self.peak_mib(), time.perf_counter() - self.started
            )
