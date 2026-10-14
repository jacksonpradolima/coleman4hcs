"""Immutable benchmark publication through existing native artifact infrastructure."""

from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.benchmarks.report import BenchmarkReport


def publish_benchmark_report(
    report: BenchmarkReport, writer: ArtifactWriter, *, max_output_bytes: int = 33554432
) -> Path:
    """Publish bounded self-contained JSON and a matching human report idempotently.

    Each file uses existing unique-temp/fsync/atomic immutable publication. JSON is
    the completion authority and is written last, after Markdown. Interrupted
    Markdown-only output is not a completed report; replay repairs it. Concurrent
    identical writes are idempotent, conflicting content errors propagate. Exact
    measured reports have content identities distinct from mathematical identities.
    The byte limit covers both UTF-8 files; serialization/validation memory has
    bounded manifest/observation overhead and is not an RSS isolation guarantee.
    """
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or not 0 <= max_output_bytes <= 67108864
    ):
        raise ValueError("Benchmark output limit must be an integer from zero through 64 MiB")
    checked = BenchmarkReport.model_validate(report)
    payload, human = checked.to_json(), checked.markdown()
    if len(payload.encode("utf-8")) + len(human.encode("utf-8")) > max_output_bytes:
        raise ValueError("Benchmark report exceeds its output byte limit")
    root = writer.run_root / "benchmark-reports"
    if root.is_symlink():
        raise ValueError("Benchmark report directory must not be a symlink")
    digest = checked.fingerprint.removeprefix("sha256:")
    path = root / f"{digest}.json"
    writer.write_immutable(root / f"{digest}.md", human)
    writer.write_immutable(path, payload)
    return path
