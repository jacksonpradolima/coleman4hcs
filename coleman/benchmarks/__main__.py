"""Local installed-package benchmark transport; all decision logic remains in library services."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.benchmarks import (
    BenchmarkExecution,
    BenchmarkRunError,
    collect_environment,
    load_suite,
    publish_benchmark_report,
    run_benchmarks,
)


def main(argv: list[str] | None = None) -> int:
    """Run a bundled bounded profile and publish success/interruption/failure with explicit exit status.

    Output is restricted to the current workspace and immutable artifact names.
    Profiles contain no imports, provider URLs or arbitrary filesystem inputs.
    Source metadata is caller supplied, never silently inferred from a checkout.
    Exit 0 means a complete mathematically checked suite, 1 a failed/interrupted
    attempt with a report, and 2 invalid input or publication failure. No timing
    threshold or model/credentials/network connection is required.
    """
    parser = argparse.ArgumentParser(description="Run independent finite-model COLEMAN decision benchmarks")
    parser.add_argument("--profile", choices=("correctness", "performance"), default="correctness")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-dirty", choices=("true", "false", "unknown"), default="unknown")
    args = parser.parse_args(argv)
    workspace = Path.cwd().resolve()
    output = (workspace / args.output).resolve()
    if not output.is_relative_to(workspace):
        parser.error("Benchmark output must remain within the current workspace")
    try:
        environment = collect_environment(
            source_commit=args.source_commit,
            source_dirty=None if args.source_dirty == "unknown" else args.source_dirty == "true",
        )
        suite = load_suite(args.profile)
        try:
            report = run_benchmarks(suite, execution=BenchmarkExecution(environment=environment))
        except BenchmarkRunError as exc:
            report = exc.report
        path = publish_benchmark_report(report, ArtifactWriter(output))
    except (ValueError, OSError) as exc:
        print(f"Benchmark input/publication error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report.status,
                "reproducibility": report.reproducibility,
                "mathematical_identity": report.mathematical_identity,
                "artifact": str(path),
                "observations": len(report.observations),
                "failed_attempts": len(report.prior_failures) + int(report.failure is not None),
            },
            sort_keys=True,
        )
    )
    return 0 if report.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
