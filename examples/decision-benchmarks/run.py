"""Run installed-package correctness benchmarks without a checkout, model or network service."""

import json
import sys
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.benchmarks import load_suite, publish_benchmark_report, run_benchmarks


def main(output):
    """Persist a complete typed report; repeated invocations share math, not timings or content digests."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must remain in the current workspace")
    report = run_benchmarks(load_suite())
    artifact = publish_benchmark_report(report, ArtifactWriter(root))
    print(
        json.dumps(
            {
                "status": report.status,
                "mathematical_identity": report.mathematical_identity,
                "reproducibility": report.reproducibility,
                "artifact": str(artifact),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
