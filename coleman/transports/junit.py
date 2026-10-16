"""Optional local entry point for pinned JUnit3 source collection, separate from native TCPCI execution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from importlib.resources import files
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.transports.junit_capture import capture_junit_source
from coleman.transports.junit_contracts import JunitCaptureRecipe


def load_junit_recipe(path: Path | None = None) -> JunitCaptureRecipe:
    """Read a bounded versioned recipe or the bundled arm64 CLI-13 preset; never acquire/execute inputs.

    Missing files, unsupported versions, unsafe declarations and files over
    2 MiB reject. The preset contains public integrity metadata, not subject code.
    Runtime preparation and optional acquisition remain explicit caller actions.
    """
    resource = path or files("coleman.corpus.resources").joinpath("apache-cli13-junit3-arm64-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("JUnit recipe exceeds 2 MiB")
    return JunitCaptureRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Collect a local source trial and print paths/counts; source failures stay outcomes rather than successes.

    Requires a preinstalled pinned Docker image; fetch is opt-in and only for
    declared public source/dependency packages. Errors propagate with a nonzero
    process exit. Incomplete role collection returns 2 after evidence publication.
    No learning experiment, historical evaluation or image pull is performed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    result = capture_junit_source(
        load_junit_recipe(args.recipe),
        SourceCache(args.cache),
        ArtifactWriter(args.output),
        args.trial,
        fetch=args.fetch,
    )
    print(
        json.dumps(
            {
                "dataset": str(result.dataset_path),
                "state": str(result.state_directory),
                "role_statuses": {capture.role: capture.status for capture in result.captures},
                "pairs": dict(sorted(Counter(pair.classification for pair in result.dataset.pairs).items())),
                "admission": result.dataset.admission,
                "generalization": result.dataset.generalization,
            },
            sort_keys=True,
        )
    )
    return 0 if all(capture.status == "collected" for capture in result.captures) else 2


if __name__ == "__main__":
    raise SystemExit(main())
