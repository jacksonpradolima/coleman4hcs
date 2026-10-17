"""Optional local Python/pytest source collection; native TCPCI experiments use the existing application API."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib.resources import files
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.transports.python_capture import capture_python_source
from coleman.transports.python_contracts import PythonCaptureRecipe

PRESETS = ("pysnooper-1-ascii-v1", "pysnooper-1-utf8-v1")


def load_python_recipe(preset: str = PRESETS[0]) -> PythonCaptureRecipe:
    """Read one bounded built-in metadata preset, never acquiring or executing foreign inputs.

    Unknown presets and missing/future schema versions reject. The resource
    contains public integrity/terms metadata, not subject source or wheel bytes.
    Operator-defined recipes can use PythonCaptureRecipe.from_json separately.
    """
    if preset not in PRESETS:
        raise ValueError("Unknown Python source preset")
    resource = files("coleman.corpus.resources").joinpath("python-" + preset + ".json")
    with resource.open("rb") as stream:
        payload = stream.read(PythonCaptureRecipe.MAXIMUM_BYTES + 1)
    if len(payload) > PythonCaptureRecipe.MAXIMUM_BYTES:
        raise ValueError("Python recipe exceeds its byte limit")
    return PythonCaptureRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Collect/replay one paired trial and print paths/accounting; incomplete collection returns 2.

    Built-in presets or a bounded stdin recipe are explicit choices. A pinned
    local runtime must already exist for new capture; fetch only acquires the
    declared public inputs. Completed replay is offline and requires no Docker.
    No image pull, policy training or historical CI evaluation occurs here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    recipes = parser.add_mutually_exclusive_group()
    recipes.add_argument("--preset", choices=PRESETS, default=PRESETS[0])
    recipes.add_argument("--recipe-stdin", action="store_true")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    if args.recipe_stdin:
        payload = sys.stdin.buffer.read(PythonCaptureRecipe.MAXIMUM_BYTES + 1)
        if len(payload) > PythonCaptureRecipe.MAXIMUM_BYTES:
            raise ValueError("Python stdin recipe exceeds its byte limit")
        recipe = PythonCaptureRecipe.from_json(payload)
    else:
        recipe = load_python_recipe(args.preset)
    result = capture_python_source(
        recipe, SourceCache(args.cache), ArtifactWriter(args.output), args.trial, fetch=args.fetch
    )
    print(
        json.dumps(
            {
                "dataset": str(result.dataset_path),
                "state": str(result.state_directory),
                "receipt": str(result.state_directory / "receipt.json"),
                "status": result.receipt.status,
                "role_statuses": {capture.role: capture.status for capture in result.captures},
                "planned_pairs": len(recipe.test_ids),
                "unobserved_pairs": len(recipe.test_ids) - len(result.dataset.pairs),
                "pairs": dict(sorted(Counter(pair.classification for pair in result.dataset.pairs).items())),
                "history": result.dataset.history,
                "admission": result.dataset.admission,
                "generalization": result.dataset.generalization,
            },
            sort_keys=True,
        )
    )
    return 0 if result.receipt.status == "collected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
