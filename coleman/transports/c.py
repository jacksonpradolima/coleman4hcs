"""Optional local C function source collection; native TCPCI experiments use the existing application API."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib.resources import files
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.transports.c_capture import capture_c_source
from coleman.transports.c_contracts import CCaptureRecipe

PRESETS = ("juliet-uint32-add-v1",)


def load_c_recipe(preset: str = PRESETS[0]) -> CCaptureRecipe:
    """Read one bounded built-in metadata preset, never acquiring or executing foreign inputs.

    Unknown presets and missing/future schema versions reject. The resource
    contains public integrity/terms metadata, not subject source or wheel bytes.
    Operator-defined recipes can use CCaptureRecipe.from_json separately.
    """
    if preset not in PRESETS:
        raise ValueError("Unknown C source preset")
    resource = files("coleman.corpus.resources").joinpath("c-" + preset + ".json")
    with resource.open("rb") as stream:
        payload = stream.read(CCaptureRecipe.MAXIMUM_BYTES + 1)
    if len(payload) > CCaptureRecipe.MAXIMUM_BYTES:
        raise ValueError("C recipe exceeds its byte limit")
    return CCaptureRecipe.from_json(payload)


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
        payload = sys.stdin.buffer.read(CCaptureRecipe.MAXIMUM_BYTES + 1)
        if len(payload) > CCaptureRecipe.MAXIMUM_BYTES:
            raise ValueError("C stdin recipe exceeds its byte limit")
        recipe = CCaptureRecipe.from_json(payload)
    else:
        recipe = load_c_recipe(args.preset)
    result = capture_c_source(
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
