"""Optional local Python/pytest source collection; native TCPCI experiments use the existing application API."""

from __future__ import annotations

import sys
from importlib.resources import files

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.transports.python_capture import capture_python_source
from coleman.transports.python_contracts import PythonCaptureRecipe
from coleman.transports.source_cli import source_arguments, source_summary

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
    args = source_arguments(__doc__, PRESETS, argv)
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
        source_summary(
            result.dataset,
            result.dataset_path,
            result.state_directory,
            result.receipt.status,
            {capture.role: capture.status for capture in result.captures},
            len(recipe.test_ids),
        )
    )
    return 0 if result.receipt.status == "collected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
