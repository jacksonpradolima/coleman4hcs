"""Explicit public original-source acquisition and offline Westermo Parquet conversion."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import cast

from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import SourceCache, materialize_source
from coleman.corpus.westermo_contracts import WestermoReceipt, WestermoRecipe
from coleman.corpus.westermo_etl import (
    export_westermo_parquet,
    load_westermo_recipe,
    preflight_westermo_etl,
    validate_westermo_etl,
)
from coleman.transports.corpus_http import public_https_source


def _recipe(path: Path | None) -> WestermoRecipe:
    """Read bounded v1 metadata, defaulting to the packaged pinned source; never acquire data implicitly."""
    if path is None:
        return load_westermo_recipe()
    with path.open("rb") as stream:
        payload = stream.read(1024**2 + 1)
    if len(payload) > 1024**2:
        raise ValueError("Westermo recipe exceeds 1 MiB")
    return WestermoRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Fetch only on explicit --fetch, or verify local/archive cache bytes and publish analytical tables.

    Paths/metadata are preflighted before cache writes. Existing cache retry is
    offline by default; local input forces re-verification even on a cache hit.
    HTTP is the existing public bounded unauthenticated allowlist transport.
    No source code is executed and no test-duration seconds, historical feature
    availability, policy result or admission is invented. Errors propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path)
    source.add_argument("--fetch", action="store_true")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = _recipe(args.recipe)
    cache_root, output = args.cache.resolve(), args.output.resolve()
    preflight_westermo_etl(recipe, cache_root, output)
    reader = None
    if args.source is not None:
        reader = partial(args.source.open, "rb")
    elif args.fetch:
        size = cast(int, recipe.source.source.size_bytes)
        reader = partial(public_https_source, recipe.source.source.uri, size)
    cache = SourceCache(cache_root)
    materialize_source(recipe.source, cache, reader, refresh_source=args.source is not None)
    path = export_westermo_parquet(recipe, cache, ArtifactWriter(output))
    receipt = WestermoReceipt.from_json(path.read_bytes())
    validate_westermo_etl(receipt, cache, path.parent)
    print(
        json.dumps(
            {
                "completion": str(path),
                "receipt": receipt.fingerprint,
                "logical_rows": receipt.logical_rows_sha256,
                "counts": receipt.counts.model_dump(mode="json"),
                "admission": receipt.admission,
                "blockers": receipt.blockers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
