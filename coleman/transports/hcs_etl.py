"""Explicit local-original-file HCS ETL and optional native WTS/VTS replay; offline by default."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from functools import partial
from importlib.resources import files
from pathlib import Path

from coleman.application.hcs import run_hcs_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.hcs_contracts import HCSReceipt, HCSRecipe
from coleman.corpus.hcs_etl import export_hcs_parquet, preflight_hcs_etl, validate_hcs_etl
from coleman.corpus.materialization import (
    SourceCache,
    materialize_source,
    validate_cache_separation,
    validate_source_storage,
)


def load_hcs_recipe(path: Path | None = None) -> HCSRecipe:
    """Read at most 2 MiB of metadata; default to the pinned first-sixteen-pipeline LibSSH data recipe.

    No source bytes, network or subject processes are accessed. The bundled
    preset contains only public source pins/terms and the frozen pipeline IDs.
    Missing/future schemas, unknown fields and oversized metadata raise ValueError.
    """
    resource = path or files("coleman.corpus.resources").joinpath("libssh-first16-pipelines-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("HCS recipe exceeds 2 MiB")
    return HCSRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Verify explicit original CSV/JSON inputs, export Parquet and optionally execute both native strategies.

    There is no implicit fetch or authenticated transport. Download the documented
    public originals separately; an ingested TAB, modified CSV, partial download
    or mismatched file fails its frozen digest. Verified cache hits permit an
    offline retry without input paths. Paths are preflighted before cache writes;
    native stdout is redirected to stderr so one final JSON summary remains
    machine-readable. Native failures propagate without claiming a completed
    native experiment; ETL completion remains independently verifiable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--executions", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--strategy", choices=("both", "wts", "vts"), default="both")
    args = parser.parse_args(argv)
    recipe = load_hcs_recipe(args.recipe)
    cache_root, output = args.cache.resolve(), args.output.resolve()
    etl_root = preflight_hcs_etl(recipe, cache_root, output)
    strategies = ("wts", "vts") if args.strategy == "both" else (args.strategy,)
    if args.run:
        for strategy in strategies:
            native = output / "native" / strategy
            if native.is_symlink() or not native.resolve().is_relative_to(output):
                raise ValueError("HCS native output escapes its trusted tree")
            validate_source_storage(native)
            validate_cache_separation(cache_root, native)
            validate_cache_separation(etl_root, native)
    cache = SourceCache(cache_root)
    for package, local in ((recipe.executions, args.executions), (recipe.jobs, args.jobs)):
        reader = None if local is None else partial(local.open, "rb")
        materialize_source(package, cache, reader, refresh_source=local is not None)
    path = export_hcs_parquet(recipe, cache, ArtifactWriter(output))
    receipt = HCSReceipt.from_json(path.read_bytes())
    validate_hcs_etl(receipt, cache, path.parent)
    summary = {
        "completion": str(path),
        "recipe": recipe.fingerprint,
        "accounting": receipt.accounting.model_dump(mode="json"),
        "variants": [v.model_dump() for v in receipt.variants],
        "scope": receipt.scope,
        "evidence": receipt.evidence.model_dump(mode="json"),
        "admission": receipt.admission,
    }
    runs = []
    if args.run:
        with redirect_stdout(sys.stderr):
            for strategy in strategies:
                result = run_hcs_replay(path, cache, output / "native" / strategy, strategy=strategy)
                runs.append(dict(strategy=strategy, run_id=result.run_id, artifacts_dir=result.artifacts_dir))
    summary["native_runs"] = runs
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
