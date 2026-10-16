"""Download a pinned real LRTS slice, export Parquet and optionally execute native TCPCI replay."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from functools import partial
from importlib.resources import files
from pathlib import Path

from coleman.application.corpus import run_lrts_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import materialize_selected_packages
from coleman.corpus.lrts_etl import export_lrts_parquet, validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt, LRTSETLRecipe
from coleman.corpus.materialization import (
    SourceCache,
    materialize_source,
    validate_cache_separation,
    validate_source_storage,
)
from coleman.corpus.materialization_contracts import MaterializationSpec
from coleman.transports.corpus_http import public_https_source


def load_lrts_etl_recipe(path: Path | None = None) -> LRTSETLRecipe:
    """Read at most 2 MiB of v1 recipe metadata, defaulting to eight pinned real Karaf reports.

    No network, source extraction or native run starts. Unknown/missing/future
    schemas, unsafe/contradictory members and oversized inputs raise ValueError.
    The bundled resource contains only public provenance/pins, never data rows.
    """
    resource = path or files("coleman.corpus.resources").joinpath("lrts-karaf-first-eight-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("LRTS ETL recipe exceeds 2 MiB")
    return LRTSETLRecipe.from_json(payload)


def _reader(package: MaterializationSpec):
    """Construct a lazy pinned public reader with a bounded budget; cache hits never open it."""
    size = package.source.size_bytes
    if size is None:
        raise ValueError("LRTS acquisition requires a frozen source byte count")
    return partial(public_https_source, package.source.uri, size, elapsed_budget=300)


def main(argv: list[str] | None = None) -> int:
    """Print completion/data paths and conserved counts; explicit fetch/run options perform their named I/O.

    Offline conversion is default. Fetch accesses only pinned public raw GitHub
    metadata and Zenodo bytes, without authentication/redirects/proxies. Cache,
    Parquet and native output must be separate and outside Git, checked before
    acquisition. Native execution uses the existing API/checkpoint with the
    manifest's explicit serial-simulation assumptions. Errors propagate nonzero;
    no partial ETL is reported as complete and no benchmark admission is granted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run/resume the declared post-hoc serial simulation")
    args = parser.parse_args(argv)
    recipe = load_lrts_etl_recipe(args.recipe)
    cache_root, output = args.cache.resolve(), args.output.resolve()
    validate_source_storage(cache_root)
    validate_source_storage(output)
    validate_cache_separation(cache_root, output)
    cache = SourceCache(cache_root)
    materialize_source(recipe.metadata, cache, _reader(recipe.metadata) if args.fetch else None)
    materialize_selected_packages(recipe.reports, cache, _reader(recipe.reports.source) if args.fetch else None)
    path = export_lrts_parquet(recipe, cache, ArtifactWriter(output))
    receipt = LRTSETLReceipt.from_json(path.read_bytes())
    validate_lrts_etl(receipt, cache, path.parent)
    summary = {
        "completion": str(path),
        "observations": str(path.parent / "observations.parquet"),
        "datasets_dir": str(path.parent / "datasets"),
        "observed_cases": receipt.observed_cases,
        "replay_cases": receipt.replay_cases,
        "excluded_not_run": receipt.excluded_not_run,
        "excluded_unknown_outcome": receipt.excluded_unknown_outcome,
        "excluded_unavailable_duration": receipt.excluded_unavailable_duration,
        "modeled_rounds": receipt.native_builds,
        "evidence": receipt.evidence,
        "admission": receipt.admission,
    }
    if args.run:
        # The native runner owns its logging; a fresh CLI process directs it to
        # stderr so stdout remains a usable machine-readable summary.
        with redirect_stdout(sys.stderr):
            result = run_lrts_replay(path, cache, output / "native")
        summary.update(native_run_id=result.run_id, native_artifacts=str(result.artifacts_dir))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
