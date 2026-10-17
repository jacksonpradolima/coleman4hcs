"""Explicit local LRTS archive acquisition, stage-preserving Parquet and optional native replay."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from functools import partial
from importlib.resources import files
from pathlib import Path

from coleman.application.corpus import run_lrts_stage_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import materialize_selected_packages
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt, LRTSStageRecipe
from coleman.corpus.lrts_stage_etl import export_lrts_stage_parquet, preflight_lrts_stage_etl, validate_lrts_stage_etl
from coleman.corpus.materialization import (
    SourceCache,
    materialize_source,
    validate_cache_separation,
    validate_source_storage,
)
from coleman.transports.corpus_http import public_https_source

_PRESETS = ("activemq", "log4j-ubuntu", "log4j-windows")


def load_lrts_stage_recipe(path: Path | None = None, *, preset: str = "activemq") -> LRTSStageRecipe:
    """Read at most 2 MiB of typed metadata; default to four pinned ActiveMQ source builds.

    Bundled presets contain public source pins/bindings only. No acquisition,
    extraction or execution occurs. Unknown preset/schema or oversized/invalid
    metadata raises ValueError. Existing single-stage recipes use their old API.
    """
    if preset not in _PRESETS:
        raise ValueError("Unknown LRTS stage preset")
    resource = path or files("coleman.corpus.resources").joinpath(f"lrts-{preset}-first-four-stage-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("LRTS stage recipe exceeds 2 MiB")
    return LRTSStageRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Acquire explicitly supplied local bytes and print verified completion/counts; optional native logs use stderr.

    Offline by default. --fetch-metadata downloads only the pinned public metadata
    CSV through the existing bounded HTTPS adapter. Large archives require an
    explicit local path and complete source verification; no download limit is
    widened. --metadata may supply the exact CSV locally. Cache/output/concrete
    native paths are preflighted before acquisition. Invalid/missing source or
    unsafe paths fail nonzero; partial files never mean a completed export.
    Native --run reuses existing API/checkpoints; no scientific admission occurs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--recipe", type=Path)
    group.add_argument("--preset", choices=_PRESETS, default="activemq")
    parser.add_argument("--archive", type=Path)
    metadata = parser.add_mutually_exclusive_group()
    metadata.add_argument("--metadata", type=Path)
    metadata.add_argument("--fetch-metadata", action="store_true")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    recipe = load_lrts_stage_recipe(args.recipe, preset=args.preset)
    cache_root, output = args.cache.resolve(), args.output.resolve()
    etl_root = preflight_lrts_stage_etl(recipe, cache_root, output)
    if args.run:
        native = output / "native"
        if native.is_symlink() or not native.resolve().is_relative_to(output):
            raise ValueError("Native LRTS stage output escapes its trusted root")
        validate_source_storage(native)
        validate_cache_separation(cache_root, native)
        validate_cache_separation(etl_root, native)
    cache = SourceCache(cache_root)
    source = recipe.source
    metadata_reader = None
    if args.metadata:
        metadata_reader = partial(args.metadata.open, "rb")
    elif args.fetch_metadata:
        size = source.metadata.source.size_bytes
        if size is None:
            raise ValueError("LRTS stage metadata requires a frozen byte count")
        metadata_reader = partial(public_https_source, source.metadata.source.uri, size, elapsed_budget=300)
    materialize_source(source.metadata, cache, metadata_reader, refresh_source=args.metadata is not None)
    if args.archive:
        materialize_source(source.reports.source, cache, partial(args.archive.open, "rb"), refresh_source=True)
    materialize_selected_packages(source.reports, cache)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(output))
    receipt = LRTSStageReceipt.from_json(path.read_bytes())
    validate_lrts_stage_etl(receipt, cache, path.parent)
    summary = {
        "completion": str(path),
        "observations": str(path.parent / "observations.parquet"),
        "datasets_dir": str(path.parent / "datasets"),
        "dataset_id": recipe.dataset_id,
        "native_stage_id": recipe.native_stage_id,
        "observed_cases": receipt.observed_cases,
        "replay_cases": receipt.replay_cases,
        "excluded_outside_stage": receipt.excluded_outside_stage,
        "excluded_not_run": receipt.excluded_not_run,
        "excluded_unknown_outcome": receipt.excluded_unknown_outcome,
        "excluded_unavailable_duration": receipt.excluded_unavailable_duration,
        "modeled_rounds": receipt.native_builds,
        "source_stages": receipt.source_stages,
        "temporal": receipt.temporal.model_dump(mode="json"),
        "evidence": receipt.evidence,
        "admission": receipt.admission,
    }
    if args.run:
        with redirect_stdout(sys.stderr):
            result = run_lrts_stage_replay(path, cache, output / "native")
        summary.update(native_run_id=result.run_id, native_artifacts=str(result.artifacts_dir))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
