"""Explicit local-archive/offline RTPTorrent conversion to Parquet and optional native TCPCI replay."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from importlib.resources import files
from pathlib import Path

from coleman.application.corpus import run_rtptorrent_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import (
    SourceCache,
    materialize_source,
    validate_cache_separation,
    validate_source_storage,
)
from coleman.corpus.rtptorrent_etl import export_rtptorrent_parquet, preflight_rtptorrent_etl, validate_rtptorrent_etl
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt, RTPTorrentETLRecipe
from coleman.corpus.zip64_selection import materialize_zip64_files


def load_rtptorrent_etl_recipe(path: Path | None = None) -> RTPTorrentETLRecipe:
    """Read at most 2 MiB of explicit v1 metadata; default to eight frozen multi-execution HikariCP job labels.

    The preset retains the complete raw history analytically. Job selection uses
    ascending numeric labels with at least two raw class executions, never an
    outcome filter or an observed chronology claim. No source acquisition or
    execution occurs here. Missing/future versions and oversized metadata reject.
    """
    resource = path or files("coleman.corpus.resources").joinpath("rtptorrent-hikaricp-first-eight-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("RTPTorrent ETL recipe exceeds 2 MiB")
    return RTPTorrentETLRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Print complete Parquet/count/provenance paths and optionally run/resume the declared serial simulation.

    No network is opened. Operators may explicitly download/resume the frozen
    public archive separately; --archive verifies the supplied file even on cache
    hits. Offline cached conversion is default. Small-source HTTP adapter budgets
    remain unchanged. All enclosing/member bytes are verified before export.
    Cache/output roots must be separate and outside Git before acquisition.
    Native execution uses existing RunSpec/API/checkpoints and budget-censored
    RNFail assumptions. Format/source/I/O/native errors propagate nonzero; no
    partial ETL is reported complete and no temporal admission is inferred.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--archive", type=Path, help="Explicit complete local archive; full source SHA-256 is verified")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Run/resume the explicitly modeled source-job serial replay")
    args = parser.parse_args(argv)
    recipe = load_rtptorrent_etl_recipe(args.recipe)
    cache_root, output = args.cache.resolve(), args.output.resolve()
    validate_source_storage(cache_root)
    validate_source_storage(output)
    validate_cache_separation(cache_root, output)
    etl_root = preflight_rtptorrent_etl(recipe, cache_root, output)
    if args.run:
        native = output / "native"
        if native.is_symlink() or not native.resolve().is_relative_to(output):
            raise ValueError("Native RTPTorrent output escapes its trusted root")
        validate_source_storage(native)
        validate_cache_separation(etl_root, native)
    cache = SourceCache(cache_root)
    if args.archive is not None:
        materialize_source(recipe.source.source, cache, lambda: args.archive.open("rb"), refresh_source=True)
    materialize_zip64_files(recipe.source, cache)
    path = export_rtptorrent_parquet(recipe, cache, ArtifactWriter(output))
    receipt = RTPTorrentETLReceipt.from_json(path.read_bytes())
    validate_rtptorrent_etl(receipt, cache, path.parent)
    summary = {
        "completion": str(path),
        "observations": str(path.parent / "observations.parquet"),
        "jobs": str(path.parent / "jobs.parquet"),
        "datasets_dir": str(path.parent / "datasets"),
        "input_rows": receipt.input_rows,
        "duplicate_rows_excluded": receipt.duplicate_rows_excluded,
        "observed_executions": receipt.observed_executions,
        "source_jobs": receipt.source_jobs,
        "replay_executions": receipt.replay_executions,
        "excluded_outside_jobs": receipt.excluded_outside_jobs,
        "excluded_not_run": receipt.excluded_not_run,
        "excluded_unknown_outcome": receipt.excluded_unknown_outcome,
        "excluded_unavailable_duration": receipt.excluded_unavailable_duration,
        "modeled_rounds": receipt.native_rounds,
        "evidence": receipt.evidence,
        "admission": receipt.admission,
        "generalization": receipt.generalization,
    }
    if args.run:
        with redirect_stdout(sys.stderr):
            result = run_rtptorrent_replay(path, cache, output / "native")
        summary.update(native_run_id=result.run_id, native_artifacts=str(result.artifacts_dir))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
