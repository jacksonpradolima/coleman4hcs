"""Bounded offline LRTS stage-view export; existing raw reader, cache and immutable writer are reused."""

from __future__ import annotations

import hashlib
import io
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple, TypedDict

import polars as pl

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import validate_selected_packages
from coleman.corpus.lrts import convert_lrts_source
from coleman.corpus.lrts_contracts import LRTSDataset, LRTSSpec
from coleman.corpus.lrts_etl import lrts_etl_producer
from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt, LRTSStageRecipe
from coleman.corpus.lrts_stage_projection import LRTSStageTables, project_lrts_stages
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.results.domain import fingerprint

_SHA256 = "sha256:"
_COMPLETION = "complete.json"


class _Counts(TypedDict):
    """Derived receipt counters at the typed-model construction boundary."""

    observed_cases: int
    replay_cases: int
    excluded_outside_stage: int
    excluded_not_run: int
    excluded_unknown_outcome: int
    excluded_unavailable_duration: int
    native_builds: int


class _Plan(NamedTuple):
    """Preflighted namespace and producer; paths are never caller-controlled receipt fields."""

    root: Path
    producer: str


def lrts_stage_producer() -> str:
    """Bind actual new/legacy conversion, contracts and storage code; no caller-supplied producer label."""
    modules = {
        "coleman.corpus": ("lrts_stage_etl.py", "lrts_stage_contracts.py", "lrts_stage_projection.py", "contracts.py"),
        "coleman": ("artifacts.py",),
        "coleman.results": ("domain.py", "versioned.py"),
        "coleman.selection": ("contracts.py",),
    }
    return fingerprint(
        [
            lrts_etl_producer(),
            [
                [package + "/" + name, hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()]
                for package, names in sorted(modules.items())
                for name in names
            ],
        ]
    )


def _output(root: Path, relative: str) -> Path:
    """Require containment, no symlink and no nested Git worktree before concrete output access."""
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("LRTS stage artifact escapes its trusted output tree")
    validate_source_storage(path)
    return path


def _paths(recipe: LRTSStageRecipe) -> tuple[str, str, str]:
    """Bind the complete analytical/build/native role set to the explicit safe dataset alias."""
    return ("observations.parquet", "builds.parquet", f"datasets/{recipe.dataset_id}/features-engineered.parquet")


def _plan(recipe: LRTSStageRecipe, cache_root: Path, output_root: Path) -> _Plan:
    """Check all concrete descendants before source access; create no cache/output files."""
    validate_cache_separation(cache_root, output_root)
    validate_source_storage(cache_root)
    validate_source_storage(output_root)
    producer = lrts_stage_producer()
    identity = fingerprint(["lrts-stage-parquet-export-v1", recipe.fingerprint, producer, pl.__version__])
    root = _output(output_root, "corpus-etl/" + identity.removeprefix(_SHA256))
    for relative in (*_paths(recipe), _COMPLETION):
        _output(root, relative)
    return _Plan(root, producer)


def preflight_lrts_stage_etl(recipe: LRTSStageRecipe, cache_root: Path, output_root: Path) -> Path:
    """Validate concrete output/cache separation before acquisition; no source or cache writes.

    Trusted-local cooperative filesystem checks are not an OS sandbox or lock.
    Unsupported profile/path contradictions raise ValueError; I/O errors propagate.
    """
    return _plan(LRTSStageRecipe.model_validate(recipe), cache_root, output_root).root


def _tables(recipe: LRTSStageRecipe, cache: SourceCache) -> LRTSStageTables:
    """Verify full enclosing membership and reuse the raw source reader without fake partial reports."""
    source = recipe.source
    validate_selected_packages(source.reports, cache)
    selected = {item.member.name: item.package for item in source.reports.selections}
    metadata = MaterializationReceipt(schema_version=1, spec=source.metadata)
    datasets: list[LRTSDataset] = []
    total = 0
    for build in source.builds:
        package = selected[build.archive_member]
        dataset = convert_lrts_source(
            LRTSSpec(
                schema_version=1,
                metadata_materialization=metadata,
                metadata_member=source.metadata.members[0].name,
                report_materialization=MaterializationReceipt(schema_version=1, spec=package),
                report_member=package.members[0].name,
                project=source.project,
                pull_request=build.pull_request,
                build_id=build.build_id,
                report_origin_path=build.archive_member.removesuffix(".zip"),
                history=source.history,
                maximum_cases=source.maximum_cases,
            ),
            cache,
        )
        total += len(dataset.cases)
        if total > source.maximum_cases:
            raise ValueError("LRTS stage total source-case limit exceeded")
        datasets.append(dataset)
    return project_lrts_stages(recipe, tuple(datasets))


def _counts(tables: LRTSStageTables) -> _Counts:
    """Derive conserved primary dispositions once at receipt construction/verification."""
    return {
        "observed_cases": tables.observations.height,
        "replay_cases": tables.native.height,
        "excluded_outside_stage": tables.exclusions["outside_stage"],
        "excluded_not_run": tables.exclusions["not_run"],
        "excluded_unknown_outcome": tables.exclusions["unknown_outcome"],
        "excluded_unavailable_duration": tables.exclusions["unavailable_duration"],
        "native_builds": tables.native["BuildId"].n_unique(),
    }


def _parquet_bytes(table: pl.DataFrame) -> bytes:
    """Encode a fixed uncompressed profile within 64 MiB; never decode untrusted caller Parquet."""
    buffer = io.BytesIO()
    table.write_parquet(buffer, compression="uncompressed", statistics=False, row_group_size=8192, data_page_size=65536)
    payload = buffer.getvalue()
    if len(payload) > 64 * 1024**2:
        raise ValueError("LRTS stage Parquet exceeds 64 MiB")
    return payload


def export_lrts_stage_parquet(recipe: LRTSStageRecipe, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Export full analytical evidence and one explicit native stage; publish complete.json last.

    Original raw/processed identities, seconds, missingness and primary exclusions
    are conserved. Source verification precedes publication. Identical concurrent
    retries converge through immutable writes. Interrupted operations may retain
    individual files, never a new completion receipt. Namespace binds recipe,
    actual implementation and writer. Old LRTS recipes/artifacts are unchanged.
    This post-hoc simulation grants no temporal admission or policy promotion.
    Cache/output must be separate and outside Git. Invalid bytes/mappings raise
    ValueError; filesystem errors propagate. No network/CLI/runner is imported.
    """
    recipe = LRTSStageRecipe.model_validate(recipe)
    plan = _plan(recipe, cache.root, writer.run_root)
    tables = _tables(recipe, cache)
    records = []
    for relative, table in zip(_paths(recipe), tables[:3], strict=True):
        payload = _parquet_bytes(table)
        writer.write_immutable_stream(_output(plan.root, relative), (payload,), maximum_bytes=len(payload))
        records.append(
            ETLFile(
                path=relative,
                sha256=_SHA256 + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=table.height,
            )
        )
    receipt = LRTSStageReceipt(
        schema_version=1,
        recipe=recipe,
        producer_sha256=plan.producer,
        parquet_writer=pl.__version__,
        source_datasets=tables.source_datasets,
        files=tuple(records),
        logical_rows_sha256=tables.logical_digest,
        metadata_accounting=tables.metadata_accounting,
        source_stages=len(recipe.stage_bindings),
        **_counts(tables),
    )
    completion = _output(plan.root, _COMPLETION)
    writer.write_immutable(completion, receipt.to_json() + "\n")
    return completion


def validate_lrts_stage_etl(receipt: LRTSStageReceipt, cache: SourceCache, root: Path) -> None:
    """Reverify actual source, stage mapping, logical rows/counts and fixed-profile artifact bytes offline.

    Metadata parsing alone does not certify source truth. Unknown producer or
    writer identities require the recorded implementation; no migration silently
    changes old meanings. Expected Parquet bytes are regenerated from source,
    rather than decoding caller-provided Parquet. No repair/network/admission.
    Invalid/missing/oversized files raise ValueError/OSError.
    """
    receipt = LRTSStageReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, root)
    validate_source_storage(root)
    for relative in (*_paths(receipt.recipe), _COMPLETION):
        _output(root, relative)
    if receipt.producer_sha256 != lrts_stage_producer():
        raise ValueError("Unsupported LRTS stage producer; use its recorded implementation")
    if receipt.parquet_writer != pl.__version__:
        raise ValueError("Unsupported LRTS stage Parquet writer; use its recorded version")
    tables = _tables(receipt.recipe, cache)
    if (receipt.source_datasets, receipt.logical_rows_sha256, receipt.metadata_accounting) != (
        tables.source_datasets,
        tables.logical_digest,
        tables.metadata_accounting,
    ):
        raise ValueError("LRTS stage source/logical identities or metadata accounting disagree")
    if any(getattr(receipt, key) != value for key, value in _counts(tables).items()):
        raise ValueError("LRTS stage receipt counters disagree with source")
    expected = dict(zip(_paths(receipt.recipe), tables[:3], strict=True))
    for item in receipt.files:
        payload_expected = _parquet_bytes(expected[item.path])
        sha = _SHA256 + hashlib.sha256(payload_expected).hexdigest()
        if (item.sha256, item.size_bytes) != (sha, len(payload_expected)):
            raise ValueError("LRTS stage artifact identity disagrees with pinned source")
        with _output(root, item.path).open("rb") as stream:
            payload = stream.read(item.size_bytes + 1)
        if len(payload) != item.size_bytes or _SHA256 + hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("LRTS stage file bytes disagree with completion")
