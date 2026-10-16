"""Offline verified RTPTorrent analytical/native Parquet, independent of CLI/MCP/network/runner imports."""

from __future__ import annotations

import hashlib
import io
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple, TypedDict

import polars as pl

from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.rtptorrent import convert_rtptorrent_source
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt, RTPTorrentETLRecipe
from coleman.corpus.rtptorrent_projection import RTPTorrentTables, project_rtptorrent
from coleman.corpus.zip64_selection import validate_zip64_files
from coleman.results.domain import fingerprint

_SHA256 = "sha256:"


class _Counts(TypedDict):
    """Exact derived receipt fields at the typed-model construction boundary."""

    observed_executions: int
    source_jobs: int
    replay_executions: int
    excluded_outside_jobs: int
    excluded_not_run: int
    excluded_unknown_outcome: int
    excluded_unavailable_duration: int
    native_rounds: int


class _OutputPlan(NamedTuple):
    """Preflighted concrete output namespace and the exact producer used to derive it."""

    root: Path
    producer: str


def rtptorrent_etl_producer() -> str:
    """Digest actual conversion/contract/archive/storage implementation bytes, not a caller-supplied producer tag."""
    modules = {
        "coleman": ("artifacts.py",),
        "coleman.corpus": (
            "rtptorrent_etl.py",
            "rtptorrent_etl_contracts.py",
            "rtptorrent_projection.py",
            "rtptorrent.py",
            "rtptorrent_contracts.py",
            "zip64_selection.py",
            "zip64_records.py",
            "zip64_contracts.py",
            "zip_metadata.py",
            "archives.py",
            "materialization.py",
            "materialization_contracts.py",
            "lrts_etl_contracts.py",
        ),
        "coleman.results": ("domain.py", "versioned.py"),
        "coleman.selection": ("contracts.py",),
    }
    return fingerprint(
        [
            [package + "/" + name, hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()]
            for package, names in sorted(modules.items())
            for name in names
        ]
    )


def _output(root: Path, relative: str) -> Path:
    """Enforce trusted-root containment, symlink and nested-Git exclusion before concrete output I/O."""
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("RTPTorrent ETL artifact escapes its trusted output tree")
    validate_source_storage(path)
    return path


def _output_plan(recipe: RTPTorrentETLRecipe, cache_root: Path, output_root: Path) -> _OutputPlan:
    """Check all concrete output descendants before source/cache operations; create no files."""
    validate_cache_separation(cache_root, output_root)
    validate_source_storage(cache_root)
    validate_source_storage(output_root)
    producer = rtptorrent_etl_producer()
    identity = fingerprint(["rtptorrent-parquet-export-v1", recipe.fingerprint, producer, pl.__version__])
    root = _output(output_root, "corpus-etl/" + identity.removeprefix(_SHA256))
    for relative in (
        "observations.parquet",
        "jobs.parquet",
        f"datasets/{recipe.dataset_id}/features-engineered.parquet",
        "complete.json",
    ):
        _output(root, relative)
    return _OutputPlan(root, producer)


def preflight_rtptorrent_etl(recipe: RTPTorrentETLRecipe, cache_root: Path, output_root: Path) -> Path:
    """Reject unsafe/overlapping cache and concrete artifact destinations before acquisition or cache construction.

    Reads implementation bytes and trusted path metadata only; creates no cache,
    output or source artifacts. Namespace/Parquet/receipt/native-input descendants
    are checked for escapes and nested Git repositories. Returns the checked ETL
    namespace so callers can preflight separate native output before acquisition.
    ValueError/OSError propagate; this is a cooperative-FS preflight, not a lock.
    """
    recipe = RTPTorrentETLRecipe.model_validate(recipe)
    return _output_plan(recipe, cache_root, output_root).root


def _verified_tables(recipe: RTPTorrentETLRecipe, cache: SourceCache) -> tuple[RTPTorrentTables, str, int, int]:
    """Verify actual enclosing membership first, then reuse the complete raw converter and pure projection."""
    validate_zip64_files(recipe.source, cache)
    dataset = convert_rtptorrent_source(recipe.conversion, cache)
    tables = project_rtptorrent(dataset, recipe)
    return tables, dataset.fingerprint, dataset.input_rows, dataset.duplicate_rows_excluded


def _roles(recipe: RTPTorrentETLRecipe, tables: RTPTorrentTables) -> tuple[tuple[str, pl.DataFrame], ...]:
    """Bind exactly three stable Parquet roles; the native name remains a validated directory component."""
    return (
        ("observations.parquet", tables.observations),
        ("jobs.parquet", tables.jobs),
        (f"datasets/{recipe.dataset_id}/features-engineered.parquet", tables.native),
    )


def _counts(tables: RTPTorrentTables) -> _Counts:
    """Derive completion counters at one typed-model construction boundary; no independently editable totals."""
    return {
        "observed_executions": tables.observations.height,
        "source_jobs": tables.jobs.height,
        "replay_executions": tables.native.height,
        "excluded_outside_jobs": tables.exclusions["outside_jobs"],
        "excluded_not_run": tables.exclusions["not_run"],
        "excluded_unknown_outcome": tables.exclusions["unknown_outcome"],
        "excluded_unavailable_duration": tables.exclusions["unavailable_duration"],
        "native_rounds": tables.native["BuildId"].n_unique(),
    }


def _parquet_bytes(table: pl.DataFrame) -> bytes:
    """Encode one fixed-profile table in at most 64 MiB; no caller-provided Parquet is decoded."""
    buffer = io.BytesIO()
    table.write_parquet(buffer, compression="uncompressed", statistics=False, row_group_size=8192, data_page_size=65536)
    payload = buffer.getvalue()
    if len(payload) > 64 * 1024**2:
        raise ValueError("RTPTorrent ETL Parquet exceeds 64 MiB")
    return payload


def export_rtptorrent_parquet(recipe: RTPTorrentETLRecipe, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Verify full source, write deterministic analytical/native Parquet and publish complete.json last.

    Source bytes, all canonical jobs/executions and every projection exclusion
    are conserved; unavailable quantities remain null and exact decimal seconds
    remain strings analytically. Only the native boundary uses binary64 seconds.
    Fixed uncompressed Parquet settings plus the recorded writer/producer bind
    physical bytes; canonical rows separately bind semantic reproducibility.
    Concurrent identical retries converge via existing immutable writes. Failed
    operations may retain individual files, never a new completion receipt.
    Cache and output must be separate and outside Git. Source/format/row/size
    contradictions raise ValueError; I/O errors propagate. No network, subject
    code, runner, checkpoint or temporal/admission experiment is started here.
    """
    recipe = RTPTorrentETLRecipe.model_validate(recipe)
    plan = _output_plan(recipe, cache.root, writer.run_root)
    tables, dataset, input_rows, duplicates = _verified_tables(recipe, cache)
    root = plan.root
    artifacts = []
    for relative, table in _roles(recipe, tables):
        payload = _parquet_bytes(table)
        writer.write_immutable_stream(_output(root, relative), (payload,), maximum_bytes=len(payload))
        artifacts.append(
            ETLFile(
                path=relative,
                sha256=_SHA256 + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=table.height,
            )
        )
    receipt = RTPTorrentETLReceipt(
        schema_version=1,
        recipe=recipe,
        producer_sha256=plan.producer,
        parquet_writer=pl.__version__,
        source_dataset_sha256=dataset,
        files=tuple(artifacts),
        logical_rows_sha256=tables.logical_digest,
        input_rows=input_rows,
        duplicate_rows_excluded=duplicates,
        **_counts(tables),
    )
    completion = _output(root, "complete.json")
    writer.write_immutable(completion, receipt.to_json() + "\n")
    return completion


def validate_rtptorrent_etl(receipt: RTPTorrentETLReceipt, cache: SourceCache, root: Path) -> None:
    """Reverify actual full source, producer, counts, canonical rows and pinned Parquet bytes without repair/network.

    Metadata reads alone never certify correspondence. Unknown producer digests
    require the recorded implementation; no old data are silently reinterpreted.
    Recorded writer and source-derived fixed-profile bytes must agree. Unknown
    writer versions require their recorded implementation; equivalent rows may
    not substitute different physical bytes under an old receipt. Verification
    regenerates expected bytes from source instead of decoding caller-provided
    Parquet. Missing/corrupt/oversized artifacts raise ValueError or OSError.
    No runner, checkpoint, clock or scientific admission is involved.
    """
    receipt = RTPTorrentETLReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, root)
    validate_source_storage(root)
    if receipt.producer_sha256 != rtptorrent_etl_producer():
        raise ValueError("Unsupported RTPTorrent ETL producer; verify with its recorded implementation")
    if receipt.parquet_writer != pl.__version__:
        raise ValueError("Unsupported RTPTorrent Parquet writer; verify with its recorded writer version")
    tables, dataset, input_rows, duplicates = _verified_tables(receipt.recipe, cache)
    if (
        receipt.source_dataset_sha256,
        receipt.logical_rows_sha256,
        receipt.input_rows,
        receipt.duplicate_rows_excluded,
    ) != (dataset, tables.logical_digest, input_rows, duplicates):
        raise ValueError("RTPTorrent ETL source/logical identities or physical-row counts disagree")
    if any(getattr(receipt, key) != value for key, value in _counts(tables).items()):
        raise ValueError("RTPTorrent ETL reported row/exclusion counts disagree with source")
    expected = dict(_roles(receipt.recipe, tables))
    for item in receipt.files:
        payload_expected = _parquet_bytes(expected[item.path])
        expected_sha = _SHA256 + hashlib.sha256(payload_expected).hexdigest()
        if item.sha256 != expected_sha or item.size_bytes != len(payload_expected):
            raise ValueError("RTPTorrent ETL artifact identity disagrees with source and fixed writer profile")
        with _output(root, item.path).open("rb") as stream:
            payload = stream.read(item.size_bytes + 1)
        if len(payload) != item.size_bytes or _SHA256 + hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("RTPTorrent ETL file bytes disagree with completion")
