"""Offline real-source LRTS Parquet export and verification, independent of network/CLI/runner imports."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import polars as pl

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import validate_selected_packages
from coleman.corpus.lrts import convert_lrts_source
from coleman.corpus.lrts_contracts import LRTSCase, LRTSDataset, LRTSSpec, LRTSSuite
from coleman.corpus.lrts_etl_contracts import ETLFile, ETLMetadataAccounting, LRTSETLReceipt, LRTSETLRecipe
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.results.domain import canonical_json, fingerprint

_SHA256 = "sha256:"
_OBSERVATIONS = {
    "execution_id": pl.String,
    "replay_name": pl.String,
    "round": pl.Int64,
    "project": pl.String,
    "pull_request": pl.String,
    "source_build_id": pl.Int64,
    "suite_position": pl.Int64,
    "case_position": pl.Int64,
    "suite_name": pl.String,
    "suite_id": pl.String,
    "node_id": pl.String,
    "enclosing_blocks": pl.List(pl.String),
    "enclosing_block_names": pl.List(pl.String),
    "raw_suite_timestamp": pl.String,
    "class_name": pl.String,
    "method_name": pl.String,
    "source_status": pl.String,
    "outcome": pl.String,
    "duration": pl.Float64,
    "duration_status": pl.String,
    "duration_unit": pl.String,
    "duration_meaning": pl.String,
    "replay_exclusion": pl.String,
}
_NATIVE = {
    "BuildId": pl.Int64,
    "Name": pl.String,
    "Duration": pl.Float64,
    "CalcPrio": pl.Int64,
    "LastRun": pl.String,
    "Verdict": pl.Int64,
    "SourceExecutionId": pl.String,
}
_BUILDS = {
    "round": pl.Int64,
    "project": pl.String,
    "pull_request": pl.String,
    "source_build_id": pl.Int64,
    "scheduled_label_epoch_seconds": pl.Float64,
    "outcome_available_at": pl.String,
    "metadata_json": pl.String,
    "source_dataset_sha256": pl.String,
    "replay_cases": pl.Int64,
    "replay_status": pl.String,
}


class _Tables(NamedTuple):
    """Typed table bundle at the Parquet boundary; source DTOs remain authoritative."""

    observations: pl.DataFrame
    builds: pl.DataFrame
    native: pl.DataFrame
    source_datasets: tuple[str, ...]
    logical_digest: str
    exclusions: Counter[str]
    metadata_accounting: ETLMetadataAccounting


def lrts_etl_producer() -> str:
    """Digest actual conversion/contract/selection bytes, not a caller-supplied producer label."""
    modules = (
        "lrts_etl.py",
        "lrts_etl_contracts.py",
        "lrts.py",
        "lrts_contracts.py",
        "lrts_raw.py",
        "archive_selection.py",
        "archives.py",
        "zip_metadata.py",
        "materialization.py",
        "materialization_contracts.py",
    )
    return fingerprint(
        [[name, hashlib.sha256(files("coleman.corpus").joinpath(name).read_bytes()).hexdigest()] for name in modules]
    )


def _source_datasets(recipe: LRTSETLRecipe, cache: SourceCache) -> tuple[LRTSDataset, ...]:
    """Reverify selected bytes and metadata order, enforcing one stage and the aggregate case budget."""
    validate_selected_packages(recipe.reports, cache)
    selected = {item.member.name: item.package for item in recipe.reports.selections}
    metadata = MaterializationReceipt(schema_version=1, spec=recipe.metadata)
    result = []
    total = 0
    prior = None
    for build in recipe.builds:
        package = selected[build.archive_member]
        dataset = convert_lrts_source(
            LRTSSpec(
                schema_version=1,
                metadata_materialization=metadata,
                metadata_member=recipe.metadata.members[0].name,
                report_materialization=MaterializationReceipt(schema_version=1, spec=package),
                report_member=package.members[0].name,
                project=recipe.project,
                pull_request=build.pull_request,
                build_id=build.build_id,
                report_origin_path=build.archive_member.removesuffix(".zip"),
                history=recipe.history,
                maximum_cases=recipe.maximum_cases,
            ),
            cache,
        )
        if len(dataset.metadata) != 1:
            raise ValueError("ETL v1 requires one explicit metadata stage; no stage alias inference")
        scheduled = dataset.metadata[0].measurement("lrts_scheduled_epoch_seconds").value
        if scheduled is None:
            raise ValueError("ETL replay order requires an observed scheduled source label")
        key = (scheduled, build.pull_request, build.build_id)
        if prior is not None and key <= prior:
            raise ValueError("ETL build selection disagrees with declared scheduled-label order")
        prior = key
        total += len(dataset.cases)
        if total > recipe.maximum_cases:
            raise ValueError("ETL exceeds total source-case limit")
        result.append(dataset)
    return tuple(result)


def _exclusion(case: LRTSCase) -> str | None:
    """Assign one primary projection exclusion, preserving unknown/not-run before missing duration."""
    if case.outcome == "not_run":
        return "not_run"
    if case.outcome == "unknown":
        return "unknown_outcome"
    if case.duration.value is None:
        return "unavailable_duration"
    return None


def _case_record(
    dataset: LRTSDataset, suite: LRTSSuite, case: LRTSCase, round_id: int, name: str, execution: str, reason: str | None
) -> dict[str, object]:
    """Serialize validated source DTO fields at the analytical table boundary, without coercing absent values."""
    return {
        "execution_id": execution,
        "replay_name": name,
        "round": round_id,
        "project": dataset.spec.project,
        "pull_request": dataset.spec.pull_request,
        "source_build_id": dataset.spec.build_id,
        "suite_position": suite.position,
        "case_position": case.position,
        "suite_name": suite.name,
        "suite_id": suite.reported_id,
        "node_id": suite.node_id,
        "enclosing_blocks": suite.enclosing_blocks,
        "enclosing_block_names": suite.enclosing_block_names,
        "raw_suite_timestamp": suite.raw_timestamp,
        "class_name": case.class_name,
        "method_name": case.method_name,
        "source_status": case.source_status,
        "outcome": case.outcome,
        "duration": case.duration.value,
        "duration_status": case.duration.status,
        "duration_unit": case.duration.unit,
        "duration_meaning": case.duration.meaning,
        "replay_exclusion": reason,
    }


class _ProjectedBuild(NamedTuple):
    """Internal serialization rows for one build; source DTOs and explicit table schemas remain authoritative."""

    observations: list[dict[str, object]]
    native: list[dict[str, object]]
    exclusions: Counter[str]


def _build_rows(dataset: LRTSDataset, round_id: int) -> _ProjectedBuild:
    """Keep source positions/occurrences independent of eligibility, costs and outcomes."""
    observations: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    for suite in dataset.suites:
        for case in suite.cases:
            blocks = None if suite.enclosing_block_names is None else list(suite.enclosing_block_names)
            label = fingerprint(
                [dataset.spec.project, suite.name, suite.reported_id, blocks, case.class_name, case.method_name]
            )
            name = fingerprint(["lrts-label-occurrence-v1", label, occurrences[label]])
            occurrences[label] += 1
            execution = case.execution_id(
                dataset.spec.project, dataset.spec.pull_request, dataset.spec.build_id, suite.position
            )
            reason = _exclusion(case)
            observations.append(_case_record(dataset, suite, case, round_id, name, execution, reason))
            if reason is not None:
                exclusions[reason] += 1
            else:
                native.append(
                    {
                        "BuildId": round_id,
                        "Name": name,
                        "Duration": case.duration.value,
                        "CalcPrio": 0,
                        "LastRun": "",
                        "Verdict": int(case.outcome == "failed"),
                        "SourceExecutionId": execution,
                    }
                )
    return _ProjectedBuild(observations, native, exclusions)


def _build_record(dataset: LRTSDataset, round_id: int, eligible: int) -> dict[str, object]:
    """Serialize source metadata and explicit native eligibility, including entirely excluded rounds."""
    metadata = dataset.metadata[0]
    return {
        "round": round_id,
        "project": dataset.spec.project,
        "pull_request": metadata.pull_request,
        "source_build_id": metadata.build_id,
        "scheduled_label_epoch_seconds": metadata.measurement("lrts_scheduled_epoch_seconds").value,
        "outcome_available_at": None,
        "metadata_json": canonical_json(metadata.model_dump(mode="json")),
        "source_dataset_sha256": dataset.fingerprint,
        "replay_cases": eligible,
        "replay_status": "eligible" if eligible else "no_compatible_reported_cases",
    }


def _tables(recipe: LRTSETLRecipe, cache: SourceCache) -> _Tables:
    """Project validated source DTOs once; dictionaries exist only at the typed Polars boundary."""
    observations, builds, native = [], [], []
    exclusions: Counter[str] = Counter()
    datasets = _source_datasets(recipe, cache)
    for round_id, dataset in enumerate(datasets, 1):
        rows = _build_rows(dataset, round_id)
        observations.extend(rows.observations)
        native.extend(rows.native)
        builds.append(_build_record(dataset, round_id, len(rows.native)))
        exclusions.update(rows.exclusions)
    frames = (
        pl.DataFrame(observations, schema=_OBSERVATIONS),
        pl.DataFrame(builds, schema=_BUILDS),
        pl.DataFrame(native, schema=_NATIVE),
    )
    logical = fingerprint([frame.to_dicts() for frame in frames])
    count = datasets[0].metadata_input_rows
    selected = sum(len(item.metadata) for item in datasets)
    duplicates = sum(item.metadata_duplicates_excluded for item in datasets)
    accounting = ETLMetadataAccounting(
        input_rows=count,
        selected_rows=selected,
        duplicates_excluded=duplicates,
        outside_slice=count - selected - duplicates,
    )
    return _Tables(*frames, tuple(item.fingerprint for item in datasets), logical, exclusions, accounting)


def _output(root: Path, relative: str) -> Path:
    """Preflight concrete content paths against containment, symlinks and nested Git worktrees."""
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("ETL artifact must remain within the trusted output tree")
    validate_source_storage(path)
    return path


def _namespace(recipe: LRTSETLRecipe, producer: str, writer: str) -> str:
    """Separate logical recipe/implementation/writer identities from mutable absolute output locations."""
    return fingerprint(["lrts-parquet-export-v1", recipe.fingerprint, producer, writer]).removeprefix(_SHA256)


def export_lrts_parquet(recipe: LRTSETLRecipe, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Create real-source analytical/native Parquet and write complete.json last, atomically.

    Source verification and conversion precede output publication. Reported zero,
    unavailable quantities and every skipped/unknown observation remain in the
    analytical table; native exclusions have conserved primary reasons. Native
    rounds/feedback/costs/label correspondence are declared simulation assumptions.
    Three uncompressed Parquet tables use fixed settings and record Polars version;
    logical identities remain independent of physical Parquet bytes. Identical
    concurrent retries converge through the existing immutable writer. A failed
    operation may retain individual files but never a new completion receipt.
    Cache/output must be separate and outside Git; trusted-local FS limits apply.
    Invalid source/profile/bytes raise ValueError; I/O errors propagate.
    """
    recipe = LRTSETLRecipe.model_validate(recipe)
    recipe.require_native_feedback()
    validate_cache_separation(cache.root, writer.run_root)
    validate_source_storage(writer.run_root)
    tables = _tables(recipe, cache)
    producer = lrts_etl_producer()
    identity = _namespace(recipe, producer, pl.__version__)
    root = _output(writer.run_root, "corpus-etl/" + identity)
    records = []
    for relative, table in (
        ("observations.parquet", tables.observations),
        ("builds.parquet", tables.builds),
        (f"datasets/{recipe.project}/features-engineered.parquet", tables.native),
    ):
        buffer = io.BytesIO()
        table.write_parquet(
            buffer, compression="uncompressed", statistics=False, row_group_size=8192, data_page_size=65536
        )
        payload = buffer.getvalue()
        if len(payload) > 64 * 1024**2:
            raise ValueError("ETL Parquet exceeds 64 MiB file limit")
        writer.write_immutable_stream(_output(root, relative), (payload,), maximum_bytes=len(payload))
        records.append(
            ETLFile(
                path=relative,
                sha256=_SHA256 + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=table.height,
            )
        )
    receipt = LRTSETLReceipt(
        schema_version=1,
        recipe=recipe,
        producer_sha256=producer,
        parquet_writer=pl.__version__,
        source_datasets=tables.source_datasets,
        files=tuple(records),
        logical_rows_sha256=tables.logical_digest,
        observed_cases=tables.observations.height,
        replay_cases=tables.native.height,
        excluded_not_run=tables.exclusions["not_run"],
        excluded_unknown_outcome=tables.exclusions["unknown_outcome"],
        excluded_unavailable_duration=tables.exclusions["unavailable_duration"],
        native_builds=tables.native["BuildId"].n_unique(),
        metadata_accounting=tables.metadata_accounting,
    )
    completion = _output(root, "complete.json")
    writer.write_immutable(completion, receipt.to_json() + "\n")
    return completion


def validate_lrts_etl(receipt: LRTSETLReceipt, cache: SourceCache, root: Path) -> None:
    """Recompute logical rows and verify actual source/table bytes offline; never repair or grant admission.

    Different writer versions may encode equivalent rows differently; recorded
    file digests bind the original physical artifacts and canonical rows establish
    semantic correspondence. Missing/corrupt/oversized files, changed schema/rows
    or contradictory metadata reject. No network or runner dependency is loaded.
    """
    receipt = LRTSETLReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, root)
    validate_source_storage(root)
    tables = _tables(receipt.recipe, cache)
    expected = {
        "observations.parquet": tables.observations,
        "builds.parquet": tables.builds,
        f"datasets/{receipt.recipe.project}/features-engineered.parquet": tables.native,
    }
    if tables.logical_digest != receipt.logical_rows_sha256 or tables.source_datasets != receipt.source_datasets:
        raise ValueError("ETL logical rows/source identities disagree")
    if tables.metadata_accounting != receipt.metadata_accounting:
        raise ValueError("ETL metadata selection counts disagree with actual source")
    if (
        receipt.observed_cases,
        receipt.replay_cases,
        receipt.excluded_not_run,
        receipt.excluded_unknown_outcome,
        receipt.excluded_unavailable_duration,
        receipt.native_builds,
    ) != (
        tables.observations.height,
        tables.native.height,
        tables.exclusions["not_run"],
        tables.exclusions["unknown_outcome"],
        tables.exclusions["unavailable_duration"],
        tables.native["BuildId"].n_unique(),
    ):
        raise ValueError("ETL reported row/exclusion counts disagree with actual source")
    for item in receipt.files:
        path = _output(root, item.path)
        with path.open("rb") as stream:
            payload = stream.read(item.size_bytes + 1)
        if len(payload) != item.size_bytes or _SHA256 + hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("ETL file bytes disagree with completion receipt")
        if not pl.read_parquet(io.BytesIO(payload)).equals(expected[item.path]):
            raise ValueError("ETL Parquet schema/rows disagree with pinned source")
