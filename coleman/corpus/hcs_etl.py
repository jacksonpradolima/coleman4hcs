"""Offline, deterministic HCS Parquet export using existing source cache and immutable artifact publication."""

from __future__ import annotations

import hashlib
import io
from importlib.resources import files
from pathlib import Path

import polars as pl

from coleman.artifacts import ArtifactWriter
from coleman.corpus.hcs_contracts import HCSReceipt, HCSRecipe
from coleman.corpus.hcs_projection import HCSProjection, project_hcs
from coleman.corpus.hcs_raw import read_hcs_slice
from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.materialization_contracts import safe_member_name
from coleman.results.domain import fingerprint

_SHA256_PREFIX = "sha256:"


def hcs_producer() -> str:
    """Identify actual conversion/contract/storage code, independently of user labels or transport.

    An implementation change creates a new export namespace. Retain the original
    wheel/Polars version to verify old receipts; no automatic content migration.
    """
    modules = {
        "coleman.corpus": (
            "hcs_contracts.py",
            "hcs_raw.py",
            "hcs_projection.py",
            "hcs_etl.py",
            "contracts.py",
            "materialization.py",
            "materialization_contracts.py",
            "lrts_contracts.py",
            "lrts_etl_contracts.py",
        ),
        "coleman": ("artifacts.py",),
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


def _target(root: Path, relative: str) -> Path:
    """Resolve only portable contained paths outside Git; cooperative local parents must remain trusted."""
    safe_member_name(relative)
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("HCS output escapes its trusted artifact tree")
    validate_source_storage(path)
    return path


def preflight_hcs_etl(recipe: HCSRecipe, cache: Path, output: Path) -> Path:
    """Check source/output isolation and derive a deterministic namespace without creating or acquiring files.

    This is a trusted-local path preflight, not a hostile-filesystem sandbox.
    Invalid metadata/overlap/symlinks raise ValueError; I/O failures propagate.
    """
    recipe = HCSRecipe.model_validate(recipe)
    validate_source_storage(cache)
    validate_source_storage(output)
    validate_cache_separation(cache, output)
    identity = fingerprint([recipe.fingerprint, hcs_producer(), pl.__version__]).removeprefix(_SHA256_PREFIX)
    root = _target(output, "hcs-etl/" + identity)
    for relative in (
        "complete.json",
        "observations.parquet",
        "cells.parquet",
        f"datasets/{recipe.dataset_id}/features-engineered.parquet",
        f"datasets/{recipe.dataset_id}/data-variants.parquet",
        "vts",
    ):
        _target(root, relative)
    return root


def _encode(frame: pl.DataFrame) -> bytes:
    """Use an uncompressed deterministic Polars profile within 64 MiB; no untrusted Parquet is decoded."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer, compression="uncompressed", statistics=False, row_group_size=8192, data_page_size=65536)
    payload = buffer.getvalue()
    if len(payload) > 64 * 1024**2:
        raise ValueError("HCS Parquet exceeds 64 MiB")
    return payload


def _receipt(recipe: HCSRecipe, projection: HCSProjection) -> tuple[HCSReceipt, tuple[bytes, ...]]:
    """Compute every exact expected table digest and source-derived count before any artifact is committed."""
    payloads, records = [], []
    for path, frame in projection.tables:
        payload = _encode(frame)
        records.append(
            ETLFile(
                path=path,
                sha256=_SHA256_PREFIX + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=frame.height,
            )
        )
        payloads.append(payload)
    receipt = HCSReceipt(
        schema_version=1,
        recipe=recipe,
        producer_sha256=hcs_producer(),
        parquet_writer=pl.__version__,
        accounting=projection.accounting,
        variants=projection.variants,
        files=tuple(records),
        logical_rows_sha256=projection.logical_digest,
    )
    return receipt, tuple(payloads)


def export_hcs_parquet(recipe: HCSRecipe, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Verify source and publish selected observations, sparse cells, union/HCS and VTS Parquet; complete last.

    Raw original files remain in the external cache; outside-slice rows are
    conserved in the receipt. Logical identities exclude measurements/positions.
    Serial test seconds, surrogate union labels and zero-nondetection native
    RNFail are explicit simulation assumptions. Missing setup/occupancy stays
    null. No legacy artifact, temporal admission or policy ranking is altered.
    Identical concurrent retries converge through existing immutable writes;
    interruptions may leave completed individual tables, never a false completion
    receipt. Unsupported source/identity/units/path errors raise ValueError and
    I/O errors propagate. No runner, CLI, MCP or network is imported.
    """
    recipe = HCSRecipe.model_validate(recipe)
    root = preflight_hcs_etl(recipe, cache.root, writer.run_root)
    projection = project_hcs(read_hcs_slice(recipe, cache))
    receipt, payloads = _receipt(recipe, projection)
    for item in receipt.files:
        _target(root, item.path)
    for item, payload in zip(receipt.files, payloads, strict=True):
        writer.write_immutable_stream(_target(root, item.path), (payload,), maximum_bytes=item.size_bytes)
    completion = _target(root, "complete.json")
    writer.write_immutable(completion, receipt.to_json() + "\n")
    return completion


def validate_hcs_etl(receipt: HCSReceipt, cache: SourceCache, root: Path) -> None:
    """Recompute exact source projection and every table byte; self-signed modified Parquet cannot certify truth.

    Parsing completion metadata alone is insufficient. Unknown producer/writer
    requires its recorded package/version, without implicit migration. Validation
    is offline and never repairs/acquires/publishes data or starts native runs.
    Unsafe, missing, changed or oversized files fail with ValueError/OSError.
    """
    receipt = HCSReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, root)
    validate_source_storage(root)
    if (receipt.producer_sha256, receipt.parquet_writer) != (hcs_producer(), pl.__version__):
        raise ValueError("Unsupported HCS producer/writer; use the recorded implementation")
    expected, _ = _receipt(receipt.recipe, project_hcs(read_hcs_slice(receipt.recipe, cache)))
    if receipt != expected:
        raise ValueError("HCS receipt disagrees with source-derived projection")
    for item in receipt.files:
        with _target(root, item.path).open("rb") as stream:
            payload = stream.read(item.size_bytes + 1)
        if len(payload) != item.size_bytes or _SHA256_PREFIX + hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("HCS artifact bytes disagree with verified source")
