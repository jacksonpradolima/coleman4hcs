"""Materialize owned CI fixtures into an external cache and run real native TCPCI with checkpoint recovery.

Usage outside the checkout: python ci.py OUTPUT_DIRECTORY EXTERNAL_CACHE_DIRECTORY
The two constructed build copies are functional checks, never independent observations.
"""

import hashlib
import io
import json
import math
import stat
import sys
import zipfile
from decimal import Decimal
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    MaterializationSpec,
    SourceCache,
    bundled_fixture_bytes,
    load_registry,
    materialize_source,
    publish_fixture_report,
    publish_materialization,
    tiny_ci_reference,
    validate_cache_separation,
    validate_fixture_source,
    validate_materialization,
)
from coleman.manifest import generate_manifest
from coleman.pilot import SECONDS
from coleman.selection import Quantity
from coleman.spec.models import RunSpec

_SHA256_PREFIX = "sha256:"


def owned_bundle():
    """Build a deterministic tiny owned ZIP; specific fixture terms remain explicit and frozen."""
    registry = load_registry()
    names = (("ci-serial-boundary-v1", "serial.json"), ("ci-parallel-occupancy-v1", "parallel.json"))
    buffer = io.BytesIO()
    members = []
    with zipfile.ZipFile(buffer, "w") as archive:
        for source_id, name in names:
            payload = bundled_fixture_bytes(source_id)
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, payload)
            members.append(
                {
                    "name": name,
                    "sha256": _SHA256_PREFIX + hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
    payload = buffer.getvalue()
    spec = MaterializationSpec.model_validate(
        {
            "schema_version": 1,
            "package_id": "owned-native-ci-bundle-v1",
            "archive": "zip",
            "source": {
                "uri": "urn:coleman:owned-ci-bundle:v1",
                "version": "owned-bundle-v1",
                "citation": "Two repository-authored CI correctness fixtures with one shared dependence group",
                "sha256": _SHA256_PREFIX + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "availability": "not_verified",
            },
            "license": registry.card("ci-serial-boundary-v1").manifest.license,
            "members": members,
        }
    )
    return spec, payload


def _native_spec(root, source, receipt):
    """Persist immutable owned input and bind real native execution to fixture/package provenance."""
    rows = [
        {
            "BuildId": build,
            "Name": test.test_id,
            "Duration": float(test.duration.value),
            "CalcPrio": 0,
            "LastRun": "2026-01-01",
            "Verdict": int(test.outcome == "failed"),
        }
        for build in (1, 2)
        for test in source.fixture.build.tests
    ]
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    data = buffer.getvalue()
    writer = ArtifactWriter(root)
    path = root / "inputs" / "constructed-ci" / "features-engineered.parquet"
    writer.write_immutable_stream(path, (data,), maximum_bytes=len(data))
    return RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["constructed-ci"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [7]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "extensions": {
                "corpus_materialization": {
                    "spec_fingerprint": receipt.spec.fingerprint,
                    "source_fingerprint": source.manifest.fingerprint,
                    "native_source_sha256": _SHA256_PREFIX + hashlib.sha256(data).hexdigest(),
                    "history": "two-constructed-copies-of-one-owned-fixture-not-independent-observations-v1",
                }
            },
            "results": {"out_dir": str(root / "native"), "manifest_enabled": True, "top_k_prioritization": 3},
        }
    )


def _check_native(result, source):
    """Check real persisted rankings/metrics with the independent exact tiny oracle; timings are not evidence."""
    directory = Path(result.artifacts_dir)
    rows = pl.read_parquet(str(directory / "results" / "**" / "*.parquet"), hive_partitioning=True)
    if rows.height != 2:
        raise ValueError("Native checkpoint continuation must preserve two constructed build rows")
    for row in rows.iter_rows(named=True):
        ranking = tuple(json.loads(row["prioritization_order_top_k"]))
        reference = tiny_ci_reference(source.fixture.build, ranking, Quantity(value=Decimal(7), semantics=SECONDS))
        if reference.napfd.value is None or reference.apfdc.value is None:
            raise ValueError("Unexpected unavailable owned correctness reference")
        if row["detected"] != 2 or row["missed"] != 0:
            raise ValueError("Native full-budget CI lost a known failing test")
        if not math.isclose(row["fitness"], float(reference.napfd.value), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native NAPFD disagrees with independent exact reference")
        if not math.isclose(row["cost"], float(reference.apfdc.value), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native APFDc disagrees with independent exact reference")
    return directory


def main(output, cache_root):
    """Materialize, verify offline, run/reopen native checkpoints and publish metadata; identical retry is safe."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Owned CI output must remain within the current workspace")
    for relative in ("inputs/constructed-ci/features-engineered.parquet", "native", "package.json", "owned-source.zip"):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Owned CI output descendants must remain within its trusted root")
    validate_cache_separation(cache_root, root)
    spec, payload = owned_bundle()
    cache = SourceCache(cache_root)
    receipt = materialize_source(spec, cache, lambda: io.BytesIO(payload))
    validate_materialization(receipt, cache)
    registry = load_registry()
    reports = tuple(
        validate_fixture_source(registry.card(source_id).manifest, cache.member_path(receipt, name).read_bytes())
        for source_id, name in (("ci-serial-boundary-v1", "serial.json"), ("ci-parallel-occupancy-v1", "parallel.json"))
    )
    writer = ArtifactWriter(root)
    writer.write_immutable(root / "package.json", spec.to_json() + "\n")
    writer.write_immutable_stream(root / "owned-source.zip", (payload,), maximum_bytes=len(payload))
    native_spec = _native_spec(root, reports[0], receipt)
    first = run(native_spec)
    second = run(native_spec)
    if first.run_id != second.run_id:
        raise ValueError("Native recovery changed run identity")
    directory = _check_native(second, reports[0])
    if not list((directory / "checkpoints").rglob("progress_*.json")):
        raise ValueError("Real native CI did not persist continuation checkpoints")
    publication = ArtifactWriter(directory)
    paths = [publish_materialization(receipt, cache, publication)]
    paths += [publish_fixture_report(report, publication) for report in reports]
    manifest = json.loads(generate_manifest(directory, run_id=second.run_id).read_bytes())
    listed = {entry["relative_path"] for entry in manifest["artifacts"]}
    if not all(path.relative_to(directory).as_posix() in listed for path in paths):
        raise ValueError("Native manifest omitted materialization/correctness provenance")
    print(
        json.dumps(
            {
                "receipt": str(paths[0]),
                "native_root": str(directory),
                "native_rows": 2,
                "members_verified": 2,
                "integrity": "verified",
                "admission": receipt.admission,
                "generalization": "insufficient_evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
