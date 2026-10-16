"""Run an owned RTPTorrent-format slice through existing native CI and checkpoint recovery.

Usage outside the checkout: python ci.py OUTPUT_DIRECTORY EXTERNAL_CACHE_DIRECTORY
No third-party archive is downloaded. Source labels and clocks are constructed;
the isolated serial check is not a temporal evaluation or corpus admission.
"""

import csv
import hashlib
import io
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    MaterializationSpec,
    SourceCache,
    load_registry,
    materialize_source,
    validate_cache_separation,
)
from coleman.corpus.reference import tiny_ci_reference
from coleman.corpus.registry import validate_bundled_source
from coleman.corpus.rtptorrent import convert_rtptorrent_source, publish_rtptorrent_dataset
from coleman.corpus.rtptorrent_contracts import RTPTorrentSpec
from coleman.manifest import generate_manifest
from coleman.pilot import SECONDS
from coleman.selection import Quantity
from coleman.spec.models import RunSpec


def owned_source():
    """Create a format-compatible owned CSV with failure, green/repeated-class and unknown jobs."""
    fixture = validate_bundled_source("ci-serial-boundary-v1").fixture
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("travisJobId", "testName", "index", "duration", "count", "failures", "errors", "skipped"))
    for position, test in enumerate(fixture.build.tests):
        writer.writerow((101, test.test_id, position, str(test.duration.value), 1, int(test.outcome == "failed"), 0, 0))
    writer.writerow((102, "owned.Repeated", 0, "0", 1, 0, 0, 0))
    writer.writerow((102, "owned.Repeated", 1, "2", 1, 0, 0, 0))
    writer.writerow((103, "owned.Unknown", 0, "", "", 0, "", 0))
    payload = buffer.getvalue().encode()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    spec = MaterializationSpec.model_validate(
        {
            "schema_version": 1,
            "package_id": "owned-rtptorrent-format-v1",
            "archive": "raw",
            "source": {
                "uri": "urn:coleman:owned:rtptorrent-format-v1",
                "version": "1.1",
                "citation": "Repository-authored format-compatible CSV; not third-party RTPTorrent data",
                "sha256": digest,
                "size_bytes": len(payload),
                "availability": "not_verified",
            },
            "license": load_registry().card("ci-serial-boundary-v1").manifest.license,
            "members": [{"name": "class-results.csv", "sha256": digest, "size_bytes": len(payload)}],
        }
    )
    return spec, payload, fixture


def _output_root(output, cache_directory):
    """Resolve owned output containment and external-cache separation before any publication."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Owned output must remain in the current workspace")
    for relative in ("inputs/isolated-job/features-engineered.parquet", "native"):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Owned output descendants must remain inside the output tree")
    validate_cache_separation(cache_directory, root)
    return root


def main(output, cache_directory):
    """Materialize/convert owned data, isolate one known-answer job, run and reopen its native checkpoint."""
    root = _output_root(output, cache_directory)
    materialization, payload, fixture = owned_source()
    cache = SourceCache(cache_directory)
    receipt = materialize_source(materialization, cache, lambda: io.BytesIO(payload))
    spec = RTPTorrentSpec(
        schema_version=1, materialization=receipt, member="class-results.csv", project="owned/ci", history="constructed"
    )
    dataset = convert_rtptorrent_source(spec, cache)
    selected = dataset.job(101)
    # Only this constructed job is projected: no cross-job learning or inferred CI clock.
    rows = [
        {
            "BuildId": 1,
            "Name": item.test_class,
            "Duration": float(item.duration.value),
            "CalcPrio": 0,
            "LastRun": "",
            "Verdict": int(item.outcome == "failed"),
        }
        for item in selected
        if item.duration is not None and item.outcome in {"passed", "failed"}
    ]
    if len(rows) != len(selected) or len({row["Name"] for row in rows}) != len(rows):
        raise ValueError("Owned isolated replay requires complete quantities and unique class labels")
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    data = buffer.getvalue()
    path = root / "inputs/isolated-job/features-engineered.parquet"
    ArtifactWriter(root).write_immutable_stream(path, (data,), maximum_bytes=len(data))
    run_spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["isolated-job"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [7]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "results": {"out_dir": str(root / "native"), "top_k_prioritization": 3},
            "extensions": {
                "rtptorrent": {"dataset_fingerprint": dataset.fingerprint, "scope": "owned-isolated-serial"}
            },
        }
    )
    first = run(run_spec)
    second = run(run_spec)
    if first.run_id != second.run_id:
        raise ValueError("Native checkpoint continuation changed run identity")
    if second.artifacts_dir is None:
        raise ValueError("Native isolated-job replay did not return its artifact directory")
    directory = Path(second.artifacts_dir)
    native = pl.read_parquet(str(directory / "results/**/*.parquet"), hive_partitioning=True)
    if native.height != 1:
        raise ValueError("Isolated job replay must have one persisted row after continuation")
    row = native.row(0, named=True)
    reference = tiny_ci_reference(
        fixture.build,
        tuple(json.loads(row["prioritization_order_top_k"])),
        Quantity(value=Decimal(7), semantics=SECONDS),
    )
    for actual, expected in ((row["fitness"], reference.napfd.value), (row["cost"], reference.apfdc.value)):
        if expected is None or not math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native isolated-job metric disagrees with independent exact reference")
    converted = publish_rtptorrent_dataset(dataset, cache, ArtifactWriter(directory))
    manifest = json.loads(generate_manifest(directory, run_id=second.run_id).read_bytes())
    if not any(item["type"] == "corpus_rtptorrent_dataset" for item in manifest["artifacts"]):
        raise ValueError("Native manifest omitted converted source provenance")
    print(
        json.dumps(
            {
                "dataset": str(converted),
                "native_root": str(directory),
                "native_rows": native.height,
                "input_rows": dataset.input_rows,
                "source_jobs": len(dataset.job_ids),
                "repeated_class_executions": dataset.repeated_class_executions,
                "admission": dataset.admission,
                "generalization": dataset.generalization,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
