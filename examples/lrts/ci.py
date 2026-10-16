"""Run an owned LRTS/Jenkins slice through real native CI and checkpoint continuation.

Outside checkout: python ci.py OUTPUT_DIRECTORY EXTERNAL_CACHE_DIRECTORY
All input bytes are repository-authored. Only this constructed single-suite
fixture declares serial-work equivalence; external Jenkins reports remain blocked.
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
from coleman.corpus.lrts import convert_lrts_source, publish_lrts_dataset
from coleman.corpus.lrts_contracts import METADATA_COLUMNS, LRTSSpec
from coleman.corpus.reference import tiny_ci_reference
from coleman.corpus.registry import validate_bundled_source
from coleman.manifest import generate_manifest
from coleman.pilot import SECONDS
from coleman.selection import Quantity
from coleman.spec.models import RunSpec


def _owned_bytes():
    """Construct two format-compatible sources from an independent owned CI known-answer fixture."""
    fixture = validate_bundled_source("ci-serial-boundary-v1").fixture
    tests = fixture.build.tests
    total = sum(float(test.duration.value) for test in tests)
    failed = sum(test.outcome == "failed" for test in tests)
    passed = len(tests) - failed
    row = dict.fromkeys(METADATA_COLUMNS, "")
    row.update(
        project="owned-ci",
        pr_name="PR-owned",
        build_id="1",
        stage_id="owned-serial",
        build_duration=str(int(total)),
        build_timestamp="0",
        build_result="FAILURE",
        ts_duration=str(total),
        passcount=str(passed),
        failcount=str(failed),
        skipcount="0",
        stage_duration_by_suite_sum=str(total),
        test_suite_duration_s=str(total),
        num_pass_class=str(passed),
        num_fail_class=str(failed),
        num_trans_class=str(failed),
        num_pass_method=str(passed),
        num_fail_method=str(failed),
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=METADATA_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    report = {
        "_class": "hudson.tasks.junit.TestResult",
        "duration": total,
        "empty": False,
        "passCount": passed,
        "failCount": failed,
        "skipCount": 0,
        "suites": [
            {
                "name": "owned.Suite",
                "id": "owned-suite",
                "nodeId": "owned-node",
                "enclosingBlocks": ["owned-block"],
                "enclosingBlockNames": ["owned serial"],
                "timestamp": None,
                "duration": total,
                "cases": [
                    {
                        "className": test.test_id,
                        "name": "owned-case",
                        "duration": float(test.duration.value),
                        "status": "FAILED" if test.outcome == "failed" else "PASSED",
                        "skipped": False,
                    }
                    for test in tests
                ],
            }
        ],
    }
    return buffer.getvalue().encode(), json.dumps(report, sort_keys=True, allow_nan=False).encode(), fixture


def owned_dataset(cache):
    """Materialize and convert newly authored schema fixtures; never download or license third-party rows."""
    metadata, report, fixture = _owned_bytes()
    receipts = []
    for label, payload, member in (("metadata", metadata, "metadata.csv"), ("report", report, "report.json")):
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        spec = MaterializationSpec.model_validate(
            {
                "schema_version": 1,
                "package_id": f"owned-lrts-{label}-demo-v1",
                "archive": "raw",
                "source": {
                    "uri": f"urn:coleman:owned:lrts-{label}-demo-v1",
                    "version": "owned-lrts-format-v1",
                    "citation": "Newly authored format-compatible CI fixture, not third-party LRTS data",
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "availability": "not_verified",
                },
                "license": load_registry().card("ci-serial-boundary-v1").manifest.license,
                "members": [{"name": member, "sha256": digest, "size_bytes": len(payload)}],
            }
        )
        receipts.append(materialize_source(spec, cache, lambda payload=payload: io.BytesIO(payload)))
    source = LRTSSpec(
        schema_version=1,
        metadata_materialization=receipts[0],
        metadata_member="metadata.csv",
        report_materialization=receipts[1],
        report_member="report.json",
        project="owned-ci",
        pull_request="PR-owned",
        build_id=1,
        report_origin_path="owned-ci/PR/PR-owned/testReport_build1.json",
        history="constructed",
    )
    return convert_lrts_source(source, cache), fixture


def _output_root(output, cache_directory):
    """Resolve output containment and external-cache separation before writing owned artifacts."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Owned output must remain inside the current workspace")
    for relative in ("inputs/isolated-suite/features-engineered.parquet", "native"):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Owned output descendants must remain inside the output tree")
    validate_cache_separation(cache_directory, root)
    return root


def _native_input(root, dataset):
    """Project only this complete owned serial fixture, never generic external reported durations."""
    if dataset.spec.history != "constructed" or len(dataset.suites) != 1:
        raise ValueError("Native projection requires the owned single-suite constructed fixture")
    cases = dataset.cases
    if any(case.duration.value is None or case.outcome not in {"passed", "failed"} for case in cases):
        raise ValueError("Owned projection requires complete durations and verdicts")
    if len({case.class_name for case in cases}) != len(cases):
        raise ValueError("Owned projection requires exactly one method per class label")
    rows = [
        {
            "BuildId": 1,
            "Name": case.class_name,
            "Duration": case.duration.value,
            "CalcPrio": 0,
            "LastRun": "",
            "Verdict": int(case.outcome == "failed"),
        }
        for case in cases
    ]
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    data = buffer.getvalue()
    path = root / "inputs/isolated-suite/features-engineered.parquet"
    ArtifactWriter(root).write_immutable_stream(path, (data,), maximum_bytes=len(data))


def main(output, cache_directory):
    """Run and reopen the existing native checkpoint, independently checking actual persisted metrics."""
    root = _output_root(output, cache_directory)
    cache = SourceCache(cache_directory)
    dataset, fixture = owned_dataset(cache)
    _native_input(root, dataset)
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["isolated-suite"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [7]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "results": {"out_dir": str(root / "native"), "top_k_prioritization": 3},
            "extensions": {"lrts": {"dataset_fingerprint": dataset.fingerprint, "scope": "owned-isolated-serial"}},
        }
    )
    first = run(spec)
    second = run(spec)
    if first.run_id != second.run_id or second.artifacts_dir is None:
        raise ValueError("Native continuation lost its run identity or artifact directory")
    directory = Path(second.artifacts_dir)
    native = pl.read_parquet(str(directory / "results/**/*.parquet"), hive_partitioning=True)
    if native.height != 1:
        raise ValueError("Owned suite must persist one result row after checkpoint continuation")
    row = native.row(0, named=True)
    reference = tiny_ci_reference(
        fixture.build,
        tuple(json.loads(row["prioritization_order_top_k"])),
        Quantity(value=Decimal(7), semantics=SECONDS),
    )
    for actual, expected in ((row["fitness"], reference.napfd.value), (row["cost"], reference.apfdc.value)):
        if expected is None or not math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native metric disagrees with independent exact owned reference")
    converted = publish_lrts_dataset(dataset, cache, ArtifactWriter(directory))
    manifest = json.loads(generate_manifest(directory, run_id=second.run_id).read_bytes())
    if not any(item["type"] == "corpus_lrts_dataset" for item in manifest["artifacts"]):
        raise ValueError("Native manifest omitted LRTS source provenance")
    print(
        json.dumps(
            {
                "dataset": str(converted),
                "native_root": str(directory),
                "native_rows": native.height,
                "reported_cases": len(dataset.cases),
                "metadata_stages": len(dataset.metadata),
                "feedback_scope": dataset.feedback_scope,
                "admission": dataset.admission,
                "generalization": dataset.generalization,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
