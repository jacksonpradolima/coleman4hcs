"""Owned controlled-observation schema fixture beside a real native CI checkpoint workflow.

Run outside checkout: python ci.py OUTPUT_DIRECTORY EXTERNAL_CACHE_DIRECTORY.
This synthetic correctness fixture does not execute foreign subject code, pretend
its constructed clocks are CI events, or project elapsed counters into runner costs.
"""

import hashlib
import io
import json
import math
import sys
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.artifacts import ArtifactWriter
from coleman.corpus import MaterializationSpec, SourceCache, load_registry, materialize_source, validate_bundled_source
from coleman.corpus.controlled import convert_controlled_source, publish_controlled_dataset
from coleman.corpus.controlled_contracts import ControlledContext, ControlledReport, ControlledSpec
from coleman.corpus.materialization import validate_cache_separation
from coleman.corpus.reference import tiny_ci_reference
from coleman.manifest import generate_manifest
from coleman.pilot import SECONDS
from coleman.selection import Quantity
from coleman.spec.models import RunSpec


def materialized(cache, name, payload, version):
    """Cache only these newly authored bytes using the existing verified source-package contract."""
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    spec = MaterializationSpec.model_validate(
        {
            "schema_version": 1,
            "package_id": name,
            "archive": "raw",
            "source": {
                "uri": "urn:coleman:owned:" + name,
                "version": version,
                "citation": "Repository-authored controlled CI fixture; no third-party data",
                "sha256": digest,
                "size_bytes": len(payload),
                "availability": "not_verified",
            },
            "license": load_registry().card("ci-serial-boundary-v1").manifest.license.model_dump(),
            "members": [{"name": name + ".json", "sha256": digest, "size_bytes": len(payload)}],
        }
    )
    return materialize_source(spec, cache, lambda: io.BytesIO(payload))


def owned_dataset(cache):
    """Construct explicit same-test pairs, including inconclusive and repeated-trial observations."""
    inputs = []
    for kind, revision in [
        ("buggy_source", "buggy-v1"),
        ("fixed_source", "fixed-v1"),
        ("test_fixture", "fixed-v1"),
        ("harness", "harness-v1"),
    ]:
        receipt = materialized(cache, kind, json.dumps({"owned": kind}).encode(), revision)
        inputs.append({"name": kind, "kind": kind, "materialization": receipt.model_dump()})
    context = ControlledContext.model_validate(
        {
            "schema_version": 1,
            "project": "owned-ci",
            "fault_id": "owned-fault",
            "language": "Python",
            "fault_origin": "synthetic",
            "buggy_revision": "buggy-v1",
            "fixed_revision": "fixed-v1",
            "fixture_revision": "fixed-v1",
            "dependence_group": "owned-ci-single-fixture",
            "test_ids": ["ci.parse"],
            "inputs": inputs,
            "runtime": {
                "implementation": "schema-fixture",
                "version": "1",
                "platform": "constructed",
                "execution_model": "constructed",
                "image_digest": None,
                "emulated": False,
                "timezone": "UTC",
                "setup": ["No subject process is executed by this schema fixture"],
                "commands": ["Construct schema-only observations"],
                "interventions": [],
            },
            "limitations": ["Synthetic schema-only observations; no fault diversity or historical CI claim"],
        }
    )
    rows = []
    for trial, role, outcome in [
        ("trial-1", "buggy", "assertion_failed"),
        ("trial-1", "fixed", "passed"),
        ("trial-2", "buggy", "timeout"),
    ]:
        rows.append(
            {
                "trial_id": trial,
                "role": role,
                "test_id": "ci.parse",
                "outcome": outcome,
                "elapsed": {"value": None, "status": "unavailable"},
                "capture_started_at": "2026-01-01T00:00:00Z",
                "capture_finished_at": "2026-01-01T00:00:00Z",
                "diagnostics_sha256": None,
            }
        )
    report = ControlledReport.model_validate(
        {"schema_version": 1, "context_fingerprint": context.fingerprint, "observations": rows}
    )
    receipt = materialized(cache, "controlled-report", report.to_json().encode(), "owned-report-v1")
    spec = ControlledSpec(
        schema_version=1, context=context, report_materialization=receipt, report_member="controlled-report.json"
    )
    return convert_controlled_source(spec, cache)


def main(output, cache_directory):
    """Reuse native CI/policies/artifacts/checkpoints; independently verify one complete owned serial suite."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Owned output must remain within the current workspace")
    for name in ("inputs/owned-ci/features-engineered.parquet", "native"):
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError("Owned output descendants must remain inside their output tree")
    validate_cache_separation(cache_directory, root)
    cache = SourceCache(cache_directory)
    dataset = owned_dataset(cache)
    source_spec_path = root / "controlled-spec.json"
    ArtifactWriter(root).write_immutable(source_spec_path, dataset.spec.to_json() + "\n")
    fixture = validate_bundled_source("ci-serial-boundary-v1").fixture
    rows = [
        {
            "BuildId": 1,
            "Name": test.test_id,
            "Duration": float(test.duration.value),
            "Verdict": int(test.outcome == "failed"),
            "CalcPrio": 0,
            "LastRun": "",
        }
        for test in fixture.build.tests
    ]
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    payload = buffer.getvalue()
    ArtifactWriter(root).write_immutable_stream(
        root / "inputs/owned-ci/features-engineered.parquet", (payload,), maximum_bytes=len(payload)
    )
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["owned-ci"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [7]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "results": {"out_dir": str(root / "native"), "top_k_prioritization": 3},
            "extensions": {
                "controlled": {"dataset_fingerprint": dataset.fingerprint, "scope": "synthetic-schema-fixture"}
            },
        }
    )
    first = run(spec)
    second = run(spec)
    if first.run_id != second.run_id or second.artifacts_dir is None:
        raise ValueError("Native checkpoint continuation lost its identity")
    directory = Path(second.artifacts_dir)
    native = pl.read_parquet(str(directory / "results/**/*.parquet"), hive_partitioning=True)
    if native.height != 1:
        raise ValueError("Native checkpoint must retain exactly one committed result")
    row = native.row(0, named=True)
    reference = tiny_ci_reference(
        fixture.build, tuple(json.loads(row["prioritization_order_top_k"])), Quantity(value=7, semantics=SECONDS)
    )
    for actual, expected in ((row["fitness"], reference.napfd.value), (row["cost"], reference.apfdc.value)):
        if expected is None or not math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native metric disagrees with independent exact CI reference")
    path = publish_controlled_dataset(dataset, cache, ArtifactWriter(directory))
    manifest = json.loads(generate_manifest(directory, run_id=second.run_id).read_bytes())
    if not any(item["type"] == "corpus_controlled_dataset" for item in manifest["artifacts"]):
        raise ValueError("Native manifest omitted controlled provenance")
    print(
        json.dumps(
            {
                "spec": str(source_spec_path),
                "dataset": str(path),
                "native_root": str(directory),
                "native_rows": native.height,
                "classifications": [pair.classification for pair in dataset.pairs],
                "admission": dataset.admission,
                "generalization": dataset.generalization,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
