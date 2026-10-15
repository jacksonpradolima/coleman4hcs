"""Exercise corpus correctness sources through a real installed native TCPCI run.

Two explicitly constructed copies of the same tiny fixture check integration;
these are not independent CI observations or evidence of policy superiority.
Usage: python ci.py OUTPUT_DIRECTORY (inside the current workspace).
"""

import hashlib
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    AdapterReference,
    CorpusProfileInventory,
    CorpusRegistry,
    DataCard,
    ObservationIdentity,
    TimeSemantics,
    VerdictMapping,
    fixture_readiness,
    load_registry,
    profile_native_source,
    publish_fixture_readiness,
    publish_fixture_report,
    publish_profile_inventory,
    publish_source_profile,
    tiny_ci_reference,
    validate_bundled_source,
)
from coleman.manifest import generate_manifest
from coleman.pilot import SECONDS
from coleman.results.domain import Measurement
from coleman.selection import Quantity
from coleman.spec.models import RunSpec


def main(output):
    """Run real native TCPCI into a fresh workspace directory, retaining explicit fixture provenance.

    Existing roots or paths escaping the workspace raise ValueError before writes.
    Native failures propagate; individual verified evidence files use immutable
    publication, while the whole native invocation requires a new output root.
    Declared seconds are synthetic inputs, not measured production CI timings.
    """
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Native fixture output must remain within current workspace")
    if root.exists():
        raise ValueError("Native fixture requires a fresh output directory")
    registry = load_registry()
    checked = tuple(
        validate_bundled_source(name, registry) for name in ("ci-serial-boundary-v1", "ci-parallel-occupancy-v1")
    )
    source = checked[0]
    root.mkdir(parents=True)
    directory = root / "inputs" / "constructed-fixture"
    directory.mkdir(parents=True, exist_ok=True)
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
    data_path = directory / "features-engineered.parquet"
    pl.DataFrame(rows).write_parquet(data_path)
    data_sha256 = "sha256:" + hashlib.sha256(data_path.read_bytes()).hexdigest()
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["constructed-fixture"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [7]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "extensions": {
                "corpus_demo": {
                    "source_manifest_fingerprint": source.manifest.fingerprint,
                    "registry_fingerprint": registry.fingerprint,
                    "materialized_parquet_sha256": data_sha256,
                    "history": "two-constructed-copies-of-one-tiny-source-not-independent-observations-v1",
                }
            },
            "results": {"out_dir": str(root / "native"), "manifest_enabled": True, "top_k_prioritization": 3},
        }
    )
    native = run(spec)
    run_root = _resume_native(spec, native)
    native_rows = pl.read_parquet(str(run_root / "results" / "**" / "*.parquet"), hive_partitioning=True)
    if native_rows.height != 2:
        raise ValueError("Expected two explicitly constructed native build copies")
    for row in native_rows.iter_rows(named=True):
        ranking = tuple(json.loads(row["prioritization_order_top_k"]))
        exact = tiny_ci_reference(source.fixture.build, ranking, Quantity(value=Decimal(7), semantics=SECONDS))
        if exact.napfd.value is None or exact.apfdc.value is None:
            raise ValueError("Fixture reference unexpectedly unavailable")
        if row["detected"] != 2 or row["missed"] != 0:
            raise ValueError("Native full-budget run lost a known failing test")
        if not math.isclose(row["fitness"], float(exact.napfd.value), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Real native ranking NAPFD disagrees with independent exact reference")
        if not math.isclose(row["cost"], float(exact.apfdc.value), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Real native ranking APFDc disagrees with independent exact reference")
    writer = ArtifactWriter(run_root)
    paths = [publish_fixture_report(report, writer) for report in checked]
    paths.extend(_profile_constructed(source, data_path, writer, registry))
    readiness = fixture_readiness(registry, checked)
    paths.append(publish_fixture_readiness(readiness, writer))
    generated = generate_manifest(run_root, run_id=native.run_id)
    metadata = json.loads(generated.read_bytes())
    listed = {entry["relative_path"] for entry in metadata["artifacts"]}
    if not all(path.relative_to(run_root).as_posix() in listed for path in paths):
        raise ValueError("Existing native artifact manifest lost corpus validation evidence")
    print(
        json.dumps(
            {
                "native_rows": native_rows.height,
                "fixture_sources_verified": len(checked),
                "run_id": native.run_id,
                "artifacts_dir": native.artifacts_dir,
                "evidence_status": readiness.evidence_status,
                "uncovered_regimes": len(readiness.uncovered_required_regimes),
            }
        )
    )


def _resume_native(spec, native):
    """Reopen the real native checkpoint; missing progress or changed run identity raises ValueError."""
    run_root = Path(native.artifacts_dir)
    if not list((run_root / "checkpoints").rglob("progress_*.json")):
        raise ValueError("Native fixture did not persist its continuation checkpoint")
    restored = run(spec)
    if restored.run_id != native.run_id:
        raise ValueError("Checkpoint continuation changed native run identity")
    return run_root


def _profile_constructed(source, data_path, writer, registry):
    """Describe the exact native input while retaining owned-source lineage and constructed history."""
    manifest_data = source.manifest.model_dump()
    manifest_data.update(
        benchmark_id="ci-native-constructed-copies-v1",
        adapter=AdapterReference(name="native-parquet-corpus", version="planned", implementation="planned"),
        identity=ObservationIdentity(observation_fields=("BuildId", "Name"), group_fields=("BuildId",)),
        time=TimeSemantics(
            origin="constructed",
            event_field=None,
            outcome_available_field=None,
            feature_available_field=None,
            timezone="unverified",
        ),
        outcomes={
            **source.manifest.outcomes.model_dump(),
            "mapping": (VerdictMapping(source_value="0", verdict=0), VerdictMapping(source_value="1", verdict=1)),
        },
        source={
            **source.manifest.source.model_dump(),
            "uri": "urn:coleman:constructed-native-copies:v1",
            "version": "constructed-native-v1",
            "citation": "Two copies of " + source.manifest.fingerprint,
            "sha256": "sha256:" + hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "size_bytes": data_path.stat().st_size,
        },
        conversion={"status": "not_validated", "input_rows": 6, "output_rows": None, "exclusions": ()},
        limitations=(
            "Two constructed copies of one owned fixture; no independent CI observations.",
            "Profile validates local native shape, not source admission or protected evaluation.",
        ),
    )
    card = DataCard(
        schema_version=1,
        manifest=manifest_data,
        history="constructed",
        fault_origin="synthetic",
        languages=(),
        execution_models=("serial",),
        measurements=(
            Measurement(
                name="corpus_source_rows",
                value=6.0,
                unit="1",
                meaning="constructed-native-row-count-v1",
                direction="none",
                status="observed",
                minimum=0.0,
            ),
        ),
        covered_regimes=(),
        missing_regimes=registry.required_regimes,
    )
    report = profile_native_source(card, data_path)
    if report.rows != 6 or report.logical_builds != 2 or report.failed_rows != 4:
        raise ValueError("Native constructed source profile lost known fixture counts")
    extended = CorpusRegistry(
        schema_version=1,
        registry_id="ci-native-demo-v1",
        cards=(*registry.cards, card),
        required_regimes=registry.required_regimes,
    )
    inventory = CorpusProfileInventory(
        schema_version=1, registry=extended, profiles=(report,), uncovered_required_regimes=extended.required_regimes
    )
    return [publish_source_profile(report, writer), publish_profile_inventory(inventory, writer)]


if __name__ == "__main__":
    main(sys.argv[1])
