"""Owned multi-stage histories exercise actual source mapping, Parquet and native recovery."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import materialize_selected_packages
from coleman.corpus.lrts_etl import export_lrts_parquet
from coleman.corpus.lrts_stage_contracts import LRTSStageBinding, LRTSStageReceipt, LRTSStageRecipe
from coleman.corpus.lrts_stage_etl import export_lrts_stage_parquet, validate_lrts_stage_etl
from coleman.corpus.materialization import SourceCache, materialize_source
from tests.corpus.test_lrts_etl import owned_recipe
from tests.corpus.test_source_views import changed


def stage_recipe(*, transform=lambda r: r, metadata_transform=lambda rows: rows):
    """Construct two correlated stages with repeated labels and distinct unavailable/outcome states."""

    def report(value):
        value["suites"][0]["enclosingBlockNames"] = ["owned linux"]
        value["suites"][1]["enclosingBlockNames"] = ["owned windows"]
        value["suites"][1]["cases"].append(
            dict(className="owned.Repeated", name="method", duration=1, status="PASSED", skipped=False)
        )
        value["passCount"] = 4
        return transform(value)

    def metadata(rows):
        result = []
        for row in rows:
            for stage in ("linux", "windows"):
                result.append({**row, "stage_id": stage, "passcount": "4"})
        return metadata_transform(result)

    source, outer, csv = owned_recipe(transform=report, metadata_transform=metadata)
    bindings = [
        dict(
            pull_request=b.pull_request,
            build_id=b.build_id,
            stage_id=stage,
            enclosing_block_names=[context],
            node_ids=[node],
        )
        for b in source.builds
        for stage, context, node in [("linux", "owned linux", "node-a"), ("windows", "owned windows", "node-b")]
    ]
    recipe = LRTSStageRecipe(
        schema_version=1,
        source=source,
        dataset_id="owned-linux",
        native_stage_id="linux",
        stage_bindings=tuple(LRTSStageBinding.model_validate(b) for b in bindings),
    )
    return recipe, outer, csv


def prepare(tmp_path, **kwargs):
    recipe, outer, csv = stage_recipe(**kwargs)
    cache = SourceCache(tmp_path / "cache")
    materialize_source(recipe.source.metadata, cache, lambda: io.BytesIO(csv))
    materialize_selected_packages(recipe.source.reports, cache, lambda: io.BytesIO(outer))
    return recipe, cache


def test_complete_source_stage_roundtrip_conservation_and_correlated_views(tmp_path):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    path = export_lrts_stage_parquet(recipe, cache, writer)
    receipt = LRTSStageReceipt.from_json(path.read_bytes())
    assert LRTSStageReceipt.from_json(receipt.to_json()) == receipt
    assert receipt.observed_cases == 12
    assert receipt.replay_cases == 4
    assert receipt.excluded_outside_stage == 6
    assert receipt.excluded_unavailable_duration == 2
    assert receipt.source_stages == 4
    assert receipt.metadata_accounting.selected_rows == 4
    observations = pl.read_parquet(path.parent / "observations.parquet")
    assert observations["execution_id"].n_unique() == 12
    assert observations["duration"].to_list().count(0) == 2
    assert observations["duration"].null_count() == 6
    assert observations["duration_unit"].unique().to_list() == ["s"]
    builds = pl.read_parquet(path.parent / "builds.parquet")
    assert all(len(json.loads(m)) == 2 for m in builds["metadata_json"])
    assert builds["outcome_available_at"].null_count() == 2
    validate_lrts_stage_etl(receipt, cache, path.parent)
    windows = changed(recipe, native_stage_id="windows", dataset_id="owned-windows")
    other = export_lrts_stage_parquet(windows, cache, writer)
    second = LRTSStageReceipt.from_json(other.read_bytes())
    assert second.replay_cases == 2
    assert second.excluded_not_run == 2
    assert second.excluded_unknown_outcome == 2
    alternate = pl.read_parquet(other.parent / "observations.parquet")
    assert observations["execution_id"].to_list() == alternate["execution_id"].to_list()
    assert observations["source_build_group"].to_list() == alternate["source_build_group"].to_list()
    assert observations["replay_name"].to_list() == alternate["replay_name"].to_list()
    assert receipt.source_datasets == second.source_datasets
    assert receipt.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match="evaluation blocked"):
        receipt.temporal.require_temporal_evaluation()
    legacy_writer = ArtifactWriter(tmp_path / "old-output")
    with pytest.raises(ValueError, match="one explicit metadata stage"):
        export_lrts_parquet(recipe.source, cache, legacy_writer)


def test_stage_contract_golden_set_order_and_unknown_versions():
    recipe, _, _ = stage_recipe()
    assert LRTSStageRecipe.from_json(recipe.to_json()) == recipe
    assert recipe == changed(recipe, stage_bindings=tuple(reversed(recipe.stage_bindings)))
    fixture = Path(__file__).with_name("fixtures") / "lrts-stage-owned-recipe-v1.json"
    assert recipe.to_json() + "\n" == fixture.read_text()
    for version in [None, 0, 2, True, "1"]:
        value = recipe.model_dump(mode="json")
        if version is None:
            value.pop("schema_version")
        else:
            value["schema_version"] = version
        with pytest.raises(ValueError):
            LRTSStageRecipe.model_validate(value)


@pytest.mark.parametrize(
    "field,value", [("stage_id", "linux"), ("enclosing_block_names", ("owned linux",)), ("node_ids", ("node-a",))]
)
def test_ambiguous_stage_context_or_node_rejects(field, value):
    recipe, _, _ = stage_recipe()
    bindings = list(recipe.stage_bindings)
    bindings[1] = changed(bindings[1], **{field: value})
    with pytest.raises(ValueError, match="Ambiguous"):
        changed(recipe, stage_bindings=tuple(bindings))


@pytest.mark.parametrize("field,value", [("enclosing_block_names", ("owned_linux",)), ("node_ids", ("wrong-node",))])
def test_unproved_raw_correspondence_rejects_without_completion(tmp_path, field, value):
    recipe, cache = prepare(tmp_path)
    bindings = list(recipe.stage_bindings)
    bindings[0] = changed(bindings[0], **{field: value})
    recipe = changed(recipe, stage_bindings=tuple(bindings))
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError, match="raw context/node relation"):
        export_lrts_stage_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


def test_interrupted_publication_retry_and_concurrent_idempotence(tmp_path, monkeypatch):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    original = writer.write_immutable_stream
    calls = []

    def interrupted(path, chunks, *, maximum_bytes):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("Owned interrupted stage publication")
        original(path, chunks, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(writer, "write_immutable_stream", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        export_lrts_stage_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))
    monkeypatch.setattr(writer, "write_immutable_stream", original)
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: export_lrts_stage_parquet(recipe, cache, writer), range(3)))
    assert paths[0] == paths[1]
    assert paths[1] == paths[2]
    receipt = LRTSStageReceipt.from_json(paths[0].read_bytes())
    validate_lrts_stage_etl(receipt, cache, paths[0].parent)


def test_native_run_real_checkpoint_recovery_and_manifest(tmp_path):
    from coleman.application.corpus import run_lrts_stage_replay

    recipe, cache = prepare(tmp_path)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    before = run_lrts_stage_replay(path, cache, tmp_path / "native")
    after = run_lrts_stage_replay(path, cache, tmp_path / "native")
    assert before.run_id == after.run_id
    assert after.artifacts_dir is not None
    root = Path(after.artifacts_dir)
    results = pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True)
    assert results.height == 4
    assert results["step"].n_unique() == 2
    checkpoints = list((root / "checkpoints").rglob("progress_*.json"))
    assert checkpoints
    assert all(json.loads(p.read_bytes())["step_committed"] == 2 for p in checkpoints)
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert sum(a["type"] == "corpus_etl_receipt" for a in manifest["artifacts"]) == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), float("-inf")])
def test_invalid_reported_seconds_reject_without_completion(tmp_path, value):
    def invalid(report):
        report["suites"][0]["cases"][0]["duration"] = value
        return report

    recipe, cache = prepare(tmp_path, transform=invalid)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_lrts_stage_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


@pytest.mark.parametrize(
    "kind", ["missing_label", "conflicting_build", "missing_stage", "reversed_order", "case_limit"]
)
def test_unsupported_source_evidence_rejects_without_completion(tmp_path, kind):
    def metadata(rows):
        if kind == "missing_label":
            for row in rows:
                row["build_timestamp"] = ""
        elif kind == "conflicting_build":
            rows[0]["build_head_sha"] = "a"
            rows[1]["build_head_sha"] = "b"
        elif kind == "missing_stage":
            rows.pop(0)
        return rows

    recipe, cache = prepare(tmp_path, metadata_transform=metadata)
    if kind == "reversed_order":
        recipe = changed(recipe, source=changed(recipe.source, builds=tuple(reversed(recipe.source.builds))))
    elif kind == "case_limit":
        recipe = changed(recipe, source=changed(recipe.source, maximum_cases=6))
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_lrts_stage_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


@pytest.mark.parametrize(
    "kind", ["conservation", "roles", "stages", "sources", "rounds", "logical", "producer", "writer", "counts"]
)
def test_forged_completion_rejects_metadata_or_actual_verification(tmp_path, kind):
    from tests.corpus.test_source_views import digest

    recipe, cache = prepare(tmp_path)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    receipt = LRTSStageReceipt.from_json(path.read_bytes())
    value = receipt.model_dump(mode="json")
    if kind == "conservation":
        value["observed_cases"] += 1
    elif kind == "roles":
        value["files"][0]["rows"] += 1
    elif kind == "stages":
        value["source_stages"] += 1
    elif kind == "sources":
        value["source_datasets"] = []
    elif kind == "rounds":
        value["native_builds"] = 3
    elif kind == "logical":
        value["logical_rows_sha256"] = digest(b"forged")
    elif kind == "producer":
        value["producer_sha256"] = digest(b"forged")
    elif kind == "writer":
        value["parquet_writer"] = "unknown"
    else:
        value["excluded_outside_stage"] -= 1
        value["excluded_not_run"] += 1
    if kind in {"conservation", "roles", "stages", "sources", "rounds"}:
        with pytest.raises(ValueError):
            LRTSStageReceipt.model_validate(value)
    else:
        forged = LRTSStageReceipt.model_validate(value)
        with pytest.raises(ValueError):
            validate_lrts_stage_etl(forged, cache, path.parent)


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_completion_versions_fail_closed(tmp_path, version):
    recipe, cache = prepare(tmp_path)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    value = json.loads(path.read_bytes())
    if version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = version
    with pytest.raises(ValueError):
        LRTSStageReceipt.model_validate(value)


def test_source_derived_verifier_rejects_self_signed_parquet_and_corruption(tmp_path):
    from tests.corpus.test_source_views import digest

    recipe, cache = prepare(tmp_path)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    receipt = LRTSStageReceipt.from_json(path.read_bytes())
    target = path.parent / "observations.parquet"
    original = target.read_bytes()
    target.write_bytes(b"owned corruption")
    with pytest.raises(ValueError, match="file bytes"):
        validate_lrts_stage_etl(receipt, cache, path.parent)
    target.write_bytes(original)
    pl.read_parquet(target).with_columns(pl.lit("invented").alias("stage_id")).write_parquet(target)
    payload = target.read_bytes()
    records = tuple(
        changed(f, sha256=digest(payload), size_bytes=len(payload)) if f.path == "observations.parquet" else f
        for f in receipt.files
    )
    forged = changed(receipt, files=records)
    with pytest.raises(ValueError, match="artifact identity"):
        validate_lrts_stage_etl(forged, cache, path.parent)


def test_exact_metadata_duplicates_and_source_order_metamorphism(tmp_path):
    recipe, cache = prepare(tmp_path / "one")
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "one-output"))

    def duplicate_reverse(rows):
        return list(reversed(rows + [rows[0]]))

    second, cache2 = prepare(tmp_path / "two", metadata_transform=duplicate_reverse)
    other = export_lrts_stage_parquet(second, cache2, ArtifactWriter(tmp_path / "two-output"))
    a = pl.read_parquet(path.parent / "observations.parquet")
    b = pl.read_parquet(other.parent / "observations.parquet")
    assert a.equals(b)
    first_receipt = LRTSStageReceipt.from_json(path.read_bytes())
    second_receipt = LRTSStageReceipt.from_json(other.read_bytes())
    assert second_receipt.metadata_accounting.duplicates_excluded == 1
    assert second_receipt.metadata_accounting.selected_rows == 4
    assert second_receipt.metadata_accounting.input_rows == 5
    assert first_receipt.logical_rows_sha256 != second_receipt.logical_rows_sha256
    # Build-table source fingerprints retain physical duplicate evidence.
    assert first_receipt.source_datasets != second_receipt.source_datasets


def test_values_change_without_changing_execution_or_replay_identity(tmp_path):
    recipe, cache = prepare(tmp_path / "a")

    def cost(report):
        report["suites"][0]["cases"][1]["duration"] = 7
        return report

    other, cache2 = prepare(tmp_path / "b", transform=cost)
    a = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "a-output"))
    b = export_lrts_stage_parquet(other, cache2, ArtifactWriter(tmp_path / "b-output"))
    x = pl.read_parquet(a.parent / "observations.parquet")
    y = pl.read_parquet(b.parent / "observations.parquet")
    assert x["execution_id"].to_list() == y["execution_id"].to_list()
    assert x["replay_name"].to_list() == y["replay_name"].to_list()
    assert x["duration"].to_list() != y["duration"].to_list()


def test_empty_native_stage_remains_analytical_and_execution_refuses(tmp_path):
    from coleman.application.corpus import run_lrts_stage_replay

    def unavailable(report):
        for case in report["suites"][0]["cases"]:
            case["duration"] = None
        return report

    recipe, cache = prepare(tmp_path, transform=unavailable)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    receipt = LRTSStageReceipt.from_json(path.read_bytes())
    assert receipt.native_builds == 0
    assert receipt.replay_cases == 0
    assert receipt.observed_cases == 12
    validate_lrts_stage_etl(receipt, cache, path.parent)
    with pytest.raises(ValueError, match="no compatible native"):
        run_lrts_stage_replay(path, cache, tmp_path / "native")
