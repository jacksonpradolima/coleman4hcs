"""Owned HCS source, identity, deterministic files and honest sparse accounting regressions."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.hcs_contracts import HCSReceipt, HCSRecipe, HCSVariant, variant_alias
from coleman.corpus.hcs_etl import export_hcs_parquet, validate_hcs_etl
from coleman.corpus.hcs_raw import read_hcs_slice
from coleman.corpus.materialization_contracts import MaterializationReceipt
from tests.corpus.hcs_support import hcs_recipe, prepare_hcs
from tests.corpus.test_source_views import changed, digest


def export(tmp_path, **kwargs):
    recipe, cache = prepare_hcs(tmp_path, **kwargs)
    completion = export_hcs_parquet(recipe, cache, ArtifactWriter(tmp_path / "etl"))
    return HCSReceipt.from_json(completion.read_bytes()), cache, completion


def test_source_roundtrip_explicit_units_missingness_identity_and_sparse_cells(tmp_path):
    receipt, cache, path = export(tmp_path)
    assert HCSReceipt.from_json(receipt.to_json()) == receipt
    assert HCSRecipe.from_json(receipt.recipe.to_json()) == receipt.recipe
    assert receipt.accounting.model_dump() == dict(
        input_rows=18,
        outside_slice=1,
        selected_rows=17,
        duplicate_rows=1,
        unsupported_job_rows=1,
        other_attempt_rows=3,
        unavailable_outcome_rows=1,
        unavailable_duration_rows=1,
        replay_rows=10,
    )
    assert len(receipt.variants) == 4
    assert variant_alias("linux.x") != variant_alias("linux_x")
    observations = pl.read_parquet(path.parent / "observations.parquet")
    assert observations["duration_unit"].unique().to_list() == ["s"]
    assert observations["duration_decimal"].null_count() == 1
    assert observations["duration_decimal"].to_list().count("0") == 3
    assert observations["reported_nonpass"].null_count() == 1
    assert observations["outcome_available_at"].null_count() == 17
    assert observations["execution_id"].n_unique() == 16
    assert "do_not_export" not in str(observations.to_dicts())
    cells = pl.read_parquet(path.parent / "cells.parquet")
    assert cells.height == 8
    assert cells["replay_rows"].sum() == 10
    assert cells["setup_seconds"].null_count() == 8
    assert cells["shared_runner_occupancy"].null_count() == 8
    assert cells.filter(pl.col("cell_status") == "absent_parsed_source")["reported_nonpass_rows"].null_count() == 3
    validate_hcs_etl(receipt, cache, path.parent)
    with pytest.raises(ValueError, match="temporal evaluation blocked"):
        receipt.evidence.require_temporal_evaluation()


def test_recipe_golden_schema_and_unknown_versions():
    recipe, _, _ = hcs_recipe()
    fixture = Path(__file__).with_name("fixtures") / "hcs-owned-recipe-v1.json"
    assert recipe.to_json() + "\n" == fixture.read_text()
    for version in [None, 0, 2, True, "1"]:
        value = recipe.model_dump(mode="json")
        if version is None:
            value.pop("schema_version")
        else:
            value["schema_version"] = version
        with pytest.raises(ValueError):
            HCSRecipe.model_validate(value)


def test_versioned_completion_golden_keeps_wire_contract_separate_from_runtime_verification(tmp_path):
    receipt, _, _ = export(tmp_path)
    owned = changed(receipt, producer_sha256=digest(b"owned-hcs-golden-producer"))
    fixture = Path(__file__).with_name("fixtures") / "hcs-owned-completion-v1.json"
    assert owned.to_json() + "\n" == fixture.read_text()
    assert HCSReceipt.from_json(fixture.read_bytes()) == owned


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_completion_future_missing_or_coerced_versions_reject(tmp_path, version):
    receipt, _, _ = export(tmp_path)
    raw = json.loads(receipt.to_json())
    if version is None:
        raw.pop("schema_version")
    else:
        raw["schema_version"] = version
    with pytest.raises(ValueError):
        HCSReceipt.from_json(json.dumps(raw))


def test_incompatible_units_orders_unsafe_formats_and_aliases_reject():
    recipe, _, _ = hcs_recipe()
    for updates in [
        dict(duration_semantics=dict(unit="ms", meaning=recipe.duration_semantics.meaning)),
        dict(pipeline_ids=(202, 101)),
        dict(pipeline_ids=(101, 101)),
        dict(jobs=recipe.executions),
        dict(dataset_id="../escape"),
        dict(dataset_id="con"),
    ]:
        with pytest.raises(ValueError):
            changed(recipe, **updates)
    with pytest.raises(ValueError, match="alias disagrees"):
        HCSVariant(name="linux.x", alias=variant_alias("linux_x"))


def test_original_variant_names_are_not_normalized_or_merged():
    names = {
        f"{base}{separator}x86_64" for base in ("Linux", "linux", "Línux") for separator in (".", "_", "/", "-", " ")
    }
    aliases = [variant_alias(name) for name in names]
    assert len(aliases) == len(set(aliases))
    assert all(HCSVariant(name=name, alias=variant_alias(name)).name == name for name in names)


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "-Infinity", "1e-999", "1000000000001"])
def test_negative_nonfinite_or_unsupported_precision_time_rejects_without_completion(tmp_path, value):
    def invalid(rows):
        rows[0]["duration"] = value
        return rows

    with pytest.raises(ValueError):
        export(tmp_path, rows_transform=invalid)
    assert not list((tmp_path / "etl").rglob("complete.json"))


@pytest.mark.parametrize("kind", ["duplicate_conflict", "missing_job", "pipeline", "commit", "verdict"])
def test_source_correspondence_and_conflicting_duplicates_fail_closed(tmp_path, kind):
    def invalid(rows):
        if kind == "duplicate_conflict":
            rows[-1]["duration"] = "8"
        elif kind == "missing_job":
            rows[0]["job_id"] = "999"
        elif kind == "pipeline":
            rows[0]["pipeline_id"] = "202"
        elif kind == "commit":
            rows[0]["sha"] = "f" * 40
        else:
            rows[0]["verdict"] = "2"
        return rows

    with pytest.raises(ValueError):
        export(tmp_path, rows_transform=invalid)
    assert not list((tmp_path / "etl").rglob("complete.json"))


def test_logical_identity_survives_measurement_change_and_physical_order_metamorphism(tmp_path):
    def altered(rows):
        for row in rows:
            if row["job_id"] == "1" and row["test_name"] == "A":
                row["duration"] = "7"
        return list(reversed(rows))

    a, _, path_a = export(tmp_path / "a")
    b, _, path_b = export(tmp_path / "b", rows_transform=altered)
    x = pl.read_parquet(path_a.parent / "observations.parquet")
    y = pl.read_parquet(path_b.parent / "observations.parquet")
    assert sorted(x["execution_id"].to_list()) == sorted(y["execution_id"].to_list())
    assert sorted(x["replay_name"].to_list()) == sorted(y["replay_name"].to_list())
    assert a.accounting == b.accounting
    assert a.logical_rows_sha256 != b.logical_rows_sha256
    assert a.recipe.fingerprint != b.recipe.fingerprint


def test_source_order_metamorphism_keeps_native_bytes_and_original_logical_ids(tmp_path):
    a, _, path_a = export(tmp_path / "a")
    b, _, path_b = export(
        tmp_path / "b",
        rows_transform=lambda rows: list(reversed(rows)),
        jobs_transform=lambda jobs: list(reversed(jobs)),
    )
    assert a.accounting == b.accounting
    for item in a.files:
        if item.path != "observations.parquet":
            assert (path_a.parent / item.path).read_bytes() == (path_b.parent / item.path).read_bytes()
    x = pl.read_parquet(path_a.parent / "observations.parquet")
    y = pl.read_parquet(path_b.parent / "observations.parquet")
    assert sorted(x["execution_id"]) == sorted(y["execution_id"])
    assert a.recipe.fingerprint != b.recipe.fingerprint


def test_cross_variant_feedback_changes_union_but_does_not_mutate_another_vts_input(tmp_path):
    def changed_other_variant(rows):
        for row in rows:
            if row["job_id"] == "1" and row["test_name"] == "B":
                row["verdict"] = "0"
        return rows

    a, _, path_a = export(tmp_path / "a")
    b, _, path_b = export(tmp_path / "b", rows_transform=changed_other_variant)
    alias = variant_alias("linux_x")
    target = f"vts/{alias}/features-engineered.parquet"
    assert (path_a.parent / target).read_bytes() == (path_b.parent / target).read_bytes()
    union_path = "datasets/owned-hcs/features-engineered.parquet"
    x, y = pl.read_parquet(path_a.parent / union_path), pl.read_parquet(path_b.parent / union_path)
    assert x["Name"].to_list() == y["Name"].to_list()
    assert x["SourceExecutionId"].to_list() == y["SourceExecutionId"].to_list()
    assert x["Verdict"].sum() == y["Verdict"].sum() + 1
    assert a.accounting == b.accounting


@pytest.mark.parametrize(
    "kind",
    [
        "input_count",
        "selected_count",
        "observations",
        "native_rows",
        "vts_rows",
        "cells",
        "variant_duplicate",
        "order",
        "setup_zero",
    ],
)
def test_completion_impossible_counts_bindings_and_unmeasured_zero_reject(tmp_path, kind):
    receipt, _, _ = export(tmp_path)
    raw = receipt.model_dump(mode="json")
    if kind == "input_count":
        raw["accounting"]["input_rows"] += 1
    elif kind == "selected_count":
        raw["accounting"]["duplicate_rows"] += 1
    elif kind == "variant_duplicate":
        raw["variants"].append(raw["variants"][0])
    elif kind == "order":
        raw["variants"].reverse()
    elif kind == "setup_zero":
        raw["evidence"]["setup_seconds"] = 0
    else:
        target = {
            "observations": "observations.parquet",
            "native_rows": "datasets/owned-hcs/data-variants.parquet",
            "vts_rows": f"vts/{receipt.variants[0].alias}/features-engineered.parquet",
            "cells": "cells.parquet",
        }[kind]
        next(f for f in raw["files"] if f["path"] == target)["rows"] += 1
    with pytest.raises(ValueError):
        type(receipt).model_validate(raw)


def test_interruption_has_no_false_completion_concurrent_retry_converges(tmp_path, monkeypatch):
    recipe, cache = prepare_hcs(tmp_path)
    writer = ArtifactWriter(tmp_path / "etl")
    original, count = writer.write_immutable_stream, 0

    def interrupted(path, chunks, *, maximum_bytes):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("owned HCS interrupted write")
        return original(path, chunks, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(writer, "write_immutable_stream", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        export_hcs_parquet(recipe, cache, writer)
    assert not list((tmp_path / "etl").rglob("complete.json"))
    monkeypatch.setattr(writer, "write_immutable_stream", original)
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: export_hcs_parquet(recipe, cache, writer), range(3)))
    assert len(set(paths)) == 1
    receipt = HCSReceipt.from_json(paths[0].read_bytes())
    validate_hcs_etl(receipt, cache, paths[0].parent)


@pytest.mark.parametrize("kind", ["bytes", "self_signed", "producer", "writer", "logical", "counts", "roles"])
def test_forged_artifacts_and_receipts_reject(tmp_path, kind):
    receipt, cache, path = export(tmp_path)
    target = path.parent / "observations.parquet"
    updates = {}
    if kind == "bytes":
        target.write_bytes(b"owned corruption")
    elif kind == "self_signed":
        target.write_bytes(b"owned modified table")
        updates["files"] = tuple(
            changed(f, sha256=digest(target.read_bytes()), size_bytes=target.stat().st_size)
            if f.path == target.name
            else f
            for f in receipt.files
        )
    elif kind == "producer":
        updates["producer_sha256"] = digest(b"unknown")
    elif kind == "writer":
        updates["parquet_writer"] = "unknown"
    elif kind == "logical":
        updates["logical_rows_sha256"] = digest(b"forged")
    elif kind == "counts":
        updates["accounting"] = changed(receipt.accounting, duplicate_rows=2, other_attempt_rows=2)
    else:
        updates["files"] = receipt.files[:-1]
    with pytest.raises(ValueError):
        validate_hcs_etl(changed(receipt, **updates), cache, path.parent)


def test_missing_source_or_unproven_original_bytes_never_adopted(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    source = cache.member_path(
        MaterializationReceipt(schema_version=1, spec=recipe.executions),
        recipe.executions.members[0].name,
    )
    source.write_bytes(b"owned corruption")
    with pytest.raises(ValueError):
        read_hcs_slice(recipe, cache)
