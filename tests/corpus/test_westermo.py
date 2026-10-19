"""Versioned source ETL: real cache/writer, missingness, deterministic bytes and fail-closed source truth."""

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.materialization import SourceCache, materialize_source
from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.corpus.westermo_contracts import WestermoCounts, WestermoObservation, WestermoReceipt, WestermoRecipe
from coleman.corpus.westermo_etl import (
    _encode,
    export_westermo_parquet,
    load_westermo_recipe,
    preflight_westermo_etl,
    validate_westermo_etl,
)
from coleman.corpus.westermo_raw import read_westermo_slice
from tests.corpus.test_source_views import changed, source_spec
from tests.corpus.westermo_support import owned_recipe, owned_rows, prepare_westermo


def test_roundtrip_golden_units_and_all_source_outcomes(tmp_path):
    recipe, cache = prepare_westermo(tmp_path)
    golden = Path(__file__).with_name("fixtures") / "westermo-owned-recipe-v1.json"
    assert recipe.to_json() + "\n" == golden.read_text()
    assert WestermoRecipe.from_json(recipe.to_json()) == recipe
    complete = export_westermo_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    receipt = WestermoReceipt.from_json(complete.read_bytes())
    assert WestermoReceipt.from_json(receipt.to_json()) == receipt
    assert receipt.counts == WestermoCounts(
        input_rows=6,
        selected_rows=5,
        outside_selection=1,
        passed=1,
        failed=2,
        invalid=1,
        unmappable=1,
        complete_sessions=1,
        incomplete_sessions=1,
    )
    validate_westermo_etl(receipt, cache, complete.parent)
    observations = pl.read_parquet(complete.parent / "observations.parquet")
    assert observations["result_raw"].to_list() == [0, 1, 2, 3, 1]
    assert observations["verdict"].to_list() == [0, 1, None, None, 1]
    assert observations["oduration_raw"].to_list() == [0, 3, 2, 0, 4]
    assert set(observations["oduration_unit"]) == {"westermo-oduration-unverified"}
    assert observations["outcome_available_at"].null_count() == 5
    assert observations["runner_seconds"].null_count() == 5
    assert set(observations["parameters_raw"]) == {"NULL"}
    sessions = pl.read_parquet(complete.parent / "sessions.parquet")
    assert sessions["sduration"].to_list() == [None, 0]
    assert sessions["sduration_status"].to_list() == ["unavailable", "observed"]
    assert sessions["sduration_unit"].to_list() == ["s", "s"]
    assert receipt.admission == "blocked_source_evidence"
    assert len(receipt.blockers) == 5


def test_idempotent_concurrent_exports_and_complete_file_verification(tmp_path):
    recipe, cache = prepare_westermo(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: export_westermo_parquet(recipe, cache, writer), range(8)))
    assert len(set(paths)) == 1
    before = {str(p): p.read_bytes() for p in writer.run_root.rglob("*") if p.is_file()}
    assert export_westermo_parquet(recipe, cache, writer) == paths[0]
    assert before == {str(p): p.read_bytes() for p in writer.run_root.rglob("*") if p.is_file()}
    receipt = WestermoReceipt.from_json(paths[0].read_bytes())
    table = paths[0].parent / "observations.parquet"
    payload = table.read_bytes()
    table.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="artifact bytes"):
        validate_westermo_etl(receipt, cache, paths[0].parent)
    table.write_bytes(payload + b"extra")
    with pytest.raises(ValueError, match="artifact bytes"):
        validate_westermo_etl(receipt, cache, paths[0].parent)


def test_row_permutation_preserves_logical_identities_and_digest(tmp_path):
    recipe, cache = prepare_westermo(tmp_path / "a")
    other, cache_other = prepare_westermo(tmp_path / "b", list(reversed(owned_rows())))
    a = export_westermo_parquet(recipe, cache, ArtifactWriter(tmp_path / "out-a"))
    b = export_westermo_parquet(other, cache_other, ArtifactWriter(tmp_path / "out-b"))
    ra, rb = (WestermoReceipt.from_json(p.read_bytes()) for p in (a, b))
    assert ra.logical_rows_sha256 == rb.logical_rows_sha256
    assert ra.fingerprint != rb.fingerprint
    assert ra.files[0].sha256 != rb.files[0].sha256  # physical source positions differ
    original = read_westermo_slice(recipe, cache).observations[0]
    assert original.execution_id == changed(original, source_position=999, result=1).execution_id
    assert original.execution_id != changed(original, oid=999).execution_id
    assert changed(recipe, sessions=tuple(reversed(recipe.sessions))).fingerprint == recipe.fingerprint


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_unknown_recipe_and_receipt_versions_reject(tmp_path, version):
    recipe, cache = prepare_westermo(tmp_path)
    complete = export_westermo_parquet(recipe, cache, ArtifactWriter(tmp_path / "out"))
    for contract, value in (
        (WestermoRecipe, recipe.model_dump(mode="json")),
        (WestermoReceipt, json.loads(complete.read_bytes())),
    ):
        if version is None:
            value.pop("schema_version")
        else:
            value["schema_version"] = version
        payload = json.dumps(value)
        with pytest.raises(ValueError):
            contract.from_json(payload)


@pytest.mark.parametrize("field,unit", [("test_duration", "s"), ("session_duration", "ms")])
def test_incompatible_unit_rejects_before_cache_access(field, unit):
    recipe, _ = owned_recipe()
    values = recipe.model_dump()
    values[field]["unit"] = unit
    with pytest.raises(ValueError, match="Incompatible Westermo"):
        WestermoRecipe.model_validate(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("oduration", "-1"),
        ("oduration", "NaN"),
        ("oduration", "Infinity"),
        ("oduration", "1.5"),
        ("oduration", "NULL"),
        ("sduration", "-1"),
        ("sduration", "inf"),
        ("result", "4"),
        ("result", "-1"),
        ("jid", "0"),
        ("oid", "01"),
        ("tid", "9223372036854775808"),
        ("ostarted", "1970-03-04T01:02:04Z"),
        ("sstarted", "1970-02-30 01:02:03"),
        ("sstarted", "1970-3-04 01:02:03"),
        ("sessiontype", "X"),
        ("parameters", "x" * 1025),
    ],
)
def test_invalid_source_values_fail_without_false_completion(tmp_path, field, value):
    rows = owned_rows()
    rows[0][field] = value
    recipe, cache = prepare_westermo(tmp_path, rows)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_westermo_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


def test_duplicates_conflicts_missing_selection_and_unselected_invalid_rows(tmp_path):
    for name, rows in (
        ("duplicate", [*owned_rows(), owned_rows()[0]]),
        ("conflict", [{**r, "sduration": "5"} if i == 0 else r for i, r in enumerate(owned_rows())]),
        ("missing", [r for r in owned_rows() if r["system"] != "owned-arm"]),
        ("outside", [{**r, "result": "4"} if r["jid"] == "22" else r for r in owned_rows()]),
    ):
        recipe, cache = prepare_westermo(tmp_path / name, rows)
        writer = ArtifactWriter(tmp_path / (name + "-out"))
        with pytest.raises(ValueError):
            export_westermo_parquet(recipe, cache, writer)


def test_completed_receipt_cannot_launder_source_or_remove_blockers(tmp_path):
    recipe, cache = prepare_westermo(tmp_path)
    complete = export_westermo_parquet(recipe, cache, ArtifactWriter(tmp_path / "out"))
    receipt = WestermoReceipt.from_json(complete.read_bytes())
    values = receipt.model_dump()
    values["counts"]["passed"], values["counts"]["failed"] = 2, 1
    false = WestermoReceipt.model_validate(values)
    with pytest.raises(ValueError, match="source-derived"):
        validate_westermo_etl(false, cache, complete.parent)
    with pytest.raises(ValueError, match="blockers"):
        changed(receipt, blockers=())
    for fields in ({"producer_sha256": "sha256:" + "0" * 64}, {"arrow_writer": "unknown"}):
        invalid = changed(receipt, **fields)
        with pytest.raises(ValueError, match="recorded implementation"):
            validate_westermo_etl(invalid, cache, complete.parent)
    blob = cache.member_path(
        MaterializationReceipt(schema_version=1, spec=recipe.source), recipe.source.members[0].name
    )
    blob.write_bytes(b"x" * recipe.source.members[0].size_bytes)
    with pytest.raises(ValueError, match="checksum"):
        validate_westermo_etl(receipt, cache, complete.parent)


def test_interrupted_second_file_resumes_without_false_completion(tmp_path):
    recipe, cache = prepare_westermo(tmp_path)

    class InterruptedWriter(ArtifactWriter):
        def write_immutable_stream(self, path, chunks, *, maximum_bytes):
            if Path(path).name == "sessions.parquet":
                raise OSError("owned interruption before sessions publication")
            return super().write_immutable_stream(path, chunks, maximum_bytes=maximum_bytes)

    output = tmp_path / "output"
    writer = InterruptedWriter(output)
    with pytest.raises(OSError, match="owned interruption"):
        export_westermo_parquet(recipe, cache, writer)
    assert not list(output.rglob("complete.json"))
    staged = next(output.rglob("observations.parquet"))
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    complete = export_westermo_parquet(recipe, cache, ArtifactWriter(output))
    assert staged == complete.parent / "observations.parquet"
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == digest
    validate_westermo_etl(WestermoReceipt.from_json(complete.read_bytes()), cache, complete.parent)


def test_preflight_overlap_git_symlinks_and_no_side_effects(tmp_path):
    recipe, _ = owned_recipe()
    cache, output = tmp_path / "cache", tmp_path / "output"
    root = preflight_westermo_etl(recipe, cache, output)
    assert not output.exists()
    assert not cache.exists()
    with pytest.raises(ValueError, match="remain separate"):
        preflight_westermo_etl(recipe, cache, cache / "output")
    git_output = Path(__file__).parents[2] / "never-written-westermo"
    with pytest.raises(ValueError, match="outside a repository"):
        preflight_westermo_etl(recipe, cache, git_output)
    root.mkdir(parents=True)
    (root / "observations.parquet").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="escapes"):
        preflight_westermo_etl(recipe, cache, output)


def test_nested_quantities_revalidate_and_do_not_accept_negative_or_nonfinite():
    recipe, _ = owned_recipe()
    with pytest.raises(ValueError, match="Duplicate"):
        changed(recipe, sessions=recipe.sessions * 2)
    with pytest.raises(ValueError):
        WestermoCounts(
            input_rows=1,
            selected_rows=0,
            outside_selection=0,
            passed=0,
            failed=0,
            invalid=0,
            unmappable=0,
            complete_sessions=0,
            incomplete_sessions=0,
        )
    # Source durations have nonnegative integer semantics. Valid negative effects
    # remain supported by DomainRecord; this adapter must not convert them to work.
    assert WestermoObservation.model_fields["oduration"].is_required()


def test_source_and_recipe_resource_bounds_and_original_default(tmp_path):
    recipe, payload = owned_recipe()
    default = load_westermo_recipe()
    assert len(default.sessions) == 16
    assert default.source.source.sha256 == "sha256:7025cc28601fe9aa02095314a297e53059df5bcd9d331749d456b990d8976dd2"
    assert default.source.license.license_id == "CC-BY-4.0"
    values = recipe.model_dump()
    values["source"]["source"]["citation"] = "x" * 1024**2
    with pytest.raises(ValueError, match="metadata exceeds"):
        WestermoRecipe.model_validate(values)
    values = recipe.model_dump()
    values["source"]["source"]["size_bytes"] = 128 * 1024**2 + 1
    values["source"]["members"][0]["size_bytes"] = 128 * 1024**2 + 1
    values["source"]["limits"]["maximum_source_bytes"] = 129 * 1024**2
    values["source"]["limits"]["maximum_member_bytes"] = 129 * 1024**2
    with pytest.raises(ValueError, match="128 MiB"):
        WestermoRecipe.model_validate(values)
    bad_payload = payload.replace(b"jid\tsystem", b"job\tsystem", 1)
    other = changed(recipe, source=source_spec("malformed.tsv", bad_payload))
    cache = SourceCache(tmp_path / "cache")
    materialize_source(other.source, cache, lambda: io.BytesIO(bad_payload))
    writer = ArtifactWriter(tmp_path / "out")
    with pytest.raises(ValueError, match="TSV schema"):
        export_westermo_parquet(other, cache, writer)
    assert not list((tmp_path / "out").rglob("complete.json"))


def test_bounded_parquet_encoding_rejects_oversized_output():
    table = pa.table({"owned_payload": [b"x" * (64 * 1024**2)]})
    with pytest.raises(ValueError, match="Parquet exceeds 64 MiB"):
        _encode(table)


def test_malformed_quoted_source_fails_with_actionable_error_and_no_completion(tmp_path):
    recipe, payload = owned_recipe()
    malformed = payload.split(b"\n", 1)[0] + b'\n"unterminated source field\n'
    recipe = changed(recipe, source=source_spec("malformed.tsv", malformed))
    cache = SourceCache(tmp_path / "cache")
    materialize_source(recipe.source, cache, lambda: io.BytesIO(malformed))
    writer = ArtifactWriter(tmp_path / "out")
    with pytest.raises(ValueError, match="Malformed Westermo") as error:
        export_westermo_parquet(recipe, cache, writer)
    assert error.value.__cause__ is not None
    assert not list((tmp_path / "out").rglob("complete.json"))


def test_selected_row_limit_retains_no_false_completion(tmp_path):
    row = owned_rows()[0]
    rows = [{**row, "oid": str(i + 1)} for i in range(20_001)]
    recipe, cache = prepare_westermo(tmp_path, rows)
    writer = ArtifactWriter(tmp_path / "out")
    with pytest.raises(ValueError, match="20000"):
        export_westermo_parquet(recipe, cache, writer)
    assert not list((tmp_path / "out").rglob("complete.json"))


def test_receipt_counts_roles_and_legacy_artifacts_cannot_be_reinterpreted(tmp_path):
    recipe, cache = prepare_westermo(tmp_path)
    path = export_westermo_parquet(recipe, cache, ArtifactWriter(tmp_path / "out"))
    receipt = WestermoReceipt.from_json(path.read_bytes())
    for changes in (
        {"complete_sessions": 0},
        {"selected_rows": 1, "outside_selection": 5, "passed": 1, "failed": 0, "invalid": 0, "unmappable": 0},
        {
            "input_rows": 20_001,
            "selected_rows": 20_001,
            "outside_selection": 0,
            "passed": 20_001,
            "failed": 0,
            "invalid": 0,
            "unmappable": 0,
        },
    ):
        values = receipt.model_dump()
        values["counts"].update(changes)
        with pytest.raises(ValueError):
            WestermoReceipt.model_validate(values)
    reversed_files = tuple(reversed(receipt.files))
    with pytest.raises(ValueError, match="roles"):
        changed(receipt, files=reversed_files)
    with pytest.raises(ValueError, match="verdict conservation"):
        changed(receipt.counts, passed=0)
    legacy = Path(__file__).with_name("fixtures") / "hcs-owned-completion-v1.json"
    original = HCSReceipt.from_json(legacy.read_bytes())
    assert original.schema_version == 1
    assert original.recipe.duration_semantics.unit == "s"
    legacy_payload = legacy.read_bytes()
    with pytest.raises(ValueError):
        WestermoReceipt.from_json(legacy_payload)
