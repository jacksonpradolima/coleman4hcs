"""RTPTorrent ETL versioning, provenance, interruption and conserved missingness regressions."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from coleman.application.corpus import run_rtptorrent_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.rtptorrent import convert_rtptorrent_source
from coleman.corpus.rtptorrent_etl import export_rtptorrent_parquet, validate_rtptorrent_etl
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt, RTPTorrentETLRecipe
from coleman.corpus.rtptorrent_projection import project_rtptorrent
from coleman.results.domain import canonical_json
from coleman.transports import rtptorrent_etl as cli
from tests.corpus.test_rtptorrent_etl import ROWS, owned_recipe, prepare
from tests.corpus.test_source_views import changed, digest


def exported(tmp_path, **kwargs):
    recipe, cache = prepare(tmp_path, **kwargs)
    path = export_rtptorrent_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    return path, RTPTorrentETLReceipt.from_json(path.read_bytes()), cache


def test_versioned_recipe_and_semantic_rows_golden(tmp_path):
    recipe, _ = owned_recipe()
    fixture = Path(__file__).with_name("fixtures")
    assert recipe.to_json() + "\n" == (fixture / "rtptorrent-etl-owned-recipe-v1.json").read_text()
    path, _, _ = exported(tmp_path)
    tables = {name: pl.read_parquet(path.parent / name).to_dicts() for name in ("observations.parquet", "jobs.parquet")}
    tables["native"] = pl.read_parquet(path.parent / "datasets/owned-rtptorrent/features-engineered.parquet").to_dicts()
    assert canonical_json(tables) + "\n" == (fixture / "rtptorrent-etl-owned-rows-v1.json").read_text()


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
@pytest.mark.parametrize("model", [RTPTorrentETLRecipe, RTPTorrentETLReceipt])
def test_missing_legacy_or_future_versions_reject(tmp_path, version, model):
    _, receipt, _ = exported(tmp_path)
    value = receipt.recipe.model_dump(mode="json") if model is RTPTorrentETLRecipe else receipt.model_dump(mode="json")
    if version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = version
    with pytest.raises(ValueError):
        model.model_validate(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_id", "../outside"),
        ("dataset_id", "bad."),
        ("dataset_id", "dir/name"),
        ("replay_job_ids", [102, 101]),
        ("replay_job_ids", [101, 101]),
        ("replay_job_ids", [True]),
        ("replay_job_ids", []),
        ("feedback", "full-reported-labels-after-modeled-round"),
        ("order", "chronological"),
        ("costs", "observed-parallel-runner-occupancy"),
    ],
)
def test_contradictory_recipe_assumptions_reject_before_io(field, value):
    recipe, _ = owned_recipe()
    payload = {**recipe.model_dump(mode="json"), field: value}
    with pytest.raises(ValueError):
        RTPTorrentETLRecipe.model_validate(payload)


def test_conversion_child_and_duration_units_cannot_be_rebound():
    recipe, _ = owned_recipe()
    for field, value in (
        ("member", "missing.csv"),
        ("duration_semantics", {"unit": "ms", "meaning": "rtptorrent-reported-class-elapsed-v1"}),
    ):
        payload = recipe.model_dump(mode="json")
        payload["conversion"][field] = value
        with pytest.raises(ValueError):
            RTPTorrentETLRecipe.model_validate(payload)


def test_multiple_members_or_rebound_conversion_cannot_enter_one_history():
    recipe, _ = owned_recipe()
    value = recipe.model_dump(mode="json")
    extra = {
        **value["source"]["selections"][0],
        "member": {**value["source"]["selections"][0]["member"], "name": "other@ci/results.csv"},
    }
    value["source"]["selections"].append(extra)
    with pytest.raises(ValueError, match="one complete"):
        RTPTorrentETLRecipe.model_validate(value)
    value = recipe.model_dump(mode="json")
    value["conversion"]["materialization"]["spec"]["package_id"] = "another-logical-child"
    with pytest.raises(ValueError, match="selected raw child"):
        RTPTorrentETLRecipe.model_validate(value)


@pytest.mark.parametrize(
    "case", ["physical", "projected", "source_bound", "selected_rounds", "zero_rounds", "file_roles"]
)
def test_receipt_metadata_cannot_erase_rows_or_invent_native_rounds(tmp_path, case):
    _, receipt, _ = exported(tmp_path)
    value = receipt.model_dump(mode="json")
    if case == "physical":
        value["input_rows"] += 1
    elif case == "projected":
        value["excluded_outside_jobs"] += 1
    elif case == "source_bound":
        value["recipe"]["conversion"]["maximum_rows"] = 7
    elif case == "selected_rounds":
        value["native_rounds"] = 3
    elif case == "zero_rounds":
        value["native_rounds"] = 0
    else:
        value["files"][0]["path"] = "different.parquet"
    with pytest.raises(ValueError):
        RTPTorrentETLReceipt.model_validate(value)


def test_pure_projection_never_relabels_another_source_spec(tmp_path):
    recipe, cache = prepare(tmp_path)
    dataset = convert_rtptorrent_source(recipe.conversion, cache)
    other = changed(recipe, conversion=changed(recipe.conversion, project="different/ci"))
    with pytest.raises(ValueError, match="conversion identity"):
        project_rtptorrent(dataset, other)


@pytest.mark.parametrize("duration", ["-1", "NaN", "Infinity", "1e-13"])
def test_invalid_source_duration_never_completes(tmp_path, duration):
    rows = (f"101,owned.Bad,0,{duration},1,0,0,0\n", ROWS[5])
    recipe, cache = prepare(tmp_path, rows=rows)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_rtptorrent_parquet(recipe, cache, writer)
    assert not list(writer.run_root.rglob("complete.json"))


def test_unknown_selected_job_never_completes(tmp_path):
    recipe, cache = prepare(tmp_path)
    invalid = changed(recipe, replay_job_ids=(101, 555))
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError, match="unknown source job"):
        export_rtptorrent_parquet(invalid, cache, writer)
    assert not list(writer.run_root.rglob("complete.json"))


def test_duplicates_and_source_order_change_provenance_not_logical_execution_identity(tmp_path):
    first, original, _ = exported(tmp_path / "first")
    second, reordered, _ = exported(tmp_path / "second", rows=tuple(reversed(ROWS)))
    third, duplicates, _ = exported(tmp_path / "third", rows=ROWS + (ROWS[0],))
    assert original.logical_rows_sha256 == reordered.logical_rows_sha256 == duplicates.logical_rows_sha256
    assert len({record.source_dataset_sha256 for record in (original, reordered, duplicates)}) == 3
    assert duplicates.input_rows == 9 and duplicates.duplicate_rows_excluded == 1
    assert duplicates.observed_executions == original.observed_executions == 8
    left = pl.read_parquet(first.parent / "observations.parquet")
    assert left.equals(pl.read_parquet(second.parent / "observations.parquet"))
    assert left.equals(pl.read_parquet(third.parent / "observations.parquet"))


def test_concurrent_exports_and_cross_workspace_bytes_converge(tmp_path):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: export_rtptorrent_parquet(recipe, cache, writer), range(3)))
    assert paths[0] == paths[1] == paths[2]
    other = export_rtptorrent_parquet(recipe, cache, ArtifactWriter(tmp_path / "other"))
    assert paths[0].read_bytes() == other.read_bytes()
    receipt = RTPTorrentETLReceipt.from_json(other.read_bytes())
    for item in receipt.files:
        assert (paths[0].parent / item.path).read_bytes() == (other.parent / item.path).read_bytes()


def test_interrupted_parquet_write_has_no_false_completion_and_retry_converges(tmp_path, monkeypatch):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    original = ArtifactWriter.write_immutable_stream

    def interrupted(self, path, chunks, **kwargs):
        if path.name == "jobs.parquet":
            raise OSError("Owned interrupted second Parquet")
        return original(self, path, chunks, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ArtifactWriter, "write_immutable_stream", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            export_rtptorrent_parquet(recipe, cache, writer)
    assert list(writer.run_root.rglob("observations.parquet"))
    assert not list(writer.run_root.rglob("complete.json"))
    completion = export_rtptorrent_parquet(recipe, cache, writer)
    validate_rtptorrent_etl(RTPTorrentETLReceipt.from_json(completion.read_bytes()), cache, completion.parent)


@pytest.mark.parametrize("field", ["producer_sha256", "parquet_writer", "source_dataset_sha256", "logical_rows_sha256"])
def test_forged_provenance_never_verifies(tmp_path, field):
    path, receipt, cache = exported(tmp_path)
    value = "unknown-writer" if field == "parquet_writer" else digest(b"other")
    invalid = changed(receipt, **{field: value})
    with pytest.raises(ValueError):
        validate_rtptorrent_etl(invalid, cache, path.parent)


def test_conserved_but_false_exclusion_counts_do_not_verify(tmp_path):
    path, receipt, cache = exported(tmp_path)
    invalid = changed(receipt, excluded_outside_jobs=0, excluded_not_run=2)
    with pytest.raises(ValueError, match="counts disagree"):
        validate_rtptorrent_etl(invalid, cache, path.parent)
    invalid = changed(receipt, input_rows=9, duplicate_rows_excluded=1)
    with pytest.raises(ValueError, match="physical-row counts"):
        validate_rtptorrent_etl(invalid, cache, path.parent)


@pytest.mark.parametrize("case", ["missing", "corrupt", "forged-content"])
def test_missing_corrupt_or_self_signed_parquet_never_verifies(tmp_path, case):
    path, receipt, cache = exported(tmp_path)
    item = next(item for item in receipt.files if item.path == "observations.parquet")
    target = path.parent / item.path
    if case == "missing":
        target.unlink()
    elif case == "corrupt":
        target.write_bytes(b"X" + target.read_bytes()[1:])
    else:
        buffer = io.BytesIO()
        pl.read_parquet(target).with_columns(pl.lit("passed").alias("class_outcome")).write_parquet(buffer)
        payload = buffer.getvalue()
        target.write_bytes(payload)
        forged = changed(item, sha256=digest(payload), size_bytes=len(payload))
        receipt = changed(receipt, files=tuple(forged if value.path == item.path else value for value in receipt.files))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_rtptorrent_etl(receipt, cache, path.parent)


def test_entirely_excluded_rounds_stay_analytical_and_cannot_run(tmp_path):
    rows = ("101,owned.Skip,0,0,1,0,0,1\n", "102,owned.Unknown,0,,,,,\n")
    path, receipt, cache = exported(tmp_path, rows=rows)
    assert receipt.observed_executions == 2 and receipt.replay_executions == receipt.native_rounds == 0
    assert pl.read_parquet(path.parent / "jobs.parquet")["replay_status"].to_list() == [
        "no_compatible_reported_cases",
        "no_compatible_reported_cases",
    ]
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="no compatible"):
        run_rtptorrent_replay(path, cache, output)
    assert not output.exists()


def test_transport_explicit_archive_verifies_even_on_cache_hit_and_cli_repeats(tmp_path, capsys):
    recipe, archive = owned_recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(recipe.to_json())
    source = tmp_path / "owned.zip"
    source.write_bytes(archive)
    args = [
        "--recipe",
        str(recipe_path),
        "--archive",
        str(source),
        "--cache",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "output"),
    ]
    assert cli.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["observed_executions"] == 8
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["completion"] == first["completion"]
    source.write_bytes(b"invalid archive")
    with pytest.raises(ValueError):
        cli.main(args)


def test_unsafe_concrete_output_rejects_before_any_archive_acquisition(tmp_path):
    recipe, _ = owned_recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(recipe.to_json())
    output, outside, cache = tmp_path / "output", tmp_path / "outside", tmp_path / "cache"
    output.mkdir()
    outside.mkdir()
    (output / "corpus-etl").symlink_to(outside, target_is_directory=True)
    args = [
        "--recipe",
        str(recipe_path),
        "--archive",
        str(tmp_path / "missing.zip"),
        "--cache",
        str(cache),
        "--output",
        str(output),
    ]
    with pytest.raises(ValueError, match="escapes"):
        cli.main(args)
    assert not cache.exists()


@pytest.mark.parametrize("case", ["native_escape", "nested_source_in_native"])
def test_native_destination_preflight_precedes_acquisition(tmp_path, case):
    recipe, _ = owned_recipe()
    metadata = tmp_path / "recipe.json"
    metadata.write_text(recipe.to_json())
    output, outside, cache = tmp_path / "output", tmp_path / "outside", tmp_path / "cache"
    output.mkdir()
    outside.mkdir()
    if case == "native_escape":
        (output / "native").symlink_to(outside, target_is_directory=True)
    else:
        (output / "native").mkdir()
        (output / "corpus-etl").symlink_to(output / "native", target_is_directory=True)
    args = [
        "--recipe",
        str(metadata),
        "--archive",
        str(tmp_path / "missing.zip"),
        "--cache",
        str(cache),
        "--output",
        str(output),
        "--run",
    ]
    with pytest.raises(ValueError):
        cli.main(args)
    assert not cache.exists()


def test_cli_native_execution_and_bounded_recipe_loader(tmp_path, capsys):
    recipe, archive = owned_recipe()
    metadata, source = tmp_path / "recipe.json", tmp_path / "source.zip"
    metadata.write_text(recipe.to_json())
    source.write_bytes(archive)
    args = [
        "--recipe",
        str(metadata),
        "--archive",
        str(source),
        "--cache",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "output"),
        "--run",
    ]
    assert cli.main(args) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["modeled_rounds"] == 2 and summary["replay_executions"] == 4
    root = Path(summary["native_artifacts"])
    assert pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True).height == 4
    assert cli.load_rtptorrent_etl_recipe().conversion.history == "observed"
    metadata.write_bytes(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        cli.load_rtptorrent_etl_recipe(metadata)
