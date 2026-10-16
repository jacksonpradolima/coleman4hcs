"""Owned ETL fixtures exercise actual nested archives, source conversion, Parquet and native checkpoints."""

import io
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from coleman.application.corpus import run_lrts_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import (
    ArchiveSelectionSpec,
    materialize_selected_packages,
    validate_selected_packages,
)
from coleman.corpus.archives import deterministic_zip_bytes
from coleman.corpus.lrts_etl import export_lrts_parquet, validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt, LRTSETLRecipe, LRTSFeedbackModel
from coleman.corpus.materialization import SourceCache, materialize_source
from coleman.transports import lrts_etl as cli
from tests.corpus.test_lrts import metadata_bytes, metadata_row, raw_report
from tests.corpus.test_source_views import changed, digest, source_spec


def zip_bytes(entries):
    return deterministic_zip_bytes(tuple(entries), maximum_member_bytes=1024**2, maximum_total_bytes=2 * 1024**2)


def owned_recipe(*, all_skipped=False, transform=lambda value: value, metadata_transform=lambda value: value):
    """Construct two source rounds with repeated labels, observed zero, null and unknown outcomes."""
    selections, builds, metadata, entries = [], [], [], []
    for index in (1, 2):
        report = raw_report()
        report["suites"][0]["cases"].append(
            dict(className="owned.NoDuration", name="method", duration=None, status="PASSED", skipped=False)
        )
        report["passCount"] = 3
        row = metadata_row()
        row.update(build_id=str(index), build_timestamp=str(index - 1), passcount="3", num_pass_method="3")
        if all_skipped:
            for suite in report["suites"]:
                for case in suite["cases"]:
                    case.update(status="SKIPPED", skipped=True, duration=None)
            report.update(passCount=0, failCount=0, skipCount=5)
            row.update(passcount="0", failcount="0", skipcount="5")
        report = transform(report)
        payload = json.dumps(report, sort_keys=True).encode()
        filename = f"testReport_build{index}.json"
        nested = zip_bytes(((filename, payload),))
        name = f"owned-ci/PR/PR-owned/{filename}.zip"
        package = source_spec(
            "nested-" + str(index),
            nested,
            "zip",
            [dict(name=filename, sha256=digest(payload), size_bytes=len(payload))],
        )
        entries.append((name, nested))
        selections.append(dict(member=dict(name=name, sha256=digest(nested), size_bytes=len(nested)), package=package))
        builds.append(dict(pull_request="PR-owned", build_id=index, archive_member=name))
        metadata.append(row)
    entries.append(("unselected.txt", b"Owned unselected bytes."))
    outer, csv = zip_bytes(entries), metadata_bytes(metadata_transform(metadata))
    selection = ArchiveSelectionSpec.model_validate(
        dict(
            schema_version=1,
            source=source_spec("outer.zip", outer),
            selections=selections,
        )
    )
    recipe = LRTSETLRecipe.model_validate(
        dict(
            schema_version=1,
            project="owned-ci",
            history="constructed",
            feedback=LRTSFeedbackModel.NATIVE_RNFAIL,
            metadata=source_spec("dataset.csv", csv),
            reports=selection,
            builds=builds,
        )
    )
    return recipe, outer, csv


def prepare(tmp_path, **kwargs):
    recipe, outer, csv = owned_recipe(**kwargs)
    cache = SourceCache(tmp_path / "cache")
    materialize_source(recipe.metadata, cache, lambda: io.BytesIO(csv))
    receipts = materialize_selected_packages(recipe.reports, cache, lambda: io.BytesIO(outer))
    assert len(receipts) == 2
    return recipe, cache


def exported(tmp_path, **kwargs):
    recipe, cache = prepare(tmp_path, **kwargs)
    path = export_lrts_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    return path, LRTSETLReceipt.from_json(path.read_bytes()), cache


def test_real_owned_roundtrip_units_exclusions_identity_replay_and_concurrency(tmp_path):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: export_lrts_parquet(recipe, cache, writer), range(3)))
    assert paths[0] == paths[1] == paths[2]
    receipt = LRTSETLReceipt.from_json(paths[0].read_bytes())
    assert LRTSETLReceipt.from_json(receipt.to_json()) == receipt
    assert receipt.observed_cases == 10
    assert receipt.replay_cases == 4
    assert receipt.excluded_not_run == receipt.excluded_unknown_outcome == receipt.excluded_unavailable_duration == 2
    assert receipt.native_builds == 2
    observations = pl.read_parquet(paths[0].parent / "observations.parquet")
    assert observations["execution_id"].n_unique() == 10
    assert observations["replay_name"].n_unique() == 5
    assert observations["duration_unit"].unique().to_list() == ["s"]
    assert observations["duration"].to_list().count(0) == 2
    assert observations["duration"].null_count() == 6
    assert observations["source_build_id"].to_list() == [1] * 5 + [2] * 5
    builds = pl.read_parquet(paths[0].parent / "builds.parquet")
    assert builds["outcome_available_at"].null_count() == 2
    native = pl.read_parquet(paths[0].parent / "datasets/owned-ci/features-engineered.parquet")
    assert native["Name"].n_unique() == 2
    assert native["SourceExecutionId"].n_unique() == 4
    validate_lrts_etl(receipt, cache, paths[0].parent)
    validate_selected_packages(recipe.reports, cache)
    assert export_lrts_parquet(recipe, cache, writer) == paths[0]


def test_recipe_versioned_golden_and_source_order_are_preserved():
    recipe, _, _ = owned_recipe()
    assert LRTSETLRecipe.from_json(recipe.to_json()) == recipe
    fixture = Path(__file__).with_name("fixtures") / "lrts-etl-owned-censored-recipe-v1.json"
    assert recipe.to_json() + "\n" == fixture.read_text()
    assert recipe.reports.source.limits.maximum_archive_entries == 128
    assert recipe.reports.maximum_entries == 4096


@pytest.mark.parametrize("model", [ArchiveSelectionSpec, LRTSETLRecipe])
@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_unknown_versions_fail_closed(tmp_path, model, version):
    recipe, _, _ = owned_recipe()
    value = recipe.reports.model_dump(mode="json") if model is ArchiveSelectionSpec else recipe.model_dump()
    if version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = version
    with pytest.raises(ValueError):
        model.model_validate(value)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_reported_durations_reject_without_completion(tmp_path, value):
    def invalid(report):
        report["suites"][0]["cases"][0]["duration"] = value
        return report

    recipe, cache = prepare(tmp_path, transform=invalid)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_lrts_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


def test_interrupt_never_certifies_completion_and_retry_reuses_verified_source(tmp_path, monkeypatch):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    original = writer.write_immutable_stream
    calls = []

    def interrupted(path, chunks, *, maximum_bytes):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("Owned interrupted Parquet publication")
        original(path, chunks, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(writer, "write_immutable_stream", interrupted)
    with pytest.raises(OSError):
        export_lrts_parquet(recipe, cache, writer)
    assert len(list((tmp_path / "output").rglob("*.parquet"))) == 1
    assert not list((tmp_path / "output").rglob("complete.json"))
    monkeypatch.setattr(writer, "write_immutable_stream", original)
    completion = export_lrts_parquet(recipe, cache, writer)
    validate_lrts_etl(LRTSETLReceipt.from_json(completion.read_bytes()), cache, completion.parent)


def test_native_runner_and_real_checkpoint_continuation(tmp_path):
    path, receipt, cache = exported(tmp_path)
    first = run_lrts_replay(path, cache, tmp_path / "native")
    second = run_lrts_replay(path, cache, tmp_path / "native")
    assert first.run_id == second.run_id
    assert second.artifacts_dir is not None
    root = Path(second.artifacts_dir)
    rows = pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True)
    assert rows.height == 4
    assert rows["step"].n_unique() == 2
    checkpoints = list((root / "checkpoints").rglob("progress_*.json"))
    assert checkpoints
    assert all(json.loads(p.read_bytes())["step_committed"] == 2 for p in checkpoints)
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert sum(item["type"] == "corpus_etl_receipt" for item in manifest["artifacts"]) == 1
    assert receipt.admission == "blocked_source_evidence"


def test_empty_native_projection_stays_analytical_and_run_rejects(tmp_path):
    path, receipt, cache = exported(tmp_path, all_skipped=True)
    assert receipt.native_builds == receipt.replay_cases == 0
    assert receipt.excluded_not_run == 10
    validate_lrts_etl(receipt, cache, path.parent)
    with pytest.raises(ValueError, match="no compatible"):
        run_lrts_replay(path, cache, tmp_path / "native")


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_entries", True),
        ("maximum_entries", 4097),
        ("maximum_metadata_bytes", 0),
        ("maximum_metadata_bytes", 2 * 1024**2 + 1),
    ],
)
def test_selection_limits_reject_before_io(field, value):
    recipe, _, _ = owned_recipe()
    with pytest.raises(ValueError):
        changed(recipe.reports, **{field: value})


@pytest.mark.parametrize("name", ["../escape", "/absolute", "disk\\path"])
def test_unselected_unsafe_archive_paths_reject(tmp_path, name):
    recipe, outer, csv = owned_recipe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(outer)) as original, zipfile.ZipFile(buffer, "w") as archive:
        for info in original.infolist():
            archive.writestr(info, original.read(info))
        archive.writestr(name, b"Owned hostile metadata")
    recipe = changed(recipe, reports=changed(recipe.reports, source=source_spec("outer.zip", buffer.getvalue())))
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_selected_packages(recipe.reports, cache, lambda: io.BytesIO(buffer.getvalue()))


def test_changed_source_and_parquet_never_validate_as_complete(tmp_path):
    path, receipt, cache = exported(tmp_path)
    observed = path.parent / "observations.parquet"
    observed.write_bytes(b"Owned corruption")
    with pytest.raises(ValueError, match="file bytes"):
        validate_lrts_etl(receipt, cache, path.parent)
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_lrts_parquet(receipt.recipe, cache, writer)


def test_cli_actual_owned_conversion_and_input_preflight(tmp_path, capsys):
    recipe, cache = prepare(tmp_path)
    file = tmp_path / "recipe.json"
    file.write_text(recipe.to_json())
    args = ["--recipe", str(file), "--cache", str(cache.root), "--output", str(tmp_path / "cli-output")]
    assert cli.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert first["observed_cases"] == 10
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    with pytest.raises(ValueError, match="outside a repository"):
        cli.main(["--recipe", str(file), "--cache", str(cache.root), "--output", str(repository)])


@pytest.mark.parametrize("version", [None, 2, True, "1"])
def test_completed_receipt_rejects_future_missing_or_coerced_versions(tmp_path, version):
    _, receipt, _ = exported(tmp_path)
    value = receipt.model_dump(mode="json")
    if version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = version
    with pytest.raises(ValueError):
        LRTSETLReceipt.model_validate(value)


@pytest.mark.parametrize("kind", ["duplicate", "source_zip", "member_digest", "terms", "version", "selection_bytes"])
def test_archive_selection_rejects_conflicting_provenance(kind):
    recipe, _, _ = owned_recipe()
    value = recipe.reports.model_dump(mode="json")
    if kind == "duplicate":
        value["selections"].append(value["selections"][0])
    elif kind == "source_zip":
        value["source"]["archive"] = "zip"
    elif kind == "member_digest":
        value["selections"][0]["member"]["sha256"] = digest(b"other")
    elif kind == "terms":
        value["selections"][0]["package"]["license"]["reviewed_at"] = "2026-01-01T00:00:00Z"
    elif kind == "version":
        value["selections"][0]["package"]["source"]["version"] = "different"
    else:
        for s in value["selections"]:
            s["member"]["size_bytes"] = 40 * 1024**2
            s["package"]["source"]["size_bytes"] = 40 * 1024**2
    with pytest.raises(ValueError):
        ArchiveSelectionSpec.model_validate(value)


@pytest.mark.parametrize(
    "kind", ["metadata_zip", "metadata_size", "duplicate_build", "unselected_build", "wrong_origin", "decoded_name"]
)
def test_recipe_rejects_conflicting_source_bindings(kind):
    recipe, _, _ = owned_recipe()
    value = recipe.model_dump(mode="json")
    if kind == "metadata_zip":
        value["metadata"]["archive"] = "zip"
    elif kind == "metadata_size":
        value["metadata"]["source"]["size_bytes"] = 33 * 1024**2
        value["metadata"]["members"][0]["size_bytes"] = 33 * 1024**2
    elif kind == "duplicate_build":
        value["builds"].append(value["builds"][0])
    elif kind == "unselected_build":
        value["builds"].pop()
    elif kind == "wrong_origin":
        value["project"] = "other"
    else:
        value["reports"]["selections"][0]["package"]["members"][0]["name"] = "wrong.json"
    with pytest.raises(ValueError):
        LRTSETLRecipe.model_validate(value)


@pytest.mark.parametrize("kind", ["scheduled_missing", "multiple_stages", "case_limit", "order"])
def test_ambiguous_or_oversized_source_projection_has_no_completion(tmp_path, kind):
    def modify(rows):
        if kind == "scheduled_missing":
            rows[0]["build_timestamp"] = ""
        if kind == "multiple_stages":
            rows.append({**rows[0], "stage_id": "second-stage"})
        return rows

    recipe, cache = prepare(tmp_path, metadata_transform=modify)
    if kind == "case_limit":
        recipe = changed(recipe, maximum_cases=5)
    if kind == "order":
        recipe = changed(recipe, builds=tuple(reversed(recipe.builds)))
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError):
        export_lrts_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


@pytest.mark.parametrize(
    "kind", ["conservation", "roles", "sources", "builds", "case_limit", "logical", "source_fingerprint", "exclusions"]
)
def test_forged_completion_cannot_validate_against_actual_sources(tmp_path, kind):
    path, receipt, cache = exported(tmp_path)
    value = receipt.model_dump(mode="json")
    if kind == "conservation":
        value["observed_cases"] += 1
    elif kind == "roles":
        value["files"][0]["rows"] += 1
    elif kind == "sources":
        value["source_datasets"] = []
    elif kind == "builds":
        value["native_builds"] = 3
    elif kind == "case_limit":
        value["recipe"]["maximum_cases"] = 5
    elif kind == "logical":
        value["logical_rows_sha256"] = digest(b"other")
    elif kind == "source_fingerprint":
        value["source_datasets"][0] = digest(b"other")
    else:
        value["excluded_unknown_outcome"] -= 1
        value["excluded_unavailable_duration"] += 1
    if kind in {"logical", "source_fingerprint", "exclusions"}:
        forged = LRTSETLReceipt.model_validate(value)
        with pytest.raises(ValueError):
            validate_lrts_etl(forged, cache, path.parent)
    else:
        with pytest.raises(ValueError):
            LRTSETLReceipt.model_validate(value)


def test_valid_parquet_with_forged_byte_digest_still_requires_source_correspondence(tmp_path):
    path, receipt, cache = exported(tmp_path)
    target = path.parent / "observations.parquet"
    table = pl.read_parquet(target).with_columns(pl.lit("invented").alias("outcome"))
    table.write_parquet(target)
    payload = target.read_bytes()
    records = tuple(
        changed(f, sha256=digest(payload), size_bytes=len(payload)) if f.path == "observations.parquet" else f
        for f in receipt.files
    )
    forged = changed(receipt, files=records)
    with pytest.raises(ValueError, match="schema/rows"):
        validate_lrts_etl(forged, cache, path.parent)


def test_values_can_change_without_changing_logical_source_or_replay_names(tmp_path):
    first, _, _ = exported(tmp_path / "one")

    def change_cost(report):
        report["suites"][0]["cases"][1]["duration"] = 3.0
        return report

    second, _, _ = exported(tmp_path / "two", transform=change_cost)
    a = pl.read_parquet(first.parent / "observations.parquet")
    b = pl.read_parquet(second.parent / "observations.parquet")
    assert a["execution_id"].to_list() == b["execution_id"].to_list()
    assert a["replay_name"].to_list() == b["replay_name"].to_list()
    assert a["duration"].to_list() != b["duration"].to_list()


def test_concrete_output_symlink_and_git_paths_reject_before_content_writes(tmp_path):
    recipe, cache = prepare(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (output / "corpus-etl").symlink_to(elsewhere, target_is_directory=True)
    writer = ArtifactWriter(output)
    with pytest.raises(ValueError, match="trusted output"):
        export_lrts_parquet(recipe, cache, writer)
    assert not list(elsewhere.iterdir())
    (output / "corpus-etl").unlink()
    (output / ".git").mkdir()
    with pytest.raises(ValueError, match="outside a repository"):
        export_lrts_parquet(recipe, cache, writer)


def test_missing_cached_child_rejects_without_repair(tmp_path):
    recipe, cache = prepare(tmp_path)
    member = recipe.reports.selections[0].member
    path = cache.verify_blob(member.sha256, member.size_bytes)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_selected_packages(recipe.reports, cache)
    assert not path.exists()


def test_recipe_oversize_and_public_reader_count_are_bounded(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_bytes(b"x" * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        cli.load_lrts_etl_recipe(path)
    recipe, _, _ = owned_recipe()
    assert cli._reader(recipe.metadata)
    missing_size = recipe.metadata.model_copy(
        update={"source": recipe.metadata.source.model_copy(update={"size_bytes": None})}
    )
    with pytest.raises(ValueError, match="byte count"):
        cli._reader(missing_size)
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="2 MiB"):
        run_lrts_replay(path, cache, tmp_path / "native")


def test_versioned_old_parquet_fixtures_validate_in_external_storage(tmp_path):
    recipe, cache = prepare(tmp_path)
    fixture = Path(__file__).with_name("fixtures") / "lrts-etl-owned-v1"
    root = tmp_path / "legacy"
    shutil.copytree(fixture, root)
    receipt = LRTSETLReceipt.from_json((root / "complete.json").read_bytes())
    assert receipt.recipe == changed(recipe, feedback=LRTSFeedbackModel.LEGACY_FULL)
    validate_lrts_etl(receipt, cache, root)
    assert receipt.observed_cases == 10
    with pytest.raises(ValueError, match="read-only"):
        run_lrts_replay(root / "complete.json", cache, tmp_path / "native")


def test_reproducible_bytes_and_identity_across_external_workspaces(tmp_path):
    a, first, _ = exported(tmp_path / "one")
    b, second, _ = exported(tmp_path / "two")
    assert first == second
    assert a.read_bytes() == b.read_bytes()
    for f in first.files:
        assert (a.parent / f.path).read_bytes() == (b.parent / f.path).read_bytes()


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_selected_members_verify_both_supported_compressions(tmp_path, compression):
    recipe, outer, _ = owned_recipe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(outer)) as original, zipfile.ZipFile(buffer, "w", compression=compression) as out:
        for info in original.infolist():
            out.writestr(info.filename, original.read(info))
    recipe = changed(recipe, reports=changed(recipe.reports, source=source_spec("outer.zip", buffer.getvalue())))
    cache = SourceCache(tmp_path / "cache")
    materialize_selected_packages(recipe.reports, cache, lambda: io.BytesIO(buffer.getvalue()))
    validate_selected_packages(recipe.reports, cache)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "symlink", "metadata_budget", "entry_budget", "crc"])
def test_selected_archive_malformed_metadata_and_payload_reject(tmp_path, kind):
    recipe, outer, _ = owned_recipe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(outer)) as original, zipfile.ZipFile(buffer, "w") as out:
        for i, info in enumerate(original.infolist()):
            if i == 0 and kind == "missing":
                continue
            out.writestr(info, original.read(info))
        if kind == "duplicate":
            entry = original.infolist()[0]
            duplicate_bytes = original.read(entry)
            with pytest.warns(UserWarning, match="Duplicate name"):
                out.writestr(entry, duplicate_bytes)
        if kind == "symlink":
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (0o120777) << 16
            out.writestr(link, b"target")
    payload = buffer.getvalue()
    if kind == "crc":
        # Corrupt actual nested bytes while pinning the enclosing modified container.
        payload = payload[:100] + bytes([payload[100] ^ 1]) + payload[101:]
    selection = changed(recipe.reports, source=source_spec("outer.zip", payload))
    if kind == "metadata_budget":
        selection = changed(selection, maximum_metadata_bytes=22)
    if kind == "entry_budget":
        selection = changed(selection, maximum_entries=2)
    cache = SourceCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        materialize_selected_packages(selection, cache, lambda: io.BytesIO(payload))


def test_metadata_accounting_and_selected_entry_budget_are_not_free_editable_claims(tmp_path):
    path, receipt, cache = exported(tmp_path)
    with pytest.raises(ValueError):
        changed(receipt.metadata_accounting, input_rows=100)
    forged = changed(receipt.metadata_accounting, input_rows=100, outside_slice=98)
    forged_receipt = changed(receipt, metadata_accounting=forged)
    with pytest.raises(ValueError, match="metadata selection"):
        validate_lrts_etl(forged_receipt, cache, path.parent)
    with pytest.raises(ValueError, match="directory limit"):
        changed(receipt.recipe.reports, maximum_entries=1)


def test_cli_runs_native_and_preserves_manifest_and_round_eligibility(tmp_path, capsys):
    recipe, cache = prepare(tmp_path)
    file = tmp_path / "recipe.json"
    file.write_text(recipe.to_json())
    args = ["--recipe", str(file), "--cache", str(cache.root), "--output", str(tmp_path / "cli-output"), "--run"]
    assert cli.main(args) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["native_run_id"]
    assert summary["modeled_rounds"] == 2
    receipt = LRTSETLReceipt.from_json(Path(summary["completion"]).read_bytes())
    builds = pl.read_parquet(Path(summary["completion"]).parent / "builds.parquet")
    assert builds["replay_cases"].to_list() == [2, 2]
    assert builds["replay_status"].to_list() == ["eligible", "eligible"]
    assert receipt.metadata_accounting.outside_slice == 0


@pytest.mark.parametrize("kind", ["duplicate_sources", "zero_builds", "selected_metadata"])
def test_completion_cross_field_counts_reject_internal_contradictions(tmp_path, kind):
    _, receipt, _ = exported(tmp_path)
    value = receipt.model_dump(mode="json")
    if kind == "duplicate_sources":
        value["source_datasets"][1] = value["source_datasets"][0]
    elif kind == "zero_builds":
        value["native_builds"] = 0
    else:
        value["metadata_accounting"]["selected_rows"] = 1
        value["metadata_accounting"]["outside_slice"] = 1
    with pytest.raises(ValueError):
        LRTSETLReceipt.model_validate(value)


def test_fetch_option_does_not_open_any_reader_when_complete_sources_are_cached(tmp_path, monkeypatch, capsys):
    recipe, cache = prepare(tmp_path)
    file = tmp_path / "recipe.json"
    file.write_text(recipe.to_json())

    def unexpected():
        raise AssertionError("Verified cache hit made a network request")

    monkeypatch.setattr(cli, "_reader", lambda _: unexpected)
    assert (
        cli.main(["--recipe", str(file), "--cache", str(cache.root), "--output", str(tmp_path / "output"), "--fetch"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["observed_cases"] == 10


def test_feedback_contract_matches_real_budget_censored_rnfail():
    from coleman.evaluation.napfd import NAPFDVerdictMetric
    from coleman.reward.rnfail import RNFailReward

    metric = NAPFDVerdictMetric()
    metric.update_available_time(1.0)
    cases = [
        {"Name": "pass", "Duration": 1.0, "Verdict": 0},
        {"Name": "unscheduled-failure", "Duration": 1.0, "Verdict": 1},
    ]
    metric.evaluate(cases)
    assert metric.unscheduled_testcases == ["unscheduled-failure"]
    assert RNFailReward().evaluate(metric, ["pass", "unscheduled-failure"]) == [0.0, 0.0]
    assert cases[1]["Verdict"] == 1
    recipe, _, _ = owned_recipe()
    assert recipe.feedback == "budget-censored-native-RNFail-zero-nondetection"


def test_oversized_decoded_report_rejects_before_materialization():
    recipe, _, _ = owned_recipe()
    value = recipe.model_dump(mode="json")
    value["reports"]["selections"][0]["package"]["members"][0]["size_bytes"] = 16 * 1024**2 + 1
    with pytest.raises(ValueError, match="16 MiB"):
        LRTSETLRecipe.model_validate(value)


def test_legacy_recipe_requires_explicit_new_identity_before_export_or_fetch(tmp_path):
    recipe, _, _ = owned_recipe()
    legacy = changed(recipe, feedback=LRTSFeedbackModel.LEGACY_FULL)
    assert legacy.fingerprint != recipe.fingerprint
    assert LRTSETLRecipe.from_json(legacy.to_json()) == legacy
    fixture = Path(__file__).with_name("fixtures") / "lrts-etl-owned-recipe-v1.json"
    assert LRTSETLRecipe.from_json(fixture.read_bytes()) == legacy
    recipe_path = tmp_path / "legacy.json"
    recipe_path.write_text(legacy.to_json())
    cache = SourceCache(tmp_path / "cache")
    writer = ArtifactWriter(tmp_path / "out")
    with pytest.raises(ValueError, match="read-only"):
        export_lrts_parquet(legacy, cache, writer)
    arguments = ["--recipe", str(recipe_path), "--cache", str(cache.root), "--output", str(tmp_path / "out"), "--fetch"]
    with pytest.raises(ValueError, match="read-only"):
        cli.main(arguments)
    assert not list((tmp_path / "out").rglob("complete.json"))


@pytest.mark.parametrize("feedback", [None, "unknown-feedback-model"])
def test_missing_or_unknown_feedback_never_acquires_an_implicit_meaning(feedback):
    recipe, _, _ = owned_recipe()
    value = recipe.model_dump(mode="json")
    if feedback is None:
        value.pop("feedback")
    else:
        value["feedback"] = feedback
    with pytest.raises(ValueError):
        LRTSETLRecipe.model_validate(value)
