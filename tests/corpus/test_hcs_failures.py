"""Invalid source metadata, immutable completion gates and honest optional local-source transport."""

import io
import json

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.hcs_contracts import HCSRecipe
from coleman.corpus.hcs_etl import export_hcs_parquet, preflight_hcs_etl
from coleman.corpus.hcs_projection import project_hcs
from coleman.corpus.hcs_raw import _integer, read_hcs_slice
from coleman.corpus.materialization import materialize_source
from coleman.results.domain import canonical_json
from coleman.transports.hcs_etl import load_hcs_recipe, main
from tests.corpus.hcs_support import hcs_recipe, prepare_hcs
from tests.corpus.test_source_views import changed, source_spec


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "01", "1.0", "", 2**63, str(2**63)])
def test_ambiguous_or_out_of_native_bound_ids_reject(value):
    with pytest.raises(ValueError):
        _integer(value)


@pytest.mark.parametrize(
    "kind", ["duplicate", "pipeline", "commit", "sha", "name", "missing_selected", "contradictory_commit"]
)
def test_bad_original_job_metadata_rejects_without_export_completion(tmp_path, kind):
    def invalid(jobs):
        if kind == "duplicate":
            jobs.append(jobs[0].copy())
        elif kind == "pipeline":
            jobs[0]["pipeline"] = None
        elif kind == "commit":
            jobs[0]["commit"] = None
        elif kind == "sha":
            jobs[0]["pipeline"]["sha"] = "f" * 40
        elif kind == "name":
            jobs[0]["name"] = ""
        elif kind == "missing_selected":
            jobs = [j for j in jobs if j["pipeline"]["id"] != 101]
        else:
            jobs[1]["commit"]["id"] = "f" * 40
            jobs[1]["pipeline"]["sha"] = "f" * 40
        return jobs

    recipe, cache = prepare_hcs(tmp_path, jobs_transform=invalid)
    with pytest.raises(ValueError):
        export_hcs_parquet(recipe, cache, ArtifactWriter(tmp_path / "etl"))
    assert not list((tmp_path / "etl").rglob("complete.json"))


def test_missing_explicit_cost_model_and_forged_duration_lexeme_reject(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    value = recipe.model_dump(mode="json")
    value.pop("native_costs")
    with pytest.raises(ValueError):
        HCSRecipe.model_validate(value)
    case = read_hcs_slice(recipe, cache).observations[0]
    with pytest.raises(ValueError, match="original lexeme"):
        changed(case, duration_lexeme="1")
    amount = changed(case.duration, semantics=dict(unit="ms", meaning="other-unit-v1"))
    with pytest.raises(ValueError, match="Incompatible HCS observation"):
        changed(case, duration=amount)


def test_unsafe_output_separation_symlink_and_repository_fail_before_acquisition(tmp_path):
    recipe, _, _ = hcs_recipe()
    with pytest.raises(ValueError, match="separate"):
        preflight_hcs_etl(recipe, tmp_path / "same", tmp_path / "same")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    with pytest.raises(ValueError, match="outside a repository"):
        preflight_hcs_etl(recipe, repository / "cache", tmp_path / "etl")
    root = preflight_hcs_etl(recipe, tmp_path / "cache", tmp_path / "etl")
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "datasets").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        preflight_hcs_etl(recipe, tmp_path / "cache", tmp_path / "etl")
    assert not (tmp_path / "cache").exists()


def test_optional_source_cli_native_matrix_and_offline_retry_are_machine_readable(tmp_path, capsys):
    recipe, csv_bytes, jobs_bytes = hcs_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(recipe.to_json())
    csv = tmp_path / "executions.csv"
    csv.write_bytes(csv_bytes)
    jobs = tmp_path / "jobs.json"
    jobs.write_bytes(jobs_bytes)
    args = ["--recipe", str(path), "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "etl"), "--run"]
    assert main(args + ["--executions", str(csv), "--jobs", str(jobs)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["accounting"]["replay_rows"] == 10
    assert {r["strategy"] for r in first["native_runs"]} == {"wts", "vts"}
    assert load_hcs_recipe(path) == recipe
    assert load_hcs_recipe().project == "libssh"
    path.write_bytes(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="exceeds 2 MiB"):
        load_hcs_recipe(path)


def test_native_symlink_rejected_before_optional_cli_cache_creation(tmp_path):
    recipe, _, _ = hcs_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(recipe.to_json())
    output = tmp_path / "etl"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "native").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        main(["--recipe", str(path), "--cache", str(tmp_path / "cache"), "--output", str(output), "--run"])
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("kind", ["header", "short_row", "long_row", "field_size", "selected_rows", "total_rows"])
def test_raw_csv_schema_and_bounded_dispositions_reject_before_completion(tmp_path, kind):
    recipe, cache = prepare_hcs(tmp_path)
    header = b"job_id;sha;pipeline_id;last_run;test_name;verdict;duration\n"
    row = ("1;" + "a" * 40 + ";101;;A;0;0\n").encode()
    if kind == "header":
        payload = b"wrong;columns\n"
    elif kind == "short_row":
        payload = header + b"1;sha;101\n"
    elif kind == "long_row":
        payload = header + row.rstrip(b"\n") + b";extra\n"
    elif kind == "field_size":
        payload = header + row.replace(b";A;", b";" + b"A" * 1025 + b";")
    elif kind == "selected_rows":
        payload = header + row * 16001
    else:
        payload = header + b"1;a;303;;A;0;0\n" * 500001
    package = source_spec("bounded-executions.csv", payload)
    recipe = changed(recipe, executions=package)
    materialize_source(package, cache, lambda: io.BytesIO(payload))
    with pytest.raises(ValueError):
        export_hcs_parquet(recipe, cache, ArtifactWriter(tmp_path / "etl"))
    assert not list((tmp_path / "etl").rglob("complete.json"))


@pytest.mark.parametrize("kind", ["root", "record", "commit_id", "job_count"])
def test_raw_job_json_schema_and_record_budget_reject(tmp_path, kind):
    recipe, cache = prepare_hcs(tmp_path)
    _, _, jobs_bytes = hcs_recipe()
    jobs = json.loads(jobs_bytes)
    if kind == "root":
        jobs = {}
    elif kind == "record":
        jobs = [None]
    elif kind == "commit_id":
        jobs[0]["commit"]["id"] = 3
    else:
        jobs = [{}] * 16385
    payload = json.dumps(jobs).encode()
    package = source_spec("bounded-jobs.json", payload)
    recipe = changed(recipe, jobs=package)
    materialize_source(package, cache, lambda: io.BytesIO(payload))
    with pytest.raises(ValueError):
        export_hcs_parquet(recipe, cache, ArtifactWriter(tmp_path / "etl"))


@pytest.mark.parametrize("kind", ["variants", "suite"])
def test_native_projection_bounds_do_not_silently_drop_excess_members(tmp_path, kind):
    def rows_transform(rows):
        if kind == "suite":
            return [{**rows[0], "test_name": f"test-{i}"} for i in range(1001)]
        return rows + [{**rows[0], "job_id": str(100 + i)} for i in range(65)]

    def jobs_transform(jobs):
        if kind == "variants":
            jobs += [{**jobs[0], "id": 100 + i, "name": f"configuration-{i}"} for i in range(65)]
        return jobs

    recipe, cache = prepare_hcs(tmp_path, rows_transform=rows_transform, jobs_transform=jobs_transform)
    with pytest.raises(ValueError, match="variant bounds|suite exceeds"):
        export_hcs_parquet(recipe, cache, ArtifactWriter(tmp_path / "etl"))
    assert not list((tmp_path / "etl").rglob("complete.json"))


@pytest.mark.parametrize(
    "kind", ["counts", "duplicate_job", "pipeline", "commit", "positions", "order", "row_bound", "binding"]
)
def test_pure_projection_revalidates_unchecked_nested_source_objects(tmp_path, kind):
    recipe, cache = prepare_hcs(tmp_path)
    source = read_hcs_slice(recipe, cache)
    updates = {}
    if kind == "counts":
        updates["input_rows"] = source.input_rows + 1
    elif kind == "duplicate_job":
        updates["jobs"] = (*source.jobs, source.jobs[0])
    elif kind == "pipeline":
        updates["recipe"] = changed(recipe, pipeline_ids=(101,))
    elif kind == "commit":
        updates["jobs"] = (source.jobs[0], changed(source.jobs[1], commit_id="f" * 40), *source.jobs[2:])
    elif kind in {"positions", "order", "row_bound", "binding"}:
        cases = list(source.observations)
        if kind == "positions":
            cases[1] = changed(cases[1], source_position=cases[0].source_position)
        elif kind == "order":
            cases.reverse()
        elif kind == "row_bound":
            cases[-1] = changed(cases[-1], source_position=source.input_rows + 1)
        else:
            cases[0] = changed(cases[0], job=changed(cases[0].job, variant="unbound"))
        updates["observations"] = tuple(cases)
    forged = source.model_copy(update=updates)
    with pytest.raises(ValueError):
        project_hcs(forged)


def test_wire_bounded_recipe_cannot_publish_a_completion_too_large_for_its_reader(tmp_path):
    recipe, _, _ = hcs_recipe()
    raw = recipe.model_dump(mode="json")
    raw["executions"]["source"]["citation"] = "A" * (2 * 1024**2 - len(recipe.to_json().encode()) - 256)
    payload = canonical_json(raw).encode()
    assert len(payload) < 2 * 1024**2
    path = tmp_path / "recipe.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="canonical metadata exceeds 1 MiB"):
        main(["--recipe", str(path), "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "etl")])
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "etl").exists()
