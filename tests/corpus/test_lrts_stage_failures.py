"""Stage-view input, path, source-correspondence and compatibility rejection boundaries."""

import json
from pathlib import Path

import polars as pl
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_stage_contracts import LRTSStageBinding, LRTSStageRecipe
from coleman.corpus.lrts_stage_etl import export_lrts_stage_parquet, preflight_lrts_stage_etl
from coleman.transports import lrts_stage_etl as cli
from tests.corpus.test_lrts_stage_etl import prepare, stage_recipe
from tests.corpus.test_source_views import changed


def test_cli_explicit_actual_source_offline_retry_and_incorrect_explicit_archive(tmp_path, capsys):
    recipe, archive, metadata = stage_recipe()
    source = tmp_path / "outer.zip"
    source.write_bytes(archive)
    csv = tmp_path / "dataset.csv"
    csv.write_bytes(metadata)
    config = tmp_path / "recipe.json"
    config.write_text(recipe.to_json())
    args = ["--recipe", str(config), "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "output")]
    assert cli.main(args + ["--archive", str(source), "--metadata", str(csv)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert first["observed_cases"] == 12
    assert first["excluded_outside_stage"] == 6
    source.write_bytes(b"owned wrong explicit source, despite complete cached source")
    with pytest.raises(ValueError):
        cli.main(args + ["--archive", str(source)])
    csv.write_bytes(b"owned wrong explicit metadata, despite complete cached source")
    with pytest.raises(ValueError):
        cli.main(args + ["--metadata", str(csv)])


@pytest.mark.parametrize("role", ["namespace", "observations", "native", "receipt", "nested_git", "native_output"])
def test_cli_rejects_unsafe_concrete_output_before_cache_construction(tmp_path, role):
    recipe, _, _ = stage_recipe()
    config = tmp_path / "recipe.json"
    config.write_text(recipe.to_json())
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    root = preflight_lrts_stage_etl(recipe, cache, output)
    outside = tmp_path / "outside"
    outside.mkdir()
    targets = {
        "namespace": root,
        "observations": root / "observations.parquet",
        "native": root / "datasets",
        "receipt": root / "complete.json",
        "native_output": output / "native",
    }
    if role == "nested_git":
        target = root / "datasets"
        (target / ".git").mkdir(parents=True)
    else:
        target = targets[role]
        target.parent.mkdir(parents=True)
        target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        cli.main(["--recipe", str(config), "--cache", str(cache), "--output", str(output), "--run"])
    assert not cache.exists()
    assert not list(outside.iterdir())


@pytest.mark.parametrize(
    "kind",
    ["duplicate_node", "missing_build", "missing_native_stage", "future_source", "legacy_feedback", "unsafe_alias"],
)
def test_invalid_recipe_declarations_reject_before_source_access(kind):
    recipe, _, _ = stage_recipe()
    value = recipe.model_dump(mode="json")
    if kind == "duplicate_node":
        value["stage_bindings"][0]["node_ids"] *= 2
    elif kind == "missing_build":
        value["stage_bindings"] = value["stage_bindings"][:2]
    elif kind == "missing_native_stage":
        value["native_stage_id"] = "undeclared"
    elif kind == "future_source":
        value["source"]["schema_version"] = 2
    elif kind == "legacy_feedback":
        value["source"]["feedback"] = "full-reported-labels-after-modeled-round"
    else:
        value["dataset_id"] = "../escape"
    with pytest.raises(ValueError):
        LRTSStageRecipe.model_validate(value)


@pytest.mark.parametrize("kind", ["missing_context", "missing_node", "extra_context", "extra_node", "empty_suite"])
def test_raw_stage_relations_include_empty_suites_and_exact_nodes(tmp_path, kind):
    def modify(report):
        if kind == "missing_context":
            report["suites"][0]["enclosingBlockNames"] = None
        elif kind == "missing_node":
            report["suites"][0]["nodeId"] = None
        elif kind == "empty_suite":
            report["suites"].append(
                dict(
                    name="owned.Empty",
                    id=None,
                    nodeId="unbound-empty-node",
                    enclosingBlocks=[],
                    enclosingBlockNames=["unbound-empty-context"],
                    timestamp=None,
                    duration=0,
                    cases=[],
                )
            )
        return report

    recipe, cache = prepare(tmp_path, transform=modify)
    if kind in {"extra_context", "extra_node"}:
        bindings = list(recipe.stage_bindings)
        key, value = (
            ("node_ids", ("node-a", "extra-node"))
            if kind == "extra_node"
            else ("enclosing_block_names", ("owned", "linux"))
        )
        bindings[0] = changed(bindings[0], **{key: value})
        recipe = changed(recipe, stage_bindings=tuple(bindings))
    writer = ArtifactWriter(tmp_path / "output")
    with pytest.raises(ValueError, match="raw context/node relation"):
        export_lrts_stage_parquet(recipe, cache, writer)
    assert not list((tmp_path / "output").rglob("complete.json"))


def test_underscore_collision_contexts_are_distinct_exact_tuples(tmp_path):
    def contexts(report):
        report["suites"][0]["enclosingBlockNames"] = ["owned_a", "b"]
        report["suites"][1]["enclosingBlockNames"] = ["owned", "a_b"]
        return report

    recipe, cache = prepare(tmp_path, transform=contexts)
    bindings = tuple(
        changed(b, enclosing_block_names=("owned_a", "b") if b.stage_id == "linux" else ("owned", "a_b"))
        for b in recipe.stage_bindings
    )
    recipe = changed(recipe, stage_bindings=bindings)
    path = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    observations = pl.read_parquet(path.parent / "observations.parquet")
    assert observations["stage_id"].unique().sort().to_list() == ["linux", "windows"]


def test_same_source_produces_identical_artifacts_across_external_roots(tmp_path):
    recipe, cache = prepare(tmp_path / "one")
    other, cache2 = prepare(tmp_path / "two")
    a = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "out-a"))
    b = export_lrts_stage_parquet(other, cache2, ArtifactWriter(tmp_path / "out-b"))
    assert a.name == b.name
    assert a.parent.name == b.parent.name
    assert a.read_bytes() == b.read_bytes()
    for p in a.parent.rglob("*.parquet"):
        assert p.read_bytes() == (b.parent / p.relative_to(a.parent)).read_bytes()


def test_recipe_size_limit_and_loader_budgets(tmp_path):
    with pytest.raises(ValueError, match="Unknown LRTS stage preset"):
        cli.load_lrts_stage_recipe(preset="unknown")
    path = tmp_path / "recipe.json"
    path.write_bytes(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="exceeds 2 MiB"):
        cli.load_lrts_stage_recipe(path)
    recipe, _, _ = stage_recipe()
    bindings = tuple(
        LRTSStageBinding(
            pull_request=b.pull_request,
            build_id=b.build_id,
            stage_id=f"owned-{i}",
            enclosing_block_names=(str(i), *("x" * 1024 for _ in range(63))),
            node_ids=(str(i),),
        )
        for b in recipe.source.builds
        for i in range(16)
    )
    with pytest.raises(ValueError, match="1 MiB canonical metadata limit"):
        changed(recipe, native_stage_id="owned-0", stage_bindings=bindings)


def test_source_units_remain_checked_by_existing_dto():
    from coleman.corpus.lrts_contracts import MEANINGS, LRTSCase
    from coleman.results.domain import Measurement

    measure = Measurement(
        name="lrts_case_duration",
        value=0,
        unit="s",
        meaning=MEANINGS["lrts_case_duration"],
        direction="none",
        minimum=0,
        status="observed",
    )
    case = LRTSCase(
        position=0, class_name="owned.C", method_name="test", source_status="PASSED", skipped=False, duration=measure
    )
    incompatible = changed(measure, unit="ms")
    with pytest.raises(ValueError, match="units/meaning/status"):
        changed(case, duration=incompatible)


def test_published_legacy_v1_receipt_stays_readable_and_validates_original_artifacts(tmp_path):
    from coleman.corpus.lrts_etl import export_lrts_parquet, validate_lrts_etl
    from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
    from tests.corpus.test_lrts_etl import prepare as legacy_prepare

    recipe, cache = legacy_prepare(tmp_path)
    path = export_lrts_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    fixture = Path(__file__).with_name("fixtures") / "lrts-etl-owned-completion-v1.json"
    receipt = LRTSETLReceipt.from_json(fixture.read_bytes())
    assert receipt.recipe == recipe
    legacy_bytes = fixture.read_bytes()
    current = LRTSETLReceipt.from_json(path.read_bytes())
    assert (receipt.to_json() + "\n").encode() == legacy_bytes
    assert current.producer_sha256 != receipt.producer_sha256
    assert legacy_bytes.count(receipt.producer_sha256.encode()) == 1
    assert path.read_bytes() == legacy_bytes.replace(receipt.producer_sha256.encode(), current.producer_sha256.encode())
    validate_lrts_etl(receipt, cache, path.parent)
    legacy_payload = recipe.to_json()
    with pytest.raises(ValueError):
        LRTSStageRecipe.from_json(legacy_payload)


def test_dataset_alias_uses_artifact_path_contract_before_acquisition(tmp_path):
    recipe, _, _ = stage_recipe()
    with pytest.raises(ValueError, match="canonical relative path"):
        changed(recipe, dataset_id="owned.")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "alias", ["CON", "con", "CON.txt", "PRN", "AUX", "NUL", "COM1", "com9.log", "LPT1", "lpt9.txt"]
)
def test_windows_reserved_aliases_reject_on_every_host_before_acquisition(tmp_path, alias):
    recipe, _, _ = stage_recipe()
    with pytest.raises(ValueError, match="Windows-reserved"):
        changed(recipe, dataset_id=alias)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("alias", ["CONSOLE", "COM10", "LPT10", "auxiliary", "normal.con"])
def test_portable_alias_validation_does_not_reject_similar_ordinary_names(alias):
    recipe, _, _ = stage_recipe()
    assert changed(recipe, dataset_id=alias).dataset_id == alias


def test_cli_rejects_reserved_alias_before_creating_source_cache(tmp_path):
    recipe, _, _ = stage_recipe()
    value = recipe.model_dump(mode="json")
    value["dataset_id"] = "NUL"
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(value))
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Windows-reserved"):
        cli.main(["--recipe", str(config), "--cache", str(cache), "--output", str(output)])
    assert not cache.exists()
    assert not output.exists()
