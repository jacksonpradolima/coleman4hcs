"""Owned full-history ZIP64 → raw DTO → Parquet → native replay, without component mocks."""

import io
import json
import math
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from coleman.application.corpus import run_rtptorrent_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import SourceCache
from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.corpus.reference import tiny_ci_reference
from coleman.corpus.rtptorrent_contracts import RTPTorrentSpec
from coleman.corpus.rtptorrent_etl import export_rtptorrent_parquet, validate_rtptorrent_etl
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt, RTPTorrentETLRecipe
from coleman.corpus.zip64_contracts import Zip64SelectionSpec
from coleman.corpus.zip64_selection import materialize_zip64_files
from coleman.pilot.contracts import SECONDS, PilotBuild, PilotTest
from coleman.results.domain import fingerprint
from coleman.selection.contracts import Quantity
from tests.corpus.test_rtptorrent import HEADER
from tests.corpus.test_source_views import changed, digest, source_spec
from tests.corpus.test_zip64_selection import owned_zip64

ROWS = (
    "101,owned.Repeated,0,0,1,0,0,0\n",
    "101,owned.Repeated,1,2.50,1,1,0,0\n",
    "101,owned.Unknown,2,,,,,\n",
    "101,owned.Skipped,3,0,2,0,0,2\n",
    "101,owned.NoDuration,4,,1,0,0,0\n",
    "102,owned.Repeated,10,1,1,0,0,0\n",
    "102,owned.Repeated,11,3,1,1,0,0\n",
    "999,owned.Outside,0,0,1,0,0,0\n",
)


def owned_recipe(*, rows=ROWS):
    """Pin a complete owned history including repeated classes, zero, null and primary exclusions."""
    payload = (HEADER + "".join(rows)).encode()
    logical = "owned@ci/class-results.csv"
    archive = owned_zip64(((logical, payload),))

    def package(name, data):
        original = source_spec(name, data)
        return changed(original, source=changed(original.source, version="1.1"))

    child = package("class-results.csv", payload)
    selection = Zip64SelectionSpec.model_validate(
        {
            "schema_version": 1,
            "source": package("owned-history.zip", archive),
            "selections": [
                {
                    "member": {"name": logical, "sha256": digest(payload), "size_bytes": len(payload)},
                    "package": child,
                }
            ],
        }
    )
    conversion = RTPTorrentSpec(
        schema_version=1,
        materialization=MaterializationReceipt(schema_version=1, spec=child),
        member="class-results.csv",
        project="owned/ci",
        history="constructed",
    )
    recipe = RTPTorrentETLRecipe(
        schema_version=1,
        source=selection,
        conversion=conversion,
        dataset_id="owned-rtptorrent",
        replay_job_ids=(101, 102),
        feedback="budget-censored-native-RNFail-zero-nondetection",
    )
    return recipe, archive


def prepare(tmp_path, **kwargs):
    recipe, source = owned_recipe(**kwargs)
    cache = SourceCache(tmp_path / "cache")
    materialize_zip64_files(recipe.source, cache, lambda: io.BytesIO(source))
    return recipe, cache


def test_actual_parquet_source_identity_units_conservation_and_idempotence(tmp_path):
    recipe, cache = prepare(tmp_path)
    writer = ArtifactWriter(tmp_path / "output")
    path = export_rtptorrent_parquet(recipe, cache, writer)
    receipt = RTPTorrentETLReceipt.from_json(path.read_bytes())
    assert receipt.observed_executions == receipt.input_rows == 8
    assert receipt.replay_executions == 4
    assert receipt.excluded_outside_jobs == receipt.excluded_not_run == 1
    assert receipt.excluded_unknown_outcome == receipt.excluded_unavailable_duration == 1
    assert receipt.source_jobs == 3 and receipt.native_rounds == 2
    rows = pl.read_parquet(path.parent / "observations.parquet")
    assert rows["execution_id"].n_unique() == 8
    assert rows["duration_decimal"].null_count() == 2
    assert rows["duration_decimal"].to_list().count("0") == 3
    assert rows["duration_unit"].unique().to_list() == ["s"]
    native = pl.read_parquet(path.parent / "datasets/owned-rtptorrent/features-engineered.parquet")
    assert native["Name"].n_unique() == 2 and native["SourceExecutionId"].n_unique() == 4
    assert native["Duration"].to_list() == [0, 2.5, 1, 3]
    jobs = pl.read_parquet(path.parent / "jobs.parquet")
    assert jobs["job_id"].to_list() == [101, 102, 999]
    assert jobs["logical_build_id"].null_count() == jobs["event_at"].null_count() == 3
    assert jobs["modeled_round"].to_list() == [1, 2, None]
    validate_rtptorrent_etl(receipt, cache, path.parent)
    assert export_rtptorrent_parquet(recipe, cache, writer) == path
    assert RTPTorrentETLReceipt.from_json(receipt.to_json()) == receipt
    assert RTPTorrentETLRecipe.from_json(recipe.to_json()) == recipe
    assert Path(path).name == "complete.json"


def test_native_api_checkpoint_recovery_and_independent_two_job_reference(tmp_path):
    recipe, cache = prepare(tmp_path)
    completion = export_rtptorrent_parquet(recipe, cache, ArtifactWriter(tmp_path / "output"))
    first = run_rtptorrent_replay(completion, cache, tmp_path / "native", budgets=(1.0,))
    second = run_rtptorrent_replay(completion, cache, tmp_path / "native", budgets=(1.0,))
    assert first.run_id == second.run_id
    assert second.artifacts_dir is not None
    root = Path(second.artifacts_dir)
    rows = pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True)
    assert rows.height == 4 and rows["step"].n_unique() == 2
    names = tuple(fingerprint([recipe.test_identity, "owned/ci", "owned.Repeated", n]) for n in (0, 1))
    for row in rows.iter_rows(named=True):
        values = ("0", "2.5") if row["step"] == 1 else ("1", "3")
        build = PilotBuild(
            build_id="owned-model-" + str(row["step"]),
            decided_at=datetime(2026, 1, 1, tzinfo=UTC),
            outcomes_available_at=datetime(2026, 1, 1, tzinfo=UTC),
            baseline_order=names,
            tests=tuple(
                PilotTest(
                    test_id=name,
                    duration=Quantity(value=Decimal(value), semantics=SECONDS),
                    outcome="failed" if index else "passed",
                )
                for index, (name, value) in enumerate(zip(names, values, strict=True))
            ),
        )
        reference = tiny_ci_reference(
            build,
            tuple(json.loads(row["prioritization_order_top_k"])),
            Quantity(value=sum(map(Decimal, values), Decimal(0)), semantics=SECONDS),
        )
        assert row["detected"] == 1 and row["missed"] == 0
        assert reference.napfd.value is not None and reference.apfdc.value is not None
        assert math.isclose(row["fitness"], float(reference.napfd.value), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(row["cost"], float(reference.apfdc.value), rel_tol=0, abs_tol=1e-12)
    checkpoints = list((root / "checkpoints").rglob("progress_*.json"))
    assert checkpoints and all(json.loads(path.read_bytes())["step_committed"] == 2 for path in checkpoints)
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert any(item["type"] == "corpus_etl_receipt" for item in manifest["artifacts"])
