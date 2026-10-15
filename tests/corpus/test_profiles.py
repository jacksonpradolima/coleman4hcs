"""Descriptive native profiles, conservation, source integrity and immutable publication."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import polars as pl
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    CorpusProfileInventory,
    NativeSourceProfile,
    NumericProfile,
    load_registry,
    profile_native_source,
    publish_profile_inventory,
    publish_source_profile,
)
from coleman.corpus import profiles as implementation
from coleman.manifest import generate_manifest

ROOT = Path(__file__).parents[2]
ID = "native-example-e970afe1686d7fe6"


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def local_card(path, *, variants=False):
    card = load_registry().card(ID)
    source = changed(
        card.manifest.source,
        sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )
    identity = card.manifest.identity
    if variants:
        identity = changed(identity, observation_fields=("BuildId", "Name", "Variant"))
    return changed(card, manifest=changed(card.manifest, source=source, identity=identity))


def materialize(tmp_path, rows=None):
    frame = pl.DataFrame(
        rows
        or {
            "BuildId": [1, 1, 2, 2, 3, 3, 4],
            "Name": [1, 2, 1, 2, 1, 2, 1],
            "Duration": [0.0, None, -2.0, float("nan"), float("inf"), float("-inf"), 8.0],
            "Verdict": [0.0, 1.0, None, 2.0, float("nan"), 0.0, 1.0],
            "LastRun": ["not-a-timestamp"] * 7,
        }
    )
    path = tmp_path / "native.parquet"
    frame.write_parquet(path)
    return local_card(path), path


def test_golden_roundtrip_old_cards_and_versions():
    card = load_registry().card(ID)
    report = profile_native_source(card, ROOT / "examples/fakedata/features-engineered.parquet")
    golden = Path(__file__).parent / "fixtures/native-source-profile-v1.json"
    assert golden.read_text() == report.to_json() + "\n"
    assert NativeSourceProfile.from_json(report.to_json()) == report
    assert type(card).from_json(card.to_json()) == card
    assert report.rows == 29
    assert report.logical_builds == 5
    assert report.failed_rows == 14
    assert report.raw_duration.median == 0.134
    assert report.raw_duration.zero_count == 3
    assert report.card.manifest.blocking_reasons == card.manifest.blocking_reasons
    for version in (None, 0, 2, True, "1"):
        payload = json.loads(report.to_json())
        if version is None:
            del payload["schema_version"]
        else:
            payload["schema_version"] = version
        with pytest.raises(ValueError):
            NativeSourceProfile.from_json(json.dumps(payload))
    with pytest.raises(ValueError):
        NativeSourceProfile.from_json(card.to_json())
    with pytest.raises(ValueError, match="Duplicate JSON"):
        NativeSourceProfile.from_json('{"schema_version":1,"schema_version":1}')


def test_zero_null_negative_nonfinite_and_invalid_verdict_are_distinct(tmp_path):
    card, path = materialize(tmp_path)
    report = profile_native_source(card, path)
    assert (report.raw_duration.finite_count, report.raw_duration.null_count, report.raw_duration.nonfinite_count) == (
        3,
        1,
        3,
    )
    assert (report.raw_duration.negative_count, report.raw_duration.zero_count) == (1, 1)
    assert (report.raw_duration.minimum, report.raw_duration.median, report.raw_duration.p95) == (-2, 0, 8)
    assert report.duration_median.value == 0
    assert report.duration_median.status == "observed"
    assert report.duration_median.unit == "source-unit-unverified"
    assert (report.failed_rows, report.passed_rows, report.null_verdict_rows, report.invalid_verdict_rows) == (
        2,
        2,
        1,
        2,
    )
    assert report.chronology_status == "not_evaluated"
    assert "LastRun" in report.columns
    assert NativeSourceProfile.from_json(report.to_json()) == report


@pytest.mark.parametrize(
    "duration", [[None, None], [float("nan"), float("inf")], []], ids=["null", "nonfinite", "empty"]
)
def test_unavailable_summary_and_empty_source(tmp_path, duration):
    n = len(duration)
    frame = pl.DataFrame(
        {"BuildId": [1] * n, "Name": list(range(n)), "Verdict": [None] * n, "Duration": duration},
        schema_overrides={"BuildId": pl.Int64, "Name": pl.Int64},
    )
    path = tmp_path / "empty.parquet"
    frame.write_parquet(path)
    report = profile_native_source(local_card(path), path)
    assert report.raw_duration.finite_count == 0
    assert report.duration_median.value is None
    assert report.duration_median.status == "unavailable"
    assert report.rows == n


def test_variant_missingness_duplication_and_conflict_counts(tmp_path):
    card, path = materialize(
        tmp_path,
        {
            "BuildId": [1, 1, 1, 1, 2, None],
            "Name": [1, 1, 1, None, 1, 1],
            "Variant": ["a", "a", "b", "a", None, "a"],
            "Duration": [1.0, 2.0, 1.0, 0.0, 1.0, 1.0],
            "Verdict": [0, 1, 0, 0, 0, 0],
        },
    )
    card = local_card(path, variants=True)
    report = profile_native_source(card, path)
    assert (report.missing_identity_rows, report.unique_observations, report.duplicate_identity_rows) == (3, 2, 1)
    assert report.conflicting_identities == 1
    assert (report.logical_builds, report.execution_groups, report.missing_group_rows) == (1, 2, 2)
    assert report.tests_per_execution_group.minimum == 1
    assert report.tests_per_execution_group.maximum == 3


def test_row_permutation_preserves_statistics_and_replay_identity(tmp_path):
    card, path = materialize(tmp_path)
    first = profile_native_source(card, path)
    for seed in (0, 11, 83):
        shuffled = tmp_path / f"shuffled-{seed}.parquet"
        pl.read_parquet(path).sample(fraction=1.0, shuffle=True, seed=seed).write_parquet(shuffled)
        second = profile_native_source(local_card(shuffled), shuffled)
        assert second.model_dump(exclude={"card"}) == first.model_dump(exclude={"card"})
    assert profile_native_source(card, path).fingerprint == first.fingerprint
    duplicated = tmp_path / "duplicated.parquet"
    frame = pl.read_parquet(path)
    pl.concat([frame, frame]).write_parquet(duplicated)
    doubled = profile_native_source(local_card(duplicated), duplicated)
    assert doubled.rows == first.rows * 2
    assert doubled.logical_builds == first.logical_builds
    assert doubled.unique_observations == first.unique_observations
    assert doubled.duplicate_identity_rows == first.unique_observations
    assert doubled.raw_duration.median == first.raw_duration.median


def test_source_bounds_integrity_and_schema_before_publication(tmp_path, monkeypatch):
    card, path = materialize(tmp_path)
    with pytest.raises(ValueError, match="digest/size"):
        profile_native_source(
            changed(card, manifest=changed(card.manifest, source=changed(card.manifest.source, size_bytes=1))), path
        )
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "MAX_SOURCE_BYTES", 1)
        with pytest.raises(ValueError, match="byte limit"):
            profile_native_source(card, path)
    for limit, message in (("MAX_ROWS", "decoded row/byte"), ("MAX_DECODED_BYTES", "decoded row/byte")):
        with monkeypatch.context() as patch:
            patch.setattr(implementation, limit, 1)
            with pytest.raises(ValueError, match=message):
                profile_native_source(card, path)
    path.write_bytes(b"not a parquet")
    with pytest.raises(ValueError):
        profile_native_source(local_card(path), path)
    for frame in (
        pl.DataFrame({"BuildId": [1]}),
        pl.DataFrame({"BuildId": [1], "Name": [1], "Verdict": [0], "Duration": [True]}),
        pl.DataFrame({"BuildId": [1], "Name": [1.5], "Verdict": [0], "Duration": [1.0]}),
    ):
        frame.write_parquet(path)
        with pytest.raises(ValueError, match="columns|type"):
            profile_native_source(local_card(path), path)
    frame = pl.DataFrame({f"extra_{i}": [i] for i in range(65)})
    frame.write_parquet(path)
    with pytest.raises(ValueError, match="column limit"):
        profile_native_source(local_card(path), path)


@pytest.mark.parametrize("mutation", ["adapter", "identity", "mapping", "digest", "size"])
def test_unsupported_semantics_fail_closed(tmp_path, mutation):
    card, path = materialize(tmp_path)
    manifest = card.manifest
    if mutation == "adapter":
        manifest = changed(manifest, adapter=changed(manifest.adapter, name="other"))
    elif mutation == "identity":
        manifest = changed(manifest, identity=changed(manifest.identity, group_fields=("Name",)))
    elif mutation == "mapping":
        manifest = changed(manifest, outcomes=changed(manifest.outcomes, mapping=()))
    else:
        key = "sha256" if mutation == "digest" else "size_bytes"
        manifest = changed(manifest, source=changed(manifest.source, **{key: None}))
    with pytest.raises(ValueError):
        profile_native_source(changed(card, manifest=manifest), path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 1),
        ("missing_identity_rows", 1),
        ("failed_rows", 0),
        ("conflicting_identities", 100),
        ("execution_groups", 1),
        ("logical_builds", 99),
        ("columns", ("same", "same")),
    ],
)
def test_profile_rejects_forged_conservation(tmp_path, field, value):
    card, path = materialize(tmp_path)
    report = profile_native_source(card, path)
    with pytest.raises(ValueError):
        changed(report, **{field: value})


def test_numeric_profile_bounds_and_absence(tmp_path):
    card, path = materialize(tmp_path)
    raw = profile_native_source(card, path).raw_duration
    for values in (
        {"negative_count": 4},
        {"median": None},
        {"median": -3.0},
        {"minimum": float("inf")},
        {"finite_count": 0, "negative_count": 0, "zero_count": 0},
    ):
        with pytest.raises(ValueError):
            changed(raw, **values)
    empty = NumericProfile(
        semantics=implementation.RAW_DURATION,
        finite_count=0,
        null_count=0,
        nonfinite_count=0,
        negative_count=0,
        zero_count=0,
        minimum=None,
        median=None,
        p95=None,
        maximum=None,
    )
    assert empty.rows == 0


def test_inventory_deduplication_membership_and_diversity():
    registry = load_registry()
    profile = profile_native_source(registry.card(ID), ROOT / "examples/fakedata/features-engineered.parquet")
    inventory = CorpusProfileInventory(
        schema_version=1,
        registry=registry,
        profiles=(profile, profile),
        uncovered_required_regimes=registry.required_regimes,
    )
    assert inventory.profiles == (profile,)
    assert CorpusProfileInventory.from_json(inventory.to_json()) == inventory
    assert len(inventory.uncovered_required_regimes) == 7
    with pytest.raises(ValueError, match="diversity"):
        changed(inventory, uncovered_required_regimes=())
    other = changed(profile, chronology_status="not_evaluated", columns=tuple(reversed(profile.columns)))
    assert other == profile
    other = changed(profile, raw_duration=changed(profile.raw_duration, median=0.135))
    with pytest.raises(ValueError, match="Conflicting"):
        changed(inventory, profiles=(profile, other))
    other = changed(profile, card=changed(profile.card, languages=("unverified-language",)))
    with pytest.raises(ValueError, match="registered"):
        changed(inventory, profiles=(other,))


def test_immutable_concurrent_publication_interruption_and_native_manifest(tmp_path, monkeypatch):
    registry = load_registry()
    profile = profile_native_source(registry.card(ID), ROOT / "examples/fakedata/features-engineered.parquet")
    writer = ArtifactWriter(tmp_path)
    import os

    original = os.link

    def interrupted(*args, **kwargs):
        raise OSError("interrupted before publication")

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            publish_source_profile(profile, writer)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    assert os.link is original
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: publish_source_profile(profile, writer), range(8)))
    assert len(set(paths)) == 1
    inventory = CorpusProfileInventory(
        schema_version=1, registry=registry, profiles=(profile,), uncovered_required_regimes=registry.required_regimes
    )
    inventory_path = publish_profile_inventory(inventory, writer)
    assert publish_profile_inventory(inventory, writer) == inventory_path
    manifest = json.loads(generate_manifest(tmp_path, run_id="profile-ci").read_bytes())
    assert {entry["type"] for entry in manifest["artifacts"]} >= {"corpus_source_profile", "corpus_profile_inventory"}
    paths[0].write_text("conflicting bytes")
    with pytest.raises(ValueError, match="Conflicting immutable"):
        publish_source_profile(profile, writer)


def test_existing_native_inventory_preserves_variant_identity_and_blockers():
    registry = load_registry()
    reports = []
    for card in registry.cards:
        if card.manifest.adapter.name != "native-parquet-corpus":
            continue
        relative = unquote(card.manifest.source.uri.split("/b743f51d5ba444410b32f2c0e318b2587b882c43/")[1])
        report = profile_native_source(card, ROOT / relative)
        assert report.card == card
        reports.append(report)
    assert len(reports) == 10
    assert sum(report.rows for report in reports) == 1_386_730
    variants = next(report for report in reports if "Variant" in report.columns)
    assert variants.execution_groups == 1686
    assert variants.logical_builds == 1036
    assert variants.duplicate_identity_rows == 7132
    assert variants.conflicting_identities == 5154
    assert all(report.generalization == "insufficient_evidence" for report in reports)


def test_explicit_units_are_retained_and_incompatible_profiles_rejected(tmp_path):
    card, path = materialize(tmp_path)
    for unit in ("s", "ms"):
        semantics = {"unit": unit, "meaning": "declared-test-runner-occupancy-v1"}
        typed = changed(
            card, manifest=changed(card.manifest, outcomes=changed(card.manifest.outcomes, duration=semantics))
        )
        profile = profile_native_source(typed, path)
        assert profile.duration_median.unit == unit
        assert profile.raw_duration.semantics.unit == unit
        assert profile.raw_duration.minimum == -2
        incompatible = changed(profile.raw_duration, semantics=implementation.GROUP_ROWS)
        with pytest.raises(ValueError, match="Incompatible"):
            changed(profile, raw_duration=incompatible)
        incompatible = changed(profile.tests_per_execution_group, semantics=implementation.RAW_DURATION)
        with pytest.raises(ValueError, match="Incompatible"):
            changed(profile, tests_per_execution_group=incompatible)


def test_column_missingness_and_complete_group_distribution_are_validated(tmp_path):
    card, path = materialize(tmp_path)
    report = profile_native_source(card, path)
    missingness = {item.name: item.null_count for item in report.column_missingness}
    assert missingness["Duration"] == 1
    assert missingness["Verdict"] == 1
    assert missingness["LastRun"] == 0
    for columns in ((), report.columns[:-1], (*report.columns, "invented")):
        with pytest.raises(ValueError):
            changed(report, columns=columns)
    bad = changed(report.column_missingness[0], null_count=report.rows + 1)
    with pytest.raises(ValueError, match="Missingness"):
        changed(report, column_missingness=(bad, *report.column_missingness[1:]))
    with pytest.raises(ValueError, match="positive"):
        changed(report, tests_per_execution_group=changed(report.tests_per_execution_group, zero_count=1))


@pytest.mark.parametrize("count,median,p95", [(1, 0, 0), (20, 9, 18), (21, 10, 19)])
def test_inverse_empirical_quantiles_have_exact_rank_boundaries(tmp_path, count, median, p95):
    card, path = materialize(
        tmp_path,
        {"BuildId": [1] * count, "Name": list(range(count)), "Duration": list(range(count)), "Verdict": [0] * count},
    )
    report = profile_native_source(card, path)
    assert report.raw_duration.median == median
    assert report.raw_duration.p95 == p95
    assert report.raw_duration.minimum == 0
    assert report.raw_duration.maximum == count - 1


def test_inventory_interruption_resumes_without_false_completed_inventory(tmp_path, monkeypatch, capsys):
    import runpy

    example = runpy.run_path(str(ROOT / "examples/corpus-profiles/inventory.py"))
    main = example["main"]
    original = main.__globals__["publish_source_profile"]
    published = []

    def interrupted(report, writer):
        if published:
            raise OSError("interrupted between source publications")
        published.append(original(report, writer))
        return published[-1]

    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as patch:
        patch.setitem(main.__globals__, "publish_source_profile", interrupted)
        with pytest.raises(OSError, match="between source"):
            main(str(ROOT), "inventory")
    assert len(list((tmp_path / "inventory").glob("corpus-profiles/*.json"))) == 1
    assert not list((tmp_path / "inventory").glob("corpus-profile-inventories/*.json"))
    main(str(ROOT), "inventory")
    result = json.loads(capsys.readouterr().out)
    assert result["profiled_sources"] == 10
    assert result["evidence_status"] == "insufficient_diversity"
    assert len(list((tmp_path / "inventory").glob("corpus-profiles/*.json"))) == 10
    main(str(ROOT), "inventory")
    assert json.loads(capsys.readouterr().out) == result


def test_inventory_example_rejects_symlinked_artifact_parent(tmp_path, monkeypatch):
    import runpy

    main = runpy.run_path(str(ROOT / "examples/corpus-profiles/inventory.py"))["main"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = workspace / "output"
    output.mkdir()
    (output / "corpus-profiles").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.chdir(workspace)
    with pytest.raises(ValueError, match="parents"):
        main(str(ROOT), "output")
    with pytest.raises(ValueError, match="output"):
        main(str(ROOT), "../escape")


def test_signed_zero_quantiles_are_canonical_under_row_permutation(tmp_path):
    reports = []
    for index, durations in enumerate(([0.0, -0.0], [-0.0, 0.0])):
        directory = tmp_path / str(index)
        directory.mkdir()
        card, path = materialize(
            directory, {"BuildId": [1, 1], "Name": [1, 2], "Verdict": [0, 0], "Duration": durations}
        )
        report = profile_native_source(card, path)
        reports.append(report)
        assert report.raw_duration.zero_count == 2
        assert report.raw_duration.negative_count == 0
        assert "-0.0" not in report.raw_duration.model_dump_json()
    assert reports[0].raw_duration == reports[1].raw_duration
    assert reports[0].raw_duration.model_dump_json() == reports[1].raw_duration.model_dump_json()
    assert reports[0].card.manifest.source.sha256 != reports[1].card.manifest.source.sha256


def test_decimal_verdicts_are_compared_exactly_without_binary64_rounding(tmp_path):
    from decimal import Decimal

    card, path = materialize(
        tmp_path,
        {
            "BuildId": [1, 1, 1, 1, 1],
            "Name": [1, 2, 3, 4, 5],
            "Duration": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Verdict": [
                Decimal("0.99999999999999999999"),
                Decimal("1.00000000000000000000"),
                Decimal("0.00000000000000000000"),
                Decimal("1.00000000000000000001"),
                None,
            ],
        },
    )
    report = profile_native_source(card, path)
    assert report.failed_rows == 1
    assert report.passed_rows == 1
    assert report.invalid_verdict_rows == 2
    assert report.null_verdict_rows == 1
    assert NativeSourceProfile.from_json(report.to_json()) == report


def test_numeric_contract_normalizes_signed_zero_on_read():
    profile = NumericProfile(
        semantics=implementation.RAW_DURATION,
        finite_count=1,
        null_count=0,
        nonfinite_count=0,
        negative_count=0,
        zero_count=1,
        minimum=-0.0,
        median=-0.0,
        p95=-0.0,
        maximum=-0.0,
    )
    assert "-0.0" not in profile.model_dump_json()
    report = profile_native_source(load_registry().card(ID), ROOT / "examples/fakedata/features-engineered.parquet")
    payload = json.loads(report.to_json())
    payload["raw_duration"]["minimum"] = -0.0
    assert NativeSourceProfile.from_json(json.dumps(payload)).to_json() == report.to_json()


def test_null_counts_cannot_contradict_column_profiles(tmp_path):
    card, path = materialize(tmp_path)
    report = profile_native_source(card, path)
    for name in ("Duration", "Verdict"):
        columns = tuple(
            changed(item, null_count=0) if item.name == name else item for item in report.column_missingness
        )
        with pytest.raises(ValueError, match="null counts disagree"):
            changed(report, column_missingness=columns)
