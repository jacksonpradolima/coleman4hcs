"""Owned controlled-source observations: exact identities, negative outcomes and cache integrity."""

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import MaterializationSpec, SourceCache, load_registry, materialize_source
from coleman.corpus.controlled import convert_controlled_source, publish_controlled_dataset, validate_controlled_dataset
from coleman.corpus.controlled_contracts import (
    ControlledContext,
    ControlledDataset,
    ControlledElapsed,
    ControlledInput,
    ControlledObservation,
    ControlledPair,
    ControlledReport,
    ControlledRuntime,
    ControlledSpec,
)
from coleman.manifest import generate_manifest


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def materialized(cache, name, payload, version="owned-v1"):
    spec = MaterializationSpec.model_validate(
        {
            "schema_version": 1,
            "package_id": name,
            "archive": "raw",
            "source": {
                "uri": "urn:coleman:owned:" + name,
                "version": version,
                "citation": "Repository-authored CI correctness fixture; no foreign rows",
                "sha256": digest(payload),
                "size_bytes": len(payload),
                "availability": "not_verified",
            },
            "license": load_registry().card("ci-serial-boundary-v1").manifest.license.model_dump(),
            "members": [{"name": name + ".json", "sha256": digest(payload), "size_bytes": len(payload)}],
        }
    )
    return materialize_source(spec, cache, lambda: io.BytesIO(payload))


def context(cache):
    inputs = []
    for kind, revision in [
        ("buggy_source", "buggy-v1"),
        ("fixed_source", "fixed-v1"),
        ("test_fixture", "fixed-v1"),
        ("harness", "harness-v1"),
    ]:
        receipt = materialized(cache, kind, json.dumps({"owned": kind}).encode(), revision)
        inputs.append(ControlledInput.model_validate({"name": kind, "kind": kind, "materialization": receipt}))
    return ControlledContext(
        schema_version=1,
        project="owned-ci",
        fault_id="owned-parser-fault",
        language="Python",
        fault_origin="synthetic",
        buggy_revision="buggy-v1",
        fixed_revision="fixed-v1",
        fixture_revision="fixed-v1",
        dependence_group="owned-ci-single-fixture",
        test_ids=("ci.parse",),
        inputs=tuple(inputs),
        runtime=ControlledRuntime(
            implementation="owned-runtime-description",
            version="1",
            platform="constructed",
            execution_model="constructed",
            image_digest=None,
            emulated=False,
            timezone="UTC",
            setup=("Constructed schema-only observations; no source execution claimed",),
            commands=("Descriptive owned fixture; no commands executed",),
            interventions=(),
        ),
        limitations=("Synthetic schema-only evidence; no historical CI or fault independence",),
    )


def observation(role="buggy", outcome="assertion_failed", trial="trial-1"):
    return ControlledObservation(
        trial_id=trial,
        role=role,
        test_id="ci.parse",
        outcome=outcome,
        elapsed=ControlledElapsed(value=0, status="observed"),
        capture_started_at=datetime(2026, 1, 1, tzinfo=UTC),
        capture_finished_at=datetime(2026, 1, 1, tzinfo=UTC),
        diagnostics_sha256=None,
    )


def source(cache, rows=None, ctx=None):
    ctx = context(cache) if ctx is None else ctx
    rows = (observation(), observation("fixed", "passed")) if rows is None else rows
    report = ControlledReport(schema_version=1, context_fingerprint=ctx.fingerprint, observations=rows)
    receipt = materialized(cache, "observations", report.to_json().encode())
    return ControlledSpec(
        schema_version=1, context=ctx, report_materialization=receipt, report_member="observations.json"
    )


def test_roundtrip_golden_cache_source_and_manifest(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    spec = source(cache)
    result = convert_controlled_source(spec, cache)
    for item in (spec, spec.context, result):
        assert type(item).from_json(item.to_json()) == item
    golden = Path(__file__).with_name("fixtures") / "controlled-owned-pair-v1.json"
    assert result.to_json() + "\n" == golden.read_text()
    validate_controlled_dataset(result, cache)
    assert [p.classification for p in result.pairs] == ["expected"]
    assert result.history == "constructed" and result.admission == "not_evaluated"
    with pytest.raises(ValueError, match="not historical CI"):
        result.require_temporal_evaluation()
    writer = ArtifactWriter(tmp_path / "out")
    path = publish_controlled_dataset(result, cache, writer)
    manifest = json.loads(generate_manifest(writer.run_root, run_id="owned-ci").read_bytes())
    assert any(a["type"] == "corpus_controlled_dataset" for a in manifest["artifacts"])
    assert ControlledDataset.from_json(path.read_bytes()) == result


@pytest.mark.parametrize("kind", [ControlledSpec, ControlledContext, ControlledDataset, ControlledReport])
@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_unknown_versions_reject_without_legacy_migration(tmp_path, kind, version):
    cache = SourceCache(tmp_path / "cache")
    spec = source(cache)
    report = ControlledReport(schema_version=1, context_fingerprint=spec.context.fingerprint, observations=())
    objects = {
        ControlledSpec: spec,
        ControlledContext: spec.context,
        ControlledDataset: convert_controlled_source(spec, cache),
        ControlledReport: report,
    }
    payload = objects[kind].model_dump(mode="json")
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    with pytest.raises(ValueError):
        kind.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (("assertion_failed", "passed"), "expected"),
        (("passed", "passed"), "both_pass"),
        (("assertion_failed", "assertion_failed"), "both_fail"),
        (("passed", "assertion_failed"), "reversed"),
        ((None, "passed"), "inconclusive"),
        (("timeout", "passed"), "inconclusive"),
        (("test_error", "passed"), "inconclusive"),
        (("infrastructure_error", "passed"), "inconclusive"),
        (("skipped", "passed"), "inconclusive"),
        (("cancelled", "passed"), "inconclusive"),
        (("unavailable", "passed"), "inconclusive"),
    ],
)
def test_classification_preserves_negative_and_inconclusive_results(outcomes, expected):
    assert ControlledPair(trial_id="t", test_id="case", buggy=outcomes[0], fixed=outcomes[1]).classification == expected


@pytest.mark.parametrize("value", [-1, 1.1, True, float("inf"), float("nan"), "1"])
def test_ticks_are_exact_nonnegative_integer_nanoseconds(value):
    with pytest.raises(ValueError):
        ControlledElapsed(value=value, status="observed")


def test_zero_null_unavailable_not_applicable_and_units():
    assert ControlledElapsed(value=0, status="observed").value == 0
    for status in ("unavailable", "not_applicable"):
        assert ControlledElapsed(value=None, status=status).value is None
        with pytest.raises(ValueError):
            ControlledElapsed(value=0, status=status)
    with pytest.raises(ValueError):
        ControlledElapsed(value=None, status="observed")
    for change in ({"unit": "s"}, {"meaning": "runner-work"}, {"status": "unknown"}):
        with pytest.raises(ValueError):
            ControlledElapsed.model_validate({"value": 0, "status": "observed", **change})


@pytest.mark.parametrize("value", [0, 1, 10**9, 2**53 + 1, 2**63 - 1])
def test_integer_elapsed_roundtrip_metamorphic(value):
    item = ControlledElapsed(value=value, status="observed")
    assert ControlledElapsed.model_validate_json(item.model_dump_json()) == item


def test_duplicate_accounting_conflicts_input_order_and_logical_identity(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    ctx = context(cache)
    a = observation()
    b = observation("fixed", "passed")
    first = convert_controlled_source(source(cache, (a, b, a), ctx), cache)
    second = convert_controlled_source(source(cache, (b, a, a), ctx), cache)
    assert first.observations == second.observations
    assert first.input_rows == 3 and first.duplicates_excluded == 1
    assert first.spec.context.fingerprint == second.spec.context.fingerprint
    assert changed(a, outcome="passed").execution_id(ctx) == a.execution_id(ctx)
    assert changed(a, trial_id="trial-2").execution_id(ctx) != a.execution_id(ctx)
    assert changed(ctx, inputs=tuple(reversed(ctx.inputs)), test_ids=tuple(reversed(ctx.test_ids))) == ctx
    with pytest.raises(ValueError, match="Conflicting"):
        convert_controlled_source(source(cache, (a, changed(a, outcome="passed")), ctx), cache)
    single = convert_controlled_source(source(cache, (a,), ctx), cache)
    assert single.pairs[0].fixed is None and single.pairs[0].classification == "inconclusive"
    assert convert_controlled_source(source(cache, (), ctx), cache).pairs == ()


def test_context_binding_missing_inputs_revisions_and_universe(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    ctx = context(cache)
    for updates in (
        {"buggy_revision": "fixed-v1"},
        {"fixture_revision": "other"},
        {"test_ids": ("same", "same")},
        {"inputs": ctx.inputs[:-1]},
        {"inputs": (*ctx.inputs, ctx.inputs[0])},
    ):
        with pytest.raises(ValueError):
            changed(ctx, **updates)
    spec = source(cache, ctx=ctx)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        convert_controlled_source(changed(spec, context=changed(ctx, project="other")), cache)
    with pytest.raises(ValueError, match="outside declared universe"):
        convert_controlled_source(source(cache, (changed(observation(), test_id="other"),), ctx), cache)
    with pytest.raises(ValueError, match="finishes before"):
        changed(observation(), capture_finished_at="2025-12-31T23:59:59Z")
    with pytest.raises(ValueError):
        changed(observation(), capture_started_at="2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="member missing"):
        changed(spec, report_member="missing")


def test_tampering_conservation_and_duplicate_dataset_reject(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    spec = source(cache)
    result = convert_controlled_source(spec, cache)
    for updates in ({"input_rows": 0}, {"duplicates_excluded": 1}, {"observations": (result.observations[0],) * 2}):
        with pytest.raises(ValueError):
            changed(result, **updates)
    with pytest.raises(ValueError, match="disagrees"):
        validate_controlled_dataset(
            changed(result, observations=(changed(result.observations[0], outcome="passed"), result.observations[1])),
            cache,
        )
    path = cache.member_path(spec.report_materialization, spec.report_member)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        convert_controlled_source(spec, cache)


def test_concurrent_publication_replay_and_interruption(tmp_path, monkeypatch):
    cache = SourceCache(tmp_path / "cache")
    result = convert_controlled_source(source(cache), cache)
    writer = ArtifactWriter(tmp_path / "out")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: publish_controlled_dataset(result, cache, writer), range(8)))
    assert len(set(paths)) == 1 and len(list(writer.run_root.rglob("*.json"))) == 1
    other = ArtifactWriter(tmp_path / "interrupted")
    import coleman.artifacts as artifacts

    original = artifacts.os.link

    def interrupted(*args, **kwargs):
        raise OSError("interrupted before publication")

    monkeypatch.setattr(artifacts.os, "link", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        publish_controlled_dataset(result, cache, other)
    assert not list(other.run_root.rglob("*.json")) and not list(other.run_root.rglob("*.tmp"))
    monkeypatch.setattr(artifacts.os, "link", original)
    assert publish_controlled_dataset(result, cache, other).exists()
    with pytest.raises(ValueError):
        publish_controlled_dataset(result, cache, ArtifactWriter(cache.root))


def test_runtime_container_identity_and_revision_kind_cardinality(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    ctx = context(cache)
    with pytest.raises(ValueError, match="image digest"):
        changed(ctx.runtime, execution_model="container")
    runtime = changed(ctx.runtime, execution_model="container", image_digest=digest(b"owned-container-manifest"))
    assert runtime.image_digest is not None
    with pytest.raises(ValueError, match="image digest"):
        changed(runtime, execution_model="host")
    renamed = changed(ctx.inputs[0], name="extra-unique-name")
    with pytest.raises(ValueError, match="exactly one input"):
        changed(ctx, inputs=(*ctx.inputs, renamed))


def test_report_member_limit_and_context_source_tampering(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    spec = source(cache)
    receipt = spec.report_materialization
    member = changed(receipt.spec.members[0], size_bytes=8 * 1024**2 + 1)
    raw = changed(receipt.spec, source=changed(receipt.spec.source, size_bytes=member.size_bytes), members=(member,))
    with pytest.raises(ValueError, match="exceeds 8 MiB"):
        changed(spec, report_materialization=changed(receipt, spec=raw))
    source_receipt = spec.context.inputs[0].materialization
    cache.member_path(source_receipt, source_receipt.spec.members[0].name).write_bytes(b"changed input")
    with pytest.raises(ValueError):
        convert_controlled_source(spec, cache)


def test_actual_parsed_bytes_are_rechecked_after_cache_validation(tmp_path, monkeypatch):
    cache = SourceCache(tmp_path / "cache")
    spec = source(cache)
    original = cache.member_path

    def mutated(receipt, name):
        path = original(receipt, name)
        if receipt == spec.report_materialization:
            path.write_bytes(b"{}")
        return path

    monkeypatch.setattr(cache, "member_path", mutated)
    with pytest.raises(ValueError, match="Parsed controlled report bytes"):
        convert_controlled_source(spec, cache)


def test_unchecked_dataset_and_unsafe_publication_cannot_bypass_validation(tmp_path):
    cache = SourceCache(tmp_path / "cache")
    result = convert_controlled_source(source(cache), cache)
    unchecked = result.model_copy(update={"input_rows": -1})
    with pytest.raises(ValueError):
        publish_controlled_dataset(unchecked, cache, ArtifactWriter(tmp_path / "out"))
    writer = ArtifactWriter(tmp_path / "unsafe")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    writer.run_root.mkdir()
    (writer.run_root / "corpus-controlled").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="within their output tree"):
        publish_controlled_dataset(result, cache, writer)
    assert not list(outside.iterdir())
