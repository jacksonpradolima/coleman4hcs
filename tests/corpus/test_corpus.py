"""Known answers, metadata admission, deterministic artifacts and failure boundaries."""

import ast
import hashlib
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    REQUIRED_REGIMES,
    CIReferenceFixture,
    ConversionSummary,
    CorpusFixtureReport,
    CorpusManifest,
    CorpusRegistry,
    ExclusionCount,
    ObservationIdentity,
    RatioReference,
    SerialFixtureCase,
    bundled_fixture_bytes,
    fixture_bytes,
    fixture_readiness,
    load_registry,
    parallel_reference,
    publish_fixture_report,
    tiny_ci_reference,
    validate_bundled_source,
    validate_fixture_source,
)
from coleman.pilot import SECONDS, PilotTest
from coleman.selection import Quantity, UnitSemantics

ROOT = Path(__file__).parents[2]
SERIAL = "ci-serial-boundary-v1"
PARALLEL = "ci-parallel-occupancy-v1"


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def q(value, semantics=SECONDS):
    return Quantity(value=Decimal(str(value)), semantics=semantics)


@pytest.fixture
def registry():
    return load_registry()


@pytest.fixture
def fixture():
    return CIReferenceFixture.from_json(bundled_fixture_bytes(SERIAL))


def test_hand_calculated_serial_and_parallel_answers(registry):
    report = validate_bundled_source(SERIAL, registry)
    by_id = {row.case_id: row.record for row in report.observations}
    partial = by_id["serial-budget-3"]
    full = by_id["serial-full-7"]
    assert partial.measurement("corpus_failing_test_recall").value == 0.5
    assert partial.measurement("corpus_runner_work").value == 3
    assert partial.measurement("corpus_first_failure_elapsed").value == 3
    assert partial.measurement("corpus_napfd").value == 0.25
    assert partial.measurement("corpus_apfdc").value == pytest.approx(5 / 14, rel=0, abs=1e-12)
    assert full.measurement("corpus_napfd").value == pytest.approx(1 / 3, rel=0, abs=1e-12)
    assert full.measurement("corpus_apfdc").value == 0.5
    partial.require_comparable(full, "corpus_napfd")
    assert partial.available_at == report.fixture.build.outcomes_available_at
    assert partial.provenance.evaluation_fingerprint == report.manifest.fingerprint
    parallel = validate_bundled_source(PARALLEL, registry).observations[0].record
    assert parallel.measurement("corpus_runner_work").value == 7
    assert parallel.measurement("corpus_execution_elapsed").value == 4
    assert parallel.measurement("corpus_first_failure_elapsed").value == 2
    assert parallel.measurement("corpus_failing_test_recall").value == 1


def test_registration_is_not_source_admission_or_independent_evidence(registry):
    assert len(registry.cards) == 12
    readiness = fixture_readiness(registry)
    assert len(readiness.blocked_sources) == 10
    assert readiness.validated_fixture_sources == ()
    assert set(readiness.uncovered_required_regimes) == set(REQUIRED_REGIMES)
    reports = tuple(validate_bundled_source(name, registry) for name in (SERIAL, PARALLEL))
    readiness = fixture_readiness(registry, (*reports, reports[0]))
    assert len(readiness.validated_fixture_sources) == 2
    assert len(readiness.fixture_dependence_groups) == 1
    assert readiness.evidence_status == "insufficient_diversity"
    assert len(readiness.uncovered_required_regimes) == 7
    blocked = next(card for card in registry.cards if card.manifest.blocking_reasons)
    assert blocked.languages is None
    assert blocked.manifest.source.sha256 is not None
    assert "duration_units_unverified" in blocked.manifest.blocking_reasons
    assert "data_license_unverified_or_prohibited" in blocked.manifest.blocking_reasons
    with pytest.raises(ValueError, match="blocked"):
        validate_bundled_source(blocked.manifest.benchmark_id, registry)


def test_roundtrip_golden_and_fail_closed_versions(registry):
    report = validate_bundled_source(SERIAL, registry)
    path = Path(__file__).parent / "fixtures/corpus-serial-report-v1.json"
    assert report.to_json() + "\n" == path.read_text()
    assert CorpusFixtureReport.from_json(path.read_bytes()) == report
    assert CorpusRegistry.from_json(registry.to_json()) == registry
    assert CorpusManifest.from_json(report.manifest.to_json()) == report.manifest
    for version in (0, 2, 99, True, "1", None):
        data = json.loads(report.to_json())
        data["schema_version"] = version
        with pytest.raises(ValueError):
            CorpusFixtureReport.model_validate(data)
    with pytest.raises(ValueError):
        CorpusRegistry.from_json('{"schema_version":1,"schema_version":1}')
    # Old native records remain their own contract, never an implicit corpus migration.
    with pytest.raises(ValueError):
        CorpusFixtureReport.from_json('{"scenario":"ci","ttf":0,"detected":1}')


def test_digest_canonicalization_and_mutated_source_rejection(registry):
    manifest = registry.card(SERIAL).manifest
    payload = bundled_fixture_bytes(SERIAL)
    assert manifest.source.sha256 == "sha256:" + hashlib.sha256(payload).hexdigest()
    report = validate_fixture_source(manifest, payload)
    assert fixture_bytes(report.fixture) == payload
    assert changed(registry, cards=tuple(reversed(registry.cards))).fingerprint == registry.fingerprint
    assert changed(report, observations=tuple(reversed(report.observations))).fingerprint == report.fingerprint
    with localcontext() as context:
        context.prec = 3
        assert validate_fixture_source(manifest, payload).fingerprint == report.fingerprint
    mutated = payload.replace(b'"value":"4"', b'"value":"5"', 1)
    assert mutated != payload
    with pytest.raises(ValueError, match="SHA-256"):
        validate_fixture_source(manifest, mutated)
    with pytest.raises(ValueError, match="size"):
        validate_fixture_source(manifest, payload + b" ")
    future = json.loads(payload)
    future["schema_version"] = 2
    future_bytes = json.dumps(future).encode()
    source = changed(
        manifest.source, sha256="sha256:" + hashlib.sha256(future_bytes).hexdigest(), size_bytes=len(future_bytes)
    )
    manifest = changed(manifest, source=source)
    with pytest.raises(ValueError):
        validate_fixture_source(manifest, future_bytes)


def test_all_permutations_and_budgets_keep_excluded_failures(fixture):
    durations = {"t1": 1, "t2": 2, "t3": 4}
    for ranking in itertools.permutations(durations):
        for budget in range(8):
            elapsed = 0
            selected = []
            failure_times = []
            for name in ranking:
                if elapsed + durations[name] > budget:
                    break
                selected.append(name)
                elapsed += durations[name]
                if name != "t1":
                    failure_times.append(elapsed)
            answer = tiny_ci_reference(fixture.build, ranking, q(budget))
            assert answer.selected_ids == tuple(selected)
            assert answer.recall.value == Fraction(len(failure_times), 2)
            assert answer.total_failing_tests == 2
            assert answer.runner_work.value == elapsed
            if failure_times:
                assert answer.first_failure_elapsed is not None
                assert answer.first_failure_elapsed.value == failure_times[0]
            else:
                assert answer.first_failure_elapsed is None
                assert answer.first_failure_status == "right_censored"


def test_zero_null_unavailable_and_zero_failure_are_distinct(fixture):
    passed = PilotTest(test_id="zero", duration=q(0), outcome="passed")
    build = changed(fixture.build, tests=(passed,), baseline_order=("zero",))
    answer = tiny_ci_reference(build, ("zero",), q(0))
    assert answer.runner_work.value == 0
    assert answer.recall.value is None
    assert answer.recall.status == "not_applicable"
    assert answer.apfdc.value is None
    assert answer.first_failure_elapsed is None
    failed = changed(build, tests=(changed(passed, outcome="failed"),))
    answer = tiny_ci_reference(failed, ("zero",), q(0))
    assert answer.recall.value == 1
    assert answer.first_failure_elapsed is not None
    assert answer.first_failure_elapsed.value == 0
    assert answer.first_failure_status == "detected"
    assert answer.apfdc.value is None
    assert parallel_reference(failed).first_failure_elapsed == answer.first_failure_elapsed
    unresolved = changed(build, tests=(changed(passed, outcome="unknown"),))
    zero_budget = q(0)
    with pytest.raises(ValueError, match="complete binary"):
        tiny_ci_reference(unresolved, ("zero",), zero_budget)
    with pytest.raises(ValueError, match="complete binary"):
        parallel_reference(unresolved)


@pytest.mark.parametrize(
    "value",
    [
        "NaN",
        "Infinity",
        "-Infinity",
        "1/0",
        "1e-100000000",
        Decimal("Infinity"),
        Decimal("1e-100000000"),
        -1,
        2,
        True,
        0.5,
    ],
)
def test_invalid_reference_values_fail_closed(value):
    with pytest.raises(ValueError):
        RatioReference(value=value, meaning="failing-test-recall-v1", status="defined")


def test_units_and_duplicate_declarations(registry, fixture):
    incompatible = q(0, UnitSemantics(unit="ms", meaning="test-runner-occupancy-v1"))
    with pytest.raises(ValueError, match="compatible"):
        tiny_ci_reference(fixture.build, fixture.build.baseline_order, incompatible)
    with pytest.raises(ValueError, match="Duplicate registered"):
        changed(registry, cards=(registry.cards[0], registry.cards[0]))
    with pytest.raises(ValueError, match="minimum diversity"):
        changed(registry, required_regimes=("invented",) * 7)
    with pytest.raises(ValueError, match="Duplicate observation"):
        ObservationIdentity(observation_fields=("build", "build"), group_fields=("build",))
    with pytest.raises(ValueError, match="known row counts"):
        ConversionSummary(status="complete", input_rows=3, output_rows=None, exclusions=())
    with pytest.raises(ValueError, match="conservation"):
        ConversionSummary(status="complete", input_rows=3, output_rows=2, exclusions=())
    duplicate = (ExclusionCount(reason="missing", rows=1),) * 2
    with pytest.raises(ValueError, match="Duplicate primary"):
        ConversionSummary(status="complete", input_rows=3, output_rows=1, exclusions=duplicate)


def test_license_temporal_and_conversion_gates_run_before_native(registry, monkeypatch):
    from coleman.evaluation import NAPFDVerdictMetric

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Native metric cannot run before source admission checks")

    monkeypatch.setattr(NAPFDVerdictMetric, "evaluate", forbidden)
    base = registry.card(SERIAL).manifest
    payload = bundled_fixture_bytes(SERIAL)
    modifications = [
        {"license": changed(base.license, status="pending")},
        {"time": changed(base.time, outcome_available_field=None)},
        {"outcomes": changed(base.outcomes, duration=None)},
        {"conversion": changed(base.conversion, status="not_validated")},
        {"adapter": changed(base.adapter, implementation="planned")},
    ]
    for change in modifications:
        manifest = changed(base, **change)
        with pytest.raises(ValueError, match="blocked"):
            validate_fixture_source(manifest, payload)
    with pytest.raises(ValueError, match="terms URI/digest"):
        changed(base.license, terms_sha256=None)


def test_mutant_native_denominator_and_tampered_report_are_detected(registry, monkeypatch):
    from coleman.evaluation import NAPFDVerdictMetric

    original = NAPFDVerdictMetric.evaluate

    def dropped_missed_failures(self, tests):
        original(self, tests)
        self.recall = 1.0 if self.detected_failures else 0.0

    with monkeypatch.context() as mutation:
        mutation.setattr(NAPFDVerdictMetric, "evaluate", dropped_missed_failures)
        with pytest.raises(ValueError, match="independent reference"):
            validate_bundled_source(SERIAL, registry)
    report = validate_bundled_source(SERIAL, registry)
    first = report.observations[0]
    bad_record = changed(
        first.record, provenance=changed(first.record.provenance, evaluation_fingerprint="sha256:" + "0" * 64)
    )
    bad_observation = changed(first, record=bad_record)
    with pytest.raises(ValueError, match="Incompatible"):
        changed(report, observations=(bad_observation, *report.observations[1:]))
    with pytest.raises(ValueError, match="exactly one"):
        changed(report, observations=(first, first))


def test_idempotent_concurrent_publication_and_interrupted_write(registry, tmp_path, monkeypatch):
    report = validate_bundled_source(SERIAL, registry)
    writer = ArtifactWriter(tmp_path)
    path = publish_fixture_report(report, writer)
    assert CorpusFixtureReport.from_json(path.read_bytes()) == report
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: publish_fixture_report(report, writer), range(8)))
    assert paths == [path] * 8
    other = ArtifactWriter(tmp_path / "interrupted")

    def interrupted(*_):
        raise OSError("injected before atomic publication")

    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(OSError, match="injected"):
        publish_fixture_report(report, other)
    assert list((tmp_path / "interrupted").rglob("*.json")) == []
    assert list((tmp_path / "interrupted").rglob("*.tmp")) == []


def test_metadata_cannot_promote_synthetic_or_weaken_protocol(registry):
    card = registry.card(SERIAL)
    with pytest.raises(ValueError, match="observed CI"):
        changed(card, history="observed")
    with pytest.raises(ValueError, match="real faults"):
        changed(card, fault_origin="real")
    with pytest.raises(ValueError, match="contradictory"):
        changed(card, missing_regimes=card.covered_regimes)
    report = validate_bundled_source(SERIAL, registry)
    different = changed(card, manifest=changed(card.manifest, dependence_group="changed"))
    changed_registry = changed(
        registry, cards=tuple(different if c.manifest.benchmark_id == SERIAL else c for c in registry.cards)
    )
    with pytest.raises(ValueError, match="registered frozen"):
        fixture_readiness(changed_registry, (report,))
    with pytest.raises(ValueError):
        changed(report, generalization="state_of_the_art")


def test_import_boundaries_and_no_arbitrary_resource_path():
    for path in (ROOT / "coleman/corpus").glob("*.py"):
        if path.name == "__main__.py":
            continue
        modules = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        assert not any(
            name.startswith(("coleman.cli", "coleman.transports", "coleman.corpus.__main__", "mcp")) for name in modules
        )
    with pytest.raises(KeyError):
        bundled_fixture_bytes("../../outside")
    code = "import sys,coleman.corpus; assert 'coleman.cli' not in sys.modules; assert 'mcp' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, timeout=30)


def source_with_fixture(manifest, fixture):
    payload = fixture_bytes(fixture)
    source = changed(manifest.source, sha256="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    budgets = tuple(case.budget for case in fixture.cases if isinstance(case, SerialFixtureCase))
    evaluation = changed(manifest.evaluation, budgets=budgets)
    return changed(manifest, source=source, evaluation=evaluation), payload


def test_native_censored_detection_cannot_be_filled_from_reference(registry, fixture, monkeypatch):
    from coleman.evaluation import NAPFDVerdictMetric

    case = fixture.cases[0]
    assert isinstance(case, SerialFixtureCase)
    empty = changed(case, budget=q(0), expected=tiny_ci_reference(fixture.build, case.ranking, q(0)))
    source = changed(fixture, cases=(empty,))
    manifest, payload = source_with_fixture(registry.card(SERIAL).manifest, source)
    original = NAPFDVerdictMetric.evaluate

    def fake_detection(self, rows):
        original(self, rows)
        self.detection_ranks = [1]
        self.ttf_duration = 0

    monkeypatch.setattr(NAPFDVerdictMetric, "evaluate", fake_detection)
    with pytest.raises(ValueError, match="availability state"):
        validate_fixture_source(manifest, payload)


def test_new_artifact_families_remain_discoverable(registry, tmp_path):
    from coleman.manifest import generate_manifest

    report = validate_bundled_source(SERIAL, registry)
    artifact = publish_fixture_report(report, ArtifactWriter(tmp_path))
    # The pilot publisher already writes this existing artifact family; discovery
    # must not omit it when newer versioned report types are added.
    pilot = tmp_path / "pilot-reports" / "existing.json"
    pilot.parent.mkdir()
    pilot.write_text('{"schema_version":1}\n')
    manifest_path = generate_manifest(tmp_path, run_id="fixture-integration")
    manifest = json.loads(manifest_path.read_bytes())
    discovered = {(entry["type"], entry["relative_path"]) for entry in manifest["artifacts"]}
    assert ("corpus_fixture_report", artifact.relative_to(tmp_path).as_posix()) in discovered
    assert ("pilot_report", pilot.relative_to(tmp_path).as_posix()) in discovered


def test_readiness_cannot_forge_membership_diversity_or_independence(registry):
    readiness = fixture_readiness(registry, (validate_bundled_source(SERIAL, registry),))
    invalid = (
        {"registered_sources": (*readiness.registered_sources, SERIAL)},
        {"validated_fixture_sources": ("unregistered",)},
        {"validated_fixture_sources": (readiness.blocked_sources[0].benchmark_id,)},
        {"blocked_sources": (*readiness.blocked_sources, readiness.blocked_sources[0])},
        {"fixture_dependence_groups": ("independent-1", "independent-2")},
        {"fixture_dependence_groups": ()},
        {"uncovered_required_regimes": (*REQUIRED_REGIMES[1:], "invented")},
    )
    for mutation in invalid:
        with pytest.raises(ValueError):
            changed(readiness, **mutation)
    reordered = changed(readiness, registered_sources=tuple(reversed(readiness.registered_sources)))
    assert reordered.fingerprint == readiness.fingerprint


def test_readiness_rejects_fabricated_independent_groups(registry, tmp_path):
    from coleman.corpus import publish_fixture_readiness

    reports = tuple(validate_bundled_source(name, registry) for name in (SERIAL, PARALLEL))
    readiness = fixture_readiness(registry, reports)
    forged = readiness.model_copy(update={"fixture_dependence_groups": ("independent-1", "independent-2")})
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(ValueError):
        publish_fixture_readiness(forged, writer)
    assert not list(tmp_path.rglob("*.json"))


def test_cli_rejects_symlinked_publication_parent(tmp_path, monkeypatch):
    from coleman.corpus.__main__ import main

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    output = workspace / "output"
    output.mkdir()
    (output / "corpus-validations").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(workspace)
    assert main(["validate", SERIAL, "--output", "output"]) == 2
    assert not list(outside.iterdir())


def test_readiness_embedded_evidence_roundtrip_and_conflicts(registry):
    from coleman.corpus import FixtureReadiness

    reports = tuple(validate_bundled_source(name, registry) for name in (SERIAL, PARALLEL))
    readiness = fixture_readiness(registry, reports)
    assert FixtureReadiness.from_json(readiness.to_json()) == readiness
    assert fixture_readiness(registry, tuple(reversed(reports))).fingerprint == readiness.fingerprint
    forged = json.loads(readiness.to_json())
    forged["fixture_dependence_groups"] = ["independent-1", "independent-2"]
    payload = json.dumps(forged)
    with pytest.raises(ValueError, match="evidence"):
        FixtureReadiness.from_json(payload)
    with pytest.raises(ValueError, match="evidence"):
        changed(readiness, validated_reports=())
    report = reports[0]
    observation = report.observations[0]
    original = observation.record.measurement("corpus_failing_test_recall")
    assert original.value is not None
    perturbed = changed(original, value=original.value - 1e-13)
    record = changed(
        observation.record,
        measurements=tuple(perturbed if m.name == original.name else m for m in observation.record.measurements),
    )
    alternative = changed(report, observations=(changed(observation, record=record), *report.observations[1:]))
    with pytest.raises(ValueError, match="Conflicting"):
        fixture_readiness(registry, (report, alternative))
