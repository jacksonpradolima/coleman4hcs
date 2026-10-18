"""Repeated corpus accounting, provenance, protected access and adversarial mutations."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.repeated import RepeatedCorpusReport, publish_repeated_corpus
from coleman.corpus.repeated_contracts import RepetitionDesign
from coleman.corpus.validation import CorpusFixtureReport
from coleman.evaluation.protected import EvaluationAudit, FamilyReport, FrozenCandidate, evaluate
from coleman.results.domain import fingerprint
from tests.corpus.repeated_support import ASSEMBLED, changed, repeated_report


def test_hand_means_equal_build_weights_and_seed_count_is_not_sample_size(tmp_path):
    report = repeated_report()
    observations = report.paired_observations()
    means = {(r.candidate_id, r.metadata.group_id): r.record.measurement("net_credit").value for r in observations}
    assert means == {
        ("baseline", "constructed-a"): -1,
        ("baseline", "constructed-b"): -1,
        ("candidate", "constructed-a"): 1.5,
        ("candidate", "constructed-b"): 4.5,
    }
    assert len(report.rows) == 12
    audit = EvaluationAudit(tmp_path)
    audit.save_freeze(report.frozen)
    result = evaluate(report.frozen, audit, report.paired_observations, actor="owned-fixture")
    assert result.results[0].observed_pairs == 2
    assert result.results[0].independent_units == ("constructed-a", "constructed-b")
    assert result.results[0].raw_mean_difference == 4
    assert result.results[0].conclusion == "insufficient_evidence"
    assert result.results[0].ci_lower is None
    assert all(row.metadata.available_at == ASSEMBLED for row in observations)
    assert all(row.record.provenance.source_revision == report.fingerprint for row in observations)


@pytest.mark.parametrize("state", ["failed", "aborted", "skipped", "completed"])
def test_missing_cell_metric_invalidates_whole_unit_without_complete_case_selection(state, tmp_path):
    report = repeated_report()
    row = next(row for row in report.rows if row.candidate_id == "candidate" and row.member_id == "build-0")
    metric = changed(row.record.measurement("net_credit"), value=None, status="unavailable")
    missing = changed(row, state=state, record=changed(row.record, measurements=(metric,)))
    rows = tuple(missing if item == row else item for item in report.rows)
    report = changed(report, rows=rows)
    aggregate = report.paired_observations()
    unavailable = next(r for r in aggregate if r.candidate_id == "candidate" and r.metadata.group_id == "constructed-a")
    assert unavailable.record.measurement("net_credit").value is None
    assert unavailable.record.measurement("net_credit").status == "unavailable"
    result = evaluate(report.frozen, EvaluationAudit(tmp_path), report.paired_observations, actor="owned-fixture")
    assert result.results[0].reason == "incomplete_outcomes"
    assert result.results[0].observed_pairs == 1
    assert result.results[0].raw_mean_difference is None


@pytest.mark.parametrize("state", ["failed", "aborted", "skipped"])
def test_noncompleted_execution_cannot_claim_observed_zero(state):
    report = repeated_report()
    row = next(row for row in report.rows if row.record.measurement("net_credit").value == 0)
    invalid = changed(row, state=state)
    rows = tuple(invalid if item == row else item for item in report.rows)
    with pytest.raises(ValueError, match="unavailable"):
        changed(report, rows=rows)


def test_missing_extra_duplicate_and_relabelled_raw_cells_fail_closed():
    report = repeated_report()
    missing = report.rows[:-1]
    duplicate = (*report.rows, report.rows[0])
    original, second = report.rows[:2]
    reused = changed(second, record=original.record)
    reused_rows = (original, reused, *report.rows[2:])
    foreign = changed(original, seed=99)
    foreign_rows = (foreign, *report.rows[1:])
    for rows, message in (
        (missing, "Every expected"),
        (duplicate, "duplicate"),
        (reused_rows, "identity"),
        (foreign_rows, "Unexpected"),
    ):
        with pytest.raises(ValueError, match=message):
            changed(report, rows=rows)


@pytest.mark.parametrize("field", ["data_fingerprint", "spec_fingerprint", "evaluation_fingerprint"])
def test_frozen_raw_provenance_cannot_be_rebound(field):
    report = repeated_report()
    row = report.rows[0]
    if field == "data_fingerprint":
        record = changed(row.record, data_fingerprint=fingerprint(["different"]))
    else:
        provenance = changed(row.record.provenance, **{field: fingerprint(["different"])})
        record = changed(row.record, provenance=provenance)
    invalid = changed(row, record=record)
    rows = (invalid, *report.rows[1:])
    with pytest.raises(ValueError, match="frozen"):
        changed(report, rows=rows)


@pytest.mark.parametrize("change", [{"unit": "ms"}, {"meaning": "other-v1"}, {"direction": "minimize"}])
def test_unit_semantics_and_objective_cannot_silently_change(change):
    report = repeated_report()
    row = report.rows[0]
    metric = changed(row.record.measurement("net_credit"), **change)
    invalid = changed(row, record=changed(row.record, measurements=(metric,)))
    rows = (invalid, *report.rows[1:])
    with pytest.raises(ValueError, match="unit/meaning/direction"):
        changed(report, rows=rows)


@pytest.mark.parametrize("timestamp", ["2026-01-05T23:00:00Z", "2026-01-08T00:00:00Z"])
def test_mean_assembly_never_backdates_labels_or_crosses_protected_cutoff(timestamp):
    report = repeated_report()
    with pytest.raises(ValueError, match="availability|assembly"):
        changed(report, assembled_at=timestamp)


def test_design_rejects_sibling_and_variant_pseudoreplication_and_revisions():
    report = repeated_report()
    first, second, third = report.design.builds
    sibling = changed(second, metadata=changed(second.metadata, group_id=first.metadata.group_id))
    revision = changed(second, metadata=changed(first.metadata, revision_id="corrected"))
    false_unit = changed(second, unit_id="seed-pretending-to-be-independent")
    for build, message in ((sibling, "Sibling"), (revision, "revision"), (false_unit, "project")):
        with pytest.raises(ValueError, match=message):
            changed(report.design, builds=(first, build, third))


def test_new_design_cannot_reuse_legacy_freeze_or_tune_budget_after_outcomes():
    report = repeated_report()
    new_budget = changed(report.design.budget, value="0.75")
    design = changed(report.design, budget=new_budget)
    with pytest.raises(ValueError, match="fingerprint"):
        changed(report, design=design)


def test_serial_clustered_and_normality_guards_precede_raw_loader(tmp_path):
    report = repeated_report()
    for dependence, normal in (("serial", True), ("clustered", True), ("independent", False)):
        protocol = changed(report.frozen.protocol, dependence=dependence, normal_differences_assumed=normal)
        frozen = changed(report.frozen, protocol=protocol)

        def forbidden():
            raise AssertionError("Unsupported protocol loaded protected outcomes")

        with pytest.raises(NotImplementedError):
            evaluate(frozen, EvaluationAudit(tmp_path), forbidden, actor="owned-fixture")


def test_deterministic_raw_order_roundtrip_golden_and_strict_versions():
    report = repeated_report()
    reordered = changed(report, rows=tuple(reversed(report.rows)))
    assert reordered.fingerprint == report.fingerprint
    assert reordered.paired_observations() == report.paired_observations()
    assert RepeatedCorpusReport.from_json(report.to_json()) == report
    assert RepetitionDesign.from_json(report.design.to_json()) == report.design
    golden = Path(__file__).parent / "fixtures/corpus-repetitions-v1.json"
    assert report.to_json() + "\n" == golden.read_text()
    for version in (0, 2, 99, True, "1", None):
        payload = {**json.loads(report.to_json()), "schema_version": version}
        with pytest.raises(ValueError):
            RepeatedCorpusReport.model_validate(payload)


def test_publication_concurrent_retry_and_interruption_never_exposes_partial_json(tmp_path, monkeypatch):
    report = repeated_report()
    writer = ArtifactWriter(tmp_path)
    original = writer.write_immutable

    def interrupted(path, content):
        def chunks():
            yield content[:40].encode()
            raise OSError("interrupted input before immutable publication")

        writer.write_immutable_stream(path, chunks(), maximum_bytes=len(content.encode()))

    monkeypatch.setattr(writer, "write_immutable", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        publish_repeated_corpus(report, writer)
    assert not tuple(tmp_path.rglob("*.json"))
    monkeypatch.setattr(writer, "write_immutable", original)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = tuple(pool.map(lambda _: publish_repeated_corpus(report, writer), range(8)))
    assert len(set(paths)) == 1
    assert RepeatedCorpusReport.from_json(paths[0].read_bytes()) == report


def test_raw_embedded_evidence_and_access_survive_resume_and_retry(tmp_path):
    report = repeated_report()
    writer = ArtifactWriter(tmp_path)
    path = publish_repeated_corpus(report, writer)
    audit = EvaluationAudit(tmp_path)
    audit.save_freeze(report.frozen)

    def interrupted():
        RepeatedCorpusReport.from_json(path.read_bytes())
        raise RuntimeError("interrupted after protected evidence read")

    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate(report.frozen, audit, interrupted, actor="owned-fixture")
    recovered = RepeatedCorpusReport.from_json(path.read_bytes())
    result = evaluate(report.frozen, EvaluationAudit(tmp_path), recovered.paired_observations, actor="owned-fixture")
    assert result.access.prior_test_accesses == 1
    assert result.confirmatory_eligible is False
    assert publish_repeated_corpus(recovered, writer).read_bytes() == path.read_bytes()


def test_normal_unit_null_and_known_effect_survive_dependent_build_seed_replication(tmp_path):
    # These are independent simulated normal unit differences in signed credits,
    # not a normality claim about bounded NAPFD or actual CI repositories.
    rng = np.random.default_rng(310)
    rejected = covered = detected = 0
    repetitions = 40
    for index in range(repetitions):
        null = tuple(float(value) for value in rng.normal(0, 1, 24))
        for kind, differences in (("null", null), ("known", tuple(value + 1 for value in null))):
            report = repeated_report(differences)
            result = evaluate(
                report.frozen,
                EvaluationAudit(tmp_path / kind / str(index)),
                report.paired_observations,
                actor="constructed-unit-simulation",
            ).results[0]
            assert result.observed_pairs == 24
            assert len(report.rows) == 192
            if kind == "null":
                rejected += result.conclusion in {"evidence_of_benefit", "evidence_of_harm"}
                assert result.ci_lower is not None
                assert result.ci_upper is not None
                covered += result.ci_lower <= 0 <= result.ci_upper
            else:
                detected += result.conclusion == "evidence_of_benefit"
    # Broad deterministic smoke bounds complement #252's 160-experiment tests;
    # they catch pseudoreplication/incorrect means, not certify nominal coverage.
    assert rejected <= 8
    assert covered >= 32
    assert detected >= 34


def test_signed_scale_translation_and_overflow_safe_mean_metamorphisms():
    report = repeated_report()
    original = report.paired_observations()
    for scale, offset in ((-3.0, 7.0), (1.0, 0.0)):
        rows = []
        for row in report.rows:
            metric = row.record.measurement("net_credit")
            value = metric.value * scale + offset
            rows.append(changed(row, record=changed(row.record, measurements=(changed(metric, value=value),))))
        transformed = changed(report, rows=tuple(reversed(rows)))
        for before, after in zip(original, transformed.paired_observations(), strict=True):
            assert (
                after.record.measurement("net_credit").value
                == before.record.measurement("net_credit").value * scale + offset
            )
            assert after.metadata == before.metadata
    rows = tuple(
        changed(
            row, record=changed(row.record, measurements=(changed(row.record.measurement("net_credit"), value=1e308),))
        )
        for row in report.rows
    )
    large = changed(report, rows=rows)
    assert all(row.record.measurement("net_credit").value == 1e308 for row in large.paired_observations())


def test_invalid_warmup_budget_seeds_members_and_source_clocks_fail_before_loading():
    design = repeated_report().design
    first = design.builds[0]
    late = changed(first, metadata=changed(first.metadata, available_at="2026-01-08T00:00:00Z"))
    changes = (
        ({"warmup_end": "2026-01-03T00:00:00Z"}, "Warmup"),
        ({"budget": changed(design.budget, value="0")}, "budget ratio"),
        ({"budget": changed(design.budget, value="1.01")}, "budget ratio"),
        ({"seeds": (11, 11)}, "Duplicate repetition seed"),
        ({"builds": (first, first)}, "Duplicate logical"),
        ({"builds": (late, *design.builds[1:])}, "eligible"),
        ({"evidence_basis": "empirical_protected", "source_admission_fingerprint": None}, "admission"),
        ({"builds": (first,) * 257, "seeds": tuple(range(64))}, "16,384"),
    )
    for fields, message in changes:
        with pytest.raises(ValueError, match=message):
            changed(design, **fields)


def test_aggregate_requires_actual_frozen_data_mean_semantics_and_bounded_grid():
    report = repeated_report()
    foreign = changed(report.frozen, data_fingerprint=fingerprint(["foreign-source"]))
    wrong_mean = changed(report.frozen, protocol=changed(report.frozen.protocol, meaning="raw-value-v1"))
    for frozen, message in ((foreign, "data or temporal"), (wrong_mean, "mean metric")):
        with pytest.raises(ValueError, match=message):
            report.design.require_frozen(frozen)
    many = repeated_report(tuple(0.0 for _ in range(64)))
    design = changed(many.design, seeds=tuple(range(64)))
    candidates = tuple(FrozenCandidate.from_configuration(f"candidate-{index}", {}) for index in range(7))
    frozen = changed(many.frozen, candidates=(*many.frozen.candidates, *candidates))
    with pytest.raises(ValueError, match="65,536"):
        design.require_frozen(frozen)


def test_mean_cannot_be_assembled_before_source_target_interval_finishes():
    end = datetime(2026, 1, 7, tzinfo=UTC)
    with pytest.raises(ValueError, match="target interval"):
        repeated_report(target_end=end)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_observations_are_rejected_at_domain_boundary(value):
    metric = repeated_report().rows[0].record.measurement("net_credit")
    with pytest.raises(ValueError):
        changed(metric, value=value)


def test_old_artifacts_keep_their_readers_and_are_not_implicitly_migrated():
    root = Path(__file__).parents[2]
    old_corpus = Path(__file__).parent / "fixtures/corpus-serial-report-v1.json"
    old_protected = root / "tests/fixtures/protected-evaluation/report-v1.json"
    assert CorpusFixtureReport.from_json(old_corpus.read_bytes()).schema_version == 1
    assert FamilyReport.from_json(old_protected.read_bytes()).schema_version == 1
    for path in (old_corpus, old_protected):
        payload = path.read_bytes()
        with pytest.raises(ValueError):
            RepeatedCorpusReport.from_json(payload)
    design = repeated_report().design
    future = {**json.loads(design.to_json()), "schema_version": 2}
    with pytest.raises(ValueError):
        RepetitionDesign.model_validate(future)
