"""Actual owned source verification, conservative gates, immutable publication and native continuation."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from html import unescape
from pathlib import Path

import polars as pl
import pytest

from coleman.application.corpus import run_lrts_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission import (
    SourceEvidence,
    assess_source,
    audit_corpus,
    load_source_evidence,
    publish_corpus_admission,
    validate_admission_completion,
    validate_corpus_sources,
)
from coleman.corpus.admission_contracts import CorpusAdmission
from coleman.corpus.admission_report import render_corpus_admission
from coleman.corpus.controlled import convert_controlled_source
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.materialization import SourceCache
from coleman.corpus.registry import load_registry
from coleman.corpus.stress_artifacts import export_stress
from coleman.corpus.stress_contracts import StressReceipt
from coleman.results.domain import fingerprint
from tests.corpus.admission_support import changed, controlled_source, lrts_source, review
from tests.corpus.stress_support import case
from tests.corpus.test_controlled import context, observation
from tests.corpus.test_controlled import source as controlled_spec
from tests.corpus.test_hcs_etl import export as hcs_export


def test_real_owned_etl_audit_publication_and_native_checkpoint_continuation(tmp_path):
    evidence, completion = lrts_source(tmp_path)
    registry = load_registry()
    legacy_before = registry.to_json()
    report = audit_corpus(registry, (evidence,))
    assessment = report.assessments[0]
    assert assessment.profile.rows == 10 and assessment.profile.unique_observations == 10
    assert assessment.profile.source_groups == 2
    assert assessment.profile.duration.null_count == 6 and assessment.profile.duration.zero_count == 2
    assert assessment.card.history == "constructed" and assessment.scope == "blocked"
    assert report.covered_regimes == report.observed_source_models == ()
    assert len(report.uncovered_regimes) == 7 and len(report.unassessed_sources) == 12
    assert load_registry().to_json() == legacy_before
    assert str(tmp_path) not in report.to_json()
    assert CorpusAdmission.from_json(report.to_json()) == report
    marker = publish_corpus_admission(report, (evidence,), ArtifactWriter(tmp_path / "audit"))
    assert validate_admission_completion(marker, (evidence,)) == report
    human = (marker.parent / "report.md").read_text()
    assert human == render_corpus_admission(report)
    assert "Observed source-model count: 0" in human and "not_applicable" in unescape(human)
    assert "not independent sampling units" in human

    # Existing native API remains a declared post-hoc simulation, independent
    # of the new audit's explicit refusal of protected historical admission.
    assert evidence.cache is not None
    result = run_lrts_replay(completion, evidence.cache, tmp_path / "native")
    assert result.artifacts_dir is not None
    root = Path(result.artifacts_dir)
    rows = pl.read_parquet(str(root / "results/**/*.parquet"), hive_partitioning=True)
    assert rows.height == 4 and rows.select(pl.struct(["step", "policy"]).n_unique()).item() == 4
    protected = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.suffix in {".pkl", ".parquet"}}
    assert protected and any(p.suffix == ".pkl" for p in protected)
    again = run_lrts_replay(completion, evidence.cache, tmp_path / "native")
    assert again.run_id == result.run_id
    assert protected == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.suffix in {".pkl", ".parquet"}
    }
    assert all(json.loads(p.read_bytes())["step_committed"] == 2 for p in root.rglob("progress_ex1.json"))


def test_identical_bindings_and_concurrent_publication_are_idempotent(tmp_path):
    evidence, _ = lrts_source(tmp_path)
    report = audit_corpus(load_registry(), (evidence,))
    duplicate = audit_corpus(load_registry(), (evidence, evidence))
    assert duplicate == report and duplicate.fingerprint == report.fingerprint
    writer = ArtifactWriter(tmp_path / "audit")
    with ThreadPoolExecutor(max_workers=3) as pool:
        markers = list(pool.map(lambda _: publish_corpus_admission(report, (evidence,), writer), range(3)))
    assert len(set(markers)) == 1
    before = {p.relative_to(writer.run_root): p.read_bytes() for p in writer.run_root.rglob("*") if p.is_file()}
    assert publish_corpus_admission(report, (evidence,), writer) == markers[0]
    assert before == {p.relative_to(writer.run_root): p.read_bytes() for p in writer.run_root.rglob("*") if p.is_file()}


def test_interruption_never_publishes_false_completion_and_retry_reuses_verified_bytes(tmp_path, monkeypatch):
    evidence, _ = lrts_source(tmp_path)
    report = audit_corpus(load_registry(), (evidence,))
    writer = ArtifactWriter(tmp_path / "audit")
    original = ArtifactWriter.write_immutable

    def interrupted(self, path, text):
        if path.name == "report.md":
            raise OSError("owned interrupted report publication")
        return original(self, path, text)

    with monkeypatch.context() as fault:
        fault.setattr(ArtifactWriter, "write_immutable", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            publish_corpus_admission(report, (evidence,), writer)
    assert not list(writer.run_root.rglob("complete.json"))
    saved = next(writer.run_root.rglob("admission.json"))
    before = saved.read_bytes()
    completion = publish_corpus_admission(report, (evidence,), writer)
    assert saved.read_bytes() == before
    assert validate_admission_completion(completion, (evidence,)) == report


def test_missing_bindings_tampered_bytes_future_readers_and_unknown_producers_fail_closed(tmp_path):
    evidence, _ = lrts_source(tmp_path)
    report = audit_corpus(load_registry(), (evidence,))
    with pytest.raises(ValueError, match="bindings"):
        validate_corpus_sources(report, ())
    unknown = changed(report, producer=fingerprint("unrecognized-producer"))
    assert CorpusAdmission.from_json(unknown.to_json()) == unknown  # Inspectable metadata, not verified evidence.
    with pytest.raises(ValueError, match="recorded installed package"):
        validate_corpus_sources(unknown, (evidence,))
    file = evidence.root / "observations.parquet"
    original = file.read_bytes()
    file.write_bytes(original[:-1])
    with pytest.raises(ValueError):
        publish_corpus_admission(report, (evidence,), ArtifactWriter(tmp_path / "must-not-complete"))
    assert not list((tmp_path / "must-not-complete").rglob("complete.json"))
    file.write_bytes(original)
    malformed = tmp_path / "future.json"
    malformed.write_text('{"schema_version":2}')
    with pytest.raises(ValueError):
        load_source_evidence("lrts", malformed, evidence.review, cache=evidence.cache)
    with pytest.raises(ValueError, match="offline cache"):
        assess_source(replace(evidence, cache=None))


def test_completed_file_metadata_is_not_a_substitute_for_source_verification(tmp_path):
    evidence, _ = lrts_source(tmp_path)
    report = audit_corpus(load_registry(), (evidence,))
    path = publish_corpus_admission(report, (evidence,), ArtifactWriter(tmp_path / "audit"))
    human = path.parent / "report.md"
    human.write_text("forged report\n")
    with pytest.raises(ValueError, match="role bytes"):
        validate_admission_completion(path, (evidence,))
    raw = json.loads(path.read_text())
    forged_bytes = human.read_bytes()
    raw["files"][1].update(sha256="sha256:" + hashlib.sha256(forged_bytes).hexdigest(), size_bytes=len(forged_bytes))
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="human report"):
        validate_admission_completion(path, (evidence,))


def test_controlled_negatives_are_visible_but_constructed_execution_is_not_a_real_fault_corpus(tmp_path):
    evidence = controlled_source(tmp_path, rows=(observation(outcome="test_error"), observation("fixed", "passed")))
    assessment = assess_source(evidence)
    assert assessment.scope == "blocked"
    assert "constructed_executor_not_observed" in assessment.admission_reasons
    assert isinstance(assessment.payload, ControlledDataset)
    assert assessment.payload.pairs[0].classification == "inconclusive"
    assert assessment.profile.unique_faults is None
    measures = {item.name: item for item in assessment.profile.measurements}
    assert measures["source_unknown_outcomes"].value == 1
    assert measures["source_failure_fraction"].value == 0
    assert measures["source_observed_ci_span"].status == "not_applicable"


@pytest.mark.parametrize(
    "buggy,fixed", [("passed", "passed"), ("assertion_failed", "assertion_failed"), ("test_error", "passed")]
)
def test_metadata_reviews_keep_negatives_but_require_original_raw_producer_verification(tmp_path, buggy, fixed):
    # Owned producer-attestation contract fixture, not a real-source efficacy
    # claim. Actual public source qualifications are separate from unit tests.
    cache = SourceCache(tmp_path / "cache")
    ctx = context(cache)
    runtime = changed(ctx.runtime, execution_model="host")
    ctx = changed(ctx, runtime=runtime, fault_origin="real")
    spec = controlled_spec(cache, ctx=ctx, rows=(observation(outcome=buggy), observation("fixed", fixed)))
    payload = convert_controlled_source(spec, cache)
    declaration = review(payload, fault_reference="urn:coleman:owned-curation-contract-fixture")
    evidence = SourceEvidence(payload, declaration, tmp_path, cache)
    report = audit_corpus(load_registry(), (evidence,))
    assessment = report.assessments[0]
    assert assessment.scope == "blocked" and assessment.card.history == "constructed"
    assert "original_raw_producer_verification_missing" in assessment.admission_reasons
    assert report.covered_regimes == () and "controlled_non_java_real_faults" in report.uncovered_regimes
    assert "medium_historical_ci" in report.uncovered_regimes
    assert assessment.profile.unique_faults is None
    missing_reference = replace(evidence, review=changed(declaration, fault_reference=None))
    assert "curated_real_fault_reference_missing" in assess_source(missing_reference).admission_reasons


def test_hcs_requires_explicit_context_review_and_keeps_sparse_jobs_and_shared_executions(tmp_path):
    payload, cache, completion = hcs_export(tmp_path)
    evidence = SourceEvidence(payload, review(payload, history="constructed"), completion.parent, cache)
    report = audit_corpus(load_registry(), (evidence,))
    item = report.assessments[0]
    assert item.profile.rows == 17 and item.profile.unique_observations == 16 and item.profile.duplicate_rows == 1
    assert item.profile.duration.null_count == 1 and item.profile.duration.zero_count == 3
    assert item.scope == "blocked" and report.observed_source_models == ()
    assert item.card.manifest.identity.group_fields == ("job_id", "pipeline_id", "project")
    facts = {f.axis: f for f in item.reality}
    assert facts["matrix_execution"].status == "synthetic"
    assert facts["outcome_availability"].status == "missing"
    assert item.card.manifest.conversion.input_rows == 18
    assert item.card.manifest.outcomes.unit == "unverified"  # Upstream nonpass collapses unsupported statuses.


def test_one_stress_export_cannot_satisfy_the_full_adversarial_regime(tmp_path):
    value = case(failures="rare", variants="sparse")
    completion = export_stress(value, ArtifactWriter(tmp_path / "stress"))
    payload = StressReceipt.from_json(completion.read_bytes())
    evidence = SourceEvidence(payload, review(payload), completion.parent)
    report = audit_corpus(load_registry(), (evidence,))
    assert report.assessments[0].scope == "synthetic_mechanism"
    assert "deterministic_adversarial_simulation" in report.uncovered_regimes
    assert report.observed_source_models == ()
    assert report.assessments[0].profile.duration.zero_count > 0
    with pytest.raises(ValueError, match="foreign-source cache"):
        assess_source(replace(evidence, cache=SourceCache(tmp_path / "unused-cache")))


def test_source_qualification_cannot_publish_inside_git_or_inside_its_source_cache(tmp_path):
    evidence, _ = lrts_source(tmp_path)
    report = audit_corpus(load_registry(), (evidence,))
    with pytest.raises(ValueError, match="outside a repository"):
        publish_corpus_admission(report, (evidence,), ArtifactWriter(Path(__file__).parent / "must-not-write"))
    assert evidence.cache is not None
    with pytest.raises(ValueError):
        publish_corpus_admission(report, (evidence,), ArtifactWriter(evidence.cache.root / "must-not-write"))


def test_renderer_escapes_source_labels_as_data_and_is_deterministic(tmp_path):
    evidence = controlled_source(tmp_path)
    label = r"<script>bad()</script> \| [source label](https://example.invalid) *styled* _value_ ~old~ `row`"
    declaration = changed(evidence.review, limitations=(label,))
    report = audit_corpus(load_registry(), (replace(evidence, review=declaration),))
    human = render_corpus_admission(report)
    assert "<script>" not in human and "&lt;script&gt;" in human and "&#124;" in human and "&#96;" in human
    assert "[source label](" not in human and "&#91;source label&#93;" in human
    assert "&#92;&#124;" in human and "&#42;styled&#42;" in human
    assert "&#95;value&#95;" in human and "&#126;old&#126;" in human
    assert label in unescape(human)
    assert human == render_corpus_admission(CorpusAdmission.from_json(report.to_json()))
