"""Owned cross-family evidence preserves identities and never converts missing gates to success."""

from dataclasses import replace

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission import (
    SourceEvidence,
    audit_corpus,
    publish_corpus_admission,
    validate_admission_completion,
)
from coleman.corpus.admission_semantics import admitted_regimes
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.lrts_stage_etl import export_lrts_stage_parquet
from coleman.corpus.registry import load_registry
from coleman.corpus.rtptorrent_etl import export_rtptorrent_parquet
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_artifacts import export_stress
from coleman.corpus.stress_contracts import StressCase, StressReceipt
from coleman.corpus.stress_design import adversarial_cases, covering_design
from tests.corpus.admission_support import changed, lrts_source, review
from tests.corpus.test_hcs_etl import export as hcs_export
from tests.corpus.test_lrts_stage_etl import prepare as stage_prepare
from tests.corpus.test_rtptorrent_etl import prepare as rtp_prepare


def test_source_stages_and_class_units_survive_the_integrated_report(tmp_path):
    recipe, cache = stage_prepare(tmp_path / "stage")
    marker = export_lrts_stage_parquet(recipe, cache, ArtifactWriter(tmp_path / "stage-output"))
    stage = LRTSStageReceipt.from_json(marker.read_bytes())
    stage_binding = SourceEvidence(stage, review(stage), marker.parent, cache)
    recipe, cache = rtp_prepare(tmp_path / "classes")
    marker = export_rtptorrent_parquet(recipe, cache, ArtifactWriter(tmp_path / "class-output"))
    classes = RTPTorrentETLReceipt.from_json(marker.read_bytes())
    class_binding = SourceEvidence(classes, review(classes), marker.parent, cache)
    report = audit_corpus(load_registry(), (stage_binding, class_binding))
    items = {type(x.payload): x for x in report.assessments}
    assert items[LRTSStageReceipt].profile.rows == 12
    assert items[LRTSStageReceipt].profile.source_groups == 4
    assert "stage_id" in items[LRTSStageReceipt].profile.execution_group_fields
    item = items[RTPTorrentETLReceipt]
    assert item.profile.rows == 8 and item.profile.source_groups == 3
    assert item.profile.duration.semantics.unit == "s"
    assert item.profile.duration.null_count == 2 and item.profile.duration.zero_count == 3
    assert item.card.manifest.outcomes.mapping[0].source_value.startswith("class_outcome:")
    assert item.card.manifest.outcomes.unit == "unverified"  # Class aggregates are not failing-test counts.
    assert "logical_build_job_mapping_unverified" in item.historical_blockers
    assert report.covered_regimes == report.observed_source_models == ()
    completion = publish_corpus_admission(report, (stage_binding, class_binding), ArtifactWriter(tmp_path / "audit"))
    assert validate_admission_completion(completion, (class_binding, stage_binding)) == report


def test_reviewed_source_models_are_distinct_without_granting_historical_admission(tmp_path):
    # Owned metadata attestations exercise the classification rule, not authentic
    # CI execution. Existing declared histories cannot be changed by a review.
    lrts, _ = lrts_source(tmp_path / "jenkins")
    payload, cache, marker = hcs_export(tmp_path / "gitlab")
    assert isinstance(payload, HCSReceipt)
    hcs = SourceEvidence(payload, review(payload, history="observed"), marker.parent, cache)
    report = audit_corpus(load_registry(), (lrts, hcs))
    assert report.observed_source_models == ("GitLab-pipeline-job-variant-reports",)
    assert report.covered_regimes == () and len(report.uncovered_regimes) == 7
    for status in ("unverified", "rejected"):
        weaker = replace(hcs, review=changed(hcs.review, status=status))
        assert audit_corpus(load_registry(), (lrts, weaker)).observed_source_models == ()
    with pytest.raises(ValueError, match="declared history"):
        audit_corpus(load_registry(), (replace(lrts, review=changed(lrts.review, history="observed")), hcs))


def test_all_default_owned_stress_exports_are_required_and_duplicates_do_not_add_coverage(tmp_path):
    # This actual network-free integration exports/profiles the complete bounded
    # family. An eight-build case is not replaced by a tiny proxy or a seed count.
    cases = tuple(StressCase(schema_version=1, factors=row) for row in covering_design().rows) + tuple(
        case for _, case in adversarial_cases()
    )
    assert len(cases) == len({case.fingerprint for case in cases}) == 39
    bindings = []
    for case in cases:
        marker = export_stress(case, ArtifactWriter(tmp_path / "stress"))
        payload = StressReceipt.from_json(marker.read_bytes())
        bindings.append(SourceEvidence(payload, review(payload), marker.parent))
    report = audit_corpus(load_registry(), tuple(bindings))
    assert len(report.assessments) == 39
    assert report.covered_regimes == ("deterministic_adversarial_simulation",)
    assert len(report.uncovered_regimes) == 6 and report.observed_source_models == ()
    assert all(x.scope == "synthetic_mechanism" for x in report.assessments)
    assert admitted_regimes(tuple(reversed(report.assessments))) == report.covered_regimes
    assert admitted_regimes(report.assessments + report.assessments) == report.covered_regimes
    # Remove each required case in turn: even 38 qualified exports cannot cover
    # a declared family that requires all 39, regardless of retained row count.
    for index in range(len(report.assessments)):
        assert admitted_regimes(report.assessments[:index] + report.assessments[index + 1 :]) == ()
