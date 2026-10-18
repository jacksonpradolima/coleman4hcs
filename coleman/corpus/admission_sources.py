"""Pure receipt-to-card bindings for the existing corpus registry.

Primary source terms describe that source only. Full component/source terms
remain in embedded receipts and must all pass their existing cache validators.
New cards describe the retained slice explicitly; old registry cards are never
silently upgraded, retagged or routed through a different adapter.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, NamedTuple

from coleman.corpus.admission_contracts import EvidencePayload, ObservationProfile, SourceReview
from coleman.corpus.contracts import (
    AdapterReference,
    ConversionSummary,
    CorpusManifest,
    CorpusResources,
    DataCard,
    EvaluationDeclaration,
    ExclusionCount,
    LicenseReview,
    ObservationIdentity,
    OutcomeSemantics,
    SourceArtifact,
    TimeSemantics,
    VerdictMapping,
)
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.lrts_contracts import MEANINGS
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.profiles import NativeSourceProfile
from coleman.corpus.registry import load_registry
from coleman.corpus.rtptorrent_contracts import REPORTED_SECONDS
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_contracts import StressReceipt
from coleman.pilot.contracts import SECONDS
from coleman.results.domain import fingerprint
from coleman.selection.contracts import Quantity, UnitSemantics


class SourceDescription(NamedTuple):
    """Pure primary-source binding; a group is dependent project/family evidence."""

    family: str
    source: SourceArtifact
    license: LicenseReview
    project_group: str
    identity: ObservationIdentity
    conversion: ConversionSummary


class SourceClassification(NamedTuple):
    """Keep history and fault provenance independent of format and successful conversion."""

    history: Literal["observed", "constructed", "unverified"]
    fault_origin: Literal["real", "synthetic", "unverified"]
    layer: Literal["historical_ci", "long_running_ci", "controlled_real_faults", "hcs", "synthetic"]


def declared_history(payload: EvidencePayload) -> Literal["observed", "constructed", "unverified"] | None:
    """Return an existing history attestation; HCS v1 has none and requires an explicit context review."""
    if isinstance(payload, NativeSourceProfile):
        return payload.card.history
    if isinstance(payload, (ControlledDataset, StressReceipt)):
        return "constructed"
    if isinstance(payload, LRTSStageReceipt):
        return payload.recipe.source.history
    if isinstance(payload, LRTSETLReceipt):
        return payload.recipe.history
    if isinstance(payload, RTPTorrentETLReceipt):
        return payload.recipe.conversion.history
    return None


def source_classification(payload: EvidencePayload, review: SourceReview) -> SourceClassification:
    """Reject conflicting source-review history; a format never turns owned fixtures into observed CI."""
    declared = declared_history(payload)
    if review.payload_fingerprint != payload.fingerprint or (declared is not None and review.history != declared):
        raise ValueError("Source context review conflicts with its payload identity or declared history")
    if isinstance(payload, ControlledDataset):
        origin = payload.spec.context.fault_origin
        return SourceClassification(
            "constructed", origin, "controlled_real_faults" if origin == "real" else "synthetic"
        )
    if review.history == "constructed":
        return SourceClassification("constructed", "synthetic", "synthetic")
    if isinstance(payload, HCSReceipt):
        layer = "hcs"
    elif isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt)):
        layer = "long_running_ci"
    else:
        layer = "historical_ci"
    return SourceClassification(review.history, "unverified", layer)


def _conversion(input_rows: int, output_rows: int, reasons: tuple[tuple[str, int], ...]) -> ConversionSummary:
    """Preserve each adapter's primary row dispositions, including genuine zero exclusions."""
    return ConversionSummary(
        status="complete",
        input_rows=input_rows,
        output_rows=output_rows,
        exclusions=tuple(ExclusionCount(reason=name, rows=count) for name, count in reasons),
    )


def source_description(payload: EvidencePayload) -> SourceDescription:
    """Read explicit receipt fields only; no URI, path, subject process or import is executed.

    Completion payloads must already be byte-verified by the offline service.
    This function describes their declarations, not the truth of arbitrary
    attestations. It preserves original job/stage/pipeline/fault dimensions.
    """
    if isinstance(payload, NativeSourceProfile):
        manifest = payload.card.manifest
        return SourceDescription(
            "legacy-native-profile",
            manifest.source,
            manifest.license,
            manifest.dependence_group,
            manifest.identity,
            manifest.conversion,
        )
    if isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt)):
        recipe = payload.recipe.source if isinstance(payload, LRTSStageReceipt) else payload.recipe
        exclusions = [
            ("not_run", payload.excluded_not_run),
            ("unknown_outcome", payload.excluded_unknown_outcome),
            ("unavailable_duration", payload.excluded_unavailable_duration),
        ]
        dimensions = ("project", "pull_request", "source_build_id")
        family = "lrts-parquet-serial-replay-v1"
        if isinstance(payload, LRTSStageReceipt):
            exclusions.append(("outside_stage", payload.excluded_outside_stage))
            dimensions += ("stage_id",)
            family = "lrts-stage-parquet-replay-v1"
        return SourceDescription(
            family,
            recipe.metadata.source,
            recipe.metadata.license,
            "public-project:lrts:" + recipe.project,
            ObservationIdentity(
                observation_fields=dimensions + ("suite_position", "case_position"),
                group_fields=dimensions,
            ),
            _conversion(payload.observed_cases, payload.replay_cases, tuple(exclusions)),
        )
    if isinstance(payload, RTPTorrentETLReceipt):
        package = payload.recipe.conversion.materialization.spec
        return SourceDescription(
            "rtptorrent-parquet-source-job-replay-v1",
            package.source,
            package.license,
            "public-project:rtptorrent:" + payload.recipe.conversion.project,
            ObservationIdentity(
                observation_fields=("project", "job_id", "source_position"),
                group_fields=("project", "job_id"),
            ),
            _conversion(
                payload.input_rows,
                payload.replay_executions,
                (
                    ("duplicates", payload.duplicate_rows_excluded),
                    ("outside_jobs", payload.excluded_outside_jobs),
                    ("not_run", payload.excluded_not_run),
                    ("unknown_outcome", payload.excluded_unknown_outcome),
                    ("unavailable_duration", payload.excluded_unavailable_duration),
                ),
            ),
        )
    if isinstance(payload, HCSReceipt):
        counts = payload.accounting
        return SourceDescription(
            "gitlab-hcs-pipeline-parquet-v1",
            payload.recipe.executions.source,
            payload.recipe.executions.license,
            "public-project:hcs:" + payload.recipe.project,
            ObservationIdentity(
                observation_fields=("project", "pipeline_id", "job_id", "test_name"),
                group_fields=("project", "pipeline_id", "job_id"),
            ),
            _conversion(
                counts.input_rows,
                counts.replay_rows,
                tuple(
                    (name, getattr(counts, name))
                    for name in (
                        "outside_slice",
                        "duplicate_rows",
                        "unsupported_job_rows",
                        "other_attempt_rows",
                        "unavailable_outcome_rows",
                        "unavailable_duration_rows",
                    )
                ),
            ),
        )
    if isinstance(payload, ControlledDataset):
        context = payload.spec.context
        package = payload.spec.report_materialization.spec
        return SourceDescription(
            "controlled-functional-observations-v1",
            package.source,
            package.license,
            "public-controlled-project:" + context.project,
            ObservationIdentity(
                observation_fields=("context_fingerprint", "trial_id", "test_id", "role"),
                group_fields=("context.project", "context.fault_id", "trial_id"),
            ),
            _conversion(payload.input_rows, len(payload.observations), (("duplicates", payload.duplicates_excluded),)),
        )
    case = payload.case
    native = payload.files[-1]
    license_review = load_registry().card("ci-serial-boundary-v1").manifest.license
    return SourceDescription(
        "owned-ci-stress-v1",
        SourceArtifact(
            uri="urn:coleman:stress-case:" + case.fingerprint.removeprefix("sha256:"),
            version=case.generator,
            citation="COLEMAN owned constructed CI mechanism case; source identity is the complete versioned case",
            sha256=case.fingerprint,
            size_bytes=len(case.to_json().encode("utf-8")),
            availability="verified_local",
        ),
        license_review,
        "coleman-owned-ci-stress-family-v1",
        ObservationIdentity(
            observation_fields=("case_fingerprint", "build", "test_id", "variant"),
            group_fields=("case_fingerprint", "build"),
        ),
        _conversion(native.rows, native.rows, ()),
    )


def source_identity(payload: EvidencePayload) -> str:
    """Hash stable selection/context identity separately from outcomes, durations and physical output paths."""
    if isinstance(payload, NativeSourceProfile):
        return payload.card.manifest.benchmark_id
    if isinstance(payload, ControlledDataset):
        logical = [payload.spec.context.fingerprint, sorted({item.trial_id for item in payload.observations})]
    elif isinstance(payload, StressReceipt):
        logical = payload.case.fingerprint
    else:
        logical = payload.recipe.fingerprint
    return "audited-source-" + fingerprint([source_description(payload).family, logical]).removeprefix("sha256:")


def source_verdict_mapping(payload: EvidencePayload) -> tuple[VerdictMapping, VerdictMapping]:
    """Name the actual source labels; unknown/error/censored outcomes never become passing labels."""
    if isinstance(payload, HCSReceipt):
        zero, one = "reported_nonpass:0", "reported_nonpass:1"
    elif isinstance(payload, ControlledDataset):
        zero, one = "outcome:passed", "outcome:assertion_failed"
    elif isinstance(payload, RTPTorrentETLReceipt):
        zero, one = "class_outcome:passed", "class_outcome:failed"
    elif isinstance(payload, StressReceipt):
        zero, one = "verdict:0", "verdict:1"
    else:
        zero, one = "outcome:passed", "outcome:failed"
    return VerdictMapping(source_value=zero, verdict=0), VerdictMapping(source_value=one, verdict=1)


def source_duration_semantics(payload: EvidencePayload) -> UnitSemantics:
    """Return actual source units/meaning; no duration conversion or physical-cost inference occurs."""
    if isinstance(payload, NativeSourceProfile):
        return payload.raw_duration.semantics
    if isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt)):
        return UnitSemantics(unit="s", meaning=MEANINGS["lrts_case_duration"])
    if isinstance(payload, RTPTorrentETLReceipt):
        return REPORTED_SECONDS
    if isinstance(payload, HCSReceipt):
        return payload.recipe.duration_semantics
    if isinstance(payload, ControlledDataset):
        return UnitSemantics(unit="ns", meaning="observed-test-harness-monotonic-elapsed-v1")
    return SECONDS


def source_execution_group_fields(payload: EvidencePayload) -> tuple[str, ...]:
    """Separate actual execution-group dimensions from original declared evaluation/dependence groups."""
    if isinstance(payload, NativeSourceProfile):
        return ("BuildId", "Variant") if "Variant" in payload.columns else ("BuildId",)
    return source_description(payload).identity.group_fields


def source_card(payload: EvidencePayload, profile: ObservationProfile, review: SourceReview) -> DataCard:
    """Build a new descriptive card using the existing type; preserve legacy cards exactly.

    Planned confirmation remains blocked: the full-ratio budget is only an
    explicit intended complete-snapshot declaration, not an executed experiment.
    Functional/synthetic qualification lives in the separate assessment scope.
    Primary terms never replace other component terms in the embedded payload.
    """
    if isinstance(payload, NativeSourceProfile):
        source_classification(payload, review)
        return payload.card
    description = source_description(payload)
    controlled = isinstance(payload, ControlledDataset)
    synthetic = isinstance(payload, StressReceipt)
    classification = source_classification(payload, review)
    time = TimeSemantics(
        origin=classification.history,
        event_field="capture_started_at" if controlled else None,
        outcome_available_field="capture_finished_at" if controlled else None,
        feature_available_field=None,
        timezone="UTC" if controlled or synthetic else "unverified",
    )
    manifest = CorpusManifest(
        schema_version=1,
        benchmark_id=source_identity(payload),
        dependence_group=description.project_group,
        layer=classification.layer,
        source=SourceArtifact.model_validate({**description.source.model_dump(), "availability": "verified_local"}),
        license=description.license,
        adapter=AdapterReference(name=description.family, version="1", implementation="implemented"),
        identity=description.identity,
        time=time,
        outcomes=OutcomeSemantics(
            unit="unverified" if isinstance(payload, (HCSReceipt, RTPTorrentETLReceipt)) else "failing_test",
            mapping=source_verdict_mapping(payload),
            unique_fault_mapping=None,
            unknown_policy="exclude_with_counts",
            duration=source_duration_semantics(payload),
        ),
        evaluation=EvaluationDeclaration(
            purpose="planned_confirmatory",
            temporal_plan=None,
            budget_mode="suite_ratio",
            budgets=(
                Quantity(
                    value=Decimal(1), semantics=UnitSemantics(unit="1", meaning="full-suite-execution-budget-ratio-v1")
                ),
            ),
            seeds=(),
            metric_ids=("source-profile-not-a-policy-evaluation-v1",),
        ),
        conversion=description.conversion,
        resources=CorpusResources(maximum_source_bytes=8 * 1024**3, maximum_rows=10_000_000),
        limitations=(
            "Source integrity and scoped qualification are separate from protected confirmation and policy efficacy.",
            "Primary source terms do not clear other components; retain every embedded source/input receipt.",
            "Reported/synthetic/harness quantities are not independently removable CI work or developer latency.",
            "Source execution groups, curated fault labels and seeds do not establish statistical independence.",
        ),
    )
    languages = None
    if isinstance(payload, ControlledDataset):
        languages = (payload.spec.context.language,)
    elif isinstance(payload, StressReceipt):
        languages = ()
    return DataCard(
        schema_version=1,
        manifest=manifest,
        history=classification.history,
        fault_origin=classification.fault_origin,
        languages=languages,
        execution_models=None,
        measurements=profile.measurements,
        covered_regimes=(),
        missing_regimes=load_registry().required_regimes,
    )
