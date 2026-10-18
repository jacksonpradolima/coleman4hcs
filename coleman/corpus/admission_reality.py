"""Source-bound qualitative production-reality facts, with missing evidence visible.

Provider/report labels describe those labels. They never certify physical
parallelism, a PR trigger, a complete expected test universe, removable setup
cost or historical feature availability. Facts are pure derivations and do not
authorize data acquisition, subject execution or policy evaluation.
"""

from __future__ import annotations

from typing import get_args

from coleman.corpus.admission_contracts import EvidencePayload, FactAxis, RealityFact, SourceReview
from coleman.corpus.admission_sources import source_classification
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.profiles import NativeSourceProfile
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_contracts import StressReceipt


def _fact(
    axis: FactAxis,
    status: str,
    values: tuple[str, ...] | None,
    locator: str,
    limitation: str,
) -> RealityFact:
    """Validate a local construction boundary; no caller dictionaries cross the domain API."""
    return RealityFact.model_validate(
        {"axis": axis, "status": status, "values": values, "locator": locator, "limitation": limitation}
    )


def _controlled_facts(payload: ControlledDataset) -> tuple[RealityFact, ...]:
    """Retain producer-attested capture conditions without promoting them to observed historical CI."""
    context = payload.spec.context
    status = "synthetic" if context.runtime.execution_model == "constructed" else "observed"
    facts = [
        _fact("language", "observed", (context.language,), "spec.context.language", "Subject language declaration."),
        _fact(
            "source_execution_model",
            status,
            ("controlled-" + context.runtime.execution_model,),
            "spec.context.runtime",
            "Producer-attested capture runtime, not a CI provider or a historical execution model.",
        ),
        _fact(
            "shared_fixtures",
            "reconstructed",
            ("same-declared-fixed-fixture-in-both-roles",),
            "spec.context.inputs[kind=test_fixture]",
            "Common input identity does not verify test-order independence or independently removable setup cost.",
        ),
        _fact(
            "test_levels",
            "reconstructed",
            ("controlled-functional-observations",),
            "spec.context.test_ids",
            "Selected functional fixtures; full original unit/integration/end-to-end classification is unverified.",
        ),
    ]
    outcome_axes: tuple[tuple[FactAxis, set[str]], ...] = (
        ("timeouts", {"timeout"}),
        ("cancellation", {"cancelled"}),
        ("infrastructure_failures", {"infrastructure_error"}),
        ("partial_outcomes", {"unavailable", "skipped"}),
    )
    for axis, outcomes in outcome_axes:
        present = tuple(sorted({row.outcome for row in payload.observations} & outcomes))
        if present:
            facts.append(
                _fact(
                    axis,
                    "observed",
                    present,
                    "observations[].outcome",
                    "Reported outcomes in this retained capture only; absence would not prove the mechanism absent.",
                )
            )
    return tuple(facts)


def _stress_facts(payload: StressReceipt) -> tuple[RealityFact, ...]:
    """Describe declared synthetic mechanisms; none becomes observed CI or prospective evidence."""
    factors = payload.case.factors
    settings: tuple[tuple[FactAxis, str], ...] = (
        ("source_execution_model", "owned-counter-based-constructed-CI"),
        ("serial_execution", "native-serial-eligibility-is-separate"),
        ("parallel_execution", factors.workers),
        ("concurrency", factors.workers),
        ("matrix_execution", factors.variants),
        ("flakiness", factors.flakiness),
        ("renames", factors.churn),
        ("partial_outcomes", factors.feedback),
        ("delayed_outcomes", factors.feedback),
        ("corrected_outcomes", factors.feedback),
        ("outcome_availability", factors.feedback),
        ("feature_availability", factors.context),
    )
    return tuple(
        _fact(
            axis,
            "synthetic",
            (value,),
            "case.factors",
            "Constructed factor declaration; workers are modeled post-hoc and variants/seeds remain dependent.",
        )
        for axis, value in settings
    )


def _historical_facts(payload: EvidencePayload, review: SourceReview) -> tuple[RealityFact, ...]:
    """Preserve two distinct reported CI structures; do not infer topology or triggers from names."""
    history = source_classification(payload, review).history
    if history == "constructed":
        status = "synthetic"
    else:
        status = "observed" if history == "observed" and review.status == "reviewed" else "reconstructed"
    facts: list[RealityFact]
    if isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt)):
        facts = [
            _fact(
                "source_execution_model",
                status,
                ("Jenkins-method-and-stage-reports",),
                "recipe.builds"
                if isinstance(payload, LRTSETLReceipt)
                else "recipe.source.builds;recipe.stage_bindings",
                "Reported stages do not establish concurrency, executor start or label availability.",
            ),
            _fact(
                "partial_outcomes",
                status,
                ("passed-failed-not_run-unknown-retained-separately",),
                "observations.parquet:outcome;replay_exclusion",
                "A skipped/not-run report is not a passing test; the complete expected suite remains unverified.",
            ),
        ]
    elif isinstance(payload, HCSReceipt):
        facts = [
            _fact(
                "source_execution_model",
                status,
                ("GitLab-pipeline-job-variant-reports",),
                "recipe.jobs;observations.parquet:pipeline_id,job_id,variant_name",
                "Reported pipeline/job labels, not measured runner topology or predecision configuration features.",
            ),
            _fact(
                "matrix_execution",
                status,
                ("sparse-pipeline-job-label-cells",),
                "cells.parquet:metadata_jobs,parsed_rows,cell_status",
                "Metadata-only labels remain absent test evidence; shared tests across labels are dependent.",
            ),
            _fact(
                "partial_outcomes",
                status,
                ("explicit-missing-parsed-job-cells-and-primary-exclusions",),
                "accounting;cells.parquet",
                "Upstream omitted whole-job counts are unavailable; binary nonpass labels have collapsed statuses.",
            ),
        ]
    elif isinstance(payload, RTPTorrentETLReceipt):
        facts = [
            _fact(
                "source_execution_model",
                status,
                ("source-job-class-reports",),
                "observations.parquet:project,job_id,source_position",
                "Jobs are preserved; logical build mapping, provider, timestamps and physical topology are unverified.",
            ),
            _fact(
                "partial_outcomes",
                status,
                ("known-not_run-unknown-class-outcomes-retained",),
                "observations.parquet:class_outcome;replay_exclusion",
                "Class-level outcomes retain their unit; source clocks and complete expected suites remain unknown.",
            ),
        ]
    else:
        facts = []
    return tuple(facts)


def _native_facts(payload: NativeSourceProfile) -> tuple[RealityFact, ...]:
    """Retain original native-card declarations; profiling never upgrades their epistemic scope."""
    specific = []
    declarations: tuple[tuple[FactAxis, tuple[str, ...] | None], ...] = (
        ("language", payload.card.languages),
        ("source_execution_model", payload.card.execution_models),
    )
    for axis, values in declarations:
        if values is not None:
            specific.append(
                _fact(
                    axis,
                    "observed" if values else "not_applicable",
                    values or None,
                    "card.languages" if axis == "language" else "card.execution_models",
                    "Original source card declaration; descriptive profiling grants no admission.",
                )
            )
    return tuple(specific)


def _specific_facts(payload: EvidencePayload, review: SourceReview) -> tuple[RealityFact, ...]:
    """Dispatch fixed data families without dynamic modules or an inference from filename/provider names."""
    if isinstance(payload, NativeSourceProfile):
        return _native_facts(payload)
    if isinstance(payload, ControlledDataset):
        return _controlled_facts(payload)
    if isinstance(payload, StressReceipt):
        return (
            *_stress_facts(payload),
            _fact("language", "not_applicable", None, "case.generator", "Owned numerical synthetic mechanism."),
        )
    return _historical_facts(payload, review)


def production_facts(payload: EvidencePayload, review: SourceReview) -> tuple[RealityFact, ...]:
    """Return all 35 axes in canonical order, keeping unsupported/absent evidence distinct.

    Native cards retain their original declarations. Source-specific facts are
    deliberately narrower than inferred language/provider/parallelism claims.
    Inapplicable language for an owned numerical generator is not an unknown
    source language. Every missing axis has an explicit reason and locator.
    """
    facts = {
        axis: _fact(
            axis,
            "missing",
            None,
            "payload:" + axis,
            "This source payload does not establish this production-reality axis; do not impute zero or absence.",
        )
        for axis in get_args(FactAxis)
    }
    for fact in _specific_facts(payload, review):
        facts[fact.axis] = fact
    return tuple(facts[axis] for axis in sorted(facts))
