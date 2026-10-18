"""Authored repeated-measures quantities; no real repositories or independent trials are implied."""

from datetime import UTC, datetime
from decimal import Decimal

from coleman.corpus.repeated import RepeatedCorpusReport
from coleman.corpus.repeated_contracts import (
    BUDGET_RATIO,
    MEAN_SUFFIX,
    CorpusBuild,
    CorpusRepetition,
    RepetitionDesign,
    repetition_execution_id,
)
from coleman.evaluation.protected import (
    ComparisonHypothesis,
    EvaluationProtocol,
    FrozenCandidate,
    TemporalObservation,
    TemporalPlan,
    freeze,
)
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.selection import Quantity, UnitSemantics

ASSEMBLED = datetime(2026, 1, 6, 12, tzinfo=UTC)


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def repeated_report(unit_differences=None, target_end=None):
    """Return two dependent builds in one synthetic unit and one build in another."""
    plan = TemporalPlan(
        train_end=datetime.fromisoformat("2026-01-02T23:59:59Z"),
        validation_end=datetime.fromisoformat("2026-01-04T23:59:59Z"),
        test_end=datetime.fromisoformat("2026-01-07T23:59:59Z"),
    )
    projects = ("constructed-a", "constructed-a", "constructed-b")
    if unit_differences is not None:
        projects = tuple(f"constructed-{index:03d}" for index in range(len(unit_differences)) for _ in range(2))
    builds = tuple(
        CorpusBuild(
            member_id=f"build-{index}",
            project_id=project,
            unit_id=project,
            metadata=TemporalObservation(
                observation_id=f"source-build-{index}",
                group_id=f"source-build-{index}",
                event_at=datetime.fromisoformat("2026-01-05T12:00:00Z"),
                available_at=datetime.fromisoformat("2026-01-06T00:00:00Z"),
                target_end=target_end,
            ),
        )
        for index, project in enumerate(projects)
    )
    design = RepetitionDesign(
        schema_version=1,
        fold_id="owned-test-fold",
        data_fingerprint=fingerprint(
            ["owned", "repeated", "source-v1", None if unit_differences is None else list(unit_differences)]
        ),
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        plan=plan,
        warmup_end=datetime.fromisoformat("2026-01-01T23:59:59Z"),
        budget=Quantity(value=Decimal("0.5"), semantics=BUDGET_RATIO),
        seeds=(11, 42),
        builds=builds,
        raw_metric="net_credit",
        raw_semantics=UnitSemantics(unit="credit", meaning="constructed-net-ci-credit-v1"),
        cost_projection="exact_integer_seconds_v1",
    )
    candidates = tuple(
        FrozenCandidate.from_configuration(name, {"policy": name, "corpus_repetition_design": design.fingerprint})
        for name in ("candidate", "baseline")
    )
    frozen = freeze(
        candidates,
        design.data_fingerprint,
        plan,
        EvaluationProtocol(
            protocol_id="owned-repetitions-v1",
            holdout_id="owned-repetition-holdout-v1",
            metric_name=design.raw_metric,
            unit=design.raw_semantics.unit,
            meaning=design.raw_semantics.meaning + MEAN_SUFFIX,
            direction="maximize",
            independent_unit=design.independent_unit,
            dependence="independent",
            normal_differences_assumed=True,
            practical_effect=0.1,
            minimum_units=20,
        ),
        (ComparisonHypothesis(hypothesis_id="candidate-baseline", candidate_id="candidate", baseline_id="baseline"),),
    )
    rows = []
    for candidate in candidates:
        for index, build in enumerate(design.builds):
            for seed_index, seed in enumerate(design.seeds):
                value = (index * 2 + seed_index) if candidate.candidate_id == "candidate" else -1
                if unit_differences is not None:
                    value = 0.0
                    if candidate.candidate_id == "candidate":
                        value = unit_differences[index // 2] + (seed_index - 0.5) + ((index % 2) - 0.5) / 2
                rows.append(
                    CorpusRepetition(
                        candidate_id=candidate.candidate_id,
                        member_id=build.member_id,
                        seed=seed,
                        state="completed",
                        record=DomainRecord(
                            schema_version=1,
                            namespace="ci.owned.repetitions",
                            run_id=candidate.candidate_id,
                            execution_id=repetition_execution_id(candidate.candidate_id, seed),
                            decision_id=build.member_id,
                            data_fingerprint=frozen.data_fingerprint,
                            available_at=build.metadata.available_at,
                            provenance=ResultProvenance(
                                producer="owned-corpus-fixture",
                                producer_version="1",
                                spec_fingerprint=candidate.configuration_fingerprint,
                                evaluation_fingerprint=frozen.fingerprint,
                            ),
                            measurements=(
                                Measurement(
                                    name=design.raw_metric,
                                    value=float(value),
                                    status="observed",
                                    unit=design.raw_semantics.unit,
                                    meaning=design.raw_semantics.meaning,
                                    direction="maximize",
                                ),
                            ),
                        ),
                    )
                )
    return RepeatedCorpusReport(
        schema_version=1, design=design, frozen=frozen, rows=tuple(rows), assembled_at=ASSEMBLED
    )
