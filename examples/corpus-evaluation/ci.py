"""Installed native TCPCI → independent metric check → protected repository means.

Run ``python ci.py ./output`` outside checkout. Tiny authored CI histories exercise
the actual Random/UCB runner, Parquet, checkpoints and protected access. They are
constructed mechanism evidence, not admitted historical repositories or a policy
comparison establishing generalization. All durations are exact integer seconds.
"""

import json
import math
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.artifacts import ArtifactWriter
from coleman.corpus import tiny_ci_reference
from coleman.corpus.repeated import RepeatedCorpusReport, publish_repeated_corpus
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
    EvaluationAudit,
    EvaluationProtocol,
    FrozenCandidate,
    TemporalObservation,
    TemporalPlan,
    evaluate,
    freeze,
)
from coleman.pilot import SECONDS, PilotBuild, PilotTest
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.selection import Quantity, UnitSemantics
from coleman.spec.models import RunSpec


def _quantity(value):
    return Quantity(value=Decimal(str(value)), semantics=SECONDS)


def _write_inputs(root):
    """Publish two owned, related CI histories with a causal sequential replay clock."""
    data = {}
    members = []
    for project in ("constructed_ci_a", "constructed_ci_b"):
        rows = []
        for build in (1, 2):
            event_at = datetime(2026, 1, 4 + build, tzinfo=UTC)
            labels_available_at = event_at + timedelta(hours=1)
            for name, duration, verdict in (("unit", 1, 0), ("integration", 2, 1), ("system", 4, build % 2)):
                if build == 2:
                    name = {"unit": "renamed-unit", "system": "contract"}.get(name, name)
                rows.append(
                    {
                        "BuildId": build,
                        "Name": name,
                        "Duration": float(duration),
                        "CalcPrio": 0,
                        "LastRun": event_at.date().isoformat(),
                        "Verdict": verdict,
                    }
                )
            member_id = f"{project}/build-{build}"
            members.append(
                CorpusBuild(
                    member_id=member_id,
                    project_id=project,
                    # Both histories are copies of one authored mechanism design;
                    # project labels do not create independent source families.
                    unit_id="constructed-fixture-family",
                    metadata=TemporalObservation(
                        observation_id=member_id,
                        group_id=member_id,
                        event_at=event_at,
                        available_at=labels_available_at,
                    ),
                )
            )
        data[project] = rows
        directory = root / "inputs" / project
        directory.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(directory / "features-engineered.parquet")
    return data, tuple(members)


def _design(data, members):
    """Declare source meaning, complete repetition membership and the protected cutoff."""
    now = datetime.now(UTC)
    # Keep a fixed, explicit final cutoff after all assembly below. No historical
    # source clocks are substituted with this analysis-time availability bound.
    plan = TemporalPlan(
        train_end="2026-01-02T23:59:59Z",
        validation_end="2026-01-04T23:59:59Z",
        test_end=now + timedelta(hours=1),
    )
    design = RepetitionDesign(
        schema_version=1,
        fold_id="owned-ci-mechanism-v1",
        data_fingerprint=fingerprint(data),
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        plan=plan,
        warmup_end="2026-01-01T00:00:00Z",
        budget=Quantity(value=Decimal("0.5"), semantics=BUDGET_RATIO),
        seeds=(11, 42),
        builds=tuple(members),
        raw_metric="verdict_napfd",
        raw_semantics=UnitSemantics(unit="1", meaning="verdict-napfd-full-universe-v1"),
        cost_projection="exact_integer_seconds_v1",
    )
    return design


def _native_specs(root, data, design):
    """Freeze actual native settings and source-bound checkpoint identities."""
    specs = {}
    for policy in ("Random", "UCB"):
        for seed in design.seeds:
            specs[policy, seed] = RunSpec.model_validate(
                {
                    "execution": {"independent_executions": 1, "parallel_pool_size": 1, "verbose": False, "seed": seed},
                    "experiment": {
                        "datasets_dir": str(root / "inputs"),
                        "datasets": sorted(data),
                        "policies": [policy],
                        "rewards": ["RNFail"],
                        "budget": {"mode": "ratio", "values": [float(design.budget.value)]},
                    },
                    "results": {"out_dir": str(root / "native" / policy / str(seed)), "top_k_prioritization": 3},
                    "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
                    "extensions": {"corpus_source": {"data_fingerprint": design.data_fingerprint}},
                }
            )
    return specs


def _checked_build(member_id, source, metadata, raw, design):
    """Reject actual native metric disagreement against the independent rational reference."""
    build = PilotBuild(
        build_id=member_id,
        decided_at=metadata.event_at,
        outcomes_available_at=metadata.available_at,
        tests=tuple(
            PilotTest(
                test_id=row["Name"],
                duration=_quantity(row["Duration"]),
                outcome="failed" if row["Verdict"] else "passed",
            )
            for row in source
        ),
        baseline_order=tuple(row["Name"] for row in source),
    )
    order = tuple(json.loads(raw["prioritization_order_top_k"]))
    native_budget = sum((test.duration.value for test in build.tests), start=Decimal(0)) * design.budget.value
    reference = tiny_ci_reference(build, order, _quantity(native_budget))
    if reference.napfd.value is None or reference.apfdc.value is None:
        raise ValueError("This complete owned failing fixture requires defined metric references")
    if not math.isclose(raw["fitness"], float(reference.napfd.value), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Actual native NAPFD differs from independent rational reference")
    if not math.isclose(raw["cost"], float(reference.apfdc.value), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Actual native APFDc differs from independent rational reference")
    return build


def _native_rows(specs, data, design, frozen):
    """Adapt verified native rows once into typed records retaining source and analysis identity."""
    metadata_by_build = {member.member_id: member.metadata for member in design.builds}
    rows = []
    for (policy, seed), spec in specs.items():
        native = run(spec)
        table = pl.read_parquet(str(Path(native.artifacts_dir) / "results/**/*.parquet"), hive_partitioning=True)
        for raw in table.iter_rows(named=True):
            project, step = raw["scenario"], raw["step"]
            source = [row for row in data[project] if row["BuildId"] == step]
            metadata = metadata_by_build[f"{project}/build-{step}"]
            build = _checked_build(f"{project}/build-{step}", source, metadata, raw, design)
            rows.append(
                CorpusRepetition(
                    candidate_id=policy,
                    member_id=build.build_id,
                    seed=seed,
                    state="completed",
                    record=DomainRecord(
                        schema_version=1,
                        namespace="ci.corpus.native_projection",
                        run_id=frozen.fingerprint,
                        execution_id=repetition_execution_id(policy, seed),
                        decision_id=build.build_id,
                        data_fingerprint=frozen.data_fingerprint,
                        available_at=datetime.now(UTC),
                        provenance=ResultProvenance(
                            producer="owned-native-corpus-example",
                            producer_version="1",
                            spec_fingerprint=frozen.candidate(policy).configuration_fingerprint,
                            evaluation_fingerprint=frozen.fingerprint,
                            source_revision=fingerprint(raw),
                        ),
                        measurements=(
                            Measurement(
                                name=design.raw_metric,
                                unit="1",
                                meaning=design.raw_semantics.meaning,
                                direction="maximize",
                                status="observed",
                                value=raw["fitness"],
                            ),
                        ),
                    ),
                )
            )
    return tuple(rows)


def main(output):
    """Freeze and reserve protected access before executing native candidates and publishing evidence."""
    root = Path(output).resolve()
    data, members = _write_inputs(root)
    design = _design(data, members)
    plan = design.plan
    specs = _native_specs(root, data, design)
    candidates = tuple(
        FrozenCandidate.from_configuration(
            policy,
            {
                "corpus_repetition_design": design.fingerprint,
                "resolved_runs": [specs[policy, seed].model_dump(mode="json") for seed in design.seeds],
            },
        )
        for policy in ("Random", "UCB")
    )
    frozen = freeze(
        candidates,
        design.data_fingerprint,
        plan,
        EvaluationProtocol(
            protocol_id="owned-ci-unit-means-v1",
            holdout_id="owned-ci-mechanism-holdout-v1",
            metric_name=design.raw_metric,
            unit="1",
            meaning=design.raw_semantics.meaning + MEAN_SUFFIX,
            direction="maximize",
            independent_unit=design.independent_unit,
            dependence="independent",
            normal_differences_assumed=True,
            practical_effect=0.05,
            minimum_units=20,
        ),
        (ComparisonHypothesis(hypothesis_id="UCB-Random", candidate_id="UCB", baseline_id="Random"),),
    )
    audit = EvaluationAudit(root / "protected")
    audit.save_freeze(frozen)
    writer = ArtifactWriter(root / "protected")
    published = []

    def load_native():
        report = RepeatedCorpusReport(
            schema_version=1,
            design=design,
            frozen=frozen,
            rows=_native_rows(specs, data, design, frozen),
            assembled_at=datetime.now(UTC),
        )
        published.append(publish_repeated_corpus(report, writer))
        return report.paired_observations()

    comparison = evaluate(frozen, audit, load_native, actor="installed-owned-ci-qualification")
    comparison_path = audit.save_report(comparison)
    recovered = RepeatedCorpusReport.from_json(published[0].read_bytes())
    retried = evaluate(frozen, audit, recovered.paired_observations, actor="installed-owned-ci-retry")
    print(
        json.dumps(
            {
                "raw_cells": len(recovered.rows),
                "independent_units": len(comparison.results[0].independent_units),
                "conclusion": comparison.results[0].conclusion,
                "retry_confirmatory": retried.confirmatory_eligible,
                "raw_evidence": str(published[0]),
                "comparison": str(comparison_path),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
