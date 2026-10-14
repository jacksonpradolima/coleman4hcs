"""Offline protected comparison of native TCPCI baselines, through the installed package.

Run: python ci.py ./output. Synthetic repositories test integration and conservative
insufficient-evidence behavior; they establish no production algorithm superiority.
"""

import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from coleman.api import run_with_extension
from coleman.evaluation.protected import (
    ComparisonHypothesis,
    EvaluationAudit,
    EvaluationProtocol,
    FrozenCandidate,
    FrozenEvaluation,
    PairedObservation,
    TemporalObservation,
    TemporalPlan,
    evaluate,
    freeze,
    named_seed,
)
from coleman.manifest import generate_manifest
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.results.parquet_sink import ParquetSink
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector


class ProjectMonitor(MonitorCollector):
    """Commit the native raw metric without reinterpreting its learning reward."""

    def __init__(self, sink, frozen, candidate_id, run_id):
        super().__init__(sink)
        self.frozen = frozen
        self.candidate_id = candidate_id
        self.run_id = run_id

    def collect(self, params):
        record = DomainRecord(
            schema_version=1,
            namespace="ci.independent_projects",
            run_id=self.run_id,
            execution_id=params.scenario_provider.name,
            decision_id="build-1",
            data_fingerprint=self.frozen.data_fingerprint,
            available_at=datetime(2026, 1, 5, 12, tzinfo=UTC),
            provenance=ResultProvenance(
                producer="native-ci-comparison-example",
                producer_version="1",
                spec_fingerprint=self.frozen.candidate(self.candidate_id).configuration_fingerprint,
                evaluation_fingerprint=self.frozen.fingerprint,
            ),
            measurements=(
                Measurement(
                    name="napfd",
                    value=float(params.metric.fitness),
                    unit="1",
                    meaning="napfd-verdict-v1",
                    direction="maximize",
                    status="observed",
                    minimum=0.0,
                ),
            ),
        )
        super().collect(replace(params, domain_records=(record,)))


@dataclass(frozen=True)
class NativeBuilder:
    """Picklable runner callback carrying a pre-outcome frozen evaluation."""

    frozen: FrozenEvaluation

    def __call__(self, config: EnvironmentBuildConfig, runtime_metadata: dict[str, str], agent_seed: int | None):
        native = cast(EnvironmentBuildConfig, replace(config, extension=None))
        project_seed = named_seed(
            config.seed if config.seed is not None else 42, config.dataset, "native-policy-execution"
        )
        env, builds = build_environment(native, runtime_metadata, project_seed)
        if config.run_id is None:
            raise ValueError("Native run identity is required")
        env.monitor = ProjectMonitor(env.monitor.sink, self.frozen, config.policy_names[0], config.run_id)
        return env, builds


def main(output):
    """Freeze fixed baselines, produce native CI records, then audit/report comparisons."""
    root = Path(output).resolve()
    inputs = root / "inputs"
    names = [f"synthetic_project_{index}" for index in range(4)]
    corpus = {}
    for index, name in enumerate(names):
        rows = [
            {
                "BuildId": 1,
                "Name": test,
                "Duration": duration,
                "CalcPrio": 0,
                "LastRun": "2026-01-05",
                "Verdict": verdict,
            }
            for test, duration, verdict in (("unit-tests", 1.0, index % 2), ("integration-tests", 2.0, 1))
        ]
        corpus[name] = rows
        directory = inputs / name
        directory.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(directory / "features-engineered.parquet")
    specs = {}
    for candidate_id in ("Greedy", "Random"):
        specs[candidate_id] = RunSpec.model_validate(
            {
                "execution": {
                    "independent_executions": 1,
                    "parallel_pool_size": 1,
                    "verbose": False,
                    "seed": named_seed(42, candidate_id, "native-policy-execution"),
                },
                "experiment": {
                    "datasets_dir": str(inputs),
                    "datasets": names,
                    "policies": [candidate_id],
                    "rewards": ["RNFail"],
                    "budget": {"mode": "ratio", "values": [0.5]},
                },
                "results": {"out_dir": str(root / "native" / candidate_id)},
            }
        )
    temporal = TemporalPlan(
        train_end="2026-01-02T23:59:59Z", validation_end="2026-01-04T23:59:59Z", test_end="2026-01-06T23:59:59Z"
    )
    protocol = EvaluationProtocol(
        protocol_id="synthetic-ci-paired-v1",
        holdout_id="synthetic-ci-projects-test-v1",
        metric_name="napfd",
        unit="1",
        meaning="napfd-verdict-v1",
        direction="maximize",
        independent_unit="synthetic-project",
        dependence="independent",
        normal_differences_assumed=True,
        practical_effect=0.05,
        minimum_units=20,
    )
    frozen = freeze(
        tuple(FrozenCandidate.from_configuration(name, spec.model_dump(mode="json")) for name, spec in specs.items()),
        fingerprint(corpus),
        temporal,
        protocol,
        (ComparisonHypothesis(hypothesis_id="Greedy-Random", candidate_id="Greedy", baseline_id="Random"),),
    )
    audit = EvaluationAudit(root / "campaign")
    audit.save_freeze(frozen)  # persist before executing either fixed candidate
    native_results = {}
    for name, spec in specs.items():
        native_results[name] = run_with_extension(spec, RunnerExtension(build_environment_fn=NativeBuilder(frozen)))
        saved = json.loads((Path(native_results[name].artifacts_dir) / "spec.resolved.json").read_text())
        assert fingerprint(saved) == frozen.candidate(name).configuration_fingerprint

    def load_outcomes():
        records = []
        for candidate_id, result in native_results.items():
            sink = ParquetSink(out_dir=str(Path(result.artifacts_dir) / "results"))
            try:
                for record in sink.domain_records():
                    metadata = TemporalObservation(
                        observation_id=record.decision_id + ":" + record.execution_id,
                        group_id=record.execution_id,
                        event_at=record.available_at,
                        available_at=record.available_at,
                    )
                    records.append(PairedObservation(candidate_id=candidate_id, metadata=metadata, record=record))
            finally:
                sink.close()
        return tuple(records)

    report = evaluate(frozen, audit, load_outcomes, actor="native-ci-example")
    artifact = audit.save_report(report)
    sink = ParquetSink(out_dir=str(root / "campaign" / "results"))
    try:
        for record in report.domain_records("synthetic-ci-campaign"):
            sink.write_domain_record(record)
    finally:
        sink.close()
    repeated = evaluate(frozen, audit, load_outcomes, actor="native-ci-example")
    assert repeated.access.prior_test_accesses >= 1
    assert report.results[0].conclusion == "insufficient_evidence"
    assert len(report.results[0].independent_units) == 4
    manifest = generate_manifest(root / "campaign", run_id="synthetic-ci-campaign")
    print(
        json.dumps(
            {
                "result": report.results[0].conclusion,
                "independent_units": 4,
                "repeat_is_confirmatory": repeated.confirmatory_eligible,
                "report": str(artifact),
                "manifest": str(manifest),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
