"""Offline TCPCI example using the installed package and native runner extension.

Run: python tcpci.py OUTPUT_DIRECTORY. No external dataset or service is needed.
The synthetic build labels are available at declared UTC times, never wall time.
"""

import json
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl

from coleman.api import run_with_extension
from coleman.results.domain import DomainRecord, LearningReward, Measurement, ResultProvenance, fingerprint
from coleman.results.parquet_sink import ParquetSink
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector


class ObservationMonitor(MonitorCollector):
    """Add raw observations without replacing scheduling or reward computation."""

    def __init__(self, sink, run_id, provenance, data_fingerprint):
        super().__init__(sink)
        self.run_id = run_id
        self.provenance = provenance
        self.data_fingerprint = data_fingerprint

    def collect(self, params):
        # Native rows can be joined by reconstructing these keys. Physical worker
        # execution UUIDs are deliberately excluded from retry-stable identity.
        logical_execution = fingerprint(
            [params.scenario_provider.name, str(params.budget_mode), params.budget_value, params.experiment]
        )
        decision = fingerprint([params.t, params.policy, params.reward_function, params.variant])
        record = DomainRecord(
            schema_version=1,
            namespace="ci.observations",
            run_id=self.run_id,
            execution_id=logical_execution,
            decision_id=decision,
            data_fingerprint=self.data_fingerprint,
            available_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=params.t),
            provenance=self.provenance,
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
                Measurement(
                    name="apfdc",
                    value=float(params.metric.cost),
                    unit="1",
                    meaning="apfdc-verdict-v1",
                    direction="maximize",
                    status="observed",
                    minimum=0.0,
                ),
                Measurement(
                    name="suite_duration",
                    value=params.total_build_duration,
                    unit="s",
                    meaning="full-suite-serial-duration-v1",
                    direction="minimize",
                    status="observed",
                    minimum=0.0,
                ),
            ),
            learning_reward=LearningReward(value=float(params.rewards), unit="1", transform_id="RNFail-mean-v1"),
        )
        super().collect(replace(params, domain_records=(record,)))


def build_observed_environment(
    config: EnvironmentBuildConfig, runtime_metadata: dict[str, str], agent_seed: int | None
):
    """Use the native builder, replacing only the additive result collector."""
    native_config = cast(EnvironmentBuildConfig, replace(config, extension=None))
    env, max_builds = build_environment(native_config, runtime_metadata, agent_seed=agent_seed)
    spec = config.resolved_spec
    if spec is None or config.run_id is None:
        raise ValueError("This example requires run_with_extension's resolved spec")
    inputs = pl.read_parquet(Path(config.datasets_dir) / config.dataset / "features-engineered.parquet")
    env.monitor = ObservationMonitor(
        env.monitor.sink,
        config.run_id,
        ResultProvenance(
            producer="coleman-tcpci-example",
            producer_version="1",
            spec_fingerprint=fingerprint(
                json.loads((Path(config.results_config["out_dir"]).parent / "spec.resolved.json").read_text())
            ),
            evaluation_fingerprint=fingerprint(
                {
                    "protocol": "synthetic-full-feedback-v1",
                    "budget_mode": str(config.budget_mode),
                    "budget_value": config.budget_value,
                }
            ),
        ),
        fingerprint(inputs.sort(["BuildId", "Name"]).to_dicts()),
    )
    return env, max_builds


def main(output, independent_executions=1, workers=1):
    """Generate three synthetic build events and run native TCPCI."""
    root = Path(output).resolve()
    dataset = root / "inputs" / "synthetic_ci"
    dataset.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "BuildId": build,
            "Name": name,
            "Duration": duration,
            "CalcPrio": 0,
            "LastRun": "2026-01-01",
            "Verdict": verdict,
        }
        for build in (1, 2, 3)
        for name, duration, verdict in (("unit-test", 1.0, 0), ("integration-test", 2.0, build % 2))
    ]
    pl.DataFrame(rows).write_parquet(dataset / "features-engineered.parquet")
    spec = RunSpec.model_validate(
        {
            "execution": {
                "independent_executions": independent_executions,
                "parallel_pool_size": workers,
                "seed": 42,
                "verbose": False,
            },
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["synthetic_ci"],
                "policies": ["Greedy"],
                "rewards": ["RNFail"],
                "budget": {"mode": "ratio", "values": [0.5]},
            },
            "results": {"out_dir": str(root / "runs"), "manifest_enabled": True},
        }
    )
    result = run_with_extension(spec, RunnerExtension(build_environment_fn=build_observed_environment))
    sink = ParquetSink(out_dir=str(Path(result.artifacts_dir) / "results"))
    records = list(sink.domain_records())
    resolved = json.loads((Path(result.artifacts_dir) / "spec.resolved.json").read_text())
    assert len(records) == 3 * independent_executions
    assert all(record.provenance.spec_fingerprint == fingerprint(resolved) for record in records)
    for record in records:
        duration = record.measurement("suite_duration").value
        assert duration is not None
        assert math.isclose(duration, 3.0)
    sink.close()
    print(json.dumps({"run_id": result.run_id, "records": len(records), "spec_traced": True}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
