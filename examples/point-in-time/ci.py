"""Run real native TCPCI with frozen point-in-time means and conservative base-order fallback.

Usage: python ci.py OUTPUT_DIRECTORY. Owned synthetic historical quantities and
three constructed build copies establish functional integration, not operational
savings or independent/prospective evidence. The same owned output can resume
through native checkpoints; unknown roots and existing symlinks are rejected.
"""

import hashlib
import itertools
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl

from coleman.api import run_with_extension
from coleman.artifacts import ArtifactWriter
from coleman.evaluation.protected import TemporalObservation, TemporalPlan
from coleman.observations import (
    EVENT_AGE_SECONDS,
    ObservationHistory,
    ObservationRevision,
    OfflineObservationProvider,
    QuantityMeanModel,
    SignalRequest,
    fit_quantity_mean,
    prepare_training,
    publish_model,
    publish_signal,
)
from coleman.policy.base import Policy
from coleman.results.domain import Fingerprint, Measurement, canonical_json, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.selection import Quantity, UnitSemantics
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector

BASE_ORDER = ("t1", "t2", "t3")
SECONDS = UnitSemantics(unit="s", meaning="test-runner-occupancy-v1")


class DemoConfiguration(VersionedContract):
    """Typed native-extension boundary; all policy parameters enter native run identity."""

    model: QuantityMeanModel
    maximum_event_age: Quantity


class DemoInputs(VersionedContract):
    """Resume proof for owned inputs; changed source/model/config/Parquet bytes fail closed."""

    source_fingerprint: Fingerprint
    configuration: DemoConfiguration
    parquet_sha256: Fingerprint


def owned_history():
    """Declare owned historical durations and a correction received after training/one decision."""
    rows = []
    for name, duration in (("t1", 4.0), ("t2", 1.0), ("t3", 2.0)):
        rows.append(
            ObservationRevision(
                schema_version=1,
                metadata=TemporalObservation.model_validate(
                    {
                        "observation_id": "historical-build:" + name,
                        "group_id": "historical-build",
                        "revision_id": "r1",
                        "event_at": "2025-12-29T10:00:00Z",
                        "available_at": "2025-12-29T10:01:00Z",
                    }
                ),
                entity_id=name,
                received_at=datetime(2025, 12, 29, 10, 1, tzinfo=UTC),
                supersedes=None,
                measurement=Measurement(
                    name="ci_duration",
                    value=duration,
                    unit=SECONDS.unit,
                    meaning=SECONDS.meaning,
                    direction="none",
                    status="observed",
                    minimum=0,
                ),
            )
        )
    original = rows[1]
    rows.append(
        ObservationRevision.model_validate(
            {
                **original.model_dump(),
                "supersedes": "r1",
                "received_at": "2026-01-02T10:05:00Z",
                "metadata": {
                    **original.metadata.model_dump(),
                    "revision_id": "r2",
                    "available_at": "2026-01-02T10:05:00Z",
                },
                "measurement": {**original.measurement.model_dump(), "value": 99.0},
            }
        )
    )
    return ObservationHistory(
        schema_version=1, source_id="owned-synthetic-ci-duration-v1", source_version="1", revisions=tuple(rows)
    )


def frozen_model(source):
    """Fit only eligible predeclared training; the later correction never refits this model."""
    plan = TemporalPlan.model_validate(
        {
            "train_end": "2025-12-31T23:59:59Z",
            "validation_end": "2026-01-01T23:59:59Z",
            "test_end": "2026-01-04T23:59:59Z",
        }
    )
    training = prepare_training(
        source,
        plan,
        information_cutoff=plan.train_end,
        measurement_name="ci_duration",
        input_semantics=SECONDS,
    )
    return fit_quantity_mean(training, frozen_at=datetime(2026, 1, 1, tzinfo=UTC))


def signals_for_step(configuration, step):
    """Use an explicit synthetic UTC calendar; no host clock or realized native outcomes enter signals."""
    decision = datetime(2026, 1, 1, 10, tzinfo=UTC) + timedelta(days=step - 1)
    return tuple(
        configuration.model.signal(
            SignalRequest(
                schema_version=1,
                decision_id=f"build-{step}",
                entity_id=name,
                decided_at=decision,
                information_cutoff=decision,
                target_start=decision,
                target_end=decision + timedelta(minutes=1),
            ),
            maximum_event_age=configuration.maximum_event_age,
        )
        for name in BASE_ORDER
    )


def declared_order(signals):
    """Preserve all tests; any unavailable/stale estimate falls back to the declared full base order."""
    if any(signal.availability != "available" for signal in signals):
        return list(BASE_ORDER)
    return [
        signal.request.entity_id
        for signal in sorted(signals, key=lambda signal: (signal.measurement.value, signal.request.entity_id))
    ]


class MeanOrder(Policy):
    """Simple estimate-based baseline; reads only logical names and eligible frozen history."""

    def __init__(self, configuration):
        self.configuration = configuration

    def __str__(self):
        return "FrozenHistoricalMean"

    def choose_all(self, agent):
        if set(agent.actions["Name"].to_list()) != set(BASE_ORDER):
            raise ValueError("Owned fixture test universe changed")
        return declared_order(signals_for_step(self.configuration, agent.t + 1))


class SignalMonitor(MonitorCollector):
    """Publish decision signals/domain sidecars before the native checkpoint can advance."""

    def __init__(self, sink, configuration, root, run_id):
        super().__init__(sink)
        self.configuration = configuration
        self.writer = ArtifactWriter(root)
        self.run_id = run_id
        publish_model(configuration.model, self.writer)

    def collect(self, params):
        if params.t > 1:  # Native first-build warmup deliberately retains its existing initial shuffle.
            signals = signals_for_step(self.configuration, params.t)
            if list(params.prioritization_order) != declared_order(signals):
                raise ValueError("Actual native prioritization disagrees with frozen point-in-time signals")
            for signal in signals:
                publish_signal(signal, self.writer)
            logical_execution = fingerprint(
                {
                    "dataset": params.scenario_provider.name,
                    "policy": params.policy,
                    "reward": params.reward_function,
                    "experiment": params.experiment,
                }
            )
            records = tuple(
                signal.to_domain_record(
                    run_id=self.run_id,
                    execution_id=logical_execution,
                    evaluation_fingerprint=self.configuration.fingerprint,
                )
                for signal in signals
            )
            params = replace(params, domain_records=(*params.domain_records, *records))
        super().collect(params)


def build_demo_environment(config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None):
    """Reuse the native environment; consume one typed extension value at the boundary."""
    env, builds = build_environment(cast(EnvironmentBuildConfig, replace(config, extension=None)), metadata, agent_seed)
    configuration = DemoConfiguration.from_json(config.extensions["point_in_time_demo"])
    if config.run_id is None:
        raise ValueError("Missing native run identity")
    env.agents[0].policy = MeanOrder(configuration)
    env.monitor = SignalMonitor(
        env.monitor.sink, configuration, Path(config.results_config["out_dir"]).parent, config.run_id
    )
    return env, builds


def workspace_root(output):
    """Allow fresh or owned-resumable local roots; reject escapes, symlinks and unbounded trees before writes."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must remain within current workspace")
    if root.exists():
        paths = tuple(itertools.islice(root.rglob("*"), 10001))
        if len(paths) > 10000 or any(path.is_symlink() for path in paths):
            raise ValueError("Example resume requires a bounded tree without symlinks")
        if not all((root / filename).is_file() for filename in ("example-inputs.json", "native-spec.json")):
            raise ValueError("Resume requires the owned example input proof")
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_case(output, maximum_event_age):
    """Run/resume three constructed builds using the existing native checkpoint store.

    Only trusted local output/checkpoints produced by this example may be resumed.
    The proof binds all inputs/configuration before native checkpoint loading. A
    changed source, policy parameter or materialized Parquet byte fails closed.
    """
    root = workspace_root(output)
    source = OfflineObservationProvider(owned_history()).load_history()
    configuration = DemoConfiguration(
        schema_version=1,
        model=frozen_model(source),
        maximum_event_age=Quantity(value=Decimal(maximum_event_age), semantics=EVENT_AGE_SECONDS),
    )
    expected = pl.DataFrame(
        [
            {
                "BuildId": step,
                "Name": name,
                "Duration": float(seconds),
                "CalcPrio": 0,
                "LastRun": "2026-01-01",
                "Verdict": int(name != "t1"),
            }
            for step in (1, 2, 3)
            for name, seconds in (("t1", 1), ("t2", 2), ("t3", 4))
        ]
    )
    directory = root / "inputs" / "constructed-ci"
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "features-engineered.parquet"
    if data.exists():
        if not pl.read_parquet(data).equals(expected):
            raise ValueError("Owned native input observations changed")
    else:
        expected.write_parquet(data)
    inputs = DemoInputs(
        schema_version=1,
        source_fingerprint=source.fingerprint,
        configuration=configuration,
        parquet_sha256="sha256:" + hashlib.sha256(data.read_bytes()).hexdigest(),
    )
    writer = ArtifactWriter(root)
    writer.write_immutable(root / "example-inputs.json", inputs.to_json() + "\n")
    writer.write_immutable(root / "source-history.json", source.to_json() + "\n")
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["constructed-ci"],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [7]},
            },
            "extensions": {"point_in_time_demo": configuration.to_json(), "owned_inputs": inputs.fingerprint},
            "results": {"out_dir": str(root / "native"), "manifest_enabled": True, "top_k_prioritization": 3},
            "checkpoint": {"enabled": True, "interval": 1, "base_dir": str(root / "checkpoints")},
        }
    )
    writer.write_immutable(root / "native-spec.json", canonical_json(spec.model_dump(mode="json")) + "\n")
    return run_with_extension(spec, RunnerExtension(build_environment_fn=build_demo_environment))


def main(output):
    """Run fresh-estimate and stale/fallback cases, reporting functional evidence only."""
    for name, age in (("fresh", 604800), ("stale", 0)):
        result = run_case(Path(output) / name, age)
        root = Path(result.artifacts_dir)
        signals = list(root.rglob("decision-signals/*.json"))
        if len(signals) != 6:
            raise ValueError("Expected six logical signals after the native warmup")
        print(
            json.dumps(
                {
                    "case": name,
                    "run_id": result.run_id,
                    "artifacts_dir": result.artifacts_dir,
                    "signals": len(signals),
                    "evidence": "synthetic_functional_only",
                }
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
