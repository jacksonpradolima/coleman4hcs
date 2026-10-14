"""Select a synthetic CI resource plan, then execute its time budget through the actual native runner.

Modeled credits are kept separate from measured NAPFD/APFDc and learning rewards.
Run python ci.py OUTPUT_DIRECTORY using the installed base package.
"""

import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl

from coleman.allocation import (
    AllocationLimits,
    AllocationProblem,
    AllocationResult,
    CurrentAllocation,
    DiscreteAllocation,
    JointScenario,
    NamedAmount,
    ProbabilityEvidence,
    ResourceCapacity,
    RiskConstraints,
    ScenarioPayoff,
    SignedPayoff,
    allocation_record,
    publish_allocation_result,
    select_allocation,
)
from coleman.api import run_with_extension
from coleman.artifacts import ArtifactWriter
from coleman.results.domain import fingerprint
from coleman.results.parquet_sink import ParquetSink
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.selection import Quantity, UnitSemantics
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector


class PlanMonitor(MonitorCollector):
    """Commit the modeled plan only after verifying native execution respects its resource budget."""

    def __init__(self, sink, result, available_at, root, run_id):
        super().__init__(sink)
        self.result = result
        self.available_at = available_at
        self.root = root
        self.run_id = run_id

    def collect(self, params):
        if params.available_time != 3 or set(params.metric.scheduled_testcases) != {"unit-test", "integration-test"}:
            raise ValueError("Native execution differs from the declared synthetic resource plan")
        publish_allocation_result(self.result, ArtifactWriter(self.root))
        existing = self.sink.read_domain_record(
            self.run_id,
            self.result.problem_fingerprint,
            self.result.fingerprint,
            namespace="decision.scenario_allocation",
        )
        available = existing.available_at if existing is not None else self.available_at
        record = allocation_record(self.result, run_id=self.run_id, available_at=available)
        super().collect(replace(params, domain_records=(record,)))


@dataclass(frozen=True)
class NativeBuilder:
    """Picklable additive collector callback; native scheduling, rewards and metrics are unchanged."""

    result: AllocationResult
    available_at: datetime

    def __call__(self, config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None):
        native = cast(EnvironmentBuildConfig, replace(config, extension=None))
        env, builds = build_environment(native, metadata, agent_seed)
        if config.run_id is None:
            raise ValueError("Native run identity is required")
        env.monitor = PlanMonitor(
            env.monitor.sink,
            self.result,
            self.available_at,
            Path(config.results_config["out_dir"]).parent,
            config.run_id,
        )
        return env, builds


def build_problem():
    """Freeze joint modeled payoff scenarios and complete runner-time alternatives."""
    credit = UnitSemantics(unit="credit", meaning="synthetic-ci-credit-accounting-v1")
    seconds = UnitSemantics(unit="s", meaning="modeled-serial-ci-resource-time-v1")
    gross = {"no_action": [0, 0], "stable": [2, 2], "variable": [10, -8]}
    durations = {"no_action": 0, "stable": 3, "variable": 4}
    return AllocationProblem(
        schema_version=1,
        data_fingerprint=fingerprint({"gross": gross, "duration": durations, "probabilities": ["0.5", "0.5"]}),
        value_semantics=credit,
        initial_balance=Quantity(value=Decimal(10), semantics=credit),
        current=CurrentAllocation(
            holdings=(),
            resource_use=(NamedAmount(name="runner_time", amount=Quantity(value=Decimal(0), semantics=seconds)),),
            reserved_balance=Quantity(value=Decimal(0), semantics=credit),
        ),
        resources=(ResourceCapacity(name="runner_time", capacity=Quantity(value=Decimal(5), semantics=seconds)),),
        holding_definitions=(),
        scenarios=tuple(JointScenario(scenario_id=name, probability=Decimal("0.5")) for name in ("idle", "busy")),
        probability_evidence=ProbabilityEvidence(kind="assumed"),
        no_action_id="no_action",
        risk=RiskConstraints(downside_target=SignedPayoff(value=Decimal(0), semantics=credit)),
        actions=tuple(
            DiscreteAllocation(
                action_id=name,
                holdings=(),
                resource_use=(
                    NamedAmount(name="runner_time", amount=Quantity(value=Decimal(durations[name]), semantics=seconds)),
                ),
                capital_required=Quantity(value=Decimal(0), semantics=credit),
                transaction_cost=Quantity(value=Decimal(0), semantics=credit),
                gross_payoffs=tuple(
                    ScenarioPayoff(scenario_id=scenario, gross=SignedPayoff(value=Decimal(value), semantics=credit))
                    for scenario, value in zip(("idle", "busy"), values, strict=True)
                ),
            )
            for name, values in gross.items()
        ),
    )


def main(output):
    """Model a plan before native CI execution; never equate modeled credits with measured efficacy."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must be within the current working directory")
    request = build_problem()
    limits = AllocationLimits(deadline_seconds=None)
    result = select_allocation(request, limits=limits)
    available_at = datetime.now(UTC)
    if result.choice.selected_action_id is None or result.choice.status != "optimal":
        raise ValueError("Synthetic fixture has no completed declared-model optimum")
    selected = request.action(result.choice.selected_action_id)
    duration = selected.resource_use[0].amount.value
    dataset = root / "inputs" / "synthetic_ci"
    dataset.mkdir(parents=True, exist_ok=True)
    rows = [
        {"BuildId": 1, "Name": name, "Duration": time, "CalcPrio": 0, "LastRun": "2026-01-01", "Verdict": verdict}
        for name, time, verdict in (("unit-test", 1.0, 0), ("integration-test", 2.0, 1))
    ]
    pl.DataFrame(rows).write_parquet(dataset / "features-engineered.parquet")
    spec = RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": ["synthetic_ci"],
                "policies": ["Greedy"],
                "rewards": ["RNFail"],
                "budget": {"mode": "fixed_time", "values": [float(duration)]},
            },
            "extensions": {
                "allocation_problem": json.loads(request.to_json()),
                "allocation_limits": limits.model_dump(mode="json"),
                "allocation_result": result.fingerprint,
                "adapter": "synthetic-resource-plan-native-ci-v1",
            },
            "results": {"out_dir": str(root / "native"), "manifest_enabled": True},
        }
    )
    native = run_with_extension(spec, RunnerExtension(build_environment_fn=NativeBuilder(result, available_at)))
    sink = ParquetSink(out_dir=str(Path(native.artifacts_dir) / "results"))
    try:
        records = list(sink.domain_records())
        if len(records) != 1 or records[0].measurement("expected_net_payoff").value != 2:
            raise ValueError("Expected one separate modeled plan record alongside native CI measurements")
    finally:
        sink.close()
    print(
        json.dumps(
            {
                "selected_action_id": result.choice.selected_action_id,
                "status": result.choice.status,
                "native_budget_seconds": float(duration),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
