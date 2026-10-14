"""Native TCPCI finite-coverage fixture using installed COLEMAN extension interfaces.

Run python ci.py OUTPUT_DIRECTORY. Two synthetic builds retain the native initial
shuffle on build 1 and exercise selection on build 2. The selected prefix consumes
the entire fixed-time budget in this fixture; this ranking adapter is deliberately
fixture-specific and does not claim to enforce arbitrary subset constraints.
"""

import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl

from coleman.api import run_with_extension
from coleman.artifacts import ArtifactWriter
from coleman.policy.base import Policy
from coleman.results.domain import fingerprint
from coleman.results.parquet_sink import ParquetSink
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.selection import (
    CoverageCandidate,
    CoverageConstraints,
    CoverageOutcome,
    CoverageProblem,
    CoverageResult,
    Quantity,
    SearchLimits,
    UnitSemantics,
    coverage_record,
    publish_coverage_result,
    solve_coverage,
)
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector


class FixtureCoveragePolicy(Policy):
    """Turn the fully budget-consuming fixture choice into a native full ranking."""

    def __init__(self, problem, method):
        self.problem = problem
        self.method = method
        self.last_result = None

    def __str__(self):
        return "FiniteCoverageFixture"

    def choose_all(self, agent):
        self.last_result = solve_coverage(self.problem, method=self.method, limits=SearchLimits(deadline_seconds=None))
        selected = self.last_result.selected_ids
        if selected is None:
            raise ValueError("No feasible fixture incumbent")
        actual = set(agent.actions["Name"].to_list())
        if actual != set(self.problem.search_ids):
            raise ValueError("Fixture action universe changed")
        if self.last_result.cost != self.problem.constraints.budget:
            raise ValueError("This fixture adapter requires a fully consumed fixed-time budget")
        return [*selected, *sorted(actual - set(selected))]


class CoverageMonitor(MonitorCollector):
    """Verify actual native scheduling before publishing the generic coverage result."""

    def __init__(self, sink, policy, root, run_id):
        super().__init__(sink)
        self.policy = policy
        self.root = root
        self.run_id = run_id

    def collect(self, params):
        if params.t == 2:
            result = self.policy.last_result
            if result is None or tuple(params.metric.scheduled_testcases) != result.selected_ids:
                raise ValueError("Native scheduling differs from the planned fixture prefix")
            publish_coverage_result(result, ArtifactWriter(self.root))
            existing = self.sink.read_domain_record(
                self.run_id, result.scope_fingerprint, result.fingerprint, namespace="decision.coverage"
            )
            available_at = existing.available_at if existing is not None else datetime.now(UTC)
            record = coverage_record(result, run_id=self.run_id, available_at=available_at)
            params = replace(params, domain_records=(record,))
        super().collect(params)


def build_fixture_environment(config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None):
    """Keep native agents/scenarios/rewards/metrics/sinks; replace only the fixture policy."""
    native = cast(EnvironmentBuildConfig, replace(config, extension=None))
    env, builds = build_environment(native, metadata, agent_seed)
    if config.run_id is None:
        raise ValueError("Native run identity is required")
    problem = CoverageProblem.model_validate(config.extensions["finite_coverage"])
    policy = FixtureCoveragePolicy(problem, config.extensions["coverage_method"])
    env.agents[0].policy = policy
    root = Path(config.results_config["out_dir"]).parent
    env.monitor = CoverageMonitor(env.monitor.sink, policy, root, config.run_id)
    return env, builds


def main(output):
    """Run both finite solvers through the real native CI loop on overlapping capabilities."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must be within the current working directory")
    cost = UnitSemantics(unit="s", meaning="modeled-serial-test-duration-v1")
    weight = UnitSemantics(unit="1", meaning="weighted-test-capability-v1")
    definitions = [("A", 2, ("x", "y")), ("B", 2, ("y", "z")), ("C", 1, ("z",))]
    request = CoverageProblem(
        schema_version=1,
        data_fingerprint=fingerprint(
            {
                "fixture": "overlapping-ci-capabilities-v1",
                "candidates": [
                    {"id": name, "cost": str(cost), "covers": list(covers)} for name, cost, covers in definitions
                ],
                "weights": {"x": "0.5", "y": "0.3", "z": "0.2"},
            }
        ),
        objective_semantics=weight,
        candidates=tuple(
            CoverageCandidate(candidate_id=name, cost=Quantity(value=Decimal(duration), semantics=cost), covers=covers)
            for name, duration, covers in definitions
        ),
        outcomes=tuple(
            CoverageOutcome(outcome_id=name, weight=Quantity(value=Decimal(value), semantics=weight))
            for name, value in (("x", "0.5"), ("y", "0.3"), ("z", "0.2"))
        ),
        constraints=CoverageConstraints(budget=Quantity(value=Decimal(3), semantics=cost)),
    )
    directory = root / "inputs" / "synthetic_ci"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "BuildId": build,
            "Name": name,
            "Duration": float(duration),
            "CalcPrio": 0,
            "LastRun": "2026-01-01",
            "Verdict": int(name != "B"),
        }
        for build in (1, 2)
        for name, duration, _ in definitions
    ]
    pl.DataFrame(rows).write_parquet(directory / "features-engineered.parquet")
    exact_result = None
    for method in ("greedy_density_v1", "branch_bound_v1"):
        spec = RunSpec.model_validate(
            {
                "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
                "experiment": {
                    "datasets_dir": str(root / "inputs"),
                    "datasets": ["synthetic_ci"],
                    "policies": ["Greedy"],
                    "rewards": ["RNFail"],
                    "budget": {"mode": "fixed_time", "values": [3]},
                },
                "extensions": {
                    "finite_coverage": json.loads(request.to_json()),
                    "coverage_method": method,
                    "coverage_adapter": "native-full-budget-fixture-v1",
                },
                "results": {"out_dir": str(root / method), "manifest_enabled": True},
            }
        )
        native = run_with_extension(spec, RunnerExtension(build_environment_fn=build_fixture_environment))
        sink = ParquetSink(out_dir=str(Path(native.artifacts_dir) / "results"))
        try:
            records = list(sink.domain_records())
            if len(records) != 1 or records[0].measurement("selection_cost").value != 3:
                raise ValueError("Expected one audited real native selection")
        finally:
            sink.close()
        paths = list((Path(native.artifacts_dir) / "coverage-results").glob("*.json"))
        selected = CoverageResult.from_json(paths[0].read_bytes())
        if method == "branch_bound_v1":
            exact_result = selected
    if exact_result is None:
        raise ValueError("Exact fixture did not execute")
    print(
        json.dumps(
            {
                "selected_ids": exact_result.selected_ids,
                "status": exact_result.status,
                "native_scheduled": list(exact_result.selected_ids),
                "objective": str(exact_result.objective.value),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
