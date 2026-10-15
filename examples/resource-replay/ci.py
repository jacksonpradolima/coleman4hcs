"""Owned synthetic native CI plus exact counterfactual resource replay.

Run: python ci.py OUTPUT_DIRECTORY. Native CI always executes the full required
suite. Resource selections are historical simulations, not real saved CI work.
Trusted local checkpoints only; concurrent native invocations need separate roots.
"""

import hashlib
import itertools
import json
import sys
from dataclasses import replace
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import cast

import polars as pl

from coleman.api import run_with_extension
from coleman.artifacts import ArtifactWriter
from coleman.evaluation.protected.temporal import TemporalPlan
from coleman.pilot import (
    OVERHEAD_SECONDS,
    RATE,
    SECONDS,
    AdoptionThresholds,
    PilotBuild,
    PilotManifest,
    PilotRun,
    PilotTest,
    serial_events,
)
from coleman.policy.base import Policy
from coleman.resources import (
    RUNNER_MS,
    BuildEstimates,
    ResourceAmount,
    ResourceEstimate,
    ResourceReplayProtocol,
    ResourceReplayReport,
    ResourceReplayRun,
    TestEstimate,
    publish_ledger,
    publish_resource_report,
    publish_resource_run,
    render_resource_report,
)
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, canonical_json, fingerprint
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.selection import Quantity
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector


def quantity(value, semantics):
    """Explicit source boundary for finite decimal seconds/rates."""
    return Quantity(value=Decimal(str(value)), semantics=semantics)


def fixture():
    """Freeze owned synthetic estimates before two protected full-suite observations."""
    builds = tuple(
        PilotBuild.model_validate(
            {
                "build_id": str(step),
                "decided_at": f"2026-01-0{step}T10:00:00Z",
                "outcomes_available_at": f"2026-01-0{step}T10:01:00Z",
                "tests": [
                    PilotTest(test_id=name, duration=quantity(seconds, SECONDS), outcome=outcome)
                    for name, seconds, outcome in (("t1", 1, "passed"), ("t2", 2, "failed"), ("t3", 4, "failed"))
                ],
                "baseline_order": ["t1", "t2", "t3"],
            }
        )
        for step in (2, 3)
    )
    data = [build.model_dump(mode="json") for build in builds]
    source = canonical_json(data) + "\n"
    manifest = PilotManifest.model_validate(
        {
            "schema_version": 1,
            "pilot_id": "owned-resource-ci-v1",
            "source_uri": "urn:coleman:fixture:resource-ci-v1",
            "source_sha256": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
            "license_id": "MIT (COLEMAN repository LICENSE)",
            "dataset_fingerprint": fingerprint(data),
            "evidence_basis": "synthetic_replay",
            "frozen_at": "2026-01-01T00:00:00Z",
            "temporal_plan": TemporalPlan.model_validate(
                {
                    "train_end": "2025-12-30T00:00:00Z",
                    "validation_end": "2026-01-01T00:00:00Z",
                    "test_end": "2026-01-04T00:00:00Z",
                }
            ),
            "candidates": ["base-order", "shortest-estimate"],
            "baseline_id": "base-order",
            "budget": quantity(7, SECONDS),
            "variable_price": quantity("0.01", RATE),
            "thresholds": AdoptionThresholds(
                minimum_builds=2,
                minimum_recall=0.5,
                minimum_runner_saving_fraction=0.0,
                maximum_overhead=quantity(1, OVERHEAD_SECONDS),
            ),
            "builds": builds,
        }
    )
    estimation_source = fingerprint({"fixture": "owned-prior-measurements-v1", "runner_ms": [4000, 1000, 2000]})
    estimates = tuple(
        BuildEstimates(
            build_id=build.build_id,
            tests=tuple(
                TestEstimate(
                    test_id=name,
                    estimate=ResourceEstimate(
                        amount=ResourceAmount(value=ms, semantics=RUNNER_MS),
                        source_fingerprint=estimation_source,
                        observed_at=datetime.fromisoformat("2025-12-29T00:00:00Z"),
                        available_at=datetime.fromisoformat("2025-12-30T00:00:00Z"),
                        valid_until=datetime.fromisoformat("2026-01-04T00:00:00Z"),
                    ),
                )
                for name, ms in (("t1", 4000), ("t2", 1000), ("t3", 2000))
            ),
        )
        for build in builds
    )
    return ResourceReplayProtocol(
        schema_version=1,
        native_manifest=manifest,
        capacity=ResourceAmount(value=5000, semantics=RUNNER_MS),
        estimates=estimates,
    ), source


class EstimateOrder(Policy):
    """Use IDs and frozen prior measurements only; no current durations/verdicts."""

    def __init__(self, protocol, candidate):
        self.protocol = protocol
        self.candidate = candidate

    def __str__(self):
        return "ResourceEstimateOrder"

    def choose_all(self, agent):
        item = self.protocol.estimates[0]  # Owned fixture has the same prior cost vector for both builds.
        ids = sorted(test.test_id for test in item.tests)
        if sorted(agent.actions["Name"].to_list()) != ids:
            raise ValueError("Owned native test universe changed")
        if self.candidate == "base-order":
            return ids
        return [
            test.test_id for test in sorted(item.tests, key=lambda test: (test.estimate.amount.value, test.test_id))
        ]


class ResourceMonitor(MonitorCollector):
    """Commit typed replay and native sidecar before native checkpoint advancement."""

    def __init__(self, sink, protocol, candidate, root, run_id, specification_fingerprint):
        super().__init__(sink)
        self.protocol, self.candidate = protocol, candidate
        self.writer, self.run_id = ArtifactWriter(root), run_id
        self.specification_fingerprint = specification_fingerprint

    def collect(self, params):
        if params.t > 1:
            manifest = self.protocol.native_manifest
            build = next(build for build in manifest.builds if build.build_id == str(params.t))
            ranking = tuple(params.prioritization_order)
            events = serial_events(build, ranking, manifest.budget)
            if tuple(event.test_id for event in events) != tuple(params.metric.scheduled_testcases):
                raise ValueError("Actual native execution differs from full-suite source schedule")
            native = PilotRun(
                candidate_id=self.candidate,
                build_id=build.build_id,
                seed=42,
                ranking=ranking,
                events=events,
                prioritization_overhead=quantity(
                    Decimal(str(params.prioritization_time)).quantize(Decimal("0.000000001"), rounding=ROUND_CEILING),
                    OVERHEAD_SECONDS,
                ),
            )
            replay = ResourceReplayRun(
                schema_version=1,
                protocol=self.protocol,
                native_run=native,
                queue_elapsed=None,
                setup_elapsed=None,
                shared_setup_occupancy=None,
            )
            path = self.writer.run_root / "resource-runs" / f"{replay.logical_id.removeprefix('sha256:')}.json"
            if path.exists():
                previous = ResourceReplayRun.from_json(path.read_bytes())
                if (previous.native_run.ranking, previous.native_run.events) != (native.ranking, native.events):
                    raise ValueError("Native replay conflicts with completed resource observation")
                if previous.protocol.to_json() != self.protocol.to_json():
                    raise ValueError("Completed resource protocol changed")
                replay = previous  # Preserve first committed timing across interrupted native checkpoints.
            publish_resource_run(replay, self.writer)
            publish_ledger(replay.ledger, self.writer)
            summary = replay.accounting()
            logical_execution = fingerprint(
                {
                    "dataset": params.scenario_provider.name,
                    "policy": params.policy,
                    "reward": params.reward_function,
                    "experiment": params.experiment,
                    "candidate": self.candidate,
                }
            )
            record = DomainRecord.model_validate(
                {
                    "schema_version": 1,
                    "namespace": "resource.replay",
                    "run_id": self.run_id,
                    "execution_id": logical_execution,
                    "decision_id": build.build_id,
                    "data_fingerprint": manifest.dataset_fingerprint,
                    "available_at": build.outcomes_available_at,
                    "provenance": ResultProvenance(
                        producer="coleman.resource-ci-example",
                        producer_version="1",
                        spec_fingerprint=self.specification_fingerprint,
                        evaluation_fingerprint=self.protocol.fingerprint,
                    ),
                    "measurements": (
                        summary.failing_test_recall,
                        summary.first_failure_elapsed,
                        Measurement(
                            name="counterfactual_runner_work_ms",
                            value=float(summary.runner_work.value),
                            unit="ms",
                            meaning=RUNNER_MS.meaning,
                            direction="minimize",
                            status="observed",
                            minimum=0.0,
                        ),
                    ),
                }
            )
            params = replace(params, domain_records=(*params.domain_records, record))
        super().collect(params)


def build_resource_environment(config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None):
    """Reuse native environment and decode one typed configuration at the extension boundary."""
    env, builds = build_environment(cast(EnvironmentBuildConfig, replace(config, extension=None)), metadata, agent_seed)
    protocol = ResourceReplayProtocol.from_json(config.extensions["resource_protocol"])
    candidate = config.extensions["resource_candidate"]
    if config.run_id is None or config.resolved_spec is None:
        raise ValueError("Missing native run/specification provenance")
    env.agents[0].policy = EstimateOrder(protocol, candidate)
    env.monitor = ResourceMonitor(
        env.monitor.sink,
        protocol,
        candidate,
        Path(config.results_config["out_dir"]).parent,
        config.run_id,
        fingerprint(config.resolved_spec.model_dump(mode="json")),
    )
    return env, builds


def workspace_root(output):
    """Accept fresh or owned-resumable bounded workspace roots, rejecting existing symlinks."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must remain within current workspace")
    if root.exists():
        paths = tuple(itertools.islice(root.rglob("*"), 10001))
        if len(paths) > 10000 or any(path.is_symlink() for path in paths):
            raise ValueError("Example resume requires a bounded tree without symlinks")
        if not (root / "example-inputs.json").is_file():
            raise ValueError("Resume requires owned example input proof")
    root.mkdir(parents=True, exist_ok=True)
    return root


def main(output):
    """Run/resume native full suites, then compare their modeled resource selections."""
    root = workspace_root(output)
    protocol, source = fixture()
    writer = ArtifactWriter(root)
    directory = root / "inputs" / "constructed-ci"
    directory.mkdir(parents=True, exist_ok=True)
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
    data = directory / "features-engineered.parquet"
    if data.exists():
        if not pl.read_parquet(data).equals(expected):
            raise ValueError("Owned native input observations changed")
    else:
        expected.write_parquet(data)
    proof = {
        "protocol": protocol.model_dump(mode="json"),
        "parquet_sha256": "sha256:" + hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    writer.write_immutable(root / "example-inputs.json", canonical_json(proof) + "\n")
    writer.write_immutable(root / "source.json", source)
    runs = []
    for candidate in protocol.native_manifest.candidates:
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
                "extensions": {"resource_protocol": protocol.to_json(), "resource_candidate": candidate},
                "results": {
                    "out_dir": str(root / candidate / "native"),
                    "manifest_enabled": True,
                    "top_k_prioritization": 3,
                },
                "checkpoint": {"enabled": True, "interval": 1, "base_dir": str(root / candidate / "checkpoints")},
            }
        )
        writer.write_immutable(
            root / f"native-spec-{candidate}.json", canonical_json(spec.model_dump(mode="json")) + "\n"
        )
        result = run_with_extension(spec, RunnerExtension(build_environment_fn=build_resource_environment))
        runs.extend(
            ResourceReplayRun.from_json(path.read_bytes())
            for path in Path(result.artifacts_dir).rglob("resource-runs/*.json")
        )
    report = ResourceReplayReport(schema_version=1, protocol=protocol, runs=tuple(runs))
    path = publish_resource_report(report, writer)
    writer.write_immutable(root / "report.md", render_resource_report(report))
    print(
        json.dumps(
            {
                "status": report.status,
                "report": str(path),
                "runs": len(report.runs),
                "adoption": "insufficient_evidence",
            }
        )
    )
    return report


if __name__ == "__main__":
    main(sys.argv[1])
