"""Run the bounded MIT-licensed synthetic CI pilot through installed native COLEMAN.

Usage: python run.py OUTPUT_DIRECTORY. Never executes a real repository's tests.
A native warmup build precedes two protected replay builds. No genuine coverage
mapping exists, so a coverage candidate is deliberately ineligible.
"""

import hashlib
import json
import sys
from dataclasses import replace
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
    PilotReport,
    PilotRun,
    PilotTest,
    publish_pilot_report,
    render_pilot_report,
    serial_events,
    verify_run,
)
from coleman.policy.base import Policy
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, canonical_json, fingerprint
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.selection import Quantity
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector


def quantity(value, semantics):
    return Quantity(value=Decimal(str(value)), semantics=semantics)


def fixture():
    builds = tuple(
        PilotBuild(
            build_id=str(build),
            decided_at=f"2026-01-0{build}T10:00:00Z",
            outcomes_available_at=f"2026-01-0{build}T10:01:00Z",
            tests=tuple(
                PilotTest(test_id=name, duration=quantity(seconds, SECONDS), outcome=outcome)
                for name, seconds, outcome in (("t1", 1, "passed"), ("t2", 2, "failed"), ("t3", 4, "failed"))
            ),
            baseline_order=("t1", "t2", "t3"),
        )
        for build in (2, 3)
    )
    source = canonical_json([build.model_dump(mode="json") for build in builds]) + "\n"
    manifest = PilotManifest(
        schema_version=1,
        pilot_id="synthetic-native-ci-v1",
        source_uri="urn:coleman:fixture:operational-ci-v1",
        source_sha256="sha256:" + hashlib.sha256(source.encode()).hexdigest(),
        license_id="MIT (COLEMAN repository LICENSE)",
        dataset_fingerprint=fingerprint(json.loads(source)),
        evidence_basis="synthetic_replay",
        frozen_at="2026-01-01T00:00:00Z",
        temporal_plan=TemporalPlan(
            train_end="2025-12-30T00:00:00Z", validation_end="2026-01-01T00:00:00Z", test_end="2026-01-04T00:00:00Z"
        ),
        candidates=("base-order", "shortest-first", "native-random", "native-greedy"),
        baseline_id="base-order",
        budget=quantity(3, SECONDS),
        variable_price=quantity("0.01", RATE),
        thresholds=AdoptionThresholds(
            minimum_builds=2,
            minimum_recall=0.5,
            minimum_runner_saving_fraction=0.0,
            maximum_overhead=quantity(1, OVERHEAD_SECONDS),
        ),
        builds=builds,
    )
    return manifest, source


class DeclaredOrder(Policy):
    """A simple non-learning baseline; never reads protected outcomes."""

    def __init__(self, order):
        self.order = order

    def __str__(self):
        return "DeclaredOrder"

    def choose_all(self, agent):
        if set(agent.actions["Name"].to_list()) != set(self.order):
            raise ValueError("Fixture test universe changed")
        return list(self.order)


class PilotMonitor(MonitorCollector):
    """Retain native results, then audit actual scheduled tests and persist a checkpoint."""

    def __init__(self, sink, manifest, candidate, root, run_id):
        super().__init__(sink)
        self.manifest = manifest
        self.candidate = candidate
        self.writer = ArtifactWriter(root)
        self.run_id = run_id
        self.report = restore_report(Path(root), manifest)

    def completed_or_new(self, run):
        """Retain completed timing evidence, rejecting a changed replayed decision."""
        previous = next(
            (
                saved
                for saved in self.report.runs
                if saved.build_id == run.build_id and saved.candidate_id == run.candidate_id
            ),
            None,
        )
        if previous is None:
            return run
        if (previous.ranking, previous.events) != (run.ranking, run.events):
            raise ValueError("Replayed native decision conflicts with completed checkpoint")
        return previous

    def collect(self, params):
        if params.t > 1:
            build = next(build for build in self.manifest.builds if build.build_id == str(params.t))
            ranking = tuple(params.prioritization_order)
            events = serial_events(build, ranking, self.manifest.budget)
            if tuple(event.test_id for event in events) != tuple(params.metric.scheduled_testcases):
                raise ValueError("Independent replay disagrees with actual native scheduling")
            run = PilotRun(
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
            run = self.completed_or_new(run)
            metrics = verify_run(self.manifest, run)
            first = metrics.first_failure_elapsed
            record = DomainRecord(
                schema_version=1,
                namespace="pilot.ci",
                run_id=self.run_id,
                execution_id=self.candidate,
                decision_id=build.build_id,
                data_fingerprint=self.manifest.dataset_fingerprint,
                available_at=build.outcomes_available_at,
                provenance=ResultProvenance(
                    producer="coleman.operational-pilot-example",
                    producer_version="1",
                    spec_fingerprint=self.manifest.fingerprint,
                    # The comparison protocol is common to all candidates/builds.
                    # Logical execution/decision keys and measured content identify
                    # each observation; timing attempts do not define comparability.
                    evaluation_fingerprint=self.manifest.fingerprint,
                ),
                measurements=(
                    Measurement(
                        name="pilot_failing_test_recall",
                        value=metrics.failing_test_recall,
                        unit="1",
                        meaning="complete-universe-failing-test-recall-v1",
                        direction="maximize",
                        status="observed" if metrics.failing_test_recall is not None else "unavailable",
                    ),
                    Measurement(
                        name="pilot_first_failure_elapsed",
                        value=float(first.value) if first is not None else None,
                        unit="s",
                        meaning="replay-first-failure-elapsed-excluding-setup-v1",
                        direction="minimize",
                        status="observed" if first is not None else "unavailable",
                    ),
                ),
            )
            params = replace(params, domain_records=(record,))
            super().collect(params)  # synchronous replay-safe audit precedes snapshot
            updated = self.report.append(run)
            publish_pilot_report(updated, self.writer)
            self.report = updated
            return
        super().collect(params)


def build_pilot_environment(config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None):
    native = cast(EnvironmentBuildConfig, replace(config, extension=None))
    env, builds = build_environment(native, metadata, agent_seed)
    manifest = PilotManifest.from_json(config.extensions["pilot_manifest"])
    candidate = config.extensions["pilot_candidate"]
    if candidate in {"base-order", "shortest-first"}:
        # Fixed declared fixture durations are available before dispatch. No labels
        # or verdicts are supplied to the baseline policy.
        build = manifest.builds[0]
        order = (
            build.baseline_order
            if candidate == "base-order"
            else tuple(
                test.test_id for test in sorted(build.tests, key=lambda test: (test.duration.value, test.test_id))
            )
        )
        env.agents[0].policy = DeclaredOrder(order)
    if config.run_id is None:
        raise ValueError("Missing native run identity")
    env.monitor = PilotMonitor(
        env.monitor.sink, manifest, candidate, Path(config.results_config["out_dir"]).parent, config.run_id
    )
    return env, builds


def restore_report(root, manifest):
    snapshots = [PilotReport.from_json(path.read_bytes()) for path in (root / "pilot-reports").glob("*.json")]
    if any(snapshot.manifest != manifest for snapshot in snapshots):
        raise ValueError("Checkpoint manifest differs; use a new output directory")
    return PilotReport(
        schema_version=1,
        manifest=manifest,
        runs=tuple(run for snapshot in snapshots for run in snapshot.runs),
    )


def main(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(root)
    manifest, source = fixture()
    writer.write_immutable(root / "source.json", source)
    writer.write_immutable(root / "manifest.json", manifest.to_json() + "\n")
    report = restore_report(root, manifest)
    publish_pilot_report(report, writer)
    directory = root / "inputs" / "synthetic-ci"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "BuildId": build,
            "Name": test.test_id,
            "Duration": float(test.duration.value),
            "CalcPrio": 0,
            "LastRun": "2026-01-01",
            "Verdict": int(test.outcome == "failed"),
        }
        for build in (1, 2, 3)
        for test in manifest.builds[0].tests
    ]
    pl.DataFrame(rows).write_parquet(directory / "features-engineered.parquet")
    for candidate in manifest.candidates:
        if not any(pair[0] == candidate for pair in report.missing):
            continue
        spec = RunSpec.model_validate(
            {
                "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42, "verbose": False},
                "experiment": {
                    "datasets_dir": str(root / "inputs"),
                    "datasets": ["synthetic-ci"],
                    "policies": ["Random" if candidate == "native-random" else "Greedy"],
                    "rewards": ["RNFail"],
                    "budget": {"mode": "fixed_time", "values": [3]},
                },
                "extensions": {"pilot_manifest": manifest.to_json(), "pilot_candidate": candidate},
                "results": {"out_dir": str(root / candidate), "manifest_enabled": True},
            }
        )
        native = run_with_extension(spec, RunnerExtension(build_environment_fn=build_pilot_environment))
        completed = restore_report(Path(native.artifacts_dir), manifest)
        for run in completed.runs:
            report = report.append(run)
        publish_pilot_report(report, writer)
    path = publish_pilot_report(report, writer)
    writer.write_immutable(root / "report.md", render_pilot_report(report))
    print(json.dumps({"status": report.status, "artifact": str(path), "adoption": "insufficient_evidence"}))


if __name__ == "__main__":
    main(sys.argv[1])
