"""Commit independently checked baseline evidence before existing native checkpoint advancement."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from coleman.application.baseline_agents import BaselineAgent
from coleman.application.baseline_resources import BaselineResourceGuard
from coleman.artifacts import ArtifactWriter
from coleman.corpus.baseline_contracts import BaselineBuild, BaselinePlan
from coleman.corpus.baseline_reference import BaselineReference, baseline_reference
from coleman.corpus.baseline_results import BaselineCoordinate, BaselineObservation, baseline_measurement
from coleman.corpus.materialization import validate_source_storage
from coleman.environment import Environment
from coleman.results.domain import DomainRecord, ResultProvenance, fingerprint
from coleman.utils.monitor import MonitorCollector
from coleman.utils.monitor_params import CollectParams


def _check_native(params: CollectParams, build: BaselineBuild, cell: BaselineCoordinate) -> BaselineReference:
    """Reject changed budgets, incomplete rankings and metric mutations against a separate implementation."""
    if (
        params.policy != cell.method
        or params.budget_mode is None
        or params.budget_mode.value != "ratio"
        or params.budget_value != float(cell.budget.value)
        or params.available_time != params.total_build_duration * float(cell.budget.value)
    ):
        raise ValueError("Native baseline execution differs from its frozen candidate or budget")
    source_work = math.fsum(test.duration.value for test in build.tests if test.duration.value is not None)
    if params.total_build_duration != source_work:
        raise ValueError("Native full-suite work differs from the declared binary64 source")
    reference = baseline_reference(build, tuple(params.prioritization_order), params.available_time)
    metric = params.metric
    if (
        tuple(metric.scheduled_testcases) != reference.selected
        or metric.detected_failures != reference.detected
        or metric.undetected_failures != reference.failures - reference.detected
    ):
        raise ValueError("Native selected universe or failure denominator failed independent baseline verification")
    if reference.failures:
        values = ((metric.fitness, reference.napfd), (metric.cost, reference.apfdc), (metric.recall, reference.recall))
        if any(
            expected is None or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in values
        ):
            raise ValueError("Native quality metric failed independent baseline verification")
    elif metric.fitness != 1 or metric.cost != 1:
        raise ValueError("Legacy zero-failure fallback changed unexpectedly")
    if reference.first_failure is not None and not math.isclose(
        metric.ttf_duration, reference.first_failure, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Native first-failure time failed independent baseline verification")
    return reference


class BaselineMonitor(MonitorCollector):
    """Publish typed raw ranking evidence synchronously before delegating the native row write.

    The ordinary native sink, flush and checkpoints are reused. If a process stops
    after this immutable observation but before the native checkpoint, replay must
    reproduce its mathematical evidence; the first recorded assembly/timing values
    are retained explicitly. Operational native row timing on that retry may differ.
    The sidecar is not an atomic transaction with Parquet, as in existing domain
    results. Completion validation requires all committed source steps separately.
    """

    def __init__(
        self,
        environment: Environment,
        plan: BaselinePlan,
        cell: BaselineCoordinate,
        run_id: str,
        spec_digest: str,
        root: Path,
        guard: BaselineResourceGuard,
    ):
        """Cache immutable context outside the per-build hot path; retain the live environment for restored agents."""
        super().__init__(environment.monitor.sink)
        self.environment = environment
        self.builds = plan.source.builds
        self.plan_digest = plan.fingerprint
        self.source_digest = plan.source.fingerprint
        self.source_revision = plan.source.source_fingerprint
        self.build_digests = tuple(fingerprint(build.model_dump(mode="json")) for build in self.builds)
        self.cell, self.run_id, self.spec_digest = cell, run_id, spec_digest
        self.writer = ArtifactWriter(root)
        self.guard = guard

    def _observation(self, params: CollectParams) -> BaselineObservation:
        """Bind an actual native execution to its full source universe and explicit scientific quantities."""
        if params.execution_id is None:
            raise ValueError("Baseline evidence requires an explicit physical attempt identity")
        if not 1 <= params.t <= len(self.builds):
            raise ValueError("Native baseline step is outside its frozen source")
        build = self.builds[params.t - 1]
        reference = _check_native(params, build, self.cell)
        agent = self.environment.agents[0]
        if not isinstance(agent, BaselineAgent):
            raise ValueError("Native checkpoint did not restore the qualified baseline agent")
        quantities = {
            "full_universe_napfd": reference.napfd,
            "full_universe_apfdc": reference.apfdc,
            "failing_test_recall": reference.recall,
            "selected_runner_seconds": reference.selected_work,
            "projected_full_runner_seconds": reference.projected_full_work,
            "source_full_runner_seconds": params.total_build_duration,
            "declared_native_work_budget_seconds": params.available_time,
            "native_budget_excess_seconds": reference.budget_excess,
            "censored_first_failure_seconds": reference.first_failure,
            "missing_input_fallback_count": float(agent.fallback_count),
            "policy_preparation_seconds": agent.timing.preparation_seconds,
            "policy_inference_seconds": agent.timing.inference_seconds,
            "policy_learning_seconds": agent.timing.learning_seconds,
            "native_process_peak_mib": self.guard.peak_mib(),
            "native_process_cpu_seconds": params.process_cpu_time_seconds,
            "native_process_elapsed_seconds": params.wall_time_seconds,
        }
        assembled = datetime.now(UTC)
        if build.source_metadata is not None and assembled < build.source_metadata.available_at:
            raise ValueError("Baseline evidence assembly precedes declared source outcome availability")
        record = DomainRecord(
            schema_version=1,
            namespace="corpus.baseline",
            run_id=self.run_id,
            execution_id=self.cell.cell_id,
            decision_id=build.member_id,
            data_fingerprint=self.source_digest,
            available_at=assembled,
            learning_reward=None,
            provenance=ResultProvenance(
                producer="coleman.baseline-native",
                producer_version="1",
                spec_fingerprint=self.spec_digest,
                evaluation_fingerprint=self.plan_digest,
                source_revision=self.source_revision,
            ),
            measurements=tuple(baseline_measurement(name, value) for name, value in quantities.items()),
        )
        return BaselineObservation(
            schema_version=1,
            cell_id=self.cell.cell_id,
            plan_fingerprint=self.plan_digest,
            source_build_fingerprint=self.build_digests[params.t - 1],
            attempt_id=params.execution_id,
            step=params.t,
            member_id=build.member_id,
            ranking=tuple(params.prioritization_order),
            selected=reference.selected,
            first_failure_state="detected" if reference.first_failure is not None else "right_censored",
            record=record,
        )

    def collect(self, params: CollectParams) -> None:
        """Reuse identical mathematical replay, reject conflicts, and publish through the original native sink."""
        self.guard.check()
        observation = self._observation(params)
        path = self.writer.run_root / "baseline-observations" / f"step-{params.t:04d}.json"
        validate_source_storage(path)
        if path.exists():
            if path.stat().st_size > 16 * 1024**2:
                raise ValueError("Baseline observation exceeds its bounded artifact size")
            previous = BaselineObservation.from_json(path.read_bytes())
            if (
                previous.mathematical_fingerprint != observation.mathematical_fingerprint
                or previous.record.provenance.spec_fingerprint != self.spec_digest
                or previous.record.run_id != self.run_id
                or previous.record.data_fingerprint != self.source_digest
            ):
                raise ValueError("Native retry conflicts with immutable baseline evidence")
            observation = previous
        payload = observation.to_json() + "\n"
        if len(payload.encode("utf-8")) > 16 * 1024**2:
            raise ValueError("Baseline observation exceeds its bounded artifact size")
        self.writer.write_immutable(path, payload)
        super().collect(replace(params, domain_records=(*params.domain_records, observation.record)))
