"""Owned offline native shadow consumer: functional recovery, never prospective evidence.

Run python ci.py OUTPUT_DIRECTORY twice. Required native CI always runs the full
suite in base order. Two predeclared shadow candidates cannot affect that choice.
Source UTC instants are synthetic; monotonic receiver ticks describe this process.
"""

import hashlib
import itertools
import json
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl
from pydantic import TypeAdapter, model_validator

from coleman.api import run_with_extension
from coleman.artifacts import ArtifactWriter
from coleman.evaluation.protected.temporal import TemporalObservation, TemporalPlan
from coleman.observations import (
    EVENT_AGE_SECONDS,
    ObservationHistory,
    ObservationRevision,
    QuantityMeanModel,
    SignalRequest,
    fit_quantity_mean,
    prepare_training,
)
from coleman.policy.base import Policy
from coleman.resources import RUNNER_MS, ResourceAmount, ResourceEstimate, ResourceRequest, TestEstimate
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, canonical_json, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_environment
from coleman.selection import Quantity, UnitSemantics
from coleman.shadow import (
    BuildClosed,
    BuildOpened,
    CandidateDecision,
    FrozenCandidate,
    HostClock,
    ShadowProtocol,
    ShadowReport,
    ShadowSource,
    SourceStatus,
    TestFeedback,
    load_shadow_state,
    publish_shadow_report,
    publish_shadow_state,
    render_shadow_report,
)
from coleman.shadow.contracts import ShadowEvent
from coleman.spec.models import RunSpec
from coleman.utils.monitor import MonitorCollector

BASE = ("t1", "t2", "t3")
PRIOR_EVENT_AT = "2025-12-29T00:00:00Z"
PRIOR_AVAILABLE_AT = "2025-12-30T00:00:00Z"
STUDY_END = "2026-01-09T00:00:00Z"
OPENED_SUFFIX = ":opened"
SESSION = uuid.uuid4().hex
DAY = datetime.fromisoformat("2026-01-02T10:00:00Z")


def amount(ms):
    """Declare whole occupied runner milliseconds explicitly."""
    return ResourceAmount(value=ms, semantics=RUNNER_MS)


def witness(utc_at):
    """Keep synthetic source UTC separate from real local monotonic receiver time."""
    return HostClock(session_id=SESSION, tick_ns=time.monotonic_ns(), utc_at=utc_at)


def start_for(step):
    """Declare two fixture builds on one UTC decision day."""
    return DAY + timedelta(minutes=step - 2)


def owned_model():
    """Freeze one historical binary-indicator mean, not a calibrated probability."""
    semantics = UnitSemantics(unit="1", meaning="historical-ci-failure-indicator-v1")
    revisions = tuple(
        ObservationRevision.model_validate(
            {
                "schema_version": 1,
                "entity_id": name,
                "received_at": PRIOR_AVAILABLE_AT,
                "supersedes": None,
                "metadata": TemporalObservation.model_validate(
                    {
                        "observation_id": "prior:" + name,
                        "group_id": "prior-build",
                        "event_at": PRIOR_EVENT_AT,
                        "available_at": PRIOR_AVAILABLE_AT,
                    }
                ),
                "measurement": Measurement(
                    name="historical_failure",
                    value=float(name != "t1"),
                    unit="1",
                    meaning=semantics.meaning,
                    direction="none",
                    status="observed",
                    minimum=0.0,
                ),
            }
        )
        for name in BASE
    )
    history = ObservationHistory(schema_version=1, source_id="owned-prior-ci", source_version="1", revisions=revisions)
    plan = TemporalPlan.model_validate(
        {
            "train_end": "2025-12-31T00:00:00Z",
            "validation_end": "2026-01-01T00:00:00Z",
            "test_end": STUDY_END,
        }
    )
    training = prepare_training(
        history,
        plan,
        information_cutoff=plan.train_end,
        measurement_name="historical_failure",
        input_semantics=semantics,
    )
    return fit_quantity_mean(training, frozen_at=plan.validation_end)


class DemoConfiguration(VersionedContract):
    """Only predecision information: frozen model/protocol and historical resource estimates."""

    model: QuantityMeanModel
    protocol: ShadowProtocol
    estimates: tuple[TestEstimate, ...]

    @model_validator(mode="after")
    def binding(self):
        """Bind the actual frozen model body, rather than trusting an illustrative API name."""
        candidate = next(item for item in self.protocol.candidates if item.candidate_id == "historical-failure")
        if candidate.model_fingerprint != self.model.fingerprint:
            raise ValueError("Demo candidate differs from actual frozen model")
        return self


def configuration():
    """Predeclare source authorization, thresholds, trial registry and seven-day evidence gate."""
    model = owned_model()
    feature = fingerprint(
        {"used_columns": ["Name"], "prior_failure_model": model.fingerprint, "prior_duration_ms": [1000, 2000, 4000]}
    )
    candidates = tuple(
        FrozenCandidate(
            candidate_id=name,
            policy_version=name + "-v1",
            model_fingerprint=model.fingerprint if name == "historical-failure" else fingerprint({"model": None}),
            feature_fingerprint=feature,
            selection_version="serial-prefix-v1",
            fallback_version="no-action-v1",
            budget=amount(4000),
            shared_setup_reserve=amount(0),
        )
        for name in ("base-order", "historical-failure")
    )
    protocol = ShadowProtocol.model_validate(
        {
            "schema_version": 1,
            "experiment_id": "owned-native-shadow-v1",
            "source": ShadowSource(
                source_id="owned-native-ci",
                source_version="1",
                schema_fingerprint=fingerprint(TypeAdapter(ShadowEvent).json_schema()),
                authorization_reference="urn:coleman:owned-mit-functional-fixture",
                clock_assumption="synthetic-source-UTC-real-process-monotonic-no-prospective-claim-v1",
                license_id="MIT (COLEMAN repository LICENSE)",
                evidence_basis="functional_offline",
            ),
            "benchmark_fingerprint": fingerprint({"benchmark": "owned-3-test-ci-v1"}),
            "frozen_at": model.frozen_at,
            "temporal_plan": model.training.plan,
            "candidates": candidates,
            "baseline_id": "base-order",
            "independent_groups": (),
            "independence_justification": None,
            "minimum_observation_days": 7,
            "minimum_builds": 2,
            "minimum_independent_groups": 2,
            "minimum_recall": 0.5,
            "minimum_net_saving_ms": 1,
            "actual_build_capacity": amount(60000),
            "maximum_silence_ns": 60 * 10**9,
            "maximum_source_lag_seconds": 60,
            "maximum_pending": 8,
            "maximum_events": 128,
            "maximum_event_bytes": 65536,
            "maximum_journal_bytes": 1048576,
        }
    )
    estimates = tuple(
        TestEstimate(
            test_id=name,
            estimate=ResourceEstimate.model_validate(
                {
                    "amount": amount(ms),
                    "source_fingerprint": feature,
                    "observed_at": PRIOR_EVENT_AT,
                    "available_at": PRIOR_AVAILABLE_AT,
                    "valid_until": STUDY_END,
                }
            ),
        )
        for name, ms in zip(BASE, (1000, 2000, 4000), strict=True)
    )
    return DemoConfiguration(schema_version=1, model=model, protocol=protocol, estimates=estimates)


class RequiredSuiteObserver(Policy):
    """Seal research-only recommendations before evaluation; always return the full base order.

    Artifact I/O errors interrupt this offline consumer, never an upstream CI job.
    A real provider must run the consumer independently of mandatory CI execution.
    """

    def __init__(self, config, root):
        self.configuration = config
        self.root = Path(root)

    def __str__(self):
        return "RequiredSuiteShadowObserver"

    def choose_all(self, agent):
        """Read only IDs/prior models; ignore current durations and verdicts, including on retry."""
        if sorted(agent.actions["Name"].to_list()) != sorted(BASE):
            raise ValueError("Owned source test universe changed")
        step = agent.t + 1
        if step not in (2, 3):  # Native warmup/optional final forecast creates no extra research build.
            return list(BASE)
        config = self.configuration
        writer = ArtifactWriter(self.root)
        state = load_shadow_state(config.protocol, writer)
        build_id = f"build-{step}"
        if state.source_event(build_id + OPENED_SUFFIX) is not None:
            return list(BASE)  # Already sealed before an interrupted native checkpoint.
        at = start_for(step)
        heartbeat = SourceStatus(event_id=f"connected:{build_id}:{SESSION}", available_at=at, connected=True)
        state = state.ingest(heartbeat, witness(at))
        publish_shadow_state(state, writer)
        decisions = []
        if state.operational(witness(at)).status == "healthy":
            for candidate in config.protocol.candidates:
                began = time.monotonic_ns()
                signals = ()
                ranking = BASE
                if candidate.candidate_id == "historical-failure":
                    signals = tuple(
                        config.model.signal(
                            SignalRequest(
                                schema_version=1,
                                decision_id=build_id,
                                entity_id=name,
                                decided_at=at,
                                information_cutoff=at,
                                target_start=at,
                                target_end=at + timedelta(seconds=60),
                            ),
                            maximum_event_age=Quantity.model_validate(
                                {"value": "604800", "semantics": EVENT_AGE_SECONDS}
                            ),
                        )
                        for name in BASE
                    )
                    for signal in signals:
                        config.model.verify_signal(signal)
                    ranking = tuple(
                        signal.request.entity_id
                        for signal in sorted(signals, key=lambda row: (-row.measurement.value, row.request.entity_id))
                    )
                request = ResourceRequest(
                    schema_version=1,
                    decision_id=build_id,
                    configuration_fingerprint=fingerprint(candidate.model_dump(mode="json")),
                    decided_at=at,
                    capacity=candidate.budget,
                    tests=config.estimates,
                    ranking=ranking,
                )
                elapsed = time.monotonic_ns() - began
                overhead = (elapsed + 999999) // 1000000  # Explicit ceil; less than 1 ms excess, one-runner proxy.
                decisions.append(
                    CandidateDecision(
                        candidate_id=candidate.candidate_id,
                        model_fingerprint=candidate.model_fingerprint,
                        request=request,
                        signals=signals,
                        decision_overhead=amount(overhead),
                        shared_setup_reserve=candidate.shared_setup_reserve,
                        overhead_interpretation="elapsed_one_runner_proxy",
                    )
                )
        opened = BuildOpened(
            event_id=build_id + OPENED_SUFFIX,
            build_id=build_id,
            logical_build_id=build_id,
            group_id="owned-project",
            decided_at=at,
            target_end=at + timedelta(seconds=60),
            required_ids=BASE,
            actual_estimate=ResourceEstimate.model_validate(
                {
                    "amount": amount(7000),
                    "source_fingerprint": config.fingerprint,
                    "observed_at": PRIOR_EVENT_AT,
                    "available_at": PRIOR_AVAILABLE_AT,
                    "valid_until": STUDY_END,
                }
            ),
            decisions=tuple(decisions),
        )
        state = state.ingest(opened, witness(at))
        publish_shadow_state(state, writer)
        return list(BASE)


class NativeFeedbackBatch(VersionedContract):
    """Immutable original native observation batch; retries retain the first timing and payloads."""

    build_id: str
    ranking: tuple[str, ...]
    scheduled: tuple[str, ...]
    feedback: tuple[TestFeedback, ...]
    closure: BuildClosed


def feedback_batch(params):
    """Normalize the owned full-suite source after genuine native metric evaluation."""
    if tuple(params.prioritization_order) != BASE or tuple(params.metric.scheduled_testcases) != BASE:
        raise ValueError("Required native CI suite/order was changed by shadow observation")
    build_id = f"build-{params.t}"
    at = start_for(params.t)
    elapsed = 0
    feedback = []
    for name, ms in zip(BASE, (1000, 2000, 4000), strict=True):
        elapsed += ms
        instant = at + timedelta(milliseconds=elapsed)
        revision = ObservationRevision(
            schema_version=1,
            metadata=TemporalObservation(
                observation_id=f"{build_id}:{name}",
                revision_id="r1",
                group_id="owned-project",
                event_at=instant,
                available_at=instant,
            ),
            entity_id=name,
            received_at=instant,
            supersedes=None,
            measurement=Measurement(
                name="ci_failure",
                value=float(name != "t1"),
                unit="1",
                meaning="ci-test-failure-indicator-v1",
                direction="none",
                status="observed",
                minimum=0.0,
            ),
        )
        feedback.append(
            TestFeedback(
                event_id=f"{build_id}:{name}:completed:r1",
                build_id=build_id,
                test_id=name,
                revision=revision,
                occupancy=amount(ms),
            )
        )
    # Native prioritization elapsed is modeled as one occupied runner, not a provider bill.
    from decimal import ROUND_CEILING, Decimal

    dispatch_ms = int((Decimal(str(params.prioritization_time)) * 1000).to_integral_value(rounding=ROUND_CEILING))
    closed = BuildClosed(
        event_id=build_id + ":closed",
        build_id=build_id,
        available_at=at + timedelta(seconds=8),
        dispatch_occupancy=amount(dispatch_ms),
        shared_setup_occupancy=amount(0),
        queue_elapsed=None,
        setup_elapsed=None,
    )
    return NativeFeedbackBatch(
        schema_version=1, build_id=build_id, ranking=BASE, scheduled=BASE, feedback=tuple(feedback), closure=closed
    )


class ShadowMonitor(MonitorCollector):
    """Commit original source batch, admissions and domain sidecar before native checkpoint advancement."""

    def __init__(self, sink, config, root, run_id, specification_fingerprint):
        super().__init__(sink)
        self.configuration = config
        self.writer = ArtifactWriter(root)
        self.run_id = run_id
        self.specification_fingerprint = specification_fingerprint

    def collect(self, params):
        if params.t > 1:
            batch = feedback_batch(params)
            path = (
                self.writer.run_root
                / "shadow-source-batches"
                / f"{fingerprint({'build': batch.build_id}).removeprefix('sha256:')}.json"
            )
            if path.exists():
                previous = NativeFeedbackBatch.from_json(path.read_bytes())
                if (previous.ranking, previous.scheduled, previous.feedback) != (
                    batch.ranking,
                    batch.scheduled,
                    batch.feedback,
                ):
                    raise ValueError("Native source replay conflicts with its committed original batch")
                batch = previous  # Retain first measured timing even when failure preceded the native checkpoint.
            self.writer.write_immutable(path, batch.to_json() + "\n")
            state = load_shadow_state(self.configuration.protocol, self.writer)
            if state.source_event(batch.build_id + OPENED_SUFFIX) is None:
                raise ValueError("Native feedback has no pre-outcome sealed declaration")
            for event in (*batch.feedback, batch.closure):
                state = state.ingest(event, witness(batch.closure.available_at))
                publish_shadow_state(state, self.writer)
            build = next(item for item in state.builds() if item.opening.build_id == batch.build_id)
            record = DomainRecord.model_validate(
                {
                    "schema_version": 1,
                    "namespace": "shadow.required-ci",
                    "run_id": self.run_id,
                    "execution_id": fingerprint(
                        {
                            "dataset": params.scenario_provider.name,
                            "policy": params.policy,
                            "reward": params.reward_function,
                            "experiment": params.experiment,
                        }
                    ),
                    "decision_id": batch.build_id,
                    "data_fingerprint": self.configuration.protocol.fingerprint,
                    "available_at": batch.closure.available_at,
                    "provenance": ResultProvenance(
                        producer="coleman.owned-shadow-ci",
                        producer_version="1",
                        spec_fingerprint=self.specification_fingerprint,
                        evaluation_fingerprint=self.configuration.protocol.fingerprint,
                    ),
                    "measurements": (
                        Measurement(
                            name="required_ci_known_runner_ms",
                            unit="ms",
                            meaning=RUNNER_MS.meaning,
                            value=float(build.actual_ledger.balance.consumed),
                            direction="minimize",
                            status="observed",
                            minimum=0.0,
                        ),
                    ),
                }
            )
            params = replace(params, domain_records=(*params.domain_records, record))
        super().collect(params)


def build_shadow_environment(config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None):
    """Reuse actual native orchestration and checkpoints; no second experiment runner."""
    env, builds = build_environment(cast(EnvironmentBuildConfig, replace(config, extension=None)), metadata, agent_seed)
    configuration = DemoConfiguration.from_json(config.extensions["shadow_demo"])
    if config.run_id is None or config.resolved_spec is None:
        raise ValueError("Missing native run/spec provenance")
    root = Path(config.results_config["out_dir"]).parent
    env.agents[0].policy = RequiredSuiteObserver(configuration, root)
    env.monitor = ShadowMonitor(
        env.monitor.sink, configuration, root, config.run_id, fingerprint(config.resolved_spec.model_dump(mode="json"))
    )
    return env, builds


def workspace_root(output):
    """Permit owned bounded roots only; existing symlinks/unknown roots cannot redirect publication."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must remain within current workspace")
    if root.exists():
        paths = tuple(itertools.islice(root.rglob("*"), 10001))
        if len(paths) > 10000 or any(path.is_symlink() for path in paths):
            raise ValueError("Example resume requires a bounded tree without symlinks")
        if not all((root / name).is_file() for name in ("example-inputs.json", "native-spec.json")):
            raise ValueError("Resume requires owned input and actual specification proofs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def main(output):
    """Run/resume two actual native full suites and report one-day functional evidence as inconclusive."""
    root = workspace_root(output)
    config = configuration()
    writer = ArtifactWriter(root)
    directory = root / "inputs" / "constructed-ci"
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "features-engineered.parquet"
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
            for name, seconds in zip(BASE, (1, 2, 4), strict=True)
        ]
    )
    if data.exists():
        if not pl.read_parquet(data).equals(expected):
            raise ValueError("Owned native source changed")
    else:
        expected.write_parquet(data)
    proof = {
        "configuration": config.model_dump(mode="json"),
        "parquet_sha256": "sha256:" + hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    writer.write_immutable(root / "example-inputs.json", canonical_json(proof) + "\n")
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
            "extensions": {"shadow_demo": config.to_json(), "owned_source": fingerprint(proof)},
            "results": {"out_dir": str(root / "native"), "manifest_enabled": True, "top_k_prioritization": 3},
            "checkpoint": {"enabled": True, "interval": 1, "base_dir": str(root / "checkpoints")},
        }
    )
    writer.write_immutable(root / "native-spec.json", canonical_json(spec.model_dump(mode="json")) + "\n")
    native = run_with_extension(spec, RunnerExtension(build_environment_fn=build_shadow_environment))
    native_writer = ArtifactWriter(native.artifacts_dir)
    state = load_shadow_state(config.protocol, native_writer)
    report = ShadowReport(schema_version=1, state=state)
    artifact = publish_shadow_report(report, writer)
    writer.write_immutable(root / "report.md", render_shadow_report(report))
    evidence = report.evidence()
    print(
        json.dumps(
            {
                "research": evidence.outcome,
                "basis": config.protocol.source.evidence_basis,
                "days": evidence.distinct_decision_days,
                "minimum_days": config.protocol.minimum_observation_days,
                "forward_builds": evidence.forward_logical_builds,
                "report": str(artifact),
                "artifacts_dir": native.artifacts_dir,
                "run_id": native.run_id,
            }
        )
    )
    return report, native


if __name__ == "__main__":
    main(sys.argv[1])
