"""Versioned baseline evidence; stable mathematical identity excludes variable operational timing."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.baseline_contracts import BaselinePlan, Method
from coleman.corpus.baseline_eligibility import BaselineEligibility, baseline_eligibility
from coleman.corpus.repeated_contracts import BUDGET_RATIO, Seed
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import DomainRecord, Fingerprint, Identifier, Measurement, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity

MATH_MEASUREMENTS = (
    "full_universe_napfd",
    "full_universe_apfdc",
    "failing_test_recall",
    "selected_runner_seconds",
    "projected_full_runner_seconds",
    "source_full_runner_seconds",
    "declared_native_work_budget_seconds",
    "native_budget_excess_seconds",
    "censored_first_failure_seconds",
    "missing_input_fallback_count",
)
TIMING_MEASUREMENTS = (
    "policy_preparation_seconds",
    "policy_inference_seconds",
    "policy_learning_seconds",
    "native_process_peak_mib",
    "native_process_cpu_seconds",
    "native_process_elapsed_seconds",
)
_MEASUREMENTS = {
    "full_universe_napfd": ("1", "native-full-universe-verdict-napfd-v1", "maximize"),
    "full_universe_apfdc": ("1", "native-midpoint-cost-full-universe-apfdc-v1", "maximize"),
    "failing_test_recall": ("1", "failing-test-recall-full-universe-v1", "maximize"),
    "selected_runner_seconds": ("s", "modeled-native-binary32-selected-test-work-v1", "minimize"),
    "projected_full_runner_seconds": ("s", "modeled-native-binary32-full-test-work-v1", "minimize"),
    "source_full_runner_seconds": ("s", "declared-source-binary64-full-test-work-v1", "minimize"),
    "declared_native_work_budget_seconds": ("s", "source-binary64-declared-native-work-budget-v1", "minimize"),
    "native_budget_excess_seconds": ("s", "projected-exact-work-excess-over-native-binary64-budget-v1", "minimize"),
    "censored_first_failure_seconds": ("s", "native-serial-first-failure-occupancy-censored-v1", "minimize"),
    "missing_input_fallback_count": ("1", "current-decision-missing-heuristic-input-count-v1", "minimize"),
    "policy_preparation_seconds": ("s", "observed-policy-input-preparation-elapsed-v1", "minimize"),
    "policy_inference_seconds": ("s", "observed-policy-choice-elapsed-v1", "minimize"),
    "policy_learning_seconds": ("s", "observed-native-reward-and-history-update-elapsed-v1", "minimize"),
    "native_process_peak_mib": ("MiB", "own-native-worker-process-peak-resident-memory-v1", "minimize"),
    "native_process_cpu_seconds": ("s", "own-native-attempt-cumulative-process-cpu-v1", "minimize"),
    "native_process_elapsed_seconds": ("s", "own-native-attempt-cumulative-elapsed-v1", "minimize"),
    "host_attempt_elapsed_seconds": (
        "s",
        "observed-host-invocation-including-bootstrap-and-publication-v1",
        "minimize",
    ),
}


def baseline_measurement(name: str, value: float | None):
    """Construct one declared quantity; missing values are unavailable, never fabricated zero.

    Unknown names raise ValueError. All declared metrics are nonnegative, with
    finite values enforced by the shared Measurement contract. No unit conversion
    or statistical interpretation is inferred from a metric's numeric magnitude.
    """
    from coleman.results.domain import Measurement

    if name not in _MEASUREMENTS:
        raise ValueError("Unknown baseline measurement contract")
    unit, meaning, direction = _MEASUREMENTS[name]
    return Measurement.model_validate(
        {
            "name": name,
            "value": value,
            "unit": unit,
            "meaning": meaning,
            "direction": direction,
            "status": "observed" if value is not None else "unavailable",
            "minimum": 0.0,
        }
    )


def _cell_id(plan_fingerprint: str, method: Method, seed: int, budget: Quantity) -> str:
    """Encode a prevalidated cell using one cached whole-plan fingerprint per grid operation."""
    return fingerprint(["coleman.baseline-cell.v1", plan_fingerprint, method, seed, budget.model_dump(mode="json")])


def baseline_cell_id(plan: BaselinePlan, method: Method, seed: int, budget: Quantity) -> str:
    """Hash a declared candidate/seed/budget cell, independent of attempt, output path and timing.

    The whole frozen plan binds source membership and every candidate. Invalid or
    undeclared dimensions raise ValueError. No implicit budget conversion occurs.
    This identifies logical development work, not a fresh protected observation.
    """
    if method not in plan.methods or type(seed) is not int or seed not in plan.seeds or budget not in plan.budgets:
        raise ValueError("Baseline cell dimensions are absent from the frozen plan")
    return _cell_id(plan.fingerprint, method, seed, budget)


class BaselineCoordinate(Immutable):
    """One predeclared execution coordinate; construction alone does not bind it to a plan."""

    method: Method
    seed: Seed
    budget: Quantity
    cell_id: Fingerprint


def baseline_coordinates(plan: BaselinePlan) -> tuple[BaselineCoordinate, ...]:
    """Expand the frozen grid with one whole-plan validation/hash, independent of execution order."""
    plan = BaselinePlan.model_validate(plan)
    digest = plan.fingerprint
    return tuple(
        BaselineCoordinate(method=method, seed=seed, budget=budget, cell_id=_cell_id(digest, method, seed, budget))
        for method in plan.methods
        for seed in plan.seeds
        for budget in plan.budgets
    )


class BaselineObservation(VersionedContract):
    """One raw completed build with ranking, selected universe and typed scientific/operational quantities.

    The original native Parquet artifact remains separate and retains its legacy
    conventions. Here, a zero-failure denominator makes quality ratios unavailable;
    nondetection is censored/null, not an observed zero time. Native binary32
    runner work remains separate from source binary64 declared work. Original
    label clocks stay in the plan; record availability is actual assembly time.
    """

    cell_id: Fingerprint
    plan_fingerprint: Fingerprint
    source_build_fingerprint: Fingerprint
    attempt_id: Identifier
    step: Annotated[int, Field(strict=True, ge=1, le=512)]
    member_id: Identifier
    ranking: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=10000)]
    selected: Annotated[tuple[Identifier, ...], Field(max_length=10000)]
    first_failure_state: Literal["detected", "right_censored"]
    record: DomainRecord

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Reject duplicate/nonprefix selections, ambiguous raw identities and incomplete quantity contracts."""
        if len(set(self.ranking)) != len(self.ranking) or self.selected != self.ranking[: len(self.selected)]:
            raise ValueError("Baseline selection must be a prefix of its complete unique ranking")
        if (
            self.record.namespace != "corpus.baseline"
            or self.record.execution_id != self.cell_id
            or self.record.decision_id != self.member_id
            or self.record.provenance.evaluation_fingerprint != self.plan_fingerprint
            or self.record.provenance.producer != "coleman.baseline-native"
            or self.record.provenance.producer_version != "1"
        ):
            raise ValueError("Baseline raw identity is not bound to its frozen cell/member/plan")
        self._validate_quantities()
        return self

    def _validate_quantities(self) -> None:
        """Keep complete named units, censoring, work availability and integral fallback counts explicit."""
        names = {item.name for item in self.record.measurements}
        if names != set(MATH_MEASUREMENTS + TIMING_MEASUREMENTS):
            raise ValueError("Baseline observation requires every declared mathematical and operational measurement")
        for measurement in self.record.measurements:
            expected = baseline_measurement(measurement.name, measurement.value)
            if measurement != expected:
                raise ValueError("Baseline measurement units, meaning, direction or availability are incompatible")
        first = next(item for item in self.record.measurements if item.name == "censored_first_failure_seconds")
        if (first.status == "observed") != (self.first_failure_state == "detected"):
            raise ValueError("Baseline first-failure censoring disagrees with measurement availability")
        required: set[str] = set(MATH_MEASUREMENTS)
        required -= {
            "full_universe_napfd",
            "full_universe_apfdc",
            "failing_test_recall",
            "censored_first_failure_seconds",
        }
        if any(item.value is None for item in self.record.measurements if item.name in required):
            raise ValueError("Completed baseline build requires observed work, budget and fallback quantities")
        fallback = self.record.measurement("missing_input_fallback_count").value
        if fallback is None or not fallback.is_integer() or fallback > len(self.ranking):
            raise ValueError("Baseline fallback count must be an integer within the test universe")

    @property
    def mathematical_fingerprint(self) -> str:
        """Hash measured mathematical evidence and membership, excluding assembly/attempt/timing noise."""
        return fingerprint(
            {
                "cell": self.cell_id,
                "plan": self.plan_fingerprint,
                "source_build": self.source_build_fingerprint,
                "step": self.step,
                "member": self.member_id,
                "ranking": list(self.ranking),
                "selected": list(self.selected),
                "first_failure_state": self.first_failure_state,
                "measurements": [
                    item.model_dump(mode="json")
                    for item in sorted(self.record.measurements, key=lambda item: item.name)
                    if item.name in MATH_MEASUREMENTS
                ],
            }
        )


class BaselineCellResult(Immutable):
    """One explicit terminal/incomplete execution cell, never an artificial zero score.

    Raw observation digests preserve partial builds for failed/aborted executions.
    ``unavailable`` includes an active or interrupted attempt needing retry.
    Failed and aborted runs remain part of the declared comparison; no aggregate
    may silently exclude them. Native run IDs may be null before execution.
    """

    method: Method
    seed: Seed
    budget: Quantity
    cell_id: Fingerprint
    state: Literal["completed", "failed", "aborted", "unavailable", "incompatible"]
    reasons: Annotated[tuple[Identifier, ...], Field(max_length=16)]
    native_run_id: Identifier | None
    observations: Annotated[tuple[Fingerprint, ...], Field(max_length=512)]
    mathematical_observations: Annotated[tuple[Fingerprint, ...], Field(max_length=512)]
    attempts: Annotated[tuple[Fingerprint, ...], Field(max_length=4096)]
    unclosed_attempts: Annotated[int, Field(strict=True, ge=0, le=4096)]
    resource_evidence: Literal["observed", "unavailable", "not_applicable"]

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Require explicit reasons and one immutable digest for each mathematical raw observation."""
        if self.budget.semantics != BUDGET_RATIO:
            raise ValueError("Baseline result requires native-work budget units")
        if (self.state == "completed") != (not self.reasons):
            raise ValueError("Incomplete/incompatible baseline cells require an explicit reason")
        if (
            len(self.observations) != len(self.mathematical_observations)
            or len(set(self.observations)) != len(self.observations)
            or len(set(self.mathematical_observations)) != len(self.mathematical_observations)
        ):
            raise ValueError("Baseline raw observation digest membership is inconsistent")
        if self.state == "incompatible" and (self.native_run_id is not None or self.observations):
            raise ValueError("Incompatible baseline cannot fabricate a native run or raw results")
        if self.state == "incompatible" and (
            self.attempts or self.unclosed_attempts or self.resource_evidence != "not_applicable"
        ):
            raise ValueError("Incompatible baseline has no native attempts or resource observations")
        if self.state != "incompatible" and self.resource_evidence == "not_applicable":
            raise ValueError("An attempted baseline cannot discard its resource evidence")
        if self.unclosed_attempts and self.resource_evidence != "unavailable":
            raise ValueError(
                "Interrupted attempts with missing terminal costs cannot certify complete resource evidence"
            )
        return self


class BaselineAttempt(VersionedContract):
    """One durable physical invocation event, separate from the stable logical matrix cell.

    A UUID identifies an invocation, not a new independent trial. Started events
    publish before the native call. Missing terminal events remain visible after
    process death; their unknown cost is never zero or silently discarded on
    checkpoint continuation. Event times denote actual recording, not inferred
    original CI clocks. Native completion and logical cell state remain separate.
    """

    cell_id: Fingerprint
    attempt_id: Identifier
    event: Literal["started", "terminal"]
    state: Literal["running", "completed", "failed", "aborted", "blocked"]
    recorded_at: Timestamp
    reasons: Annotated[tuple[Identifier, ...], Field(max_length=16)]
    host_elapsed: Measurement
    worker_elapsed: Measurement
    worker_peak: Measurement

    @model_validator(mode="after")
    def _event(self) -> Self:
        """Require actual typed units and prevent a started event from inventing terminal measurements."""
        if (self.event == "started") != (self.state == "running"):
            raise ValueError("Baseline attempt event/state disagree")
        fields = (
            (self.host_elapsed, "host_attempt_elapsed_seconds"),
            (self.worker_elapsed, "native_process_elapsed_seconds"),
            (self.worker_peak, "native_process_peak_mib"),
        )
        for value, name in fields:
            if value != baseline_measurement(name, value.value):
                raise ValueError("Baseline attempt measurement contract is incompatible")
        if self.event == "started" and (self.reasons or any(value.value is not None for value, _ in fields)):
            raise ValueError("A started attempt has no terminal cost or failure reason")
        if self.event == "terminal" and self.host_elapsed.status != "observed":
            raise ValueError("A terminal host invocation requires its observed elapsed time")
        if self.state in {"failed", "aborted", "blocked"} and not self.reasons:
            raise ValueError("Failed/interrupted/blocked attempts require an explicit reason")
        if self.state == "completed" and self.reasons:
            raise ValueError("Completed attempt cannot carry failure reasons")
        return self


class BaselineFilePin(Immutable):
    """One native relative artifact's actual complete bytes; no path is opened by the contract."""

    path: Annotated[str, Field(strict=True, min_length=1, max_length=4096)]
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=1073741824)]

    @model_validator(mode="after")
    def _relative(self) -> Self:
        """Reject absolute, traversal, control and noncanonical paths before application storage access."""
        parsed = PurePosixPath(self.path)
        if (
            parsed.is_absolute()
            or not parsed.parts
            or str(parsed) != self.path
            or ".." in parsed.parts
            or "\\" in self.path
            or ":" in self.path
            or any(ord(character) < 32 or ord(character) == 127 for character in self.path)
        ):
            raise ValueError("Baseline file pin requires a canonical safe relative artifact path")
        return self


class BaselineNativeCompletion(VersionedContract):
    """Native work and immutable byte inventory completed before the child releases its execution lock.

    The measured resource frontier belongs to this completion invocation. After
    interrupted checkpoint recovery, earlier invocations remain separate attempt
    evidence; missing earlier costs cannot be invented or reset to zero. Mutable
    operational manifests/resolved specs and the lock file are excluded from pins.
    Reading a receipt alone does not validate actual bytes or authenticate execution.
    """

    cell_id: Fingerprint
    plan_fingerprint: Fingerprint
    runtime_fingerprint: Fingerprint
    native_run_id: Identifier
    recorded_at: Timestamp
    completed_steps: Annotated[int, Field(strict=True, ge=1, le=512)]
    observations: Annotated[tuple[Fingerprint, ...], Field(min_length=1, max_length=512)]
    mathematical_observations: Annotated[tuple[Fingerprint, ...], Field(min_length=1, max_length=512)]
    files: Annotated[tuple[BaselineFilePin, ...], Field(min_length=1, max_length=8192)]
    worker_elapsed: Measurement
    worker_cpu: Measurement
    worker_peak: Measurement

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Require complete distinct raw membership, consistent resource units and unique canonical file roles."""
        if (
            len(self.observations) != self.completed_steps
            or len(self.mathematical_observations) != self.completed_steps
            or len(set(self.observations)) != self.completed_steps
            or len(set(self.mathematical_observations)) != self.completed_steps
            or len({entry.path for entry in self.files}) != len(self.files)
        ):
            raise ValueError("Native baseline completion has incomplete or duplicated evidence")
        for value, name in (
            (self.worker_elapsed, "native_process_elapsed_seconds"),
            (self.worker_cpu, "native_process_cpu_seconds"),
            (self.worker_peak, "native_process_peak_mib"),
        ):
            if value != baseline_measurement(name, value.value):
                raise ValueError("Native baseline completion resource semantics are incompatible")
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda item: item.path)))
        return self


class BaselineMatrixReport(VersionedContract):
    """Complete declared development grid, retaining all failures and incompatibilities.

    Raw evidence is referenced by immutable digest and validated separately against
    actual files at the application boundary. Reading alone does not authenticate
    provenance or certify completion. A report containing unavailable cells is
    incomplete and cannot publish a completion receipt. V1 grants no promotion,
    statistical comparison, empirical independence or protected test access.
    """

    plan: BaselinePlan
    producer_fingerprint: Fingerprint
    cells: Annotated[tuple[BaselineCellResult, ...], Field(min_length=1, max_length=65536)]
    conclusion: Literal["development_only_insufficient_evidence"] = "development_only_insufficient_evidence"

    @model_validator(mode="after")
    def _grid(self) -> Self:
        """Reject missing/duplicate/forged cell identities or omitted failed/incompatible candidates."""
        expected = {cell.cell_id: (cell.method, cell.seed, cell.budget) for cell in baseline_coordinates(self.plan)}
        ids = [cell.cell_id for cell in self.cells]
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("Baseline report requires the entire unique frozen candidate/seed/budget grid")
        eligibility = {item.method: item for item in baseline_eligibility(self.plan)}
        for cell in self.cells:
            self._check_cell(cell, expected[cell.cell_id], eligibility[cell.method])
        object.__setattr__(self, "cells", tuple(sorted(self.cells, key=lambda item: item.cell_id)))
        return self

    def _check_cell(
        self, cell: BaselineCellResult, dimensions: tuple[str, int, Quantity], compatible: BaselineEligibility
    ) -> None:
        """Verify each frozen coordinate retains its exact compatibility and complete raw membership."""
        if (cell.method, cell.seed, cell.budget) != dimensions:
            raise ValueError("Baseline cell identity does not match its declared dimensions")
        if not compatible.compatible and (cell.state != "incompatible" or cell.reasons != compatible.reasons):
            raise ValueError("Baseline incompatible source/candidate must retain its exact eligibility reasons")
        if compatible.compatible and cell.state == "incompatible":
            raise ValueError("An eligible baseline cannot be silently excluded")
        if cell.state == "completed" and (
            cell.native_run_id is None or len(cell.observations) != len(self.plan.source.builds)
        ):
            raise ValueError("Completed baseline requires every raw build and its native run")

    @property
    def complete(self) -> bool:
        """Return false for unavailable/aborted runs; failure remains a terminal result."""
        return all(cell.state not in {"unavailable", "aborted"} for cell in self.cells)

    @property
    def mathematical_fingerprint(self) -> str:
        """Retain failures/incompatibilities and mathematical raw digests while excluding operational noise."""
        return fingerprint(
            {
                "plan": self.plan.fingerprint,
                "cells": [
                    {
                        "id": cell.cell_id,
                        "state": cell.state,
                        "reasons": list(cell.reasons),
                        "observations": list(cell.mathematical_observations),
                    }
                    for cell in self.cells
                ],
            }
        )


class BaselineMatrixReceipt(VersionedContract):
    """Publish last after a terminal grid's immutable report; reading does not verify referenced files."""

    plan_fingerprint: Fingerprint
    runtime_fingerprint: Fingerprint
    report_fingerprint: Fingerprint
    mathematical_fingerprint: Fingerprint
    report_size_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)]
