"""Auditable paired resource comparisons and publication using existing artifacts."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.pilot.contracts import AdoptionThresholds
from coleman.resources.replay import ResourceReplayProtocol, ResourceReplayRun
from coleman.results.domain import Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable


class ResourceComparison(Immutable):
    """One same-period/budget/outcome pair, retaining signed work and net differences.

    Positive differences mean candidate uses fewer declared milliseconds. Overhead
    is included only in net difference, and may erase apparent work savings.
    Null quality or incomplete accounting prevents a passing descriptive gate.
    A gate never promotes a candidate or establishes independent observations.
    """

    candidate_id: Identifier
    build_id: Identifier
    work_saving_ms: Annotated[int, Field(strict=True)]
    net_saving_ms: Annotated[int, Field(strict=True)]
    gate: Literal["pass", "fail", "inconclusive"]
    reasons: tuple[Identifier, ...]
    adoption: Literal["insufficient_evidence"] = "insufficient_evidence"


def _compare_pair(
    run: ResourceReplayRun,
    baseline: ResourceReplayRun,
    thresholds: AdoptionThresholds,
) -> ResourceComparison:
    """Apply declared per-build descriptive gates to one verified pair, retaining signed losses."""
    current, base = run.accounting(), baseline.accounting()
    work_saving = base.charged_ledger.balance.runner_occupancy - current.charged_ledger.balance.runner_occupancy
    net_saving = base.charged_ledger.balance.consumed - current.charged_ledger.balance.consumed
    reasons = []
    failed = False
    if current.failing_test_recall.value is None or base.failing_test_recall.value is None:
        reasons.append("unavailable_or_not_applicable_recall")
    elif current.failing_test_recall.value < max(base.failing_test_recall.value, thresholds.minimum_recall):
        reasons.append("recall_regression")
        failed = True
    if net_saving <= 0:
        reasons.append("no_positive_net_resource_saving")
        failed = True
    baseline_work = base.charged_ledger.balance.runner_occupancy
    if not baseline_work:
        reasons.append("zero_baseline_work")
    elif Fraction(work_saving, baseline_work) < Fraction(str(thresholds.minimum_runner_saving_fraction)):
        reasons.append("runner_saving_threshold_failure")
        failed = True
    if run.native_run.prioritization_overhead.value > thresholds.maximum_overhead.value:
        reasons.append("overhead_threshold_failure")
        failed = True
    if current.charged_ledger.balance.containment != "normal":
        reasons.append("resource_overrun_containment")
        failed = True
    if base.charged_ledger.balance.containment != "normal":
        reasons.append("baseline_resource_overrun_containment")
    if "partial_unmeasured_setup" in {current.accounting_status, base.accounting_status}:
        reasons.append("unmeasured_setup_occupancy")
    return ResourceComparison(
        candidate_id=run.native_run.candidate_id,
        build_id=run.native_run.build_id,
        work_saving_ms=work_saving,
        net_saving_ms=net_saving,
        gate="fail" if failed else "inconclusive" if reasons else "pass",
        reasons=tuple(reasons),
    )


class ResourceReplayReport(VersionedContract):
    """Verified completed native runs and counterfactual ledgers under one protocol.

    Runs are keyed by registered candidate/build, not seed or native attempt UUID.
    Duplicate canonical payloads deduplicate; changed observations/timing conflict.
    Missing pairs remain partial; repeated seeds create no independent evidence.
    The existing pilot protocol provides temporal cutoffs and complete candidate
    trial registry. This report supplies no additional statistical inference.
    """

    protocol: ResourceReplayProtocol
    runs: Annotated[tuple[ResourceReplayRun, ...], Field(max_length=64000)] = ()

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Reject mixed protocols and conflicting logical attempts on every read."""
        unique: dict[tuple[str, str], ResourceReplayRun] = {}
        for run in self.runs:
            if run.protocol.to_json() != self.protocol.to_json():
                raise ValueError("Resource comparison requires identical frozen protocol")
            key = run.native_run.candidate_id, run.native_run.build_id
            if key in unique and unique[key].to_json() != run.to_json():
                raise ValueError("Conflicting resource candidate/build replay")
            unique[key] = run
        object.__setattr__(self, "runs", tuple(unique[key] for key in sorted(unique)))
        return self

    @property
    def missing(self) -> tuple[tuple[str, str], ...]:
        """Return absent registered pairs without counting trials/seeds as independent builds."""
        seen = {(run.native_run.candidate_id, run.native_run.build_id) for run in self.runs}
        manifest = self.protocol.native_manifest
        return tuple(
            (candidate, build.build_id)
            for candidate in manifest.candidates
            for build in manifest.builds
            if (candidate, build.build_id) not in seen
        )

    @property
    def status(self) -> Literal["partial", "completed"]:
        """Declare matrix completion only; completed replay is not prospective evidence."""
        return "partial" if self.missing else "completed"

    def append(self, run: ResourceReplayRun) -> ResourceReplayReport:
        """Return a new validated snapshot; invalid input cannot mutate acknowledged state."""
        validated = ResourceReplayReport.model_validate(self)
        run = ResourceReplayRun.model_validate(run)
        key = run.native_run.candidate_id, run.native_run.build_id
        for previous in validated.runs:
            if (previous.native_run.candidate_id, previous.native_run.build_id) == key:
                if previous.to_json() != run.to_json():
                    raise ValueError("Conflicting resource candidate/build replay")
                return validated
        return ResourceReplayReport(schema_version=1, protocol=validated.protocol, runs=(*validated.runs, run))

    def comparisons(self) -> tuple[ResourceComparison, ...]:
        """Pair only same-build frozen baseline/candidate observations, keeping losses.

        The gate fails an observed recall regression, nonpositive net saving or
        overrun. Otherwise null/incomplete accounting is inconclusive. Passing
        means descriptive fixture thresholds only; adoption remains insufficient.
        Missing pairs appear in `missing` and cannot contribute favorable evidence.
        """
        report = ResourceReplayReport.model_validate(self)
        indexed = {(run.native_run.candidate_id, run.native_run.build_id): run for run in report.runs}
        baseline_id = report.protocol.native_manifest.baseline_id
        thresholds = report.protocol.native_manifest.thresholds
        results = []
        for (candidate, build_id), run in indexed.items():
            baseline = indexed.get((baseline_id, build_id))
            if candidate == baseline_id or baseline is None:
                continue
            results.append(_compare_pair(run, baseline, thresholds))
        return tuple(results)


def publish_resource_run(run: ResourceReplayRun, writer: ArtifactWriter) -> Path:
    """Commit one logical run atomically; identical retries succeed and changed bytes fail.

    Publish before native checkpoint advancement. A process crash before commit
    leaves no acknowledged result; after commit reuse the original timing rather
    than remeasure it. Existing ArtifactWriter filesystem/durability limits apply.
    """
    run = ResourceReplayRun.model_validate(run)
    path = writer.run_root / "resource-runs" / f"{run.logical_id.removeprefix('sha256:')}.json"
    writer.write_immutable(path, run.to_json() + "\n")
    return path


def publish_resource_report(report: ResourceReplayReport, writer: ArtifactWriter) -> Path:
    """Atomically publish a content-addressed verified matrix without guessing completeness."""
    report = ResourceReplayReport.model_validate(report)
    path = writer.run_root / "resource-reports" / f"{report.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, report.to_json() + "\n")
    return path


def render_resource_report(report: ResourceReplayReport) -> str:
    """Render human CI accounting with signed results and explicit measurement limits."""
    report = ResourceReplayReport.model_validate(report)
    lines = [
        "# CI resource replay",
        f"Matrix: {report.status}; missing pairs: {len(report.missing)}.",
        f"Frozen protocol: {report.protocol.fingerprint}.",
        "This is counterfactual historical/synthetic replay, not prospective observed CI savings. "
        "Native required suites remain fully executed. Resource units are whole occupied runner ms; "
        "dispatch elapsed is rounded up to milliseconds and modeled as one occupied runner. "
        "Modeled credits exclude fixed/unmeasured costs. Queue/setup elapsed is separate from runner work.",
        "| Candidate | Build | Runner ms | Execution elapsed ms | Charged overhead ms | Recall | Queue ms | "
        "Setup elapsed ms | Modeled credits subtotal | Accounting |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in report.runs:
        current = run.accounting()
        lines.append(
            f"| {run.native_run.candidate_id} | {run.native_run.build_id} | {current.runner_work.value} | "
            f"{current.execution_elapsed.value} | {current.overhead.value} | "
            f"{current.failing_test_recall.value} ({current.failing_test_recall.status}) | "
            f"{run.queue_elapsed.value if run.queue_elapsed is not None else 'null (unavailable)'} | "
            f"{run.setup_elapsed.value if run.setup_elapsed is not None else 'null (unavailable)'} | "
            f"{current.accounted_modeled_cost.value} | {current.accounting_status} |"
        )
    for comparison in report.comparisons():
        lines.append(
            f"{comparison.candidate_id}/{comparison.build_id}: work saving {comparison.work_saving_ms} ms; "
            f"net saving including charged overhead {comparison.net_saving_ms} ms; gate {comparison.gate}; "
            f"reasons {', '.join(comparison.reasons) or 'descriptive gates satisfied'}."
        )
    lines.append(
        "Adoption: **insufficient evidence**. No IID claim, confidence interval, calibrated probability, "
        "production savings or commercial value is inferred. Builds may be dependent; seeds do not add "
        "observations. Negative and inconclusive comparisons remain valid results. No-action/full-suite "
        "fallbacks apply only to the modeled selection and never waive required CI execution."
    )
    return ("\n\n".join(lines) + "\n").replace("|\n\n|", "|\n|")
