"""Validated pilot checkpoints and reports on the existing immutable artifact writer."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.pilot.contracts import PilotManifest, PilotRun
from coleman.pilot.replay import verify_run
from coleman.results.domain import Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, ticks


class CandidateFinding(Immutable):
    """Descriptive predeclared safety gate; never a statistical promotion decision.

    Per-build ranges show heterogeneity, not confidence intervals. The gate can
    fail on a measured regression, but insufficient observations never pass.
    Financial figures remain modeled; willingness to adopt/pay is unmeasured.
    """

    candidate_id: Identifier
    gate: Literal["pass", "fail", "inconclusive"]
    distinct_builds: int
    recall_range: tuple[float, float] | None
    runner_saving_fraction_range: tuple[float, float] | None
    reasons: tuple[str, ...]
    adoption: Literal["insufficient_evidence"] = "insufficient_evidence"
    uncertainty: Literal["descriptive_range_only_no_independence_assumption"] = (
        "descriptive_range_only_no_independence_assumption"
    )
    commercial_demand: Literal["unmeasured"] = "unmeasured"


class PilotReport(VersionedContract):
    """Full provenance plus verified completed runs; also a restart checkpoint.

    Missing candidate/build pairs remain explicit and prevent completion. Same
    identity/same content retries deduplicate; conflicts fail. Interrupted writes
    use ArtifactWriter's local immutable publication semantics. Restore validates
    all completed events, then callers run only missing pairs through the native
    runner. This does not save learning state: use native checkpoints for that.
    No old native metrics/artifacts are rewritten or implicitly migrated.
    """

    manifest: PilotManifest
    runs: tuple[PilotRun, ...] = Field(default=(), max_length=640000)

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Recompute every admitted event and fail conflicting logical retries."""
        unique: dict[tuple[str, str], PilotRun] = {}
        for run in self.runs:
            verify_run(self.manifest, run)
            key = run.candidate_id, run.build_id
            if key in unique and unique[key] != run:
                raise ValueError("Conflicting candidate/build retry")
            unique[key] = run
        object.__setattr__(self, "runs", tuple(unique[key] for key in sorted(unique)))
        return self

    @property
    def missing(self) -> tuple[tuple[str, str], ...]:
        """List absent logical candidate/build pairs; repeated seeds count once."""
        present = {(run.candidate_id, run.build_id) for run in self.runs}
        return tuple(
            (candidate, build.build_id)
            for candidate in self.manifest.candidates
            for build in self.manifest.builds
            if (candidate, build.build_id) not in present
        )

    @property
    def status(self) -> Literal["partial", "completed"]:
        """Distinguish a complete replay matrix from missing/aborted experiments."""
        return "partial" if self.missing else "completed"

    def append(self, run: PilotRun) -> PilotReport:
        """Return a revalidated checkpoint; never mutate state or run a scheduler."""
        return PilotReport(schema_version=1, manifest=self.manifest, runs=(*self.runs, run))

    def findings(self) -> tuple[CandidateFinding, ...]:
        """Apply frozen per-build gates with every missing or unresolved run visible.

        These descriptive gates do not account for dependence, generalization or
        real deployment overhead. Adoption remains insufficient_evidence even if
        observed safety gates pass. Confirmatory claims require an independently
        justified protected comparison and representative operational evidence.
        """
        metrics = {(run.candidate_id, run.build_id): verify_run(self.manifest, run) for run in self.runs}
        runs = {(run.candidate_id, run.build_id): run for run in self.runs}
        thresholds = self.manifest.thresholds
        output = []
        for candidate in self.manifest.candidates:
            if candidate == self.manifest.baseline_id:
                continue
            recalls: list[float] = []
            savings: list[float] = []
            reasons: set[str] = set()
            failed = False
            paired = 0
            for build in self.manifest.builds:
                key = candidate, build.build_id
                baseline_key = self.manifest.baseline_id, build.build_id
                if key not in metrics or baseline_key not in metrics:
                    reasons.add("missing_candidate_or_baseline")
                    continue
                paired += 1
                metric, baseline = metrics[key], metrics[baseline_key]
                if metric.failing_test_recall is None or baseline.failing_test_recall is None:
                    reasons.add("unresolved_or_zero_failure_outcomes")
                else:
                    recalls.append(metric.failing_test_recall)
                    if metric.failing_test_recall < max(thresholds.minimum_recall, baseline.failing_test_recall):
                        failed = True
                        reasons.add("detection_regression_or_threshold_failure")
                baseline_work = ticks(baseline.runner_work.value)
                if baseline_work == 0:
                    reasons.add("zero_baseline_work")
                else:
                    saving = Fraction(baseline_work - ticks(metric.runner_work.value), baseline_work)
                    savings.append(float(saving))
                    if saving < Fraction(str(thresholds.minimum_runner_saving_fraction)):
                        failed = True
                        reasons.add("runner_saving_threshold_failure")
                if runs[key].prioritization_overhead.value > thresholds.maximum_overhead.value:
                    failed = True
                    reasons.add("overhead_threshold_failure")
            if paired < thresholds.minimum_builds:
                reasons.add("too_few_distinct_builds")
            if self.manifest.evidence_basis != "observed_ci":
                reasons.add("synthetic_replay_is_not_operational_evidence")
            gate = "fail" if failed else "inconclusive" if reasons else "pass"
            output.append(
                CandidateFinding(
                    candidate_id=candidate,
                    gate=gate,
                    distinct_builds=paired,
                    recall_range=(min(recalls), max(recalls)) if recalls else None,
                    runner_saving_fraction_range=(min(savings), max(savings)) if savings else None,
                    reasons=tuple(sorted(reasons)),
                )
            )
        return tuple(output)


def publish_pilot_report(report: PilotReport, writer: ArtifactWriter) -> Path:
    """Publish a complete validated snapshot; partial runs stay explicitly partial.

    Digest paths make identical concurrent retries idempotent. I/O errors propagate;
    no partial file receives a completed name. JSON contains source runs; summaries
    are recomputed on read rather than trusted. Unknown versions fail closed.
    """
    report = PilotReport.model_validate(report)
    path = writer.run_root / "pilot-reports" / f"{report.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, report.to_json() + "\n")
    return path


def render_pilot_report(report: PilotReport) -> str:
    """Render replayed per-build evidence, limitations and predeclared safety gates."""
    report = PilotReport.model_validate(report)
    lines = [
        f"# CI operational pilot: {report.manifest.pilot_id}",
        f"Matrix: {report.status}; missing pairs: {len(report.missing)}.",
        f"Evidence: {report.manifest.evidence_basis}. Source: {report.manifest.source_uri}.",
        f"License: {report.manifest.license_id}. Source digest: {report.manifest.source_sha256}.",
        f"Frozen manifest: {report.manifest.fingerprint}.",
        "Runner work/elapsed values are replay accounting from declared durations; "
        "prioritization overhead is separately measured. No queue/setup/fixed cost is claimed saved.",
        "| Candidate | Build | Recall | First failure s | Runner s | Elapsed s | Modeled credits | Overhead s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in report.runs:
        metric = verify_run(report.manifest, run)
        first = metric.first_failure_elapsed
        lines.append(
            f"| {run.candidate_id} | {run.build_id} | {metric.failing_test_recall} | "
            f"{first.value if first is not None else 'null (censored)'} | {metric.runner_work.value} | "
            f"{metric.execution_elapsed.value} | {metric.modeled_variable_cost.value} | "
            f"{run.prioritization_overhead.value} |"
        )
    for finding in report.findings():
        lines.append(
            f"Candidate {finding.candidate_id}: descriptive gate **{finding.gate}**; "
            f"reasons: {', '.join(finding.reasons) or 'predeclared descriptive gates satisfied'}. "
            f"Recall range {finding.recall_range}; runner saving fraction range {finding.runner_saving_fraction_range}."
        )
    lines.extend(
        [
            "Adoption: **insufficient evidence**. Commercial demand: **unmeasured**.",
            "Ranges describe observed heterogeneity, not confidence intervals. Repeated seeds do not add "
            "independent builds. No IID confidence interval or savings/profitability claim is made.",
            "Before adoption: verify source/coverage availability, representative independent projects or "
            "justified temporal blocks, protected comparisons, real end-to-end overhead, setup/queue/retry costs, "
            "required suites and conservative full-suite fallback when evidence or inputs are insufficient.",
        ]
    )
    rendered = "\n\n".join(lines) + "\n"
    # Markdown tables require adjacent rows, while prose needs blank lines.
    return rendered.replace("|\n\n|", "|\n|")
