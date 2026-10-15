"""Descriptive prospective gates with explicit observed, modeled and unavailable evidence."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.resources import RUNNER_MS
from coleman.results.domain import Identifier, Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity
from coleman.shadow.contracts import BuildOpened, TestFeedback
from coleman.shadow.decisions import resolve_selection
from coleman.shadow.state import ShadowBuildSnapshot, ShadowState


class ShadowFinding(Immutable):
    """One attempt/candidate's modeled membership against observed required-CI labels.

    Cost is a proxy: selected source occupancy plus one shared setup and the
    candidate's declared inference cost. It is never actual saved CI work. Missing
    declarations, censored labels or unknown costs remain null. Source raw work,
    queue/setup elapsed and actual booking stay in the accompanying typed state.
    """

    build_id: Identifier
    logical_build_id: Identifier
    group_id: Identifier
    candidate_id: Identifier
    selected_ids: tuple[Identifier, ...] | None
    recall: Measurement
    modeled_work: Measurement
    modeled_cost: Measurement
    net_saving: Measurement
    reasons: tuple[Identifier, ...]


class ShadowEvidence(Immutable):
    """Predeclared descriptive forward gates; no IID estimate or automatic promotion.

    Distinct logical CI decision days/builds and declared source groups count;
    delivery retries, source corrections, seeds and job retries add no independent
    units. Functional fixtures always remain inconclusive with zero genuine
    prospective builds, even when their simulated dates span the whole window.
    """

    outcome: Literal["pass", "fail", "inconclusive"]
    reasons: tuple[Identifier, ...]
    received_logical_builds: Annotated[int, Field(strict=True, ge=0, le=4096)]
    scored_logical_builds: Annotated[int, Field(strict=True, ge=0, le=4096)]
    distinct_decision_days: Annotated[int, Field(strict=True, ge=0, le=4096)]
    forward_logical_builds: Annotated[int, Field(strict=True, ge=0, le=4096)]
    independent_source_groups: tuple[Identifier, ...]
    known_adverse_findings: bool
    actual_known_required_runner: Quantity
    actual_accounting_complete: bool
    uncertainty: Literal["not_estimated_no_iid_inference"] = "not_estimated_no_iid_inference"
    adoption: Literal["insufficient_evidence"] = "insufficient_evidence"


def _metric(name: str, value: int | float | None, *, meaning: str, unit: str = "ms") -> Measurement:
    """Project exact bounded integer ms or a recall ratio, preserving null unavailable states."""
    return Measurement(
        name=name,
        value=float(value) if value is not None else None,
        unit=unit,
        meaning=meaning,
        direction="maximize" if name.endswith(("recall", "saving")) else "minimize",
        status="observed" if value is not None else "unavailable",
    )


def _recall(build: ShadowBuildSnapshot, cutoff, selected: tuple[str, ...] | None) -> tuple[Measurement, bool, bool]:
    """Score only complete application-visible raw labels; no failures makes recall undefined."""
    labels = {row.entity_id: row.measurement for row in build.history.as_of(cutoff).observations}
    complete = (
        build.closure is not None
        and build.application_closed_at is not None
        and build.application_closed_at.utc_at <= cutoff
        and set(labels) == set(build.opening.required_ids)
    )
    known = complete and all(measure.status == "observed" for measure in labels.values())
    failures = {key for key, measure in labels.items() if measure.value is not None and int(measure.value) == 1}
    value = None
    if selected is not None and known and failures:
        value = float(Fraction(len(failures.intersection(selected)), len(failures)))
    metric = _metric(
        "shadow_membership_recall", value, meaning="selected-membership-recall-against-required-suite-v1", unit="1"
    )
    if known and not failures:
        metric = Measurement.model_validate({**metric.model_dump(), "status": "not_applicable"})
    return metric, complete, known


def _modeled_cost(state: ShadowState, build: ShadowBuildSnapshot, decision, selected, complete):
    """Use first physical occupancy once; corrections never add work or infer unknown cost."""
    work_by_test = {}
    for entry in state.journal:
        event = entry.event
        if (
            entry.disposition == "accepted"
            and isinstance(event, TestFeedback)
            and event.build_id == build.opening.build_id
            and event.test_id not in work_by_test
        ):
            work_by_test[event.test_id] = event.occupancy
    work = None
    if selected is not None and complete and all(work_by_test.get(key) is not None for key in selected):
        work = sum(work_by_test[key].value for key in selected)
    cost = None
    closure = build.closure
    if (
        work is not None
        and decision is not None
        and decision.decision_overhead is not None
        and closure is not None
        and closure.shared_setup_occupancy is not None
    ):
        cost = work + decision.decision_overhead.value + closure.shared_setup_occupancy.value
    return work, cost


def _finding(state: ShadowState, build: ShadowBuildSnapshot, candidate_id: str) -> ShadowFinding:
    """Compute one sealed subset without treating actual required-suite execution as avoided."""
    cutoff = min(state.journal[-1].clock.utc_at, build.opening.target_end, state.protocol.temporal_plan.test_end)
    decision = next((item for item in build.opening.decisions if item.candidate_id == candidate_id), None)
    selected = resolve_selection(decision).selected_ids if decision is not None else None
    recall, complete, known = _recall(build, cutoff, selected)
    work, cost = _modeled_cost(state, build, decision, selected, complete)
    reasons = []
    if decision is None:
        reasons.append("recommendation_not_recorded")
    if not known:
        reasons.append("incomplete_or_censored_required_labels")
    elif recall.status == "not_applicable":
        reasons.append("zero_failure_recall_not_applicable")
    if cost is None:
        reasons.append("unknown_modeled_resource_cost")
    return ShadowFinding(
        build_id=build.opening.build_id,
        logical_build_id=build.opening.logical_build_id,
        group_id=build.opening.group_id,
        candidate_id=candidate_id,
        selected_ids=selected,
        recall=recall,
        modeled_work=_metric("shadow_modeled_work", work, meaning="selected-membership-source-runner-work-proxy-v1"),
        modeled_cost=_metric(
            "shadow_modeled_cost", cost, meaning="selected-source-work-plus-shared-setup-and-inference-proxy-v1"
        ),
        net_saving=_metric(
            "shadow_modeled_net_saving", None, meaning="baseline-minus-candidate-modeled-runner-cost-v1"
        ),
        reasons=tuple(reasons),
    )


def _evidence_units(state: ShadowState, rows: tuple[ShadowFinding, ...]):
    """Count paired logical builds and first-attempt days/groups, never delivery or job retries."""
    builds = state.builds()
    complete_attempts = {
        build.opening.build_id
        for build in builds
        if all(not row.reasons for row in rows if row.build_id == build.opening.build_id)
    }
    logical = {build.opening.logical_build_id for build in builds}
    scored = {build.opening.logical_build_id for build in builds if build.opening.build_id in complete_attempts}
    first: dict[str, BuildOpened] = {}
    for entry in state.journal:
        if entry.disposition == "accepted" and isinstance(entry.event, BuildOpened):
            first.setdefault(entry.event.logical_build_id, entry.event)
    days = {first[key].decided_at.date() for key in scored}
    groups = tuple(sorted({first[key].group_id for key in scored}.intersection(state.protocol.independent_groups)))
    return logical, scored, days, groups


def _beneficial(row: ShadowFinding, minimum_ms: int) -> bool:
    """Require a positive practical benefit; zero savings cannot pass."""
    value = row.net_saving.value
    return value is not None and value > 0 and value >= minimum_ms


def _outcome(reasons: list[str], candidates, adverse: bool, minimum_ms: int):
    """Keep insufficient evidence inconclusive before considering descriptive quality or cost."""
    if reasons:
        return "inconclusive"
    if adverse:
        return "fail"
    if candidates and all(_beneficial(row, minimum_ms) for row in candidates):
        return "pass"
    reasons.append("practical_benefit_not_demonstrated")
    return "inconclusive"


class ShadowReport(VersionedContract):
    """Deterministic snapshot of accepted state, with summaries recomputed rather than trusted.

    This historical report describes health as of the last committed admission.
    Query ShadowState.operational(current_clock) for live health after restart or
    silence. No current clock is read or silently inserted into mathematical output.
    """

    state: ShadowState

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Require a nonempty admitted journal, without inventing an evaluation timestamp."""
        if not self.state.journal:
            raise ValueError("Shadow report requires at least one source admission")
        return self

    def findings(self) -> tuple[ShadowFinding, ...]:
        """Return all frozen trials per observed attempt, including missing recommendations."""
        state = ShadowState.model_validate(self.state)
        output = []
        for build in state.builds():
            rows = [_finding(state, build, candidate.candidate_id) for candidate in state.protocol.candidates]
            baseline = next(row for row in rows if row.candidate_id == state.protocol.baseline_id)
            for row in rows:
                saving = None
                if baseline.modeled_cost.value is not None and row.modeled_cost.value is not None:
                    saving = int(baseline.modeled_cost.value) - int(row.modeled_cost.value)
                output.append(
                    ShadowFinding.model_validate(
                        {
                            **row.model_dump(),
                            "net_saving": _metric(
                                "shadow_modeled_net_saving",
                                saving,
                                meaning="baseline-minus-candidate-modeled-runner-cost-v1",
                            ),
                        }
                    )
                )
        return tuple(output)

    def evidence(self) -> ShadowEvidence:
        """Apply frozen minimum days/builds/groups before any descriptive conclusion."""
        state = ShadowState.model_validate(self.state)
        builds = state.builds()
        rows = self.findings()
        candidates = tuple(row for row in rows if row.candidate_id != state.protocol.baseline_id)
        logical, scored, days, groups = _evidence_units(state, rows)
        reasons = _insufficiency_reasons(state, rows, len(days), len(scored), len(groups))
        adverse = _known_adverse(state, rows)
        outcome = _outcome(reasons, candidates, adverse, state.protocol.minimum_net_saving_ms)
        return ShadowEvidence(
            outcome=outcome,
            reasons=tuple(reasons),
            received_logical_builds=len(logical),
            scored_logical_builds=len(scored),
            distinct_decision_days=len(days),
            independent_source_groups=groups,
            known_adverse_findings=adverse,
            forward_logical_builds=len(scored)
            if state.protocol.source.evidence_basis == "authorized_forward_ci"
            else 0,
            actual_known_required_runner=Quantity(
                value=Decimal(sum(build.actual_ledger.balance.consumed for build in builds)),
                semantics=RUNNER_MS,
            ),
            actual_accounting_complete=bool(builds)
            and all(build.resource_accounting == "settled" for build in builds)
            and all(entry.disposition == "accepted" for entry in state.journal),
        )


def _known_adverse(state: ShadowState, rows: tuple[ShadowFinding, ...]) -> bool:
    """Keep known sticky resource/quality breaches even after a later label correction.

    Unavailable inference is operational containment, not a measured adverse
    effect. Quarantined inputs likewise cannot manufacture a numerical result.
    """
    known_breaches = state._engine().risk_reasons - {"unavailable_candidate_inference_cost"}
    return bool(known_breaches) or any(_adverse(row, state.protocol.minimum_recall) for row in rows)


def _adverse(row: ShadowFinding, minimum_recall: float) -> bool:
    """Retain known negative costs/recall without turning unavailable values into zero."""
    return (
        row.recall.value is not None
        and row.recall.value < minimum_recall
        or row.net_saving.value is not None
        and row.net_saving.value < 0
    )


def _insufficiency_reasons(
    state: ShadowState, rows: tuple[ShadowFinding, ...], days: int, builds: int, groups: int
) -> list[str]:
    """Keep functional, partial, dependent and too-short evidence explicitly inconclusive."""
    protocol = state.protocol
    reasons = []
    if protocol.source.evidence_basis != "authorized_forward_ci":
        reasons.append("offline_functional_recovery_is_not_prospective_evidence")
    if days < protocol.minimum_observation_days:
        reasons.append("insufficient_distinct_logical_ci_decision_days")
    if builds < protocol.minimum_builds:
        reasons.append("too_few_scored_logical_builds")
    if groups < protocol.minimum_independent_groups or protocol.independence_justification is None:
        reasons.append("insufficient_justified_independent_source_groups")
    if any(row.reasons for row in rows):
        reasons.append("missing_or_censored_candidate_or_baseline_evidence")
    if any(entry.disposition == "quarantined" for entry in state.journal):
        reasons.append("quarantined_source_inputs")
    if any(build.resource_accounting != "settled" for build in state.builds()):
        reasons.append("incomplete_actual_required_ci_accounting")
    return reasons


def publish_shadow_report(report: ShadowReport, writer: ArtifactWriter) -> Path:
    """Atomically publish verified source-backed evidence, never a fabricated forward observation."""
    report = ShadowReport.model_validate(report)
    path = writer.run_root / "shadow-reports" / f"{report.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, report.to_json() + "\n")
    return path


def render_shadow_report(report: ShadowReport) -> str:
    """Render actual work separately from proxy selections and insufficient prospective evidence."""
    report = ShadowReport.model_validate(report)
    evidence = report.evidence()
    clock = report.state.journal[-1].clock
    health = report.state.operational(clock)
    lines = [
        "# Frozen CI shadow evaluation",
        f"Experiment: {report.state.logical_id}.",
        f"Source basis: {report.state.protocol.source.evidence_basis}. Operational snapshot: {health.status}.",
        f"Research outcome: **{evidence.outcome}**. "
        f"Reasons: {', '.join(evidence.reasons) or 'descriptive gates only'}.",
        f"Distinct logical decision days: {evidence.distinct_decision_days}; "
        f"required: {report.state.protocol.minimum_observation_days}. "
        f"Scored logical builds: {evidence.scored_logical_builds}; "
        f"genuine forward builds: {evidence.forward_logical_builds}.",
        f"Actual known required-CI runner subtotal: {evidence.actual_known_required_runner.value} ms; "
        f"complete accounting: {evidence.actual_accounting_complete}.",
        "| Attempt | Candidate | Selected IDs | Membership recall | Modeled work ms | "
        "Modeled cost ms | Net saving ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.findings():
        lines.append(
            f"| {row.build_id} | {row.candidate_id} | {row.selected_ids} | "
            f"{row.recall.value} ({row.recall.status}) | {row.modeled_work.value} | "
            f"{row.modeled_cost.value} | {row.net_saving.value} |"
        )
    lines.append(
        "Actual required CI execution remains unchanged. Selected-membership outcomes and additive source-duration "
        "costs are counterfactual proxies, not observed rerun outcomes, calibrated probabilities, wall-clock savings "
        "or provider bills. Shared setup is counted once, never removed per test. Unknown costs/labels remain null. "
        "Queue/setup elapsed remain in the typed source closures and are not inferred from runner work. "
        "Repeated seeds/retries/corrections do not create independent builds or days. "
        "Uncertainty is not estimated and no IID inference is made. Adoption: **insufficient evidence**. "
        "Offline recovery cannot establish prospective or commercial value. Query a current host clock for live health."
    )
    return ("\n\n".join(lines) + "\n").replace("|\n\n|", "|\n|")
