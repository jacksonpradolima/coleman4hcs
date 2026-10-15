"""Private source admission checks; validation never dispatches a job or reads a clock."""

from coleman.observations import DecisionSignal, ObservationHistory, ObservationRevision
from coleman.results.domain import fingerprint
from coleman.shadow._build import _Build
from coleman.shadow.contracts import (
    BuildClosed,
    BuildOpened,
    CandidateDecision,
    FrozenCandidate,
    HostClock,
    ShadowProtocol,
    TestFeedback,
)


def opening_reason(
    protocol: ShadowProtocol, event: BuildOpened, clock: HostClock, builds: dict[str, _Build], healthy: bool
) -> str | None:
    """Require protected forward declarations and seal all available candidate trials."""
    if event.build_id in builds:
        return "conflicting_build_declaration"
    plan = protocol.temporal_plan
    if not plan.validation_end < event.decided_at <= plan.test_end:
        return "build_outside_forward_window"
    if event.target_end > plan.test_end:
        return "build_target_exceeds_forward_window"
    if event.decided_at > clock.utc_at:
        raise ValueError("Build declaration precedes its declared decision instant")
    retry_issue = retry_reason(event, builds)
    if retry_issue is not None:
        return retry_issue
    estimate = event.actual_estimate
    if estimate.amount.semantics != protocol.actual_build_capacity.semantics:
        return "incompatible_actual_estimate_units"
    if not estimate.available_at <= event.decided_at <= estimate.valid_until:
        return "actual_estimate_unavailable_at_upstream_start"
    if estimate.amount.value > protocol.actual_build_capacity.value:
        return "actual_estimate_exceeds_declared_capacity"
    if event.decisions:
        if not healthy:
            return "recommendations_blocked_by_source_or_risk_state"
        if event.decided_at != clock.utc_at:
            return "recommendation_must_be_sealed_at_application_decision"
        return decisions_reason(protocol, event)
    return None


def decisions_reason(protocol: ShadowProtocol, event: BuildOpened) -> str | None:
    """Validate all frozen candidate bindings without receiving future outcomes."""
    candidates = {candidate.candidate_id: candidate for candidate in protocol.candidates}
    if {decision.candidate_id for decision in event.decisions} != set(candidates):
        return "incomplete_frozen_candidate_trial_registry"
    for decision in event.decisions:
        issue = decision_reason(protocol, event, decision, candidates[decision.candidate_id])
        if issue is not None:
            return issue
    return None


def retry_reason(event: BuildOpened, builds: dict[str, _Build]) -> str | None:
    """Require full-suite retries of the latest closed attempt; do not inflate independent builds."""
    attempts = [build for build in builds.values() if build.opening.logical_build_id == event.logical_build_id]
    if event.retry_of is None:
        return "new_attempt_requires_explicit_retry_parent" if attempts else None
    parent = builds.get(event.retry_of)
    if parent is None or parent.closure is None or parent.opening.logical_build_id != event.logical_build_id:
        return "retry_requires_closed_parent_and_same_logical_build"
    if not attempts or parent is not attempts[-1]:
        return "retry_parent_must_be_latest_attempt"
    closure = parent.closure.event
    if not isinstance(closure, BuildClosed) or event.decided_at < closure.available_at:
        return "retry_precedes_upstream_parent_closure"
    if event.group_id != parent.opening.group_id or event.required_ids != parent.opening.required_ids:
        return "retry_changes_required_universe_or_independent_group"
    return None


def decision_reason(
    protocol: ShadowProtocol,
    event: BuildOpened,
    decision: CandidateDecision,
    candidate: FrozenCandidate,
) -> str | None:
    """Bind one predecision request to its frozen model, units, context and complete universe."""
    request = decision.request
    fallback = "no_action" if candidate.fallback_version == "no-action-v1" else "full_suite"
    actual = (
        decision.model_fingerprint,
        request.configuration_fingerprint,
        request.capacity,
        decision.shared_setup_reserve,
        request.decided_at,
        request.decision_id,
        sorted(request.ranking),
        request.fallback,
    )
    expected = (
        candidate.model_fingerprint,
        fingerprint(candidate.model_dump(mode="json")),
        candidate.budget,
        candidate.shared_setup_reserve,
        event.decided_at,
        event.build_id,
        sorted(event.required_ids),
        fallback,
    )
    if actual != expected:
        return "decision_differs_from_frozen_candidate_or_required_universe"
    if any(
        item.estimate is not None and item.estimate.source_fingerprint != candidate.feature_fingerprint
        for item in request.tests
    ):
        return "estimate_differs_from_frozen_feature_evidence"
    for signal in decision.signals:
        issue = signal_reason(protocol, event, candidate, signal)
        if issue is not None:
            return issue
        if signal.availability != "available" and request.priority_status == "available":
            return "unavailable_signal_cannot_support_available_ranking"
    return None


def signal_reason(
    protocol: ShadowProtocol,
    event: BuildOpened,
    candidate: FrozenCandidate,
    signal: DecisionSignal,
) -> str | None:
    """Prevent source/model/cutoff/horizon substitution without executing code by fingerprint."""
    request_binding = (
        signal.request.decision_id,
        signal.request.target_start,
        signal.request.target_end,
        signal.request.decided_at,
    )
    expected_binding = (event.build_id, event.decided_at, event.target_end, event.decided_at)
    if (
        signal.provenance.model_fingerprint != candidate.model_fingerprint
        or signal.provenance.frozen_at > protocol.frozen_at
        or signal.provenance.training_cutoff > protocol.temporal_plan.train_end
        or request_binding != expected_binding
        or signal.request.entity_id not in event.required_ids
    ):
        return "signal_differs_from_frozen_model_or_decision_cutoff"
    return None


def feedback_reason(protocol: ShadowProtocol, event: TestFeedback, clock: HostClock, build: _Build) -> str | None:
    """Keep missing parents, identity collisions and unsupported work corrections visible."""
    revision = event.revision
    if revision.received_at > clock.utc_at:
        raise ValueError("Provider first receipt is later than application admission")
    if event.test_id not in build.opening.required_ids:
        return "unknown_required_test"
    if revision.entity_id != event.test_id or revision.metadata.group_id != build.opening.group_id:
        return "feedback_entity_or_group_mismatch"
    if revision.metadata.event_at < build.opening.decided_at:
        return "feedback_precedes_sealed_build"
    if (
        build.closure is not None
        and isinstance(build.closure.event, BuildClosed)
        and revision.metadata.event_at > build.closure.event.available_at
    ):
        return "physical_completion_after_upstream_closure"
    return lineage_reason(protocol, event, clock, build)


def lineage_reason(protocol: ShadowProtocol, event: TestFeedback, clock: HostClock, build: _Build) -> str | None:
    """Validate correction parents, immutable physical work and application-visible lineage."""
    revision = event.revision
    previous = build.feedback.get(event.test_id, [])
    if not previous and revision.supersedes is not None:
        return "correction_parent_unavailable"
    if previous:
        first = previous[0].event
        if not isinstance(first, TestFeedback):
            raise ValueError("Internal feedback identity is invalid")
        if first.occupancy != event.occupancy:
            return "occupancy_corrections_unsupported_by_resource_v1"
        if first.revision.metadata.observation_id != revision.metadata.observation_id:
            return "physical_test_has_conflicting_logical_observation"
    try:
        history = build.history(protocol)
        admitted = ObservationRevision.model_validate({**revision.model_dump(), "received_at": clock.utc_at})
        ObservationHistory.model_validate({**history.model_dump(), "revisions": (*history.revisions, admitted)})
    except ValueError:
        return "conflicting_or_incomplete_observation_lineage"
    return None
