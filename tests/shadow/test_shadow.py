"""Owned forward-arrival fixtures for immutable shadow admission and required-CI work."""

from datetime import datetime, timedelta

import pytest

from coleman.evaluation.protected.temporal import TemporalObservation, TemporalPlan
from coleman.observations import ObservationRevision
from coleman.resources import RUNNER_MS, ResourceAmount, ResourceEstimate, ResourceRequest
from coleman.resources import TestEstimate as CostEstimate
from coleman.results.domain import Measurement, fingerprint
from coleman.shadow.contracts import (
    BuildClosed,
    BuildOpened,
    CandidateDecision,
    FrozenCandidate,
    HostClock,
    ShadowProtocol,
    ShadowSource,
    SourceStatus,
)
from coleman.shadow.contracts import (
    TestFeedback as CIFeedback,
)
from coleman.shadow.state import ShadowState

FP = fingerprint({"owned": "shadow-ci-v1"})
START = datetime.fromisoformat("2026-01-02T10:00:00Z")


def amount(ms):
    return ResourceAmount(value=ms, semantics=RUNNER_MS)


def estimate(ms):
    return ResourceEstimate(
        amount=amount(ms),
        source_fingerprint=FP,
        observed_at=datetime.fromisoformat("2025-12-29T00:00:00Z"),
        available_at=datetime.fromisoformat("2025-12-30T00:00:00Z"),
        valid_until=datetime.fromisoformat("2026-01-10T00:00:00Z"),
    )


def clock(seconds, tick=0, session="first"):
    return HostClock(session_id=session, tick_ns=tick, utc_at=START + timedelta(seconds=seconds))


def protocol(basis="functional_offline", maximum_events=64, pending=4):
    candidates = tuple(
        FrozenCandidate(
            candidate_id=name,
            policy_version="declared-prior-order-v1",
            model_fingerprint=FP,
            feature_fingerprint=FP,
            selection_version="serial-prefix-v1",
            fallback_version="no-action-v1",
            budget=amount(3000),
            shared_setup_reserve=amount(0),
        )
        for name in ("base", "estimate")
    )
    return ShadowProtocol.model_validate(
        {
            "schema_version": 1,
            "experiment_id": "owned-shadow-ci-v1",
            "source": ShadowSource(
                source_id="owned-ci",
                source_version="1",
                schema_fingerprint=FP,
                authorization_reference="urn:coleman:owned-fixture",
                clock_assumption="owned-synchronized-fixture-v1",
                license_id="MIT",
                evidence_basis=basis,
            ),
            "benchmark_fingerprint": FP,
            "frozen_at": "2026-01-01T00:00:00Z",
            "temporal_plan": TemporalPlan.model_validate(
                {
                    "train_end": "2025-12-31T00:00:00Z",
                    "validation_end": "2026-01-01T00:00:00Z",
                    "test_end": "2026-01-09T00:00:00Z",
                }
            ),
            "candidates": candidates,
            "baseline_id": "base",
            "independent_groups": (),
            "independence_justification": None,
            "minimum_observation_days": 7,
            "minimum_builds": 2,
            "minimum_independent_groups": 2,
            "minimum_recall": 0.5,
            "minimum_net_saving_ms": 1,
            "actual_build_capacity": amount(8000),
            "maximum_silence_ns": 1000000000,
            "maximum_source_lag_seconds": 20,
            "maximum_pending": pending,
            "maximum_events": maximum_events,
            "maximum_event_bytes": 65536,
            "maximum_journal_bytes": 1048576,
        }
    )


def opening(proto, build_id="build-2", seconds=0, with_decisions=True):
    decisions = []
    for candidate in proto.candidates:
        request = ResourceRequest(
            schema_version=1,
            decision_id=build_id,
            configuration_fingerprint=fingerprint(candidate.model_dump(mode="json")),
            decided_at=START + timedelta(seconds=seconds),
            capacity=candidate.budget,
            tests=tuple(
                CostEstimate(test_id=name, estimate=estimate(ms))
                for name, ms in (("t1", 1000), ("t2", 2000), ("t3", 4000))
            ),
            ranking=("t1", "t2", "t3") if candidate.candidate_id == "base" else ("t2", "t3", "t1"),
        )
        decisions.append(
            CandidateDecision(
                candidate_id=candidate.candidate_id,
                model_fingerprint=candidate.model_fingerprint,
                request=request,
                decision_overhead=amount(0),
                shared_setup_reserve=candidate.shared_setup_reserve,
                overhead_interpretation="occupied_runner",
            )
        )
    return BuildOpened(
        event_id=build_id + ":opened",
        build_id=build_id,
        logical_build_id=build_id,
        group_id="owned-project",
        decided_at=START + timedelta(seconds=seconds),
        target_end=START + timedelta(seconds=seconds + 60),
        required_ids=("t1", "t2", "t3"),
        actual_estimate=estimate(7000),
        decisions=tuple(decisions) if with_decisions else (),
    )


def feedback(test_id, seconds, ms, label, build_id="build-2", revision_id="r1", supersedes=None):
    at = START + timedelta(seconds=seconds)
    row = ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation(
            observation_id=build_id + ":" + test_id,
            revision_id=revision_id,
            group_id="owned-project",
            event_at=at,
            available_at=at,
        ),
        entity_id=test_id,
        received_at=at,
        supersedes=supersedes,
        measurement=Measurement(
            name="ci_failure",
            value=label,
            unit="1",
            meaning="ci-test-failure-indicator-v1",
            direction="none",
            status="observed" if label is not None else "unknown",
            minimum=0.0,
        ),
    )
    return CIFeedback(
        event_id=build_id + ":" + test_id + ":" + revision_id,
        build_id=build_id,
        test_id=test_id,
        revision=row,
        occupancy=amount(ms) if ms is not None else None,
    )


def closure(build_id="build-2", seconds=8):
    return BuildClosed(
        event_id=build_id + ":closed",
        build_id=build_id,
        available_at=START + timedelta(seconds=seconds),
        dispatch_occupancy=amount(0),
        shared_setup_occupancy=amount(0),
        queue_elapsed=None,
        setup_elapsed=None,
    )


def connected(proto=None):
    state = ShadowState(schema_version=1, protocol=proto or protocol())
    heartbeat = SourceStatus(event_id="connected-1", available_at=clock(-1).utc_at, connected=True)
    return state.ingest(heartbeat, clock(-1))


def completed(proto=None):
    state = connected(proto)
    state = state.ingest(opening(state.protocol), clock(0, 10))
    for name, seconds, ms, label, tick in (
        ("t1", 1, 1000, 0.0, 20),
        ("t2", 3, 2000, 1.0, 30),
        ("t3", 7, 4000, 1.0, 40),
    ):
        state = state.ingest(feedback(name, seconds, ms, label), clock(seconds, tick))
    return state.ingest(closure(), clock(8, 50))


def test_actual_required_suite_accounting_is_separate_from_shadow_selection():
    state = completed()
    (build,) = state.builds()
    assert build.resource_accounting == "settled"
    assert build.actual_ledger.balance.consumed == 7000
    assert build.actual_ledger.balance.available == 1000
    assert build.actual_ledger.balance.reserved == 0
    assert len(build.history.revisions) == 3
    assert state.operational(clock(9, 60)).status == "healthy"
    assert ShadowState.from_json(state.to_json()).to_json() == state.to_json()
    assert ShadowState.from_json(state.to_json()).fingerprint == state.fingerprint


def test_same_source_retry_retains_first_clock_and_never_double_counts():
    state = completed()
    event = state.source_event("build-2:t2:r1")
    assert isinstance(event, CIFeedback)
    retried = state.ingest(event, clock(10000, 1, "new-session"))
    assert retried.to_json() == state.to_json()
    assert retried.builds()[0].actual_ledger.balance.consumed == 7000
    changed = event.model_dump(mode="json")
    changed["revision"]["measurement"]["value"] = 0.0
    conflict = CIFeedback.model_validate(changed)
    receipt = clock(9, 60)
    with pytest.raises(ValueError, match="Conflicting shadow source event"):
        state.ingest(conflict, receipt)
    assert state.builds()[0].actual_ledger.balance.consumed == 7000


def test_local_monotonic_freshness_and_clock_restart_are_not_source_timestamps():
    state = completed()
    assert state.operational(clock(9, 1000000001)).status == "stopped"
    assert state.operational(clock(9, 0, "second")).status == "recovering"
    heartbeat = SourceStatus(event_id="connected-2", available_at=clock(9).utc_at, connected=True)
    resumed = state.ingest(heartbeat, clock(9, 1, "second"))
    assert resumed.operational(clock(9, 2, "second")).status == "healthy"
    assert resumed.builds()[0].actual_ledger.balance == state.builds()[0].actual_ledger.balance


def test_new_id_stale_heartbeat_cannot_make_a_source_healthy():
    state = completed()
    old = SourceStatus(event_id="old-heartbeat", available_at=clock(-1).utc_at, connected=True)
    stale = state.ingest(old, clock(100, 60))
    assert stale.operational(clock(100, 61)).status == "stopped"
    assert stale.builds()[0].actual_ledger.balance.consumed == 7000


def test_out_of_order_independent_test_completions_keep_original_times_and_book_once():
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 10))
    later = feedback("t2", 3, 2000, 1.0)
    earlier = feedback("t1", 1, 1000, 0.0)
    state = state.ingest(later, clock(4, 20))
    state = state.ingest(earlier, clock(5, 30))
    (build,) = state.builds()
    assert build.actual_ledger.balance.consumed == 3000
    assert build.actual_ledger.balance.reserved == 4000
    assert earlier.revision.received_at == START + timedelta(seconds=1)
    view = build.history.as_of(clock(3).utc_at)
    assert view.observations == ()
    view = build.history.as_of(clock(5).utc_at)
    assert len(view.observations) == 2
    stored = state.source_event(earlier.event_id)
    assert isinstance(stored, CIFeedback)
    assert earlier.revision.to_json() == stored.revision.to_json()


def test_missing_build_is_quarantined_and_bounded_instead_of_invented():
    state = connected(protocol(pending=1))
    unseen = feedback("t1", 1, 1000, 0.0)
    blocked = state.ingest(unseen, clock(1, 10))
    assert blocked.journal[-1].disposition == "quarantined"
    assert blocked.operational(clock(1, 11)).status == "stopped"
    assert blocked.builds() == ()
    second = feedback("t2", 2, 2000, 1.0)
    receipt = clock(2, 20)
    with pytest.raises(ValueError, match="quarantine is full"):
        blocked.ingest(second, receipt)
    assert len(blocked.journal) == 2


def test_changed_protocol_cannot_reuse_accepted_protected_outcomes():
    state = completed()
    payload = state.model_dump(mode="json")
    payload["protocol"]["candidates"][0]["model_fingerprint"] = fingerprint({"changed": "model"})
    with pytest.raises(ValueError, match="Protocol changed"):
        ShadowState.model_validate(payload)


def test_explicit_disconnect_and_utc_source_lag_are_independent_of_monotonic_ticks():
    state = completed()
    stale = state.operational(clock(30, 60))
    assert stale.status == "stopped"
    assert "declared_source_clock_lag" in stale.reasons
    disconnected = SourceStatus(event_id="disconnected", available_at=clock(9).utc_at, connected=False)
    stopped = state.ingest(disconnected, clock(9, 60))
    assert stopped.operational(clock(9, 61)).status == "stopped"
    assert stopped.builds()[0].actual_ledger.balance.consumed == 7000


def test_shadow_artifact_roundtrip_and_checkpoint_prefix(tmp_path):
    from coleman.artifacts import ArtifactWriter
    from coleman.shadow import load_shadow_state, publish_shadow_state, verify_shadow_checkpoint

    state = completed()
    writer = ArtifactWriter(tmp_path)
    checkpoint = publish_shadow_state(state, writer)
    assert load_shadow_state(state.protocol, writer).to_json() == state.to_json()
    assert verify_shadow_checkpoint(checkpoint, state.protocol, writer).fingerprint == state.fingerprint
    assert publish_shadow_state(state, writer) == checkpoint
    earlier = ShadowState(schema_version=1, protocol=state.protocol, journal=state.journal[:2])
    old_checkpoint = publish_shadow_state(earlier, writer)
    assert verify_shadow_checkpoint(old_checkpoint, state.protocol, writer).to_json() == earlier.to_json()
    assert load_shadow_state(state.protocol, writer).to_json() == state.to_json()


@pytest.mark.parametrize("stage", ["before_frame", "after_frame_before_reference"])
def test_committed_event_recovery_before_and_after_atomic_commit(tmp_path, monkeypatch, stage):
    import coleman.artifacts
    from coleman.artifacts import ArtifactWriter
    from coleman.shadow import load_shadow_state, publish_shadow_state

    state = completed()
    writer = ArtifactWriter(tmp_path)
    original = coleman.artifacts.os.link

    def interrupted(source, destination):
        if stage == "before_frame" and destination.parent.name == "frames" and destination.name == "0003.json":
            raise OSError("injected before frame commit")
        if stage == "after_frame_before_reference" and destination.parent.name == "checkpoints":
            raise OSError("injected after committed frames")
        original(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(coleman.artifacts.os, "link", interrupted)
        with pytest.raises(OSError, match="injected"):
            publish_shadow_state(state, writer)
    assert not list(tmp_path.rglob("checkpoints/*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    restored = load_shadow_state(state.protocol, writer)
    assert len(restored.journal) == (2 if stage == "before_frame" else len(state.journal))
    for entry in state.journal:
        restored = restored.ingest(entry.event, entry.clock)
    assert restored.to_json() == state.to_json()
    publish_shadow_state(restored, writer)
    assert load_shadow_state(state.protocol, writer).builds()[0].actual_ledger.balance.consumed == 7000


def test_concurrent_identical_shadow_publications_converge(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from coleman.artifacts import ArtifactWriter
    from coleman.shadow import load_shadow_state, publish_shadow_state

    state = completed()
    writer = ArtifactWriter(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: publish_shadow_state(state, writer), range(8)))
    assert len(set(paths)) == 1
    assert load_shadow_state(state.protocol, writer).to_json() == state.to_json()


def test_seven_day_requirement_with_one_received_day_is_inconclusive():
    from coleman.shadow import ShadowReport, render_shadow_report

    state = completed()
    report = ShadowReport(schema_version=1, state=state)
    evidence = report.evidence()
    assert evidence.outcome == "inconclusive"
    assert evidence.distinct_decision_days == 1
    assert evidence.received_logical_builds == 1
    assert evidence.scored_logical_builds == 1
    assert evidence.forward_logical_builds == 0
    assert evidence.actual_known_required_runner.value == 7000
    assert "insufficient_distinct_logical_ci_decision_days" in evidence.reasons
    assert "offline_functional_recovery_is_not_prospective_evidence" in evidence.reasons
    assert "required: 7" in render_shadow_report(report)
    assert ShadowReport.from_json(report.to_json()).evidence() == evidence


def test_shadow_report_has_actual_required_work_and_distinct_modeled_cost():
    from coleman.shadow import ShadowReport

    state = completed()
    rows = ShadowReport(schema_version=1, state=state).findings()
    base, estimate = rows
    assert base.candidate_id == "base"
    assert base.modeled_work.value == 3000
    assert estimate.candidate_id == "estimate"
    assert estimate.modeled_work.value == 2000
    assert estimate.net_saving.value == 1000
    assert base.recall.value == estimate.recall.value == 0.5
    assert state.builds()[0].actual_ledger.balance.consumed == 7000
    assert estimate.modeled_cost.meaning.endswith("proxy-v1")
    assert estimate.net_saving.unit == "ms"
