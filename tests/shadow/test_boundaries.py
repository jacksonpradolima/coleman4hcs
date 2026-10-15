"""Fail-closed source, temporal, frozen-protocol and recovery boundaries."""

from datetime import timedelta

import pytest
from test_contracts import replace, synthetic_gate_state
from test_shadow import START, clock, closure, completed, connected, feedback, opening, protocol

from coleman.artifacts import ArtifactWriter
from coleman.resources import QUEUE_MS, ResourceAmount
from coleman.results.domain import fingerprint
from coleman.shadow import (
    ShadowProtocol,
    ShadowReport,
    ShadowState,
    SourceStatus,
    load_shadow_state,
    publish_shadow_state,
)


@pytest.mark.parametrize("case", ["baseline", "freeze", "duration", "budget", "reserve", "units", "bytes", "groups"])
def test_frozen_protocol_rejects_incomparable_or_impossible_studies(case):
    data = protocol().model_dump(mode="json")
    if case == "baseline":
        data["baseline_id"] = "missing"
    elif case == "freeze":
        data["frozen_at"] = START.isoformat()
    elif case == "duration":
        data["minimum_observation_days"] = 9
    elif case == "budget":
        data["candidates"][0]["budget"]["value"] = 3001
    elif case == "reserve":
        data["candidates"][0]["shared_setup_reserve"]["value"] = 1
    elif case == "units":
        data["actual_build_capacity"]["semantics"] = QUEUE_MS.model_dump(mode="json")
    elif case == "bytes":
        data.update(maximum_event_bytes=8192, maximum_journal_bytes=4096)
    else:
        data["independent_groups"] = ["unjustified-group"]
    with pytest.raises(ValueError):
        ShadowProtocol.model_validate(data)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("past", "build_outside_forward_window"),
        ("target", "build_target_exceeds_forward_window"),
        ("units", "incompatible_actual_estimate_units"),
        ("future_estimate", "actual_estimate_unavailable_at_upstream_start"),
        ("capacity", "actual_estimate_exceeds_declared_capacity"),
        ("late_seal", "recommendation_must_be_sealed_at_application_decision"),
        ("partial_trials", "incomplete_frozen_candidate_trial_registry"),
        ("changed_model", "decision_differs_from_frozen_candidate_or_required_universe"),
        ("unhealthy", "recommendations_blocked_by_source_or_risk_state"),
    ],
)
def test_opening_cannot_invent_predecision_eligibility(case, reason):
    state = connected()
    event = opening(state.protocol)
    receipt = clock(0, 1)
    if case == "past":
        event = replace(event, decided_at=state.protocol.temporal_plan.validation_end)
    elif case == "target":
        event = replace(event, target_end=state.protocol.temporal_plan.test_end + timedelta(seconds=1))
    elif case == "units":
        wrong = ResourceAmount(value=7000, semantics=QUEUE_MS)
        event = replace(event, actual_estimate=replace(event.actual_estimate, amount=wrong))
    elif case == "future_estimate":
        event = replace(
            event, actual_estimate=replace(event.actual_estimate, available_at=START + timedelta(seconds=1))
        )
    elif case == "capacity":
        event = replace(
            event,
            actual_estimate=replace(event.actual_estimate, amount=replace(event.actual_estimate.amount, value=9000)),
        )
    elif case == "late_seal":
        receipt = clock(1, 1)
    elif case == "partial_trials":
        event = replace(event, decisions=event.decisions[:1])
    elif case == "changed_model":
        changed = replace(event.decisions[0], model_fingerprint=fingerprint({"other": "model"}))
        event = replace(event, decisions=(changed, event.decisions[1]))
    else:
        state = ShadowState(schema_version=1, protocol=state.protocol)
    changed = state.ingest(event, receipt)
    assert changed.journal[-1].reason == reason
    assert changed.builds() == ()
    assert state.builds() == ()


@pytest.mark.parametrize(
    "case", ["future_decision", "future_status", "future_closure", "future_provider", "early_host", "reverse_utc"]
)
def test_impossible_source_and_host_availability_raises_without_acknowledgment(case):
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    receipt = clock(1, 2)
    if case == "future_decision":
        event = opening(state.protocol, "build-future", 2)
    elif case == "future_status":
        event = SourceStatus(event_id="future", connected=True, available_at=clock(2).utc_at)
    elif case == "future_closure":
        event = closure()
    elif case == "future_provider":
        event = feedback("t1", 2, 1000, 0.0)
    elif case == "early_host":
        state = ShadowState(schema_version=1, protocol=state.protocol)
        receipt = clock(-200000, 0)
        event = SourceStatus(event_id="early", connected=True, available_at=receipt.utc_at)
    else:
        receipt = clock(-1, 2)
        event = SourceStatus(event_id="reverse", connected=True, available_at=receipt.utc_at)
    before = state.to_json()
    with pytest.raises(ValueError):
        state.ingest(event, receipt)
    assert state.to_json() == before


@pytest.mark.parametrize("case", ["duplicate_build", "missing_retry", "unknown_parent", "early_retry", "changed_group"])
def test_build_and_retry_identity_cannot_inflate_or_change_independent_units(case):
    state = completed()
    event = opening(state.protocol, "new-attempt", 10, False)
    if case == "duplicate_build":
        event = replace(event, build_id="build-2")
    elif case == "missing_retry":
        event = replace(event, logical_build_id="build-2")
    elif case == "unknown_parent":
        event = replace(event, logical_build_id="build-2", retry_of="absent")
    elif case == "early_retry":
        event = replace(event, logical_build_id="build-2", retry_of="build-2", decided_at=START + timedelta(seconds=7))
    else:
        event = replace(event, logical_build_id="build-2", retry_of="build-2", group_id="other-group")
    changed = state.ingest(event, clock(10, 60))
    assert changed.journal[-1].disposition == "quarantined"
    assert len(changed.builds()) == 1
    assert ShadowReport(schema_version=1, state=changed).evidence().received_logical_builds == 1


@pytest.mark.parametrize("case", ["test", "entity", "group", "past", "observation_collision", "fork"])
def test_invalid_physical_source_identity_and_causal_lineage_are_audited(case):
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    event = feedback("t1", 1, 1000, 0.0)
    if case == "test":
        event = replace(event, test_id="absent")
    elif case == "entity":
        event = replace(event, revision=replace(event.revision, entity_id="different"))
    elif case == "group":
        metadata = replace(event.revision.metadata, group_id="different")
        event = replace(event, revision=replace(event.revision, metadata=metadata))
    elif case == "past":
        event = feedback("t1", -1, 1000, 0.0)
    else:
        state = state.ingest(event, clock(1, 2))
        parent = "r1" if case == "observation_collision" else None
        event = feedback("t1", 1, 1000, 0.0, revision_id="r2", supersedes=parent)
        if case == "observation_collision":
            metadata = replace(event.revision.metadata, observation_id="different")
            event = replace(event, revision=replace(event.revision, metadata=metadata))
    changed = state.ingest(event, clock(2, 3))
    assert changed.journal[-1].disposition == "quarantined"
    assert changed.builds()[0].actual_ledger.balance.consumed == (
        1000 if case in {"observation_collision", "fork"} else 0
    )


def test_unknown_inference_contains_operation_without_inventing_an_adverse_measurement():
    state = connected()
    event = opening(state.protocol)
    decision = replace(event.decisions[0], decision_overhead=None, overhead_interpretation="unavailable")
    event = replace(event, decisions=(decision, event.decisions[1]))
    state = state.ingest(event, clock(0, 1))
    assert state.operational(clock(1, 2)).status == "stopped"
    assert not ShadowReport(schema_version=1, state=state).evidence().known_adverse_findings
    assert state.admitted_event_ids == ("connected-1", "build-2:opened")
    assert state.protocol_fingerprint == state.protocol.fingerprint


def test_unknown_and_observed_zero_inference_cannot_share_an_interpretation():
    event = opening(protocol()).decisions[0]
    payload = {**event.model_dump(), "decision_overhead": None}
    contract_class = type(event)
    with pytest.raises(ValueError, match="availability and interpretation disagree"):
        contract_class.model_validate(payload)


def test_no_data_cannot_create_a_report_or_evaluation_timestamp():
    state = ShadowState(schema_version=1, protocol=protocol())
    with pytest.raises(ValueError, match="at least one source admission"):
        ShadowReport(schema_version=1, state=state)


@pytest.mark.parametrize("encoding", ["ascii", "utf8"])
def test_portable_state_rejects_byte_oversize_before_json_parse(encoding):
    payload = b" " * (20 * 1024 * 1024 + 1) if encoding == "ascii" else "é" * (10 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="serialized byte limit"):
        ShadowState.from_json(payload)


def test_empty_committed_prefix_and_absent_journal_restore_without_inventing_events(tmp_path):
    from coleman.shadow import verify_shadow_checkpoint

    proto = protocol()
    writer = ArtifactWriter(tmp_path)
    state = load_shadow_state(proto, writer)
    assert state.journal == ()
    path = publish_shadow_state(state, writer)
    assert verify_shadow_checkpoint(path, proto, writer).to_json() == state.to_json()


@pytest.mark.parametrize(
    "case", ["missing_header", "oversized_header", "changed_header", "too_many_frames", "oversized_frame"]
)
def test_reader_rejects_incomplete_or_oversized_artifact_trees(tmp_path, case):
    proto = replace(protocol(maximum_events=1), maximum_event_bytes=1024)
    writer = ArtifactWriter(tmp_path)
    state = connected(proto)
    publish_shadow_state(state, writer)
    root = tmp_path / "shadow-journals" / proto.fingerprint.removeprefix("sha256:")
    header = root / "protocol.json"
    frame = root / "frames/0001.json"
    if case == "missing_header":
        header.unlink()
    elif case == "oversized_header":
        header.write_text(" " * (1024 * 1024 + 1))
    elif case == "changed_header":
        header.write_text(replace(proto, experiment_id="other-experiment").to_json())
    elif case == "too_many_frames":
        (root / "frames/0002.json").write_bytes(frame.read_bytes())
    else:
        frame.write_bytes(frame.read_bytes() + b" " * 6000)
    with pytest.raises(ValueError):
        load_shadow_state(proto, writer)


def test_serialized_protocol_header_limit_applies_before_publication(tmp_path):
    source = replace(protocol().source, authorization_reference="x" * (1024 * 1024))
    proto = replace(protocol(), source=source)
    state = ShadowState(schema_version=1, protocol=proto)
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(ValueError, match="protocol header exceeds"):
        publish_shadow_state(state, writer)
    assert not list(tmp_path.rglob("*.json"))


def test_insufficient_practical_benefit_cannot_pass_despite_sufficient_source_counts():
    state = synthetic_gate_state(minimum_net_saving_ms=2000)
    evidence = ShadowReport(schema_version=1, state=state).evidence()
    assert evidence.scored_logical_builds == evidence.distinct_decision_days == 2
    assert not evidence.known_adverse_findings
    assert evidence.outcome == "inconclusive"
    assert evidence.reasons == ("practical_benefit_not_demonstrated",)
