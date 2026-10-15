"""Compatibility, unknown quantities, causal revisions and bounded admission regressions."""

import ast
import itertools
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_shadow import CIFeedback, amount, clock, closure, completed, connected, feedback, opening, protocol

from coleman.resources import QUEUE_MS, ResourceAmount
from coleman.shadow import ShadowProtocol, ShadowReport, ShadowState, resolve_selection
from coleman.shadow.contracts import SourceStatus

ROOT = Path(__file__).parents[2]


def replace(model, **changes):
    return type(model).model_validate({**model.model_dump(), **changes})


def test_versioned_golden_protocol_and_completed_state():
    for name, model, expected in (
        ("protocol-v1.json", ShadowProtocol, protocol()),
        ("state-v1.json", ShadowState, completed()),
    ):
        raw = (ROOT / "tests/fixtures/shadow" / name).read_text()
        assert model.from_json(raw).to_json() == expected.to_json()
        assert model.from_json(raw).fingerprint == expected.fingerprint
        assert raw == expected.to_json() + "\n"


@pytest.mark.parametrize("version", [None, 0, 2, 100, True, "1"])
def test_unsupported_versions_fail_closed(version):
    models = (protocol(), completed(), ShadowReport(schema_version=1, state=completed()))
    for model in models:
        data = model.model_dump(mode="json")
        if version is None:
            data.pop("schema_version")
        else:
            data["schema_version"] = version
        with pytest.raises(ValueError):
            type(model).from_json(json.dumps(data))


def test_duplicate_json_keys_and_legacy_artifacts_are_never_reinterpreted(tmp_path):
    with pytest.raises(ValueError):
        ShadowState.from_json('{"schema_version":1,"schema_version":1}')
    legacy = (ROOT / "tests/fixtures/domain-results/native-v3.2.json").read_bytes()
    with pytest.raises(ValueError):
        ShadowState.from_json(legacy)
    # Reading shadow contracts must not change the established legacy reader.
    from coleman.results.domain_artifacts import DomainArtifacts

    assert tuple(DomainArtifacts(tmp_path).records()) == ()


@pytest.mark.parametrize("value", [-1, True, 1.5, float("inf"), float("nan"), -float("inf")])
def test_occupied_milliseconds_require_finite_nonnegative_integers(value):
    with pytest.raises(ValidationError):
        amount(value)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_outcomes_and_protocol_thresholds_are_rejected(value):
    with pytest.raises(ValidationError):
        feedback("t1", 1, 0, value)
    with pytest.raises(ValidationError):
        replace(protocol(), minimum_recall=value)


def test_queue_delay_cannot_be_used_as_runner_work_or_inference_cost():
    wrong = ResourceAmount(value=1, semantics=QUEUE_MS)
    for original, change in (
        (closure(), {"shared_setup_occupancy": wrong}),
        (feedback("t1", 1, 0, 0.0), {"occupancy": wrong}),
        (opening(protocol()).decisions[0], {"decision_overhead": wrong}),
    ):
        with pytest.raises(ValidationError):
            replace(original, **change)


@pytest.mark.parametrize("label,status", [(0.0, "observed"), (None, "unknown")])
def test_zero_label_and_unknown_label_retain_distinct_semantics(label, status):
    event = feedback("t1", 1, 0, label)
    assert event.revision.measurement.value == label
    assert event.revision.measurement.status == status
    assert event.occupancy == amount(0)
    assert feedback("t1", 1, None, label).occupancy is None


def test_unknown_inference_and_shared_setup_reserves_cannot_create_free_work():
    decision = opening(protocol()).decisions[0]
    assert resolve_selection(decision).selected_ids == ("t1", "t2")
    reserved = replace(decision, shared_setup_reserve=amount(1000), decision_overhead=amount(1))
    assert resolve_selection(reserved).selected_ids == ("t1",)
    unknown = replace(decision, decision_overhead=None, overhead_interpretation="unavailable")
    assert resolve_selection(unknown).selected_ids == ()
    exhausted = replace(decision, shared_setup_reserve=amount(3001))
    assert resolve_selection(exhausted).selected_ids == ()


def test_actual_overrun_is_retained_and_stops_only_recommendations():
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    state = state.ingest(feedback("t1", 1, 9000, 0.0), clock(1, 2))
    assert state.builds()[0].actual_ledger.balance.consumed == 9000
    assert state.operational(clock(1, 3)).status == "stopped"
    assert ShadowReport(schema_version=1, state=state).evidence().known_adverse_findings
    observed = opening(state.protocol, "build-3", 2, with_decisions=False)
    state = state.ingest(observed, clock(2, 4))
    assert len(state.builds()) == 2
    assert state.journal[-1].disposition == "accepted"


def test_late_label_correction_keeps_asof_history_without_recharging_work():
    state = completed()
    change = feedback("t2", 3, 2000, 0.0, revision_id="r2", supersedes="r1")
    data = change.model_dump(mode="json")
    data["revision"]["metadata"]["available_at"] = clock(9).utc_at.isoformat()
    data["revision"]["received_at"] = clock(9).utc_at.isoformat()
    change = CIFeedback.model_validate(data)
    corrected = state.ingest(change, clock(9, 60))
    build = corrected.builds()[0]
    assert build.actual_ledger.balance.consumed == 7000
    assert build.history.as_of(clock(8).utc_at).observations != build.history.as_of(clock(9).utc_at).observations
    assert ShadowReport(schema_version=1, state=corrected).findings()[0].recall.value == 0.0
    assert corrected.operational(clock(9, 61)).status == "stopped"
    assert "counterfactual_selection_recall_limit" in corrected.operational(clock(9, 61)).reasons


def test_occupancy_correction_and_missing_parent_are_audited_without_mutation():
    state = completed()
    event = feedback("t2", 3, 1000, 0.0, revision_id="r2", supersedes="r1")
    updated = state.ingest(event, clock(9, 60))
    assert updated.journal[-1].reason == "occupancy_corrections_unsupported_by_resource_v1"
    assert updated.builds()[0].actual_ledger.balance.consumed == 7000
    fresh = connected()
    fresh = fresh.ingest(opening(fresh.protocol), clock(0, 1))
    event = feedback("t2", 3, 2000, 1.0, revision_id="r2", supersedes="r1")
    assert fresh.ingest(event, clock(3, 2)).journal[-1].reason == "correction_parent_unavailable"


def test_null_work_remains_pending_and_closed_partial_labels_cannot_score():
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    state = state.ingest(feedback("t1", 1, None, None), clock(1, 2))
    state = state.ingest(closure(), clock(8, 3))
    build = state.builds()[0]
    assert build.resource_accounting == "pending_or_unknown"
    assert build.actual_ledger.balance.consumed == 0
    assert build.actual_ledger.balance.reserved == 7000
    report = ShadowReport(schema_version=1, state=state)
    assert all(row.recall.value is None and row.modeled_cost.value is None for row in report.findings())
    assert report.evidence().outcome == "inconclusive"
    assert not report.evidence().actual_accounting_complete
    assert state.operational(clock(9, 4)).status == "stopped"


def test_zero_failures_are_not_perfect_recall():
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    for index, name in enumerate(("t1", "t2", "t3"), 1):
        state = state.ingest(feedback(name, index, index * 1000, 0.0), clock(index, index + 1))
    state = state.ingest(closure(), clock(8, 5))
    report = ShadowReport(schema_version=1, state=state)
    assert all(row.recall.status == "not_applicable" and row.recall.value is None for row in report.findings())
    assert report.evidence().scored_logical_builds == 0


def test_negative_net_savings_are_valid_adverse_evidence():
    state = connected()
    event = opening(state.protocol)
    decisions = list(event.decisions)
    # The frozen budget remains identical; only the actually measured inference cost differs.
    decisions[1] = replace(decisions[1], decision_overhead=amount(2000))
    state = state.ingest(replace(event, decisions=decisions), clock(0, 1))
    for name, seconds, ms, label in (("t1", 1, 0, 0.0), ("t2", 2, 0, 1.0), ("t3", 3, 0, 1.0)):
        state = state.ingest(feedback(name, seconds, ms, label), clock(seconds, seconds + 1))
    state = state.ingest(closure(), clock(8, 5))
    report = ShadowReport(schema_version=1, state=state)
    assert report.findings()[1].net_saving.value == -2000.0
    assert report.evidence().known_adverse_findings
    assert report.evidence().outcome == "inconclusive"


def test_duplicate_at_event_limit_is_idempotent_but_new_event_is_not_acknowledged():
    state = connected(protocol(maximum_events=1))
    entry = state.journal[0]
    assert state.ingest(entry.event, clock(100, 100)).to_json() == state.to_json()
    event = SourceStatus(event_id="new", connected=True, available_at=clock(0).utc_at)
    with pytest.raises(ValueError, match="event retention"):
        state.ingest(event, clock(0, 1))
    assert len(state.journal) == 1


def test_payload_and_journal_byte_limits_reject_without_changing_prior_state():
    limited = replace(protocol(), maximum_event_bytes=1024, maximum_journal_bytes=4096)
    state = connected(limited)
    with pytest.raises(ValueError, match="event exceeds its byte"):
        state.ingest(opening(limited), clock(0, 1))
    assert len(state.journal) == 1
    state = connected(replace(protocol(), maximum_event_bytes=4096, maximum_journal_bytes=4096))
    with pytest.raises(ValueError, match="journal byte retention"):
        for index in range(1, 40):
            event = SourceStatus(event_id=f"heartbeat-{index}", connected=True, available_at=clock(index).utc_at)
            state = state.ingest(event, clock(index, index))
    assert len(state.to_json().encode()) < 10000


def test_quarantined_only_source_never_claims_complete_actual_accounting():
    state = connected().ingest(feedback("t1", 1, 1, 0.0), clock(1, 1))
    assert not ShadowReport(schema_version=1, state=state).evidence().actual_accounting_complete


def test_same_logical_retry_cannot_create_another_independent_build_or_day():
    state = completed()
    event = opening(state.protocol, "retry-1", seconds=10, with_decisions=False)
    event = replace(event, logical_build_id="build-2", retry_of="build-2")
    state = state.ingest(event, clock(10, 60))
    evidence = ShadowReport(schema_version=1, state=state).evidence()
    assert evidence.received_logical_builds == evidence.scored_logical_builds == evidence.distinct_decision_days == 1
    changed = replace(opening(state.protocol, "retry-2", 11, False), logical_build_id="build-2", retry_of="build-2")
    assert state.ingest(changed, clock(11, 70)).journal[-1].reason == "retry_parent_must_be_latest_attempt"


def test_metamorphic_independent_delivery_permutations_preserve_labels_and_balance():
    reports = []
    for order in itertools.permutations(("t1", "t2", "t3")):
        state = connected()
        state = state.ingest(opening(state.protocol), clock(0, 1))
        for index, name in enumerate(order, 2):
            event = feedback(
                name, {"t1": 1, "t2": 3, "t3": 7}[name], {"t1": 1000, "t2": 2000, "t3": 4000}[name], float(name != "t1")
            )
            state = state.ingest(event, clock(8, index))
        state = state.ingest(closure(), clock(8, 6))
        report = ShadowReport(schema_version=1, state=state)
        reports.append((report.findings(), report.evidence()))
        assert state.builds()[0].actual_ledger.balance.consumed == 7000
    assert all(result == reports[0] for result in reports)


def test_domain_import_boundary_excludes_cli_transports_and_system_clocks():
    for path in (ROOT / "coleman/shadow").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("coleman.cli", "coleman.transports", "mcp"))
            if isinstance(node, ast.Import):
                assert all(
                    not item.name.startswith(("coleman.cli", "coleman.transports", "mcp", "time"))
                    for item in node.names
                )


@pytest.mark.parametrize("field", ["model_fingerprint", "feature_fingerprint", "fallback_version"])
def test_frozen_candidate_mutation_changes_experiment_and_rejects_old_evidence(field):
    from coleman.results.domain import fingerprint

    state = completed()
    data = state.model_dump(mode="json")
    value = fingerprint({"changed": field}) if field.endswith("fingerprint") else "changed-fallback-v2"
    data["protocol"]["candidates"][0][field] = value
    changed = ShadowProtocol.model_validate(data["protocol"])
    assert changed.logical_id != state.logical_id
    with pytest.raises(ValueError, match="Protocol changed"):
        ShadowState.model_validate(data)


def test_censored_feedback_with_known_work_settles_cost_but_not_quality():
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    for index, name in enumerate(("t1", "t2", "t3"), 1):
        state = state.ingest(
            feedback(name, index, 1000, None if name == "t3" else float(index == 2)), clock(index, index + 1)
        )
    state = state.ingest(closure(), clock(8, 5))
    report = ShadowReport(schema_version=1, state=state)
    assert report.evidence().actual_accounting_complete
    assert report.evidence().scored_logical_builds == 0
    assert all(row.recall.status == "unavailable" for row in report.findings())
    assert all(row.modeled_cost.status == "observed" for row in report.findings())


def test_late_required_feedback_can_finish_closed_pending_accounting():
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    state = state.ingest(closure(), clock(8, 2))
    assert state.builds()[0].resource_accounting == "pending_or_unknown"
    for index, name in enumerate(("t1", "t2", "t3"), 1):
        state = state.ingest(feedback(name, index, 1000, float(index > 1)), clock(9, index + 2))
    assert state.builds()[0].resource_accounting == "settled"
    assert state.builds()[0].actual_ledger.balance.consumed == 3000


def test_changed_same_ordinal_concurrent_history_conflicts_without_overwriting(tmp_path):
    from coleman.artifacts import ArtifactWriter
    from coleman.shadow import load_shadow_state, publish_shadow_state

    first = connected()
    changed = replace(first.journal[0].event, event_id="different-heartbeat")
    fork = ShadowState(schema_version=1, protocol=first.protocol).ingest(changed, first.journal[0].clock)
    writer = ArtifactWriter(tmp_path)
    publish_shadow_state(first, writer)
    with pytest.raises(ValueError, match="Conflicting immutable"):
        publish_shadow_state(fork, writer)
    assert load_shadow_state(first.protocol, writer).to_json() == first.to_json()


@pytest.mark.parametrize("corruption", ["future_frame", "missing_frame", "changed_predecessor", "changed_reference"])
def test_committed_artifact_corruption_is_never_a_successful_restore(tmp_path, corruption):
    from coleman.artifacts import ArtifactWriter
    from coleman.shadow import load_shadow_state, publish_shadow_state, verify_shadow_checkpoint

    state = completed()
    writer = ArtifactWriter(tmp_path)
    checkpoint = publish_shadow_state(state, writer)
    frames = sorted(tmp_path.rglob("frames/*.json"))
    if corruption == "missing_frame":
        frames[0].unlink()
    elif corruption == "changed_reference":
        data = json.loads(checkpoint.read_bytes())
        data["state_fingerprint"] = state.protocol.fingerprint
        checkpoint.write_text(json.dumps(data))
    else:
        data = json.loads(frames[1].read_bytes())
        if corruption == "future_frame":
            data["schema_version"] = 2
        else:
            data["previous_frame"] = state.protocol.fingerprint
        frames[1].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_shadow_checkpoint(checkpoint, state.protocol, writer)
    if corruption != "changed_reference":
        with pytest.raises(ValueError):
            load_shadow_state(state.protocol, writer)


@pytest.mark.parametrize("adverse", [False, True])
def test_synthetic_gate_branch_fixture_requires_all_predeclared_days_and_groups(adverse):
    # This unit fixture exercises the authorized-source contract's descriptive
    # branch. It is simulated, not collected forward evidence or a study result.
    proto = replace(
        protocol("authorized_forward_ci"),
        minimum_observation_days=2,
        independent_groups=("group-a", "group-b"),
        independence_justification="unit-fixture-only-no-empirical-independence-claim",
    )
    state = ShadowState(schema_version=1, protocol=proto)
    for day, group in enumerate(("group-a", "group-b")):
        offset = day * 86400
        status = SourceStatus(event_id=f"connected-{day}", connected=True, available_at=clock(offset).utc_at)
        session = f"session-{day}"
        state = state.ingest(status, clock(offset, 0, session))
        event = replace(opening(proto, f"build-{day}", offset), group_id=group)
        state = state.ingest(event, clock(offset, 1, session))
        for index, name in enumerate(("t1", "t2", "t3"), 1):
            label = float(name != "t1")
            if adverse and day == 1:
                label = float(name == "t3")
            source = feedback(
                name, offset + index, {"t1": 1000, "t2": 2000, "t3": 4000}[name], label, build_id=event.build_id
            )
            revision = replace(source.revision, metadata=replace(source.revision.metadata, group_id=group))
            state = state.ingest(replace(source, revision=revision), clock(offset + index, index + 1, session))
        state = state.ingest(closure(event.build_id, offset + 8), clock(offset + 8, 5, session))
        if day == 0:
            assert ShadowReport(schema_version=1, state=state).evidence().outcome == "inconclusive"
    evidence = ShadowReport(schema_version=1, state=state).evidence()
    if adverse:
        assert evidence.known_adverse_findings
        assert evidence.outcome == "fail"
    else:
        assert evidence.outcome == "pass"
        assert evidence.distinct_decision_days == evidence.forward_logical_builds == 2
        assert evidence.independent_source_groups == ("group-a", "group-b")


@pytest.mark.parametrize("restored", [False, True])
def test_reused_clock_session_retains_its_admitted_high_watermark(restored):
    state = completed()
    second = SourceStatus(event_id="session-b", available_at=clock(9).utc_at, connected=True)
    state = state.ingest(second, clock(9, 100, "second"))
    if restored:
        state = ShadowState.from_json(state.to_json())
    reused = SourceStatus(event_id="reused-session-a", available_at=clock(10).utc_at, connected=True)
    before = state.to_json()
    with pytest.raises(ValueError, match="Monotonic host tick moved backwards"):
        state.ingest(reused, clock(10, 1, "first"))
    with pytest.raises(ValueError, match="Monotonic host tick moved backwards"):
        state.operational(clock(10, 1, "first"))
    assert state.to_json() == before
    # Legitimately continuing that clock is safe; ticks are never compared to B's 100.
    resumed = state.ingest(reused, clock(10, 60, "first"))
    assert resumed.operational(clock(10, 61, "first")).status == "healthy"


@pytest.mark.parametrize("case", ["before_start", "before_known_completion", "completion_after_close"])
def test_upstream_closure_cannot_precede_physical_work(case):
    state = connected()
    state = state.ingest(opening(state.protocol), clock(0, 1))
    if case == "before_start":
        state = state.ingest(closure(seconds=-1), clock(1, 2))
        expected = "closure_precedes_upstream_start"
    elif case == "before_known_completion":
        state = state.ingest(feedback("t3", 7, 4000, 1.0), clock(7, 2))
        state = state.ingest(closure(seconds=6), clock(8, 3))
        expected = "closure_precedes_known_physical_completion"
    else:
        state = state.ingest(closure(seconds=6), clock(6, 2))
        state = state.ingest(feedback("t3", 7, 4000, 1.0), clock(8, 3))
        expected = "physical_completion_after_upstream_closure"
    assert state.journal[-1].disposition == "quarantined"
    assert state.journal[-1].reason == expected
