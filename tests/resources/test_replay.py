"""Exact resource replay, temporal mutation, losses and legacy compatibility."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from coleman.pilot import OVERHEAD_SECONDS, PilotRun, serial_events
from coleman.resources import (
    RUNNER_MS,
    ResourceAmount,
    ResourceReplayProtocol,
    ResourceReplayReport,
    ResourceReplayRun,
    render_resource_report,
)
from coleman.results.domain import fingerprint
from coleman.selection import Quantity


def native_run(protocol, candidate="base-order", overhead="0", ranking=None):
    build = protocol.native_manifest.builds[0]
    ranking = ranking or (build.baseline_order if candidate == "base-order" else ("t2", "t3", "t1"))
    return PilotRun(
        candidate_id=candidate,
        build_id=build.build_id,
        seed=42,
        ranking=ranking,
        events=serial_events(build, ranking, protocol.native_manifest.budget),
        prioritization_overhead=Quantity(value=Decimal(overhead), semantics=OVERHEAD_SECONDS),
    )


def replay(protocol, candidate="base-order", overhead="0", setup=0):
    return ResourceReplayRun(
        schema_version=1,
        protocol=protocol,
        native_run=native_run(protocol, candidate, overhead),
        queue_elapsed=None,
        setup_elapsed=None,
        shared_setup_occupancy=None if setup is None else ResourceAmount(value=setup, semantics=RUNNER_MS),
    )


def test_native_verified_source_counterfactual_accounting_and_golden(example):
    protocol, _ = example.fixture()
    run = replay(protocol)
    assert run.selection.selected_ids == ("t1", "t2")
    assert run.accounting().selected_ids == ("t1", "t2")
    assert run.ledger.balance.consumed == 3000
    assert run.ledger.balance.available == 2000
    assert run.ledger.balance.overrun_execution_ids == ("t2",)
    assert run.accounting().failing_test_recall.value == 0.5
    assert run.accounting().first_failure_elapsed.value == 3000
    assert run.accounting().execution_elapsed.value == 3000
    assert run.accounting().accounted_modeled_cost.value == Decimal("0.03")
    golden = Path(__file__).with_name("fixtures") / "replay-v1.json"
    assert run.to_json() + "\n" == golden.read_text()
    assert ResourceReplayRun.from_json(golden.read_bytes()) == run
    assert ResourceReplayRun.from_json(run.to_json()).fingerprint == run.fingerprint
    assert run.queue_elapsed is None
    assert run.setup_elapsed is None


def test_overrun_stops_new_hypothetical_dispatches_but_native_suite_is_complete(example):
    protocol, _ = example.fixture()
    run = replay(protocol, "shortest-estimate")
    assert len(run.native_run.events) == 3
    assert run.selection.selected_ids == ("t2", "t3")
    assert run.accounting().selected_ids == ("t2",)
    assert run.ledger.balance.runner_occupancy == 2000
    assert run.ledger.balance.containment == "stop_new_reservations"
    assert run.accounting().failing_test_recall.value == 0.5


def test_negative_payoff_and_overhead_can_erase_apparent_work_savings(example):
    protocol, _ = example.fixture()
    baseline = replay(protocol)
    candidate = replay(protocol, "shortest-estimate", "1.5")
    report = ResourceReplayReport(schema_version=1, protocol=protocol, runs=(baseline, candidate))
    (pair,) = report.comparisons()
    assert pair.work_saving_ms == 1000
    assert pair.net_saving_ms == -500
    assert pair.gate == "fail"
    assert pair.adoption == "insufficient_evidence"
    assert report.status == "partial"
    assert len(report.missing) == 2
    text = render_resource_report(report)
    assert "net saving including charged overhead -500 ms" in text
    assert "null (unavailable)" in text
    assert "not prospective" in text


def test_unknown_setup_is_partial_and_not_zero(example):
    protocol, _ = example.fixture()
    run = replay(protocol, setup=None)
    assert run.accounting().accounting_status == "partial_unmeasured_setup"
    assert replay(protocol).accounting().accounting_status == "declared_runner_scope"
    assert run.to_json() != replay(protocol).to_json()


def test_future_realizations_cannot_enter_estimated_selection(example):
    protocol, _ = example.fixture()
    original = replay(protocol).request()
    raw = protocol.model_dump(mode="json")
    for build in raw["native_manifest"]["builds"]:
        for test in build["tests"]:
            test["duration"]["value"] = "0"
            test["outcome"] = "failed" if test["outcome"] == "passed" else "passed"
    raw["native_manifest"]["dataset_fingerprint"] = fingerprint(raw["native_manifest"]["builds"])
    changed = ResourceReplayProtocol.model_validate(raw)
    after = replay(changed).request()
    assert original.tests == after.tests
    assert original.ranking == after.ranking
    assert replay(protocol).selection == replay(changed).selection
    assert replay(protocol).ledger.balance != replay(changed).ledger.balance


def test_estimate_source_unavailable_at_freeze_rejected(example):
    protocol, _ = example.fixture()
    raw = protocol.model_dump(mode="json")
    raw["estimates"][0]["tests"][0]["estimate"]["available_at"] = "2026-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable when protocol was frozen"):
        ResourceReplayProtocol.model_validate(raw)


@pytest.mark.parametrize("outcome", ["unknown", "timeout"])
def test_unobserved_labels_remain_unavailable_even_when_excluded(example, outcome):
    protocol, _ = example.fixture()
    raw = protocol.model_dump(mode="json")
    for build in raw["native_manifest"]["builds"]:
        build["tests"][-1]["outcome"] = outcome
    raw["native_manifest"]["dataset_fingerprint"] = fingerprint(raw["native_manifest"]["builds"])
    run = replay(ResourceReplayProtocol.model_validate(raw))
    assert run.accounting().failing_test_recall.value is None
    assert run.accounting().failing_test_recall.status == "unavailable"


def test_no_failure_is_not_applicable_and_first_feedback_censored(example):
    protocol, _ = example.fixture()
    raw = protocol.model_dump(mode="json")
    for build in raw["native_manifest"]["builds"]:
        for test in build["tests"]:
            test["outcome"] = "passed"
    raw["native_manifest"]["dataset_fingerprint"] = fingerprint(raw["native_manifest"]["builds"])
    run = replay(ResourceReplayProtocol.model_validate(raw))
    assert run.accounting().failing_test_recall.status == "not_applicable"
    assert run.accounting().first_failure_elapsed.value is None
    assert run.accounting().first_failure_status == "right_censored"


def test_modeled_cost_is_exact_under_decimal_context_mutation(example):
    protocol, _ = example.fixture()
    run = replay(protocol, overhead="0.000000001")
    assert run.ledger.balance.overhead == 1
    with localcontext() as context:
        context.prec = 2
        assert run.modeled_cost().value == Decimal("0.03001")


def test_canonical_membership_duplicate_conflict_and_version_rejection(example):
    protocol, _ = example.fixture()
    raw = protocol.model_dump(mode="json")
    raw["estimates"].reverse()
    for item in raw["estimates"]:
        item["tests"].reverse()
    assert ResourceReplayProtocol.model_validate(raw).to_json() == protocol.to_json()
    run = replay(protocol)
    report = ResourceReplayReport(schema_version=1, protocol=protocol, runs=(run, run))
    assert len(report.runs) == 1
    changed = replay(protocol, overhead="0.001")
    with pytest.raises(ValueError, match="Conflicting resource candidate/build"):
        report.append(changed)
    payload = json.loads(run.to_json())
    payload["schema_version"] = 2
    encoded = json.dumps(payload)
    with pytest.raises(ValueError):
        ResourceReplayRun.from_json(encoded)


def test_old_pilot_golden_remains_readable_unmodified():
    from coleman.pilot import PilotReport

    path = Path(__file__).parents[1] / "pilot/fixtures/pilot-v1.json"
    payload = path.read_bytes()
    old = PilotReport.from_json(payload)
    assert old.schema_version == 1
    assert path.read_bytes() == payload


def expanded_protocol(example, count):
    protocol, _ = example.fixture()
    raw = protocol.model_dump(mode="json")
    raw["native_manifest"]["builds"] = raw["native_manifest"]["builds"][:1]
    raw["estimates"] = raw["estimates"][:1]
    build = raw["native_manifest"]["builds"][0]
    duration = build["tests"][0]["duration"]
    duration["value"] = "0.001"
    build["tests"] = [
        {"test_id": f"test-{i:04d}", "duration": duration, "outcome": "failed" if i % 3 == 0 else "passed"}
        for i in range(count)
    ]
    build["baseline_order"] = [test["test_id"] for test in build["tests"]]
    prior = raw["estimates"][0]["tests"][0]["estimate"]
    prior["amount"]["value"] = 1
    raw["estimates"][0]["tests"] = [{"test_id": test["test_id"], "estimate": prior} for test in build["tests"]]
    raw["native_manifest"]["dataset_fingerprint"] = fingerprint(raw["native_manifest"]["builds"])
    return raw


def test_bounded_thousand_test_replay_reconciles_every_event(example):
    protocol = ResourceReplayProtocol.model_validate(expanded_protocol(example, 1000))
    run = replay(protocol)
    ledger = run.ledger
    assert len(ledger.events) == 3002
    assert ledger.balance.consumed == 1000
    assert ledger.balance.available == 4000
    assert ledger.balance.reserved == 0
    assert ledger.balance.containment == "normal"
    assert len(ledger.balance.terminal_execution_ids) == 1000
    assert run.accounting().failing_test_recall.value == 1.0
    assert ResourceReplayRun.from_json(run.to_json()).ledger.to_json() == ledger.to_json()


def test_unsupported_unbounded_test_universe_fails_at_contract_boundary(example):
    raw = expanded_protocol(example, 1001)
    with pytest.raises(ValueError, match="at most 1000"):
        ResourceReplayProtocol.model_validate(raw)


def test_zero_work_feedback_stays_observed_and_fractional_savings_remain_inconclusive(example):
    protocol, _ = example.fixture()
    raw = protocol.model_dump(mode="json")
    for build in raw["native_manifest"]["builds"]:
        for test in build["tests"]:
            test["duration"]["value"] = "0"
    for estimates in raw["estimates"]:
        for test in estimates["tests"]:
            test["estimate"]["amount"]["value"] = 0
    raw["native_manifest"]["dataset_fingerprint"] = fingerprint(raw["native_manifest"]["builds"])
    changed = ResourceReplayProtocol.model_validate(raw)
    baseline = replay(changed, overhead="0.001")
    candidate = replay(changed, "shortest-estimate")
    first = candidate.accounting().first_failure_elapsed
    assert first.status == "observed"
    assert first.value == 0
    assert candidate.accounting().failing_test_recall.value == 1.0
    report = ResourceReplayReport(schema_version=1, protocol=changed, runs=(baseline, candidate))
    (pair,) = report.comparisons()
    assert pair.work_saving_ms == 0
    assert pair.net_saving_ms == 1
    assert pair.gate == "inconclusive"
    assert "zero_baseline_work" in pair.reasons


def test_detached_selection_and_comparison_carry_units_and_semantics(example):
    protocol, _ = example.fixture()
    baseline = replay(protocol)
    candidate = replay(protocol, "shortest-estimate")
    selection = baseline.selection
    assert selection.semantics == RUNNER_MS
    assert selection.model_dump(mode="json")["semantics"] == RUNNER_MS.model_dump(mode="json")
    report = ResourceReplayReport(schema_version=1, protocol=protocol, runs=(baseline, candidate))
    (pair,) = report.comparisons()
    assert pair.semantics == RUNNER_MS
    wrong = pair.model_dump(mode="json")
    wrong["semantics"]["meaning"] = "execution-elapsed-v1"
    from coleman.resources import ResourceComparison

    with pytest.raises(ValueError, match="occupied runner milliseconds"):
        ResourceComparison.model_validate(wrong)
