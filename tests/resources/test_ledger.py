"""Independent accounting fixtures, exact replay properties and interrupted publication."""

import ast
import concurrent.futures
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from coleman.artifacts import ArtifactWriter
from coleman.resources import (
    RUNNER_MS,
    IntentEvent,
    OccupancyEvent,
    OverheadEvent,
    ReserveEvent,
    ResourceAmount,
    ResourceEstimate,
    ResourceLedger,
    ResourceRequest,
    publish_ledger,
    restore_ledger,
    select_resources,
)
from coleman.resources import (
    TestEstimate as CostEstimate,
)
from coleman.results.domain import fingerprint
from coleman.selection.contracts import UnitSemantics

SOURCE = fingerprint({"owned": "ci-resource-fixture-v1"})
TIME = datetime.fromisoformat("2026-01-02T00:00:00Z")


def amount(value):
    return ResourceAmount(value=value, semantics=RUNNER_MS)


def estimate(value):
    return ResourceEstimate(
        amount=amount(value),
        source_fingerprint=SOURCE,
        observed_at=datetime.fromisoformat("2025-12-30T00:00:00Z"),
        available_at=datetime.fromisoformat("2025-12-31T00:00:00Z"),
        valid_until=datetime.fromisoformat("2026-01-03T00:00:00Z"),
    )


def metadata(sequence) -> dict[str, Any]:
    return dict(event_id=f"event-{sequence}", sequence=sequence, event_at=TIME, recorded_at=TIME)


def empty(capacity=2000):
    return ResourceLedger(
        schema_version=1,
        ledger_id="ci-build-2",
        source_fingerprint=SOURCE,
        configuration_fingerprint=SOURCE,
        capacity=amount(capacity),
    )


def reserved(value=1000, capacity=2000):
    ledger = empty(capacity).append(
        IntentEvent(**metadata(1), execution_id="test-a:attempt-1", estimate=estimate(value))
    )
    return ledger.append(ReserveEvent(**metadata(2), execution_id="test-a:attempt-1", amount=amount(value)))


def complete(ledger, value=900, terminal="passed"):
    return ledger.append(
        OccupancyEvent(
            **metadata(len(ledger.events) + 1),
            execution_id="test-a:attempt-1",
            amount=amount(value),
            terminal=terminal,
        )
    )


def balances(ledger):
    b = ledger.balance
    assert ledger.capacity.value == b.available + b.reserved + b.consumed
    assert b.consumed == b.runner_occupancy + b.overhead
    return b.available, b.reserved, b.consumed


def fixture():
    ledger = complete(reserved())
    return ledger.append(OverheadEvent(**metadata(4), component="dispatch", amount=amount(1)))


def test_independent_issue_fixture_and_golden():
    ledger = empty()
    assert balances(ledger) == (2000, 0, 0)
    ledger = reserved()
    assert balances(ledger) == (1000, 1000, 0)
    ledger = complete(ledger)
    assert balances(ledger) == (1100, 0, 900)
    ledger = fixture()
    assert balances(ledger) == (1099, 0, 901)
    golden = Path(__file__).with_name("fixtures") / "ledger-v1.json"
    assert ledger.to_json() + "\n" == golden.read_text()
    assert ResourceLedger.from_json(golden.read_bytes()) == ledger
    assert ledger.fingerprint == ResourceLedger.from_json(ledger.to_json()).fingerprint
    assert ledger.logical_id == empty().logical_id
    assert ledger.fingerprint != empty().fingerprint


def test_duplicate_conflict_rejects_preserving_previous_balances():
    ledger = fixture()
    assert ledger.append(ledger.events[2]).to_json() == ledger.to_json()
    changed = ledger.events[2].model_dump()
    changed["amount"] = amount(950)
    conflict = OccupancyEvent.model_validate(changed)
    with pytest.raises(ValueError, match="Conflicting resource event"):
        ledger.append(conflict)
    assert balances(ledger) == (1099, 0, 901)


def test_actual_overrun_cannot_be_clipped_or_erased():
    ledger = complete(reserved(900, 1000), 1100)
    assert balances(ledger) == (-100, 0, 1100)
    assert ledger.balance.overrun_execution_ids == ("test-a:attempt-1",)
    assert ledger.balance.containment == "stop_new_reservations"
    ledger = ledger.append(IntentEvent(**metadata(4), execution_id="b", estimate=estimate(0)))
    reservation = ReserveEvent(**metadata(5), execution_id="b", amount=amount(0))
    with pytest.raises(ValueError, match="Containment"):
        ledger.append(reservation)
    ledger = ledger.append(OverheadEvent(**metadata(5), component="shared_setup", amount=amount(1)))
    assert balances(ledger) == (-101, 0, 1101)


@pytest.mark.parametrize("terminal", ["passed", "failed", "unknown", "timeout", "partial", "cancelled"])
def test_partial_occupancy_then_terminal_releases_only_unused_reservation(terminal):
    ledger = complete(reserved(), 200, None)
    assert balances(ledger) == (1000, 800, 200)
    assert ledger.balance.pending_execution_ids == ("test-a:attempt-1",)
    ledger = complete(ledger, 300, terminal)
    assert balances(ledger) == (1500, 0, 500)
    assert ledger.balance.pending_execution_ids == ()
    with pytest.raises(ValueError, match="terminal"):
        complete(ledger, 1)


def test_timeout_retry_requires_new_identity_and_charges_both_attempts():
    ledger = complete(reserved(), 900, "timeout")
    ledger = ledger.append(
        IntentEvent(**metadata(4), execution_id="test-a:attempt-2", retry_of="test-a:attempt-1", estimate=estimate(500))
    )
    ledger = ledger.append(ReserveEvent(**metadata(5), execution_id="test-a:attempt-2", amount=amount(500)))
    ledger = ledger.append(
        OccupancyEvent(**metadata(6), execution_id="test-a:attempt-2", amount=amount(400), terminal="failed")
    )
    assert balances(ledger) == (700, 0, 1300)
    assert len(ledger.balance.terminal_execution_ids) == 2
    initial = empty()
    retry = IntentEvent(**metadata(1), execution_id="retry", retry_of="missing", estimate=estimate(1))
    with pytest.raises(ValueError, match="previously terminal"):
        initial.append(retry)


@pytest.mark.parametrize("bad", [None, True, False, -1, 1.5, 1.0, "1", float("nan"), float("inf"), 10**12 + 1])
def test_integer_input_rejects_ambiguous_unsupported_values(bad):
    with pytest.raises(ValidationError):
        amount(bad)


def test_zero_cancel_is_not_null_or_unavailable():
    ledger = complete(reserved(0, 0), 0, "cancelled")
    assert balances(ledger) == (0, 0, 0)
    assert ledger.balance.terminal_execution_ids == ("test-a:attempt-1",)
    with pytest.raises(ValidationError):
        amount(None)


@pytest.mark.parametrize("version", [None, 0, 2, 99, True, "1"])
def test_versions_fail_closed_without_implicit_migration(version):
    payload = json.loads(fixture().to_json())
    payload["schema_version"] = version
    encoded = json.dumps(payload)
    with pytest.raises(ValueError):
        ResourceLedger.from_json(encoded)


def test_sequence_time_staleness_units_and_unsupported_mechanics():
    initial = empty()
    skipped = IntentEvent(**metadata(2), execution_id="a", estimate=estimate(1))
    with pytest.raises(ValueError, match="contiguous"):
        initial.append(skipped)
    stale_metadata = metadata(1)
    stale_metadata.update(
        event_at=datetime.fromisoformat("2026-01-04T00:00:00Z"),
        recorded_at=datetime.fromisoformat("2026-01-04T00:00:00Z"),
    )
    stale_intent = IntentEvent(**stale_metadata, execution_id="a", estimate=estimate(1))
    ledger = empty()
    with pytest.raises(ValueError, match="unavailable or stale"):
        ledger.append(stale_intent)
    wrong = ResourceAmount(value=1, semantics=UnitSemantics(unit="s", meaning=RUNNER_MS.meaning))
    held = reserved()
    incompatible = OverheadEvent(**metadata(3), component="dispatch", amount=wrong)
    with pytest.raises(ValueError, match="Incompatible"):
        held.append(incompatible)
    unsupported = empty().to_json().replace('"events":[]', '"events":[{"kind":"refund"}]')
    with pytest.raises(ValueError):
        ResourceLedger.from_json(unsupported)
    intent_only = empty().append(IntentEvent(**metadata(1), execution_id="test-a:attempt-1", estimate=estimate(1)))
    with pytest.raises(ValueError, match="prior reservation"):
        complete(intent_only, 1)


@pytest.mark.parametrize("first", [0, 1, 7, 99, 200, 499, 500])
@pytest.mark.parametrize("second", [0, 1, 7, 99, 200, 499, 500])
def test_metamorphic_split_occupancy_and_duplicate_deliveries(first, second):
    one = complete(reserved(), first + second)
    split = complete(complete(reserved(), first, None), second)
    assert one.balance == split.balance
    for event in split.events:
        assert split.append(event).to_json() == split.to_json()


def test_interrupted_publication_and_concurrent_retry(tmp_path, monkeypatch):
    ledger = fixture()
    writer = ArtifactWriter(tmp_path)
    original = writer.write_immutable

    def fail_checkpoint(path, content):
        if path.parent.name == "checkpoints":
            raise OSError("injected before checkpoint publication")
        original(path, content)

    with monkeypatch.context() as fault:
        fault.setattr(writer, "write_immutable", fail_checkpoint)
        with pytest.raises(OSError, match="injected"):
            publish_ledger(ledger, writer)
    assert not list(tmp_path.rglob("checkpoints/*.json"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: publish_ledger(ledger, writer), range(8)))
    assert len(set(paths)) == 1
    assert restore_ledger(paths[0]) == ledger
    changed = complete(reserved(), 901)
    with pytest.raises(ValueError, match="Conflicting immutable"):
        publish_ledger(changed, writer)
    assert restore_ledger(paths[0]).balance == ledger.balance


def test_predispatch_fallback_and_unavailable_estimates():
    request = ResourceRequest(
        schema_version=1,
        decision_id="build-2",
        configuration_fingerprint=SOURCE,
        decided_at=TIME,
        capacity=amount(1000),
        tests=(CostEstimate(test_id="a", estimate=estimate(900)), CostEstimate(test_id="b", estimate=estimate(1000))),
        ranking=("a", "b"),
    )
    assert select_resources(request).selected_ids == ("a",)
    raw = request.model_dump()
    raw.update(priority_status="stale", fallback="full_suite", capacity=amount(2000))
    assert select_resources(ResourceRequest.model_validate(raw)).disposition == "full_suite_fallback"
    raw["capacity"] = amount(1000)
    assert select_resources(ResourceRequest.model_validate(raw)).selected_ids == ()
    raw["tests"][0]["estimate"] = None
    assert select_resources(ResourceRequest.model_validate(raw)).reason == "ineligible_estimate"


def test_import_boundary_has_no_cli_or_transports():
    root = Path(__file__).parents[2] / "coleman/resources"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert all(not module.startswith(("coleman.cli", "coleman.transports", "mcp")) for module in modules)


@pytest.mark.parametrize("scale", [1, 2, 7, 1000])
def test_metamorphic_explicit_integer_scaling_preserves_accounting(scale):
    original = fixture()
    payload = json.loads(original.to_json())
    payload["capacity"]["value"] *= scale
    for event in payload["events"]:
        if event["kind"] == "intent":
            event["estimate"]["amount"]["value"] *= scale
        else:
            event["amount"]["value"] *= scale
    changed = ResourceLedger.model_validate(payload)
    assert balances(changed) == tuple(value * scale for value in balances(original))
    assert changed.balance.containment == original.balance.containment


def test_new_receipt_out_of_order_and_estimate_expiry_before_dispatch_fail():
    ledger = reserved()
    invalid = OverheadEvent.model_validate(
        {
            **metadata(3),
            "event_at": "2026-01-01T00:00:00Z",
            "recorded_at": "2026-01-01T00:00:00Z",
            "component": "dispatch",
            "amount": amount(0),
        }
    )
    with pytest.raises(ValueError, match="receipt is out of order"):
        ledger.append(invalid)
    pending = empty().append(IntentEvent(**metadata(1), execution_id="a", estimate=estimate(1)))
    stale = ReserveEvent.model_validate(
        {
            **metadata(2),
            "event_at": "2026-01-04T00:00:00Z",
            "recorded_at": "2026-01-04T00:00:00Z",
            "execution_id": "a",
            "amount": amount(1),
        }
    )
    with pytest.raises(ValueError, match="stale before reservation"):
        pending.append(stale)
    assert pending.balance.pending_execution_ids == ("a",)
    assert balances(pending) == (2000, 0, 0)


def test_cancel_and_partial_overrun_stay_observed_and_immutable():
    cancelled = complete(reserved(), 0, "cancelled")
    assert balances(cancelled) == (2000, 0, 0)
    pending = complete(reserved(900), 1100, None)
    assert pending.balance.overrun_execution_ids == ("test-a:attempt-1",)
    assert pending.balance.pending_execution_ids == ("test-a:attempt-1",)
    settled = complete(pending, 100, "partial")
    assert balances(settled) == (800, 0, 1200)
    assert settled.balance.containment == "stop_new_reservations"
    assert balances(pending) == (900, 0, 1100)


def test_atomic_link_failure_cannot_publish_a_completed_checkpoint(tmp_path, monkeypatch):
    import coleman.artifacts

    writer = ArtifactWriter(tmp_path)
    ledger = fixture()
    original = coleman.artifacts.os.link

    def fail_final_link(source, destination):
        if destination.parent.name == "checkpoints":
            raise OSError("injected interrupted final link")
        original(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(coleman.artifacts.os, "link", fail_final_link)
        with pytest.raises(OSError, match="interrupted final link"):
            publish_ledger(ledger, writer)
    assert not list(tmp_path.rglob("checkpoints/*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    path = publish_ledger(ledger, writer)
    assert restore_ledger(path).to_json() == ledger.to_json()


def test_terminal_and_retry_cannot_precede_prior_execution_events():
    ledger = reserved()
    later = OccupancyEvent.model_validate(
        {
            **metadata(3),
            "event_at": "2026-01-02T00:00:02Z",
            "recorded_at": "2026-01-02T00:00:03Z",
            "execution_id": "test-a:attempt-1",
            "amount": amount(100),
            "terminal": None,
        }
    )
    ledger = ledger.append(later)
    reversed_terminal = OccupancyEvent.model_validate(
        {
            **metadata(4),
            "event_at": "2026-01-02T00:00:01Z",
            "recorded_at": "2026-01-02T00:00:03Z",
            "execution_id": "test-a:attempt-1",
            "amount": amount(0),
            "terminal": "timeout",
        }
    )
    with pytest.raises(ValueError, match="reversed execution"):
        ledger.append(reversed_terminal)
    valid_terminal = OccupancyEvent.model_validate(
        {
            **metadata(4),
            "event_at": "2026-01-02T00:00:02Z",
            "recorded_at": "2026-01-02T00:00:03Z",
            "execution_id": "test-a:attempt-1",
            "amount": amount(0),
            "terminal": "timeout",
        }
    )
    ledger = ledger.append(valid_terminal)
    reversed_retry = IntentEvent.model_validate(
        {
            **metadata(5),
            "event_at": "2026-01-02T00:00:01Z",
            "recorded_at": "2026-01-02T00:00:03Z",
            "execution_id": "test-a:attempt-2",
            "retry_of": "test-a:attempt-1",
            "estimate": estimate(1),
        }
    )
    with pytest.raises(ValueError, match="previously terminal"):
        ledger.append(reversed_retry)


def test_full_ledger_still_accepts_identical_retry_without_consuming_a_delivery_slot():
    events = tuple(OverheadEvent(**metadata(index), component="dispatch", amount=amount(0)) for index in range(1, 4097))
    ledger = ResourceLedger(
        schema_version=1,
        ledger_id="bounded-full-log",
        source_fingerprint=SOURCE,
        configuration_fingerprint=SOURCE,
        capacity=amount(0),
        events=events,
    )
    assert ledger.append(events[-1]).to_json() == ledger.to_json()
    conflict_payload = events[-1].model_dump()
    conflict_payload["amount"] = amount(1)
    conflict = OverheadEvent.model_validate(conflict_payload)
    with pytest.raises(ValueError, match="Conflicting resource event"):
        ledger.append(conflict)
    assert balances(ledger) == (0, 0, 0)
