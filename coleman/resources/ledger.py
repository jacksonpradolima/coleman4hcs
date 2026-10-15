"""Bounded immutable event replay, exact reconciliation and sticky containment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.resources.contracts import (
    IntentEvent,
    OccupancyEvent,
    OverheadEvent,
    ReserveEvent,
    ResourceAmount,
    ResourceEvent,
)
from coleman.results.domain import Fingerprint, Identifier, canonical_json, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, UnitSemantics

Total = Annotated[int, Field(strict=True, ge=0, le=4097 * 10**12)]


class ResourceBalance(Immutable):
    """Exact reconciled resource units; available may be negative after actual work.

    Initial = available + reserved + consumed after every accepted event. Overrun
    includes exceeding an estimate, even with remaining capacity. Containment is
    sticky within this ledger; restarting or finishing cannot erase a violation.
    """

    semantics: UnitSemantics
    available: Annotated[int, Field(strict=True, ge=-4097 * 10**12, le=10**12)]
    reserved: Total
    consumed: Total
    runner_occupancy: Total
    overhead: Total
    overrun_execution_ids: tuple[Identifier, ...]
    containment: Literal["normal", "stop_new_reservations"]
    pending_execution_ids: tuple[Identifier, ...]
    terminal_execution_ids: tuple[Identifier, ...]


@dataclass
class _Attempt:
    """Private mutable accumulator; never crosses a public contract boundary."""

    intent: IntentEvent
    reservation: ReserveEvent | None = None
    consumed: int = 0
    remaining: int = 0
    terminal: bool = False
    last_event_at: datetime | None = None


class _Replay:
    """Linear replay accumulator; constructed afresh for every validated snapshot."""

    def __init__(self, capacity: ResourceAmount):
        """Initialize declared capacity with no assumed adjustments or execution."""
        self.capacity = capacity
        self.available = capacity.value
        self.reserved = self.consumed = self.work = self.overhead = 0
        self.attempts: dict[str, _Attempt] = {}
        self.overruns: set[str] = set()
        self.contained = False

    def _units(self, amount: ResourceAmount) -> None:
        """Reject any unit or meaning change before updating balances."""
        if amount.semantics != self.capacity.semantics:
            raise ValueError("Incompatible resource units or semantics")

    def apply(self, event: ResourceEvent) -> None:
        """Apply one unique ordered event, raising on illegal state transitions."""
        if isinstance(event, IntentEvent):
            self._intent(event)
        elif isinstance(event, OverheadEvent):
            self._units(event.amount)
            self.available -= event.amount.value
            self.consumed += event.amount.value
            self.overhead += event.amount.value
        else:
            self._execution(event)
        self.contained |= bool(self.overruns) or self.available < 0
        if self.capacity.value != self.available + self.reserved + self.consumed:
            raise ValueError("Resource reconciliation failed")

    def _intent(self, event: IntentEvent) -> None:
        """Require a unique eligible attempt and a terminal parent for retries."""
        self._units(event.estimate.amount)
        if event.execution_id in self.attempts:
            raise ValueError("Execution identity already declared; retry needs a new ID")
        if not event.estimate.available_at <= event.event_at <= event.estimate.valid_until:
            raise ValueError("Estimate unavailable or stale at decision")
        if event.retry_of is not None:
            parent = self.attempts.get(event.retry_of)
            if (
                parent is None
                or not parent.terminal
                or event.event_at < (parent.last_event_at or parent.intent.event_at)
            ):
                raise ValueError("Retry requires a previously terminal attempt")
        self.attempts[event.execution_id] = _Attempt(intent=event)

    def _execution(self, event: ReserveEvent | OccupancyEvent) -> None:
        """Reserve estimates or charge observed deltas without clipping reality."""
        self._units(event.amount)
        attempt = self.attempts.get(event.execution_id)
        if attempt is None or attempt.terminal or event.event_at < (attempt.last_event_at or attempt.intent.event_at):
            raise ValueError("Unknown, terminal or reversed execution event")
        if isinstance(event, ReserveEvent):
            self._reserve(event, attempt)
            return
        if attempt.reservation is None or event.event_at < attempt.reservation.event_at:
            raise ValueError("Occupancy requires a prior reservation")
        attempt.last_event_at = event.event_at
        used = min(attempt.remaining, event.amount.value)
        attempt.remaining -= used
        attempt.consumed += event.amount.value
        self.reserved -= used
        self.available -= event.amount.value - used
        self.consumed += event.amount.value
        self.work += event.amount.value
        if attempt.consumed > attempt.reservation.amount.value:
            self.overruns.add(event.execution_id)
        if event.terminal is not None:
            self.available += attempt.remaining
            self.reserved -= attempt.remaining
            attempt.remaining = 0
            attempt.terminal = True

    def _reserve(self, event: ReserveEvent, attempt: _Attempt) -> None:
        """Check eligible estimate, available balance and existing containment."""
        if attempt.reservation is not None:
            raise ValueError("Execution already reserved")
        if self.contained:
            raise ValueError("Containment blocks new reservations")
        if event.event_at > attempt.intent.estimate.valid_until:
            raise ValueError("Estimate stale before reservation")
        if event.amount != attempt.intent.estimate.amount or event.amount.value > self.available:
            raise ValueError("Reservation must equal eligible estimate and fit available capacity")
        self.available -= event.amount.value
        self.reserved += event.amount.value
        attempt.remaining = event.amount.value
        attempt.reservation = event
        attempt.last_event_at = event.event_at

    def balance(self) -> ResourceBalance:
        """Expose verified integer totals and explicit incomplete attempts."""
        return ResourceBalance(
            semantics=self.capacity.semantics,
            available=self.available,
            reserved=self.reserved,
            consumed=self.consumed,
            runner_occupancy=self.work,
            overhead=self.overhead,
            overrun_execution_ids=tuple(sorted(self.overruns)),
            containment="stop_new_reservations" if self.contained else "normal",
            pending_execution_ids=tuple(sorted(key for key, attempt in self.attempts.items() if not attempt.terminal)),
            terminal_execution_ids=tuple(sorted(key for key, attempt in self.attempts.items() if attempt.terminal)),
        )


class ResourceLedger(VersionedContract):
    """Versioned complete audit log; pure replay also serves as a restart snapshot.

    At most 4,096 deliveries are accepted per constructor; identical IDs with
    identical canonical payload deduplicate before sequence checks. Unique events
    require contiguous sequence 1..n and nondecreasing original receipt times.
    Duplicate payload differences, unsupported operations, versions and invalid
    transitions fail without mutating a prior snapshot. No implicit v0 migration
    or legacy artifact rewrite exists. Actual work can exceed resource capacity;
    containment is observed and cannot stop an external process retroactively.
    """

    ledger_id: Identifier
    source_fingerprint: Fingerprint
    configuration_fingerprint: Fingerprint
    capacity: ResourceAmount
    events: Annotated[tuple[ResourceEvent, ...], Field(max_length=4096)] = ()

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Deduplicate exact deliveries and reconcile every ordered transition."""
        seen: dict[str, str] = {}
        unique = []
        replay = _Replay(self.capacity)
        previous = None
        for event in self.events:
            content = canonical_json(event.model_dump(mode="json"))
            if event.event_id in seen:
                if seen[event.event_id] != content:
                    raise ValueError(f"Conflicting resource event: {event.event_id}")
                continue
            if event.sequence != len(unique) + 1:
                raise ValueError("New resource events require contiguous sequence order")
            if previous is not None and event.recorded_at < previous.recorded_at:
                raise ValueError("New resource event receipt is out of order")
            replay.apply(event)
            seen[event.event_id] = content
            unique.append(event)
            previous = event
        object.__setattr__(self, "events", tuple(unique))
        return self

    @property
    def logical_id(self) -> str:
        """Identify the declared ledger independently of changing balances/content."""
        return fingerprint({"namespace": "resource-ledger-v1", "ledger_id": self.ledger_id})

    @property
    def balance(self) -> ResourceBalance:
        """Recompute verified state; no cached or supplied totals are trusted."""
        validated = ResourceLedger.model_validate(self)
        replay = _Replay(validated.capacity)
        for event in validated.events:
            replay.apply(event)
        return replay.balance()

    def append(self, event: ResourceEvent) -> ResourceLedger:
        """Return a validated new snapshot; identical retries remain unchanged.

        Invalid/conflicting events raise ValueError and leave this object intact.
        Callers persist the returned snapshot before acknowledging an event.
        """
        return ResourceLedger(
            schema_version=1,
            ledger_id=self.ledger_id,
            source_fingerprint=self.source_fingerprint,
            configuration_fingerprint=self.configuration_fingerprint,
            capacity=self.capacity,
            events=(*self.events, event),
        )
