"""Immutable resumable shadow state with independently inspectable operational views."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from coleman.observations import ObservationHistory
from coleman.resources import ResourceLedger
from coleman.results.domain import Fingerprint, Identifier, canonical_json
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.shadow.contracts import (
    AdmittedEvent,
    BuildClosed,
    BuildOpened,
    HostClock,
    ShadowEvent,
    ShadowProtocol,
)
from coleman.shadow.engine import _Engine, event_json

_EVENT_ADAPTER = TypeAdapter(ShadowEvent)
MAXIMUM_STATE_BYTES = 20 * 1024 * 1024


class OperationalView(Immutable):
    """Current ingestion health, independent of research pass/fail/inconclusive status.

    Recovering requires a fresh source heartbeat in this monotonic clock session.
    Stopped recommendations never cancel the required upstream CI suite or erase
    accepted accounting. Unknown cost/feedback remains visible in build snapshots.
    """

    status: Literal["healthy", "stopped", "recovering"]
    reasons: tuple[Identifier, ...]
    clock: HostClock
    first_admission: HostClock | None
    last_admission: HostClock | None
    last_fresh_source: HostClock | None
    accepted_events: Annotated[int, Field(strict=True, ge=0, le=4096)]
    quarantined_events: Annotated[int, Field(strict=True, ge=0, le=128)]


class ShadowBuildSnapshot(Immutable):
    """Full required-CI work and application-as-of labels, separate from shadow choices.

    The original provider revisions remain in the state journal. This history's
    local receipt is first validated application admission, explicitly namespaced
    separately from provider receipt. The resource ledger books actual required
    work once per test attempt, including work outside the modeled selection.
    """

    opening: BuildOpened
    application_opened_at: HostClock
    closure: BuildClosed | None
    application_closed_at: HostClock | None
    history: ObservationHistory
    actual_ledger: ResourceLedger
    resource_accounting: Literal["settled", "pending_or_unknown"]


class ShadowState(VersionedContract):
    """Bounded typed admission journal and the exact protocol frozen before it.

    Array order defines admission order and timestamp ties; event IDs never guess
    chronology. Accepted entries are rederived on every read. Quarantined payloads
    remain bounded and block recommendations; unsupported/missing causal parents
    require source/operator repair under a new experiment, not invented history.
    Identical source retries retain the first application clock and consume no new
    slot, including at capacity. No native pickle or transport is read here.
    """

    protocol: ShadowProtocol
    journal: Annotated[tuple[AdmittedEvent, ...], Field(max_length=4096)] = ()

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Reconcile all source admission, identity, time and resource invariants."""
        self._engine()
        return self

    def _engine(self) -> _Engine:
        """Replay into private local accumulators; never retain a mutable derived cache."""
        if len(self.journal) > self.protocol.maximum_events:
            raise ValueError("Shadow journal event retention limit reached")
        engine = _Engine(self.protocol)
        total = 0
        for entry in self.journal:
            size = len(canonical_json(entry.model_dump(mode="json")).encode("utf-8"))
            if len(event_json(entry.event).encode("utf-8")) > self.protocol.maximum_event_bytes:
                raise ValueError("Shadow source event exceeds its byte limit")
            total += size
            if total > self.protocol.maximum_journal_bytes:
                raise ValueError("Shadow journal byte retention limit reached")
            engine.accept(entry)
        return engine

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Reject oversized serialized state before parsing, then strict v1/duplicate-key validation."""
        if len(payload) > MAXIMUM_STATE_BYTES:
            raise ValueError("Shadow state exceeds supported serialized byte limit")
        if isinstance(payload, str) and len(payload.encode("utf-8")) > MAXIMUM_STATE_BYTES:
            raise ValueError("Shadow state exceeds supported serialized byte limit")
        return super().from_json(payload)

    @property
    def logical_id(self) -> str:
        """Identify the frozen experiment; model/config/source changes cannot inherit its evidence."""
        return self.protocol.logical_id

    def ingest(self, event: ShadowEvent, clock: HostClock) -> ShadowState:
        """Return a new validated state, retaining prior facts on rejection or conflict.

        Duplicate source payloads ignore later retry clocks and refresh neither
        liveness nor evidence counts. Different payload for an existing ID raises
        ValueError. Unsupported causal/source operations become typed quarantine
        entries up to the frozen limit; full byte/event/quarantine limits reject
        without acknowledging or altering the previous state. Persist the returned
        state through ArtifactWriter before acknowledging an upstream event.
        """
        state = ShadowState.model_validate(self)
        event = _EVENT_ADAPTER.validate_python(event)
        clock = HostClock.model_validate(clock)
        engine = state._engine()
        previous = engine.seen.get(event.event_id)
        if previous is not None:
            if event_json(previous.event) != event_json(event):
                raise ValueError(f"Conflicting shadow source event: {event.event_id}")
            return state
        reason = engine.reason(event, clock)
        entry = AdmittedEvent(
            event=event,
            clock=clock,
            protocol_fingerprint=state.protocol.fingerprint,
            disposition="quarantined" if reason else "accepted",
            reason=reason,
        )
        return ShadowState(schema_version=1, protocol=state.protocol, journal=(*state.journal, entry))

    def operational(self, clock: HostClock) -> OperationalView:
        """Query current health without changing source receipts, clock session or evidence."""
        engine = ShadowState.model_validate(self)._engine()
        clock = HostClock.model_validate(clock)
        status, reasons = engine.operational(clock)
        return OperationalView(
            status=status,
            reasons=reasons,
            clock=clock,
            first_admission=self.journal[0].clock if self.journal else None,
            last_admission=engine.last_clock,
            last_fresh_source=engine.last_fresh,
            accepted_events=len(self.journal) - len(engine.quarantined),
            quarantined_events=len(engine.quarantined),
        )

    def builds(self) -> tuple[ShadowBuildSnapshot, ...]:
        """Expose verified actual-work snapshots without inferring labels for missing tests."""
        state = ShadowState.model_validate(self)
        engine = state._engine()
        snapshots = []
        for build_id in sorted(engine.builds):
            build = engine.builds[build_id]
            closure = build.closure.event if build.closure is not None else None
            if closure is not None and not isinstance(closure, BuildClosed):
                raise ValueError("Internal closure is not a typed upstream build closure")
            snapshots.append(
                ShadowBuildSnapshot(
                    opening=build.opening,
                    application_opened_at=build.opened_clock,
                    closure=closure,
                    application_closed_at=build.closure.clock if build.closure is not None else None,
                    history=build.history(state.protocol),
                    actual_ledger=build.ledger(state.protocol),
                    resource_accounting="settled" if build.settled else "pending_or_unknown",
                )
            )
        return tuple(snapshots)

    def source_event(self, event_id: str) -> ShadowEvent | None:
        """Return the exact first source payload for native retry adapters, including original receipts."""
        entry = ShadowState.model_validate(self)._engine().seen.get(event_id)
        return None if entry is None else entry.event

    @property
    def admitted_event_ids(self) -> tuple[str, ...]:
        """List unique accepted source IDs; seeds and delivery attempts never add observations."""
        state = ShadowState.model_validate(self)
        return tuple(entry.event.event_id for entry in state.journal if entry.disposition == "accepted")

    @property
    def protocol_fingerprint(self) -> Fingerprint:
        """Return the exact frozen contract digest separately from changing observed state."""
        return self.protocol.fingerprint
