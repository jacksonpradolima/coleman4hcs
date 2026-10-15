"""Deterministic shadow admission and required-CI accounting; no scheduler or I/O."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from coleman.resources import (
    IntentEvent,
    OccupancyEvent,
    ReserveEvent,
)
from coleman.resources.ledger import _Replay
from coleman.results.domain import canonical_json
from coleman.shadow._build import _Build
from coleman.shadow.admission import feedback_reason, opening_reason
from coleman.shadow.contracts import (
    AdmittedEvent,
    BuildClosed,
    BuildOpened,
    HostClock,
    ShadowEvent,
    ShadowProtocol,
    SourceStatus,
    TestFeedback,
)


class _Engine:
    """Ephemeral deterministic replay, with bounded quarantine and sticky risk containment."""

    def __init__(self, protocol: ShadowProtocol):
        """Initialize no observations, no live reservations and no assumed connectivity."""
        self.protocol = protocol
        self.protocol_fingerprint = protocol.fingerprint
        self.builds: dict[str, _Build] = {}
        self.seen: dict[str, AdmittedEvent] = {}
        self.quarantined: list[AdmittedEvent] = []
        self.last_clock: HostClock | None = None
        self.session_ticks: dict[str, int] = {}
        self.last_fresh: HostClock | None = None
        self.last_status_at: datetime | None = None
        self.connected = False
        self.risk_reasons: set[str] = set()

    def operational(self, clock: HostClock) -> tuple[Literal["healthy", "stopped", "recovering"], tuple[str, ...]]:
        """Separate host monotonic liveness from source UTC age and research conclusions."""
        self._clock(clock)
        if self.risk_reasons or self.quarantined:
            return "stopped", tuple(sorted(self.risk_reasons | {"quarantined_input" for _ in self.quarantined}))
        if any(build.closure is not None and not build.settled for build in self.builds.values()):
            return "stopped", ("incomplete_required_ci_resource_accounting",)
        if clock.utc_at > self.protocol.temporal_plan.test_end:
            return "stopped", ("observation_window_closed",)
        if not self.connected and self.last_status_at is not None:
            return "stopped", ("source_disconnected_or_stale",)
        if self.last_fresh is None or self.last_fresh.session_id != clock.session_id:
            return "recovering", ("fresh_source_required_for_clock_session",)
        if self.last_status_at is not None and clock.utc_at - self.last_status_at > timedelta(
            seconds=self.protocol.maximum_source_lag_seconds
        ):
            return "stopped", ("declared_source_clock_lag",)
        if clock.tick_ns - self.last_fresh.tick_ns > self.protocol.maximum_silence_ns:
            return "stopped", ("monotonic_source_silence",)
        return "healthy", ()

    def _clock(self, clock: HostClock) -> None:
        """Reject host clock reversal; never subtract monotonic ticks across sessions."""
        if self.last_clock is not None and clock.utc_at < self.last_clock.utc_at:
            raise ValueError("Host UTC receipt clock moved backwards")
        previous_tick = self.session_ticks.get(clock.session_id)
        if previous_tick is not None and clock.tick_ns < previous_tick:
            raise ValueError("Monotonic host tick moved backwards within a session")

    def reason(self, event: ShadowEvent, clock: HostClock) -> str | None:
        """Declare admission/quarantine before state mutation; malformed clocks raise."""
        self._clock(clock)
        if clock.utc_at < self.protocol.frozen_at:
            raise ValueError("Application receipt precedes protocol freeze")
        if isinstance(event, BuildOpened):
            return opening_reason(self.protocol, event, clock, self.builds, self.operational(clock)[0] == "healthy")
        if isinstance(event, SourceStatus):
            if event.available_at > clock.utc_at:
                raise ValueError("Source status is not yet available at receipt")
            return None
        build = self.builds.get(event.build_id)
        if build is None:
            return "feedback_before_required_build_declaration"
        if isinstance(event, BuildClosed):
            if event.available_at > clock.utc_at:
                raise ValueError("Build closure is not yet available at receipt")
            if event.available_at < build.opening.decided_at:
                return "closure_precedes_upstream_start"
            if any(row.metadata.event_at > event.available_at for row in build.history(self.protocol).revisions):
                return "closure_precedes_known_physical_completion"
            return "conflicting_build_closure" if build.closure is not None else None
        return feedback_reason(self.protocol, event, clock, build)

    def accept(self, admitted: AdmittedEvent) -> None:
        """Apply one verified journal entry using the same deterministic admission decision."""
        if admitted.protocol_fingerprint != self.protocol_fingerprint:
            raise ValueError("Protocol changed; old evidence cannot become untouched data for a new experiment")
        event, clock = admitted.event, admitted.clock
        if event.event_id in self.seen:
            raise ValueError("Journal repeats an event identity instead of retaining its first admission")
        expected = self.reason(event, clock)
        if admitted.reason != expected or admitted.disposition != ("quarantined" if expected else "accepted"):
            raise ValueError("Journal admission differs from verified source/risk state")
        if expected and len(self.quarantined) >= self.protocol.maximum_pending:
            raise ValueError("Bounded quarantine is full; source event is not acknowledged")
        if expected:
            self.quarantined.append(admitted)
        else:
            self._apply(admitted)
        self.seen[event.event_id] = admitted
        self.last_clock = clock
        self.session_ticks[clock.session_id] = clock.tick_ns

    def _apply(self, admitted: AdmittedEvent) -> None:
        """Dispatch typed accepted source operations; no externally visible mutable state escapes."""
        event, clock = admitted.event, admitted.clock
        if isinstance(event, SourceStatus):
            lag = clock.utc_at - event.available_at
            ordered = self.last_status_at is None or event.available_at >= self.last_status_at
            self.connected = (
                event.connected and ordered and lag <= timedelta(seconds=self.protocol.maximum_source_lag_seconds)
            )
            if ordered:
                self.last_status_at = event.available_at
            if self.connected:
                self.last_fresh = clock
        elif isinstance(event, BuildOpened):
            self._open(event, clock)
        elif isinstance(event, TestFeedback):
            self._feedback(admitted, event)
        else:
            build = self.builds[event.build_id]
            build.closure = admitted
            build.settle(self.protocol, clock)
            self.risk_reasons.update(build.risk_reasons)

    def _feedback(self, admitted: AdmittedEvent, event: TestFeedback) -> None:
        """Book first physical work once, then settle/recheck sticky resource and quality risk."""
        clock = admitted.clock
        build = self.builds[event.build_id]
        first = event.test_id not in build.feedback
        build.feedback.setdefault(event.test_id, []).append(admitted)
        if first and event.occupancy is not None:
            build.emit(
                OccupancyEvent(
                    **build.resource_metadata(f"work:{event.event_id}", clock),
                    execution_id=event.build_id,
                    amount=event.occupancy,
                    terminal=None,
                )
            )
        if build.resources.contained:
            self.risk_reasons.add("actual_required_ci_resource_overrun")
        build.settle(self.protocol, clock)
        if build.settled:
            build.selection_risks(self.protocol)
        self.risk_reasons.update(build.risk_reasons)

    def _open(self, event: BuildOpened, clock: HostClock) -> None:
        """Model required upstream work analytically without dispatching or cancelling CI."""
        build = _Build(opening=event, opened_clock=clock, resources=_Replay(self.protocol.actual_build_capacity))
        build.emit(
            IntentEvent(
                event_id=f"intent:{event.event_id}",
                sequence=1,
                estimate=event.actual_estimate,
                event_at=event.decided_at,
                recorded_at=clock.utc_at,
                execution_id=event.build_id,
            )
        )
        build.emit(
            ReserveEvent(
                event_id=f"reserve:{event.event_id}",
                sequence=2,
                amount=event.actual_estimate.amount,
                event_at=event.decided_at,
                recorded_at=clock.utc_at,
                execution_id=event.build_id,
            )
        )
        self.builds[event.build_id] = build
        for decision in event.decisions:
            if decision.decision_overhead is None:
                self.risk_reasons.add("unavailable_candidate_inference_cost")
            elif (
                decision.decision_overhead.value + decision.shared_setup_reserve.value > decision.request.capacity.value
            ):
                self.risk_reasons.add("candidate_preselection_charges_exceed_budget")


def event_json(event: ShadowEvent) -> str:
    """Compare source payloads canonically, independently of later physical retry clocks."""
    return canonical_json(event.model_dump(mode="json"))
