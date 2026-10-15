"""Private required-CI resource projection and point-in-time source views."""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from coleman.observations import ObservationHistory, ObservationRevision
from coleman.resources import OccupancyEvent, OverheadEvent, ResourceAmount, ResourceLedger
from coleman.resources.contracts import ResourceEvent
from coleman.resources.ledger import _Replay
from coleman.results.domain import fingerprint
from coleman.shadow.contracts import AdmittedEvent, BuildClosed, BuildOpened, HostClock, ShadowProtocol, TestFeedback
from coleman.shadow.decisions import resolve_selection


@dataclass
class _Build:
    """Private single-replay accumulator; all exported state is reconstructed typed data."""

    opening: BuildOpened
    opened_clock: HostClock
    resources: _Replay
    resource_events: list[ResourceEvent] = field(default_factory=list)
    feedback: dict[str, list[AdmittedEvent]] = field(default_factory=dict)
    closure: AdmittedEvent | None = None
    settled: bool = False
    risk_reasons: set[str] = field(default_factory=set)

    def emit(self, event: ResourceEvent) -> None:
        """Use the shared exact accounting kernel, never a second balance implementation."""
        self.resources.apply(event)
        self.resource_events.append(event)

    def resource_metadata(self, event_id: str, clock: HostClock) -> dict:
        """Describe an application booking event, separately from upstream completion time."""
        return {
            "event_id": event_id,
            "sequence": len(self.resource_events) + 1,
            "event_at": clock.utc_at,
            "recorded_at": clock.utc_at,
        }

    def ledger(self, protocol: ShadowProtocol) -> ResourceLedger:
        """Freeze the complete actual required-CI audit, including live/unknown reservations."""
        return ResourceLedger(
            schema_version=1,
            ledger_id=f"{protocol.logical_id}:{self.opening.build_id}",
            source_fingerprint=fingerprint(protocol.source.model_dump(mode="json")),
            configuration_fingerprint=protocol.fingerprint,
            capacity=protocol.actual_build_capacity,
            events=tuple(self.resource_events),
        )

    def history(self, protocol: ShadowProtocol) -> ObservationHistory:
        """Retain provider receipts while exposing first validated application availability.

        The journal stores the untouched provider revision. This distinct source
        view uses application admission as its local receipt; a later admission
        cannot retrospectively make information usable at an earlier cutoff.
        """
        revisions = []
        for events in self.feedback.values():
            for admitted in events:
                feedback = admitted.event
                if not isinstance(feedback, TestFeedback):
                    raise ValueError("Internal feedback journal has an incompatible event")
                revisions.append(
                    ObservationRevision.model_validate(
                        {**feedback.revision.model_dump(), "received_at": admitted.clock.utc_at}
                    )
                )
        return ObservationHistory(
            schema_version=1,
            source_id=f"{protocol.source.source_id}:shadow-admission-v1",
            source_version=protocol.source.source_version,
            revisions=tuple(revisions),
        )

    def settle(self, protocol: ShadowProtocol, clock: HostClock) -> None:
        """Release only known unused reservation; missing occupancy keeps accounting pending."""
        if self.settled or self.closure is None:
            return
        closure = self.closure.event
        if not isinstance(closure, BuildClosed):
            raise ValueError("Internal closure journal has an incompatible event")
        complete = set(self.feedback) == set(self.opening.required_ids)
        roots = [events[0].event for events in self.feedback.values()]
        known = all(isinstance(event, TestFeedback) and event.occupancy is not None for event in roots)
        if not complete or not known or closure.dispatch_occupancy is None or closure.shared_setup_occupancy is None:
            return
        charges: tuple[tuple[Literal["dispatch", "shared_setup"], ResourceAmount], ...] = (
            ("dispatch", closure.dispatch_occupancy),
            ("shared_setup", closure.shared_setup_occupancy),
        )
        for component, amount in charges:
            self.emit(
                OverheadEvent(
                    **self.resource_metadata(f"{component}:{closure.event_id}", clock),
                    component=component,
                    amount=amount,
                )
            )
        self.emit(
            OccupancyEvent(
                **self.resource_metadata(f"settle:{closure.event_id}", clock),
                execution_id=self.opening.build_id,
                amount=ResourceAmount(value=0, semantics=protocol.actual_build_capacity.semantics),
                terminal="unknown",
            )
        )
        self.settled = True
        if self.resources.contained:
            self.risk_reasons.add("actual_required_ci_resource_overrun")
        self.selection_risks(protocol)

    def selection_risks(self, protocol: ShadowProtocol) -> None:
        """Contain modeled selection overruns without changing actual required CI work."""
        history = self.history(protocol)
        last_clock = max(rows[-1].clock.utc_at for rows in self.feedback.values())
        labels = {row.entity_id: row.measurement for row in history.as_of(last_clock).observations}
        complete_labels = len(labels) == len(self.opening.required_ids) and all(
            measure.status == "observed" for measure in labels.values()
        )
        failures = {key for key, measure in labels.items() if measure.value == 1.0}
        for decision in self.opening.decisions:
            selection = resolve_selection(decision)
            work = self.selected_cost(decision, selection.selected_ids)
            if work is not None and work > decision.request.capacity.value:
                self.risk_reasons.add("counterfactual_selection_resource_overrun")
            if complete_labels and failures:
                recall = Fraction(len(failures.intersection(selection.selected_ids)), len(failures))
                if recall < Fraction(str(protocol.minimum_recall)):
                    self.risk_reasons.add("counterfactual_selection_recall_limit")

    def selected_cost(self, decision, selected_ids: tuple[str, ...]) -> int | None:
        """Include one actual shared setup and known inference in the selected-work proxy."""
        if self.closure is None or not isinstance(self.closure.event, BuildClosed):
            return None
        setup = self.closure.event.shared_setup_occupancy
        if setup is None or decision.decision_overhead is None:
            return None
        total = setup.value + decision.decision_overhead.value
        for key, rows in self.feedback.items():
            event = rows[0].event
            if key in selected_ids:
                if not isinstance(event, TestFeedback) or event.occupancy is None:
                    return None
                total += event.occupancy.value
        return total
