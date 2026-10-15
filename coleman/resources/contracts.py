"""Exact resource events; no scheduler, transport, clock or storage side effects."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import Fingerprint, Identifier
from coleman.selection.contracts import Immutable, UnitSemantics

RUNNER_MS = UnitSemantics(unit="ms", meaning="occupied-runner-time-v1")
ELAPSED_MS = UnitSemantics(unit="ms", meaning="serial-execution-elapsed-excluding-queue-setup-v1")
QUEUE_MS = UnitSemantics(unit="ms", meaning="observed-queue-elapsed-v1")
SETUP_MS = UnitSemantics(unit="ms", meaning="observed-setup-elapsed-v1")
Whole = Annotated[int, Field(strict=True, ge=0, le=10**12)]
Sequence = Annotated[int, Field(strict=True, ge=1, le=4096)]
Outcome = Literal["passed", "failed", "unknown", "timeout", "partial", "cancelled"]


class ResourceAmount(Immutable):
    """Whole nonnegative resource units, capped at 10^12 per external input.

    Zero is observed zero; null, booleans, fractions and nonfinite values fail.
    Unit symbols and versioned meanings must match exactly. There is no implicit
    seconds/milliseconds conversion, rounding or interpretation as elapsed time.
    """

    value: Whole
    semantics: UnitSemantics


class ResourceEstimate(Immutable):
    """Predecision estimate with explicit first availability and inclusive expiry.

    The source fingerprint identifies reviewed estimation evidence, never proof
    of predictive calibration. No realized outcome belongs in this contract.
    Producers must substantiate these availability claims. Zero is valid.
    """

    amount: ResourceAmount
    source_fingerprint: Fingerprint
    observed_at: Timestamp
    available_at: Timestamp
    valid_until: Timestamp

    @model_validator(mode="after")
    def _time(self) -> Self:
        """Reject reversed observation/availability/validity intervals."""
        if not self.observed_at <= self.available_at <= self.valid_until:
            raise ValueError("Estimate requires observed_at <= available_at <= valid_until")
        return self


class EventMetadata(Immutable):
    """Ordered source event, retaining its original receipt on identical replay.

    Sequence defines order, including equal timestamps; IDs never break ties.
    A late event may have an earlier event time, but receipt order is monotone.
    New out-of-order sequences are rejected by the ledger rather than reordered.
    """

    event_id: Identifier
    sequence: Sequence
    event_at: Timestamp
    recorded_at: Timestamp

    @model_validator(mode="after")
    def _receipt(self) -> Self:
        """Require an event to exist before its first receipt."""
        if self.recorded_at < self.event_at:
            raise ValueError("Receipt cannot precede source event")
        return self


class IntentEvent(EventMetadata):
    """Declare an execution and eligible estimate without reserving or consuming.

    Retrying execution requires a new ID and a terminal prior attempt. Estimate
    expiry is checked again at reservation. A stale intent cannot dispatch.
    """

    kind: Literal["intent"] = "intent"
    execution_id: Identifier
    estimate: ResourceEstimate
    retry_of: Identifier | None = None


class ReserveEvent(EventMetadata):
    """Reserve exactly the declared estimate if available and not contained.

    This is a predispatch feasibility check, never a runtime duration guarantee.
    Reservation is not consumption. V1 has no partial reservation or top-up.
    """

    kind: Literal["reserve"] = "reserve"
    execution_id: Identifier
    amount: ResourceAmount


class OccupancyEvent(EventMetadata):
    """Charge additional actual occupied resource, optionally settling the attempt.

    Amount is a delta, not a cumulative counter. Distinct event IDs must represent
    distinct occupancy. A terminal outcome releases unconsumed reservation only;
    partial/timeout/cancelled never imply a passed or failed label. A zero terminal
    cancellation releases reservation without inventing work. Overruns remain
    charged even beyond capacity and block later reservations. Actual executions
    already running can still report occupancy under containment.
    """

    kind: Literal["occupancy"] = "occupancy"
    execution_id: Identifier
    amount: ResourceAmount
    terminal: Outcome | None


class OverheadEvent(EventMetadata):
    """Charge separately observed occupied-resource overhead, including shared setup.

    This is actual resource occupancy, not elapsed queue/setup delay. A shared
    setup is charged once under its stable event ID; v1 does not allocate it to
    tests or infer unmeasured costs. Refunds and capacity adjustments are unsupported.
    """

    kind: Literal["overhead"] = "overhead"
    component: Literal["dispatch", "shared_setup", "retry_scheduling"]
    amount: ResourceAmount


ResourceEvent = Annotated[IntentEvent | ReserveEvent | OccupancyEvent | OverheadEvent, Field(discriminator="kind")]
