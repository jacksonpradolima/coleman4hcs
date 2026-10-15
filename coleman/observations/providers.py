"""Typed authorized-provider boundary; vendor SDKs and source rules stay outside the base package."""

from dataclasses import dataclass
from typing import Protocol

from coleman.observations.contracts import ObservationHistory


class ObservationProvider(Protocol):
    """Provider-owned normalization of identities, units, clocks and immutable first receipts.

    Implementations may perform explicitly authorized I/O; the core never imports
    a provider by name or invokes credentials/transports implicitly. Adapters must
    bound source payloads before decoding and attest source completeness/clock
    assumptions. Delivery retries retain the first receipt rather than stamping a
    later time into the same logical revision. No SDK dependency belongs here.
    """

    def load_history(self) -> ObservationHistory:
        """Return a complete bounded typed history or propagate actionable provider errors."""
        ...


@dataclass(frozen=True)
class OfflineObservationProvider:
    """In-memory provider for owned offline CI fixtures, with no clock, network or mutable cache."""

    history: ObservationHistory

    def load_history(self) -> ObservationHistory:
        """Revalidate supplied history, including unchecked nested instances; perform no I/O."""
        return ObservationHistory.model_validate(self.history)
