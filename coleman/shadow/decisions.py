"""Conservative predecision resource allocation using only frozen/known charges."""

from coleman.resources import ResourceAmount, ResourceRequest, ResourceSelection, select_resources
from coleman.shadow.contracts import CandidateDecision


def resolve_selection(decision: CandidateDecision) -> ResourceSelection:
    """Reserve shared setup and known inference before selecting a test-work prefix.

    The request retains the nominal frozen total budget for provenance. Effective
    test-work capacity deducts the declared setup reservation and known inference
    cost exactly. Unknown inference or already-exhausted total capacity yields
    no-action; zero remaining capacity still permits genuine zero-cost tests.
    Realized test/setup work never enters this predecision calculation. An actual
    overrun is later retained in accounting, never clipped into feasibility.
    """
    decision = CandidateDecision.model_validate(decision)
    request = decision.request
    overhead = decision.decision_overhead
    charge = decision.shared_setup_reserve.value + (overhead.value if overhead is not None else 0)
    payload = request.model_dump()
    payload["capacity"] = ResourceAmount(
        value=max(0, request.capacity.value - charge), semantics=request.capacity.semantics
    )
    if overhead is None or charge > request.capacity.value:
        payload.update(priority_status="unavailable", fallback="no_action")
    return select_resources(ResourceRequest.model_validate(payload))
