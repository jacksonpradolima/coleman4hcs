"""Synthetic CI runner-credit allocation; signed forecasts are not measured returns or rewards."""

import json
import sys
from decimal import Decimal
from pathlib import Path

from coleman.allocation import (
    AllocationLimits,
    AllocationProblem,
    CurrentAllocation,
    DiscreteAllocation,
    JointScenario,
    NamedAmount,
    ProbabilityEvidence,
    ResourceCapacity,
    RiskConstraints,
    ScenarioPayoff,
    SignedPayoff,
    publish_allocation_result,
    select_allocation,
)
from coleman.artifacts import ArtifactWriter
from coleman.results.domain import fingerprint
from coleman.selection import Quantity, UnitSemantics


def build_problem():
    """Declare complete discrete plans and correlated joint states before evaluating any choice."""
    credit = UnitSemantics(unit="credit", meaning="synthetic-ci-credit-accounting-v1")
    slot = UnitSemantics(unit="slot", meaning="synthetic-runner-slot-use-v1")
    gross = {"no_action": [0, 0], "stable": [2, 2], "variable": [10, -8]}
    return AllocationProblem(
        schema_version=1,
        data_fingerprint=fingerprint({"gross": gross, "probabilities": ["0.5", "0.5"]}),
        value_semantics=credit,
        initial_balance=Quantity(value=Decimal(10), semantics=credit),
        current=CurrentAllocation(
            holdings=(),
            resource_use=(NamedAmount(name="runner_slots", amount=Quantity(value=Decimal(0), semantics=slot)),),
            reserved_balance=Quantity(value=Decimal(0), semantics=credit),
        ),
        resources=(ResourceCapacity(name="runner_slots", capacity=Quantity(value=Decimal(2), semantics=slot)),),
        holding_definitions=(),
        scenarios=tuple(JointScenario(scenario_id=name, probability=Decimal("0.5")) for name in ("idle", "busy")),
        probability_evidence=ProbabilityEvidence(kind="assumed"),
        no_action_id="no_action",
        risk=RiskConstraints(downside_target=SignedPayoff(value=Decimal(0), semantics=credit)),
        actions=tuple(
            DiscreteAllocation(
                action_id=name,
                holdings=(),
                resource_use=(
                    NamedAmount(
                        name="runner_slots",
                        amount=Quantity(value=Decimal(0 if name == "no_action" else 1), semantics=slot),
                    ),
                ),
                capital_required=Quantity(value=Decimal(0), semantics=credit),
                transaction_cost=Quantity(value=Decimal(0), semantics=credit),
                gross_payoffs=tuple(
                    ScenarioPayoff(scenario_id=scenario, gross=SignedPayoff(value=Decimal(value), semantics=credit))
                    for scenario, value in zip(("idle", "busy"), values, strict=True)
                ),
            )
            for name, values in gross.items()
        ),
    )


def main(output):
    """Publish a self-contained exact result, retaining variable-plan losses and variance."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must be within the current working directory")
    result = select_allocation(build_problem(), limits=AllocationLimits(deadline_seconds=None))
    path = publish_allocation_result(result, ArtifactWriter(root))
    print(
        json.dumps(
            {
                "selected_action_id": result.choice.selected_action_id,
                "status": result.choice.status,
                "artifact": str(path),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
