"""Finite joint-scenario resource allocation with signed payoffs and conservative evidence handling."""

from coleman.allocation.artifacts import allocation_record, publish_allocation_result
from coleman.allocation.contracts import (
    AllocationProblem,
    CurrentAllocation,
    DiscreteAllocation,
    HoldingDefinition,
    JointScenario,
    NamedAmount,
    ProbabilityEvidence,
    ResourceCapacity,
    RiskConstraints,
    ScenarioPayoff,
    UtilitySpec,
)
from coleman.allocation.forecasts import ActionEvaluation, ActionForecast, evaluate_allocation_action
from coleman.allocation.quantities import (
    ExactPayoff,
    NonnegativeMoment,
    SignedPayoff,
    penalty_semantics,
    variance_semantics,
)
from coleman.allocation.resources import AllocationLimits
from coleman.allocation.results import AllocationCheckpoint, AllocationResult
from coleman.allocation.solver import select_allocation

__all__ = [
    "ActionEvaluation",
    "ActionForecast",
    "AllocationCheckpoint",
    "AllocationLimits",
    "AllocationProblem",
    "AllocationResult",
    "CurrentAllocation",
    "DiscreteAllocation",
    "ExactPayoff",
    "HoldingDefinition",
    "JointScenario",
    "NamedAmount",
    "NonnegativeMoment",
    "ProbabilityEvidence",
    "ResourceCapacity",
    "RiskConstraints",
    "ScenarioPayoff",
    "SignedPayoff",
    "UtilitySpec",
    "allocation_record",
    "evaluate_allocation_action",
    "penalty_semantics",
    "publish_allocation_result",
    "select_allocation",
    "variance_semantics",
]
