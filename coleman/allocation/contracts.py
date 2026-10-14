"""Finite joint-scenario allocation, accounting roles, probability evidence and risk contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from coleman.allocation.quantities import (
    InputAmount,
    NonnegativeMoment,
    Probability,
    SignedPayoff,
    penalty_semantics,
    variance_semantics,
)
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, UnitSemantics, decimal_text, ticks


def ordered[T](values: tuple[T, ...], key: Callable[[T], str]) -> tuple[T, ...]:
    """Canonicalize validated row identities, rejecting duplicates rather than dropping them."""
    names = [key(value) for value in values]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate allocation row identity")
    return tuple(sorted(values, key=key))


class NamedAmount(Immutable):
    """One complete holding/resource component; explicit zero is different from missing data."""

    name: Identifier
    amount: InputAmount


class ResourceCapacity(Immutable):
    """Inclusive total resource ceiling for a chosen allocation, including current commitments."""

    name: Identifier
    capacity: InputAmount


class HoldingDefinition(Immutable):
    """Unit/meaning of a holding component; finite alternatives declare every component."""

    name: Identifier
    semantics: UnitSemantics


class ProbabilityEvidence(Immutable):
    """Weight provenance, not certification that learned probabilities are calibrated.

    Estimated weights require producer evidence/method and per-scenario bounds.
    Independent-unit/count metadata may be explicitly unknown; seeds are never
    inferred as observations. This v1 selector conservatively abstains from new
    allocations under estimated weights, retaining conditional forecasts/bounds.
    """

    kind: Literal["assumed", "estimated"]
    evidence_fingerprint: Fingerprint | None = None
    method_id: Identifier | None = None
    independent_unit: Identifier | None = None
    independent_units: Annotated[int, Field(strict=True, ge=1)] | None = None

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Require identifiable estimation evidence without fabricating an observation count."""
        if self.kind == "estimated" and (self.evidence_fingerprint is None or self.method_id is None):
            raise ValueError("Estimated weights require evidence and method identities")
        if self.independent_units is not None and self.independent_unit is None:
            raise ValueError("An independent observation count requires its unit definition")
        return self


class JointScenario(Immutable):
    """One joint state shared by all actions; probabilities are dimensionless, not rewards.

    Point weights are nonnegative decimals summing exactly to one across states
    (zero tolerance; never silently renormalized). Estimated lower/upper bounds
    retain supplied uncertainty and are not assigned a confidence/coverage claim.
    """

    scenario_id: Identifier
    probability: Probability
    probability_unit: Literal["1"] = "1"
    probability_meaning: Literal["joint-scenario-probability-v1"] = "joint-scenario-probability-v1"
    lower: Probability | None = None
    upper: Probability | None = None

    @field_serializer("probability", "lower", "upper")
    def _probability(self, value):
        """Use canonical decimal strings while retaining unavailable bounds as null."""
        return None if value is None else decimal_text(value)

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Require paired ordered bounds containing the point weight when supplied."""
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Probability uncertainty requires both bounds")
        if self.lower is not None and self.upper is not None and not self.lower <= self.probability <= self.upper:
            raise ValueError("Probability point lies outside its declared bounds")
        return self


class ScenarioPayoff(Immutable):
    """Total gross increment of one whole allocation in a joint state, before its one fee."""

    scenario_id: Identifier
    gross: SignedPayoff


class CurrentAllocation(Immutable):
    """Current holdings, total resource use and liquidity reserved, separate from expenses."""

    holdings: Annotated[tuple[NamedAmount, ...], Field(max_length=64)]
    resource_use: Annotated[tuple[NamedAmount, ...], Field(max_length=64)]
    reserved_balance: InputAmount

    @field_validator("holdings", "resource_use")
    @classmethod
    def _vectors(cls, values: tuple[NamedAmount, ...]) -> tuple[NamedAmount, ...]:
        """Canonicalize complete components without imputing omitted zeros."""
        return ordered(values, lambda value: value.name)


class DiscreteAllocation(Immutable):
    """One complete finite target, with signed gross increments and one deterministic fee.

    Capital_required is liquidity reserved (including fee funding), not another
    expense. Net payoff is gross minus transaction_cost exactly once. Terminal
    balance is initial_balance plus net; reserved capital is not subtracted again.
    Holdings describe the target, not an extra payoff transformation. Joint gross
    increments already describe the entire target including correlated outcomes.
    """

    action_id: Identifier
    holdings: Annotated[tuple[NamedAmount, ...], Field(max_length=64)]
    resource_use: Annotated[tuple[NamedAmount, ...], Field(max_length=64)]
    capital_required: InputAmount
    transaction_cost: InputAmount
    gross_payoffs: Annotated[tuple[ScenarioPayoff, ...], Field(min_length=1, max_length=256)]

    @field_validator("holdings", "resource_use")
    @classmethod
    def _vectors(cls, values: tuple[NamedAmount, ...]) -> tuple[NamedAmount, ...]:
        """Canonicalize declared complete holding/resource vectors."""
        return ordered(values, lambda value: value.name)

    @field_validator("gross_payoffs")
    @classmethod
    def _payoffs(cls, values: tuple[ScenarioPayoff, ...]) -> tuple[ScenarioPayoff, ...]:
        """Preserve joint-state alignment by identity rather than incoming row position."""
        return ordered(values, lambda value: value.scenario_id)

    @model_validator(mode="after")
    def _funded_fee(self) -> Self:
        """Require compatible fee/reservation units and enough reserved liquidity for the fee."""
        if self.capital_required.semantics != self.transaction_cost.semantics:
            raise ValueError("Fee and capital reservation have incompatible units/meaning")
        if self.transaction_cost.value > self.capital_required.value:
            raise ValueError("Capital reservation does not fund its transaction cost")
        return self


class RiskConstraints(Immutable):
    """Precisely defined conditional risk ceilings; null means the constraint is not requested.

    Shortfall is E[max(target-net,0)], not CVaR/tail-conditioned expected loss.
    Variance is the population scenario variance of net payoff, in squared units.
    A terminal floor applies in every positive-probability scenario. Raw negative
    outcomes remain visible even when these constraints reject an allocation.
    """

    downside_target: SignedPayoff
    maximum_variance: NonnegativeMoment | None = None
    maximum_expected_shortfall: InputAmount | None = None
    minimum_terminal_balance: InputAmount | None = None


class UtilitySpec(Immutable):
    """Expected net or E[net - coefficient*max(target-net,0)^2], in payoff units.

    The quadratic coefficient has inverse payoff units and is nonnegative. Utility
    is evaluated on joint scenarios before expectation; losses are not clipped or
    normalized into rewards. Log/CVaR/continuous objectives are unsupported in v1.
    """

    kind: Literal["expected_net", "quadratic_target"] = "expected_net"
    coefficient: InputAmount | None = None

    @model_validator(mode="after")
    def _parameter(self) -> Self:
        """Reject missing or unused coefficients rather than guessing objective semantics."""
        if (self.kind == "quadratic_target") != (self.coefficient is not None):
            raise ValueError("Only quadratic target utility requires a coefficient")
        return self


class AllocationProblem(VersionedContract):
    """Complete finite action universe over shared joint scenarios and explicit current state.

    Offline modeled decisions only; no holdings are mutated and no live action is
    executed. Optimality is conditional on this finite declared universe and its
    assumed weights, never a continuous/global or calibrated forecasting claim.
    """

    data_fingerprint: Fingerprint
    value_semantics: UnitSemantics
    initial_balance: InputAmount
    current: CurrentAllocation
    resources: Annotated[tuple[ResourceCapacity, ...], Field(max_length=64)]
    holding_definitions: Annotated[tuple[HoldingDefinition, ...], Field(max_length=64)]
    scenarios: Annotated[tuple[JointScenario, ...], Field(min_length=1, max_length=256)]
    probability_evidence: ProbabilityEvidence
    actions: Annotated[tuple[DiscreteAllocation, ...], Field(min_length=1, max_length=64)]
    no_action_id: Identifier
    risk: RiskConstraints
    utility: UtilitySpec = Field(default_factory=UtilitySpec)

    @field_validator("resources", "holding_definitions")
    @classmethod
    def _definitions(cls, values):
        """Canonicalize declared component definitions before validating full vectors."""
        return ordered(values, lambda value: value.name)

    @field_validator("scenarios")
    @classmethod
    def _scenarios(cls, values: tuple[JointScenario, ...]) -> tuple[JointScenario, ...]:
        """Keep joint states stable under declaration order changes."""
        return ordered(values, lambda value: value.scenario_id)

    @field_validator("actions")
    @classmethod
    def _actions(cls, values: tuple[DiscreteAllocation, ...]) -> tuple[DiscreteAllocation, ...]:
        """Keep finite action identities stable, rejecting duplicate alternatives."""
        return ordered(values, lambda value: value.action_id)

    @model_validator(mode="after")
    def _probabilities(self) -> Self:
        """Require an exact simplex and retain every estimated bound/evidence declaration."""
        if sum(ticks(scenario.probability) for scenario in self.scenarios) != 10**12:
            raise ValueError("Scenario probabilities must sum exactly to one; tolerance is zero")
        bounded = [scenario.lower is not None for scenario in self.scenarios]
        if self.probability_evidence.kind == "estimated" and not all(bounded):
            raise ValueError("Estimated probabilities require bounds for every joint scenario")
        if self.probability_evidence.kind == "assumed" and any(bounded):
            raise ValueError("Assumed point probabilities must not carry ambiguous estimation bounds")
        return self

    @model_validator(mode="after")
    def _accounting(self) -> Self:
        """Check complete vectors, accounting units and all joint scenario references."""
        _check_value_semantics(self)
        resources = {item.name: item.capacity.semantics for item in self.resources}
        holdings = {item.name: item.semantics for item in self.holding_definitions}
        _check_vector(self.current.resource_use, resources)
        _check_vector(self.current.holdings, holdings)
        scenarios = {item.scenario_id for item in self.scenarios}
        for action in self.actions:
            _check_vector(action.resource_use, resources)
            _check_vector(action.holdings, holdings)
            if {payoff.scenario_id for payoff in action.gross_payoffs} != scenarios:
                raise ValueError("Every action must declare every joint scenario exactly once")
            if any(payoff.gross.semantics != self.value_semantics for payoff in action.gross_payoffs):
                raise ValueError("Gross payoff units/meaning differ from the accounting context")
        return self

    @model_validator(mode="after")
    def _no_action(self) -> Self:
        """Require an explicit fee-free unchanged allocation; its modeled payoffs need not be zero."""
        action = self.action(self.no_action_id)
        if (
            action.holdings != self.current.holdings
            or action.resource_use != self.current.resource_use
            or action.capital_required != self.current.reserved_balance
            or action.transaction_cost.value != 0
        ):
            raise ValueError("No-action must preserve current holdings/resources/reservation and charge no fee")
        return self

    def action(self, action_id: str) -> DiscreteAllocation:
        """Find a declared action; unknown identity raises ValueError without creating a fallback."""
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise ValueError("Unknown allocation action")


def _check_vector(values: tuple[NamedAmount, ...], definitions: dict[str, UnitSemantics]) -> None:
    """Reject omitted/extra components and incompatible units, including for explicit zero."""
    if {value.name for value in values} != set(definitions):
        raise ValueError("Allocation vectors must include every declared component, including zeros")
    if any(value.amount.semantics != definitions[value.name] for value in values):
        raise ValueError("Allocation component units/meaning differ from their definition")


def _check_value_semantics(problem: AllocationProblem) -> None:
    """Keep initial stock, reservation, gross increment and one fee in one declared accounting unit."""
    values = [problem.initial_balance, problem.current.reserved_balance, problem.risk.downside_target]
    values.extend(action.capital_required for action in problem.actions)
    values.extend(action.transaction_cost for action in problem.actions)
    values.extend(
        value
        for value in (problem.risk.maximum_expected_shortfall, problem.risk.minimum_terminal_balance)
        if value is not None
    )
    if any(value.semantics != problem.value_semantics for value in values):
        raise ValueError("Allocation accounting units/meaning are incompatible")
    if problem.risk.maximum_variance is not None and problem.risk.maximum_variance.semantics != variance_semantics(
        problem.value_semantics
    ):
        raise ValueError("Variance limit must carry squared payoff units and variance meaning")
    if problem.utility.coefficient is not None and problem.utility.coefficient.semantics != penalty_semantics(
        problem.value_semantics
    ):
        raise ValueError("Utility coefficient must carry inverse payoff units and penalty meaning")
