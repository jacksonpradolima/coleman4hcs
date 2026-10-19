"""Conservative paired bounds for independent units with predeclared bounded metrics.

Hoeffding's inequality requires independence between units and fixed outcome
bounds, not normality or identical distributions. Bounds are never estimated from
protected outcomes. Rational Taylor lower sums and dyadic upper radii avoid
platform exp/log rounding entering evidence. These are conservative tail bounds,
not exact distribution-specific p-values. No I/O, transport, RNG or policy is used.
"""

from __future__ import annotations

from fractions import Fraction
from math import inf, isfinite, nextafter
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import Field, model_validator

from coleman.results.domain import Finite, Identifier
from coleman.results.versioned import VersionedContract


class BoundedMeanProtocol(VersionedContract):
    """Freeze known metric bounds and an independent paired-mean comparison.

    Bounds use the metric's own unit/meaning and must be fixed before protected
    access. Independent units may each aggregate dependent builds/seeds, but IDs
    alone never establish independence. Every original candidate observation must
    lie in the inclusive bounds; out-of-range observations reject rather than
    clip. Signed values and observed zero remain valid. Normality is not assumed.
    This v1 method cannot analyze serial/clustered units directly. Missing, legacy
    and future versions reject; the legacy Student-t protocol stays unchanged.
    """

    protocol_id: Identifier
    holdout_id: Identifier
    metric_name: Identifier
    unit: Identifier
    meaning: Identifier
    direction: Literal["minimize", "maximize"]
    independent_unit: Identifier
    dependence: Literal["independent"]
    normal_differences_assumed: Annotated[bool, Field(strict=True)] = False
    practical_effect: Annotated[Finite, Field(ge=0)]
    outcome_lower: Finite
    outcome_upper: Finite
    alpha: Annotated[Finite, Field(gt=0, lt=1)] = 0.05
    minimum_units: Annotated[int, Field(strict=True, ge=2)] = 20
    method: Literal["paired_bounded_mean_hoeffding_holm_v1"] = "paired_bounded_mean_hoeffding_holm_v1"
    estimand: Literal["average-expected-independent-paired-unit-difference-v1"] = (
        "average-expected-independent-paired-unit-difference-v1"
    )
    arithmetic: Literal["rational-taylor32-dyadic32-outward-binary64-v1"] = (
        "rational-taylor32-dyadic32-outward-binary64-v1"
    )

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Require nontrivial predeclared bounds and the method's explicit distribution-free assumptions."""
        if self.outcome_lower >= self.outcome_upper:
            raise ValueError("Bounded mean requires outcome_lower < outcome_upper")
        if self.normal_differences_assumed:
            raise ValueError("Bounded mean does not declare a normal-differences assumption")
        return self

    def require_observed_value(self, value: float | None) -> None:
        """Reject an observed metric outside its frozen inclusive bounds, without clipping or side effects.

        The value must already satisfy the strict finite Measurement contract.
        Null retains its original unavailable/inapplicable status and is handled
        as incomplete evidence by the evaluator, never as an observed zero.
        """
        if value is not None and not self.outcome_lower <= value <= self.outcome_upper:
            raise ValueError("Observation exceeds the frozen metric outcome bounds")


class MeanStatistics(NamedTuple):
    """Finite summary in metric units; absent uncertainty never becomes zero."""

    mean: float | None = None
    improvement: float | None = None
    low: float | None = None
    high: float | None = None
    p_value: float | None = None
    reason: str = "too_few_independent_units"


def bounded_tail_upper(exponent: Fraction) -> Fraction:
    """Bound min(1, 2 exp(-x)) upward by a positive degree-32 rational Taylor sum.

    For x>=0, exp(x)>=sum(x**k/k!,k=0..32). Its reciprocal yields a conservative
    upper bound even in extreme tails. There is no floating exp approximation or
    claim of an exact distribution-specific tail probability. Negative exponents
    raise ValueError; no unbounded precision/iteration parameter is exposed.
    """
    if not isinstance(exponent, Fraction) or exponent < 0:
        raise ValueError("Bounded tail exponent must be nonnegative")
    term = total = Fraction(1)
    for index in range(1, 33):
        term = term * exponent / index
        total += term
    return min(Fraction(1), 2 / total)


def _positive_count(value: object) -> bool:
    """Accept a positive integer count while rejecting booleans and nonintegral inputs, without coercion."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_alpha(value: object) -> bool:
    """Require an exact significance level in (0,1), comparing normalized rational integers without coercion."""
    return isinstance(value, Fraction) and 0 < value.numerator < value.denominator


def bounded_radius(units: int, family_size: int, alpha: Fraction) -> Fraction:
    """Return a conservative normalized two-sided simultaneous radius on a fixed dyadic grid.

    Inputs are positive integral unit/family counts and 0<alpha<1. A fixed 32-step
    bisection keeps the upper endpoint satisfying the full-family tail bound.
    Radius two covers the complete parameter range when the bound is too weak.
    Unit interval bounds are transformed by the caller; no data-dependent bound
    selection, stopping or numerical optimization is performed.
    """
    if not _valid_alpha(alpha) or not _positive_count(units) or not _positive_count(family_size):
        raise ValueError("Bounded radius requires positive counts and alpha in (0,1)")
    target = alpha / family_size
    low, high = Fraction(0), Fraction(2)
    if bounded_tail_upper(units * high * high / 2) > target:
        return high
    for _ in range(32):
        middle = (low + high) / 2
        if bounded_tail_upper(units * middle * middle / 2) <= target:
            high = middle
        else:
            low = middle
    return high


def outward_binary64(value: Fraction, *, upper: bool) -> float:
    """Round a rational endpoint outward to finite binary64, with subnormal/zero behavior explicit.

    Rational zero becomes canonical unsigned zero. Upper tail probabilities round
    upward; confidence lower/upper endpoints round toward negative/positive
    infinity respectively. An unrepresentable finite endpoint raises OverflowError
    rather than serializing infinity or a falsely narrower interval.
    """
    if not isinstance(value, Fraction) or type(upper) is not bool:
        raise ValueError("Outward conversion requires a rational value and a boolean direction")
    result = float(value)
    represented = Fraction(result)
    if (upper and represented < value) or (not upper and represented > value):
        result = nextafter(result, inf if upper else -inf)
    if not isfinite(result):
        raise OverflowError("Bounded uncertainty exceeds finite binary64 range")
    return result


def _outward_holm(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Apply Holm to already validated p-values with rational multiplication and outward output rounding.

    Ordinary binary64 multiplication can round an adjusted upper bound downward.
    The public Holm helper validates inputs and selects this explicit profile;
    its legacy default arithmetic remains unchanged. Empty input returns empty.
    """
    adjusted = [0.0] * len(p_values)
    running = Fraction(0)
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda index: p_values[index])):
        running = max(running, min(Fraction(1), (len(p_values) - rank) * Fraction(p_values[index])))
        adjusted[index] = outward_binary64(running, upper=True)
    return tuple(adjusted)


def summarize_bounded(
    differences: tuple[Fraction, ...],
    protocol: BoundedMeanProtocol,
    family_size: int,
    total_units: int,
) -> MeanStatistics:
    """Summarize complete independent pairs; preserve signed effects and honest unavailable uncertainty.

    Inputs are exact differences of the original binary64 candidate measurements.
    All observations must already pass the frozen inclusive source bounds. A
    missing pair invalidates the whole comparison, not just that unit. The mean
    is rounded once to binary64. Intervals control simultaneous family coverage
    through Bonferroni; the existing evaluator applies Holm to these conservative
    p-values. Extreme unrepresentable uncertainty remains null. This never
    reserves protected access, grants source admission or activates a policy.
    """
    if len(differences) != total_units:
        return MeanStatistics(reason="incomplete_outcomes")
    if not differences:
        return MeanStatistics()
    span = Fraction(protocol.outcome_upper) - Fraction(protocol.outcome_lower)
    average = _complete_mean(differences, span)
    signed = average if protocol.direction == "maximize" else -average
    try:
        mean, improvement = float(average), float(signed)
    except OverflowError as error:
        raise ValueError("Paired mean difference exceeds finite numeric range") from error
    if len(differences) < protocol.minimum_units:
        return MeanStatistics(mean=mean, improvement=improvement)
    radius = bounded_radius(len(differences), family_size, Fraction(protocol.alpha))
    lower = max(-span, signed - span * radius)
    upper = min(span, signed + span * radius)
    p_value = bounded_tail_upper(len(differences) * (average / span) ** 2 / 2)
    try:
        low = outward_binary64(lower, upper=False)
        high = outward_binary64(upper, upper=True)
        probability = outward_binary64(p_value, upper=True)
    except OverflowError:
        return MeanStatistics(mean=mean, improvement=improvement, reason="numerically_unstable_uncertainty")
    return MeanStatistics(
        mean=mean, improvement=improvement, low=low, high=high, p_value=probability, reason="estimated"
    )


def _complete_mean(differences: tuple[Fraction, ...], span: Fraction) -> Fraction:
    """Check exact paired differences and return a complete rational mean in the original metric unit."""
    if any(not isinstance(difference, Fraction) or abs(difference) > span for difference in differences):
        raise ValueError("Paired difference exceeds frozen outcome bounds")
    return sum(differences, Fraction(0)) / len(differences)
