"""Signed scenario quantities with shared exact decimal/serialization primitives."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import AfterValidator, Field

from coleman.selection.contracts import INPUT_MAXIMUM, Quantity, UnitSemantics


def _input_cap(value: Quantity) -> Quantity:
    """Bound a nonnegative declared input to 10^12 units, without clipping."""
    if value.value > INPUT_MAXIMUM:
        raise ValueError("Allocation input exceeds 10^12 units")
    return value


InputAmount = Annotated[Quantity, AfterValidator(_input_cap)]
Probability = Annotated[Decimal, Field(ge=0, le=1, max_digits=13, decimal_places=12)]


class SignedPayoff(Quantity):
    """Finite signed modeled payoff in declared units, with twelve decimal places.

    Values lie in [-10^12,10^12]. Losses stay negative; no probability/reward
    normalization or implicit transformation is applied. Signed zero uses the
    shared canonical zero before hashing/projection.
    """

    value: Annotated[Decimal, Field(ge=Decimal("-1e12"), le=Decimal("1e12"), max_digits=25, decimal_places=12)]


class ExactPayoff(Quantity):
    """Exact signed moment/utility output, allowing 48 fractional and 110 total digits."""

    value: Annotated[Decimal, Field(ge=Decimal("-1e48"), le=Decimal("1e48"), max_digits=110, decimal_places=48)]


class NonnegativeMoment(Quantity):
    """Exact nonnegative variance/shortfall or a declared risk ceiling in its own unit."""

    value: Annotated[Decimal, Field(ge=0, le=Decimal("1e48"), max_digits=110, decimal_places=48)]


def variance_semantics(value: UnitSemantics) -> UnitSemantics:
    """Declare variance in squared payoff units, separate from payoff/credit magnitudes."""
    return UnitSemantics(unit=f"({value.unit})^2", meaning=value.meaning + ":net-payoff-variance-v1")


def penalty_semantics(value: UnitSemantics) -> UnitSemantics:
    """Declare the inverse-unit coefficient keeping quadratic downside utility in payoff units."""
    return UnitSemantics(unit=f"1/({value.unit})", meaning=value.meaning + ":quadratic-target-penalty-v1")
