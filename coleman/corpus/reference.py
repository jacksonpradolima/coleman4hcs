"""Tiny exact CI reference calculations, independent of production metric/scheduling code.

These deliberately bounded references check algebra/accounting, not performance or
external validity. They operate on complete binary synthetic outcomes, never infer
unique faults, and never treat seed repetitions as independent observations.
"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.benchmarks.contracts import ExactFraction
from coleman.pilot.contracts import SECONDS, WALL_SECONDS, PilotBuild
from coleman.selection.contracts import Immutable, Quantity, decimal_value, ticks


def _check_literal_size(value) -> None:
    """Bound compact exponent and rational text forms before Fraction allocates large integers."""
    if isinstance(value, str) and (
        len(value) > 128 or re.fullmatch(r"[+-]?\d+(?:/\d+|\.\d+)?", value, flags=re.ASCII) is None
    ):
        raise ValueError("Reference literal must be bounded and contain no exponent")
    if isinstance(value, Decimal) and (
        not value.is_finite() or len(value.as_tuple().digits) > 64 or abs(int(value.as_tuple().exponent)) > 64
    ):
        raise ValueError("Reference decimal must be finite and bounded")


class RatioReference(Immutable):
    """Exact dimensionless reference and explicit undefined/defined state.

    Null denotes a mathematically inapplicable ratio, never a zero score. Meanings
    distinguish failing-test recall, normalized ranking area and cost-aware area.
    Fractions retain exact answers until an explicit production binary64 check.
    Ratio literals have <=128 characters and no exponent; decimal precision/scale
    are bounded to 64 and rational numerator/denominator to 256 bits before use.
    """

    unit: Literal["1"] = "1"
    meaning: Literal["failing-test-recall-v1", "verdict-napfd-v1", "verdict-apfdc-full-universe-cost-v1"]
    value: ExactFraction | None
    status: Literal["defined", "not_applicable"]

    @field_validator("value", mode="before")
    @classmethod
    def _bounded_literal(cls, value):
        """Bound rational parsing before exponent/denominator allocation; reject nonfinite/zero division."""
        if value is None:
            return value
        if isinstance(value, bool | float) or not isinstance(value, int | str | Decimal | Fraction):
            raise ValueError("Exact reference requires an integer, decimal or bounded rational literal")
        _check_literal_size(value)
        try:
            rational = Fraction(value)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("Reference requires a finite rational with nonzero denominator") from exc
        if rational.numerator.bit_length() > 256 or rational.denominator.bit_length() > 256:
            raise ValueError("Reference rational exceeds 256-bit fixture bounds")
        return value

    @model_validator(mode="after")
    def _availability(self) -> Self:
        """Keep a zero reference distinct from an inapplicable denominator."""
        if (self.value is not None) != (self.status == "defined"):
            raise ValueError("Defined reference requires a rational value")
        if self.value is not None and not 0 <= self.value <= 1:
            raise ValueError("CI reference ratio must be within [0, 1]")
        return self


class TinyCIReference(Immutable):
    """Complete-universe serial-prefix reference with explicit physical quantities.

    NAPFD becomes APFD for full execution. APFDc retains the native full-universe
    cost denominator, including known excluded failing tests. No parallel ranking
    measure is inferred from serial formulas. No-failure ratios are inapplicable.
    """

    selected_ids: tuple[str, ...]
    total_failing_tests: Annotated[int, Field(strict=True, ge=0, le=16)]
    detected_failing_tests: Annotated[int, Field(strict=True, ge=0, le=16)]
    recall: RatioReference
    napfd: RatioReference
    apfdc: RatioReference
    runner_work: Quantity
    full_suite_work: Quantity
    first_failure_elapsed: Quantity | None
    first_failure_status: Literal["detected", "right_censored"]


def _ratio(value: Fraction | None, meaning) -> RatioReference:
    """Declare exact ratio units and status without filling an undefined denominator."""
    return RatioReference(value=value, meaning=meaning, status="defined" if value is not None else "not_applicable")


def tiny_ci_reference(build: PilotBuild, ranking: tuple[str, ...], budget: Quantity) -> TinyCIReference:
    """Calculate independent exact NAPFD/APFDc and serial work for at most 16 tests.

    Stop at the first nonfitting test. Complete known binary outcomes, a full unique
    ranking and compatible occupancy-second quantities are required. Unknown or
    timeout outcomes and oversized suites raise ValueError, never become passes.
    All arithmetic uses integral ticks/Fractions, independent of Decimal precision.
    No production scheduler, metric, replay service, RNG or filesystem is invoked.
    """
    build = PilotBuild.model_validate(build)
    budget = Quantity.model_validate(budget)
    if len(build.tests) > 16:
        raise ValueError("Exact CI reference is limited to sixteen tests")
    if budget.semantics != SECONDS or sorted(ranking) != sorted(build.baseline_order):
        raise ValueError("Reference requires compatible budget and full unique ranking")
    tests = {test.test_id: test for test in build.tests}
    if any(test.outcome not in {"passed", "failed"} for test in build.tests):
        raise ValueError("Reference requires complete binary outcomes")
    durations = [ticks(tests[name].duration.value) for name in ranking]
    full_work = sum(durations)
    failures = sum(test.outcome == "failed" for test in build.tests)
    occupied, selected, failure_ranks, area, first_failure = _serial_prefix(
        ranking, tests, durations, full_work, ticks(budget.value)
    )
    recall = Fraction(len(failure_ranks), failures) if failures else None
    napfd = None
    if recall is not None:
        count = len(ranking)
        napfd = recall - Fraction(sum(failure_ranks), failures * count) + recall / (2 * count)
    apfdc = area / (full_work * failures) if full_work and failures else None
    return TinyCIReference(
        selected_ids=tuple(selected),
        total_failing_tests=failures,
        detected_failing_tests=len(failure_ranks),
        recall=_ratio(recall, "failing-test-recall-v1"),
        napfd=_ratio(napfd, "verdict-napfd-v1"),
        apfdc=_ratio(apfdc, "verdict-apfdc-full-universe-cost-v1"),
        runner_work=Quantity(value=decimal_value(occupied), semantics=SECONDS),
        full_suite_work=Quantity(value=decimal_value(full_work), semantics=SECONDS),
        first_failure_elapsed=Quantity(value=decimal_value(first_failure), semantics=WALL_SECONDS)
        if first_failure is not None
        else None,
        first_failure_status="detected" if first_failure is not None else "right_censored",
    )


def _serial_prefix(ranking, tests, durations, full_work, budget_ticks):
    """Integrate the fitting serial prefix using only exact integer ticks and rational areas."""
    occupied = 0
    selected: list[str] = []
    failure_ranks: list[int] = []
    area = Fraction(0)
    first_failure = None
    for index, name in enumerate(ranking):
        duration = durations[index]
        if occupied + duration > budget_ticks:
            break
        selected.append(name)
        if tests[name].outcome == "failed":
            failure_ranks.append(index + 1)
            area += full_work - occupied - Fraction(duration, 2)
            if first_failure is None:
                first_failure = occupied + duration
        occupied += duration
    return occupied, selected, failure_ranks, area, first_failure
