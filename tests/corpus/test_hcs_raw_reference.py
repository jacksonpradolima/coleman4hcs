"""Adversarial checks of the independent reference itself, using only authored tiny quantities."""

import json
from decimal import Decimal

import pytest

from tests.corpus.hcs_raw_reference import _check_row


def example():
    cases = {"zero": (Decimal(0), 0), "first": (Decimal(1), 1), "later": (Decimal(2), 1)}
    row = dict(
        budget_mode="ratio",
        experiment=1,
        step=1,
        reward_function="Reward Based on Failures",
        budget_value=0.5,
        prioritization_order_top_k=json.dumps(["zero", "first", "later"]),
        total_build_duration=3.0,
        tests_ran=2,
        tests_not_ran=1,
        detected=1,
        missed=1,
    )
    return cases, row


def test_owned_zero_duration_and_budget_boundary_agree_exactly():
    cases, row = example()
    assert _check_row(cases, row) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("tests_ran", 3),
        ("tests_ran", -1),
        ("tests_not_ran", 2),
        ("detected", 0),
        ("missed", 2),
        ("total_build_duration", 4.0),
        ("total_build_duration", float("nan")),
        ("total_build_duration", float("inf")),
        ("step", 0),
        ("experiment", 2),
        ("budget_mode", "fixed_time"),
        ("reward_function", "RNFail"),
        ("prioritization_order_top_k", '["zero","zero","later"]'),
        ("prioritization_order_top_k", '["zero","first"]'),
    ],
)
def test_fake_accounting_or_order_mutations_fail_independent_guards(field, value):
    cases, row = example()
    row[field] = value
    with pytest.raises(AssertionError):
        _check_row(cases, row)


def test_original_decimal_boundary_difference_is_retained_instead_of_tolerated():
    cases = {"third": (Decimal("0.3"), 1), "first": (Decimal("0.1"), 0), "second": (Decimal("0.2"), 0)}
    _, row = example()
    row.update(
        prioritization_order_top_k='["third","first","second"]',
        total_build_duration=0.6000000163912773,
        tests_ran=0,
        tests_not_ran=3,
        detected=0,
        missed=1,
    )
    # Original decimal budget fits 0.3; projected binary32 half-budget does not.
    assert _check_row(cases, row) is True
