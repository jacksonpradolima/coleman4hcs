"""Adversarial checks of the independent reference itself, using only authored tiny quantities."""

import json
from collections import Counter
from decimal import Decimal

import polars as pl
import pytest

from tests.corpus.hcs_raw_reference import _check_cells, _check_row


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


@pytest.mark.parametrize("mutation", [None, "missing_cell", "metadata_count", "invented_nonpass_zero"])
def test_independent_sparse_reference_requires_known_but_unobserved_cells(mutation):
    rows = [
        dict(
            pipeline_id=1,
            variant_name="seen",
            variant_alias="v_seen",
            metadata_jobs=1,
            parsed_rows=1,
            replay_rows=1,
            reported_nonpass_rows=0,
            cell_status="eligible",
        ),
        dict(
            pipeline_id=1,
            variant_name="metadata",
            variant_alias="v_metadata",
            metadata_jobs=1,
            parsed_rows=0,
            replay_rows=0,
            reported_nonpass_rows=None,
            cell_status="absent_parsed_source",
        ),
    ]
    if mutation == "missing_cell":
        rows.pop()
    elif mutation == "metadata_count":
        rows[1]["metadata_jobs"] = 0
    elif mutation == "invented_nonpass_zero":
        rows[1]["reported_nonpass_rows"] = 0
    table = pl.DataFrame(rows)
    inputs = (
        (1,),
        {"v_seen": "seen", "v_metadata": "metadata"},
        {(1, "seen"): {"A": (Decimal(1), 0)}},
        Counter({(1, "seen"): 1, (1, "metadata"): 1}),
        Counter({(1, "seen"): 1}),
    )
    if mutation is None:
        _check_cells(table, *inputs)
    else:
        with pytest.raises(AssertionError):
            _check_cells(table, *inputs)


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
