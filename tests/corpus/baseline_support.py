"""Owned CI fixtures reuse the complete public example, preserving a frozen wider test grid."""

from decimal import Decimal

from coleman.corpus.repeated_contracts import BUDGET_RATIO
from coleman.selection import Quantity
from examples.comparable_baselines import baseline_plan as example_plan


def change(model, **values):
    """Revalidate the complete changed model instead of bypassing a public contract."""
    return type(model).model_validate({**model.model_dump(), **values})


def baseline_plan():
    """Keep the same owned source with both budgets and seeds used by the golden contract fixtures."""
    return change(
        example_plan(),
        seeds=(11, 42),
        budgets=tuple(Quantity(value=Decimal(value), semantics=BUDGET_RATIO) for value in ("0.5", "1")),
    )
