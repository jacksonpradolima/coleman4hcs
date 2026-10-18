"""Owned small CI source; its repeated builds are one constructed mechanism family."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from coleman.corpus.baseline_contracts import (
    ESTIMATE_MEANING,
    FAILURE_MEANING,
    BaselineBuild,
    BaselineFeature,
    BaselinePlan,
    BaselineSource,
    BaselineTest,
)
from coleman.corpus.repeated_contracts import BUDGET_RATIO
from coleman.evaluation.protected import TemporalObservation
from coleman.pilot.contracts import SECONDS
from coleman.results.domain import Measurement, fingerprint
from coleman.selection import Quantity


def change(model, **values):
    """Revalidate the complete changed model instead of bypassing a public contract."""
    return type(model).model_validate({**model.model_dump(), **values})


def measure(name, value, meaning, *, unit="1", direction="none", status="observed", minimum=0.0):
    """Declare a source measurement; null states never receive a numeric placeholder."""
    return Measurement(
        name=name, value=value, unit=unit, meaning=meaning, direction=direction, status=status, minimum=minimum
    )


def baseline_plan():
    """Three causal build clocks, nonlexical base order and independent prior cost/context declarations."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    prior = fingerprint(["owned", "prior-known-inputs", 1])
    builds = []
    for step in (1, 2, 3):
        rows = []
        for name, seconds, cost in (("a", 1, 4), ("b", 2, 1), ("c", 4, 2)):
            feature = BaselineFeature(
                measurement=measure("cost_estimate", float(cost), ESTIMATE_MEANING, unit="s", direction="minimize"),
                source_fingerprint=prior,
                available_at=start,
                available_before_step=0,
            )
            context = BaselineFeature(
                measurement=measure("prior_change_hint", float(name == "b"), "owned-prior-change-hint-v1"),
                source_fingerprint=prior,
                available_at=start,
                available_before_step=0,
            )
            rows.append(
                BaselineTest(
                    test_id=name,
                    duration=measure(
                        "observed_occupancy", float(seconds), SECONDS.meaning, unit="s", direction="minimize"
                    ),
                    failure=measure("observed_binary_failure", float(name == "b" and step == 2), FAILURE_MEANING),
                    prior_cost=feature,
                    context=(context,),
                )
            )
        decision = start + timedelta(days=step)
        builds.append(
            BaselineBuild(
                step=step,
                member_id=f"owned-build-{step}",
                source_metadata=TemporalObservation(
                    observation_id=f"source-{step}",
                    group_id=f"source-{step}",
                    event_at=decision,
                    available_at=decision + timedelta(hours=1),
                ),
                decided_at=decision,
                tests=tuple(rows),
                original_order=("c", "a", "b"),
            )
        )
    return BaselinePlan(
        schema_version=1,
        fold_id="owned-development-fold-v1",
        source=BaselineSource(
            schema_version=1,
            source_id="owned-baseline-mechanism-v1",
            source_uri="urn:coleman:owned:baseline-mechanism-v1",
            source_fingerprint=fingerprint(["owned", "baseline-source", 1]),
            license_id="MIT (COLEMAN repository LICENSE)",
            evidence_basis="constructed_mechanism",
            clock_basis="source_point_in_time",
            admission_fingerprint=None,
            builds=tuple(builds),
        ),
        seeds=(11, 42),
        budgets=tuple(Quantity(value=Decimal(value), semantics=BUDGET_RATIO) for value in ("0.5", "1")),
    )
