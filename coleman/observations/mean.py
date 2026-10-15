"""Frozen simple means of eligible historical quantities; no calibrated probability or inference."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self, cast

from pydantic import Field, TypeAdapter, model_validator

from coleman.evaluation.protected.temporal import SplitDecision, TemporalPlan, Timestamp
from coleman.observations.contracts import (
    EVENT_AGE_SECONDS,
    MAXIMUM_REVISIONS,
    DecisionSignal,
    ObservationHistory,
    ObservationRevision,
    ObservationSnapshot,
    SignalProvenance,
    SignalRequest,
)
from coleman.results.domain import Identifier, Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

_ROUNDING_SCALE = 10**9


class TrainingSet(VersionedContract):
    """Eligible frozen training observations and auditable temporal exclusions.

    The view's cutoff cannot exceed the predeclared training boundary. Source
    availability includes first receipt. Labels must be available, finite,
    nonnegative and <=10^12 in exactly the declared units/meaning; unknown labels
    fail instead of becoming zero or silently disappearing from a fit.
    """

    plan: TemporalPlan
    snapshot: ObservationSnapshot
    measurement_name: Identifier
    input_semantics: UnitSemantics
    exclusions: Annotated[tuple[SplitDecision, ...], Field(max_length=MAXIMUM_REVISIONS)] = ()

    @model_validator(mode="after")
    def _training(self) -> Self:
        """Reuse protected temporal checks and reject future, incompatible or unavailable labels."""
        if self.snapshot.information_cutoff > self.plan.train_end:
            raise ValueError("Training cutoff exceeds the frozen training boundary")
        decisions = self.effective_plan.split(tuple(row.effective_metadata for row in self.snapshot.observations))
        if any(decision.partition != "train" or not decision.eligible for decision in decisions):
            raise ValueError("Ineligible training observation")
        for row in self.snapshot.observations:
            _training_quantity(row, self.measurement_name, self.input_semantics)
        _exclusions(self)
        return self

    @property
    def effective_plan(self) -> TemporalPlan:
        """Retain declared folds while purging targets at this fit's actual information cutoff."""
        return _training_plan(self.plan, self.snapshot.information_cutoff)


def _training_plan(plan: TemporalPlan, cutoff: Timestamp) -> TemporalPlan:
    """Reuse protected target/group eligibility with an earlier effective training boundary."""
    return TemporalPlan.model_validate({**plan.model_dump(), "train_end": cutoff})


def _training_quantity(row: ObservationRevision, name: str, semantics: UnitSemantics) -> None:
    """Bound physical-quantity fitting inputs without coercing units or missingness."""
    measure = row.measurement
    if measure.name != name or (measure.unit, measure.meaning) != (semantics.unit, semantics.meaning):
        raise ValueError("Incompatible training measurement units or meaning")
    if measure.status != "observed" or measure.value is None:
        raise ValueError("Unavailable training labels cannot be fitted")
    if not 0 <= measure.value <= 10**12:
        raise ValueError("Quantity-mean training requires nonnegative values <=10^12")


def _exclusions(training: TrainingSet) -> None:
    """Validate explicit temporal exclusion reasons without exposing excluded label values."""
    selected = {row.metadata.observation_id for row in training.snapshot.observations}
    excluded = [item.observation.observation_id for item in training.exclusions]
    if len(excluded) != len(set(excluded)) or selected & set(excluded):
        raise ValueError("Duplicate or contradictory training exclusion")
    plan = training.effective_plan
    for item in training.exclusions:
        actual = plan.decide(item.observation)
        if actual != item or (item.eligible and item.partition == "train"):
            raise ValueError("Invalid training exclusion reason")
        if item.observation.available_at > training.snapshot.information_cutoff:
            raise ValueError("Training exclusion contains metadata unavailable at its cutoff")
    object.__setattr__(
        training, "exclusions", tuple(sorted(training.exclusions, key=lambda item: item.observation.observation_id))
    )


def prepare_training(
    history: ObservationHistory,
    plan: TemporalPlan,
    *,
    information_cutoff: Timestamp,
    measurement_name: str,
    input_semantics: UnitSemantics,
) -> TrainingSet:
    """Select latest eligible training revisions, frozen at min(request cutoff, train_end).

    New validation/test outcomes and post-training corrections never refit this
    view. Temporal exclusions retain reasons/countable metadata, not labels. An
    unavailable label inside otherwise eligible training fails validation. A source
    with no matching observations yields an empty set, supporting explicit fallback.
    """
    plan = TemporalPlan.model_validate(plan)
    cutoff = TypeAdapter(Timestamp).validate_python(information_cutoff)
    eligibility_plan = _training_plan(plan, min(cutoff, plan.train_end))
    snapshot = history.as_of(eligibility_plan.train_end)
    rows = tuple(row for row in snapshot.observations if row.measurement.name == measurement_name)
    decisions = eligibility_plan.split(tuple(row.effective_metadata for row in rows))
    eligible_ids = {
        item.observation.observation_id for item in decisions if item.partition == "train" and item.eligible
    }
    selected = tuple(row for row in rows if row.metadata.observation_id in eligible_ids)
    return TrainingSet(
        schema_version=1,
        plan=plan,
        snapshot=ObservationSnapshot.model_validate({**snapshot.model_dump(), "observations": selected}),
        measurement_name=measurement_name,
        input_semantics=input_semantics,
        exclusions=tuple(item for item in decisions if item.observation.observation_id not in eligible_ids),
    )


class MeanEstimate(Immutable):
    """Historical sample mean with exact rounded decimal units and event-time freshness.

    Sample count is a count of logical observations, not independent evidence.
    Arithmetic begins from exact binary64 input rationals; upward rounding to
    10^-9 of the declared unit is explicit. Zero remains a genuine mean.
    """

    entity_id: Identifier
    quantity: Quantity
    observations: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_REVISIONS)]
    latest_event_at: Timestamp


def _mean_estimates(training: TrainingSet) -> tuple[MeanEstimate, ...]:
    """Independently rederive bounded exact arithmetic and declared decimal ceiling."""
    groups: dict[str, list[ObservationRevision]] = defaultdict(list)
    for row in training.snapshot.observations:
        groups[row.entity_id].append(row)
    semantics = UnitSemantics(
        unit=training.input_semantics.unit,
        meaning=training.input_semantics.meaning + ":historical-mean-ceil-1e-9-v1",
    )
    estimates = []
    for entity in sorted(groups):
        rows = groups[entity]
        total = sum((Fraction(float(cast(float, row.measurement.value))) for row in rows), Fraction(0))
        scaled = total * _ROUNDING_SCALE / len(rows)
        rounded = -(-scaled.numerator // scaled.denominator)
        decimal = Decimal(f"{rounded // _ROUNDING_SCALE}.{rounded % _ROUNDING_SCALE:09d}")
        estimates.append(
            MeanEstimate(
                entity_id=entity,
                quantity=Quantity(value=decimal, semantics=semantics),
                observations=len(rows),
                latest_event_at=max(row.metadata.event_at for row in rows),
            )
        )
    return tuple(estimates)


class QuantityMeanModel(VersionedContract):
    """A frozen, revalidated simple historical mean; no learned normalization or efficacy claim.

    Serialized models embed their eligible training evidence and rederive all
    means/counts/freshness on read. Post-cutoff source corrections cannot alter an
    existing model. This estimator supports nonnegative quantities; signed priority
    scores remain supported by DecisionSignal, without pretending to fit this model.
    """

    training: TrainingSet
    frozen_at: Timestamp
    estimates: Annotated[tuple[MeanEstimate, ...], Field(max_length=MAXIMUM_REVISIONS)]

    @model_validator(mode="after")
    def _model(self) -> Self:
        """Reject future-fitted or altered means, observation counts and freshness declarations."""
        if self.frozen_at < self.training.snapshot.information_cutoff:
            raise ValueError("Model freeze precedes its training cutoff")
        expected = _mean_estimates(self.training)
        if tuple(sorted(self.estimates, key=lambda item: item.entity_id)) != expected:
            raise ValueError("Model estimates disagree with frozen eligible training evidence")
        object.__setattr__(self, "estimates", expected)
        return self

    def signal(self, request: SignalRequest, *, maximum_event_age: Quantity) -> DecisionSignal:
        """Emit a future quantity estimate or explicit missing/stale signal, with no I/O.

        Age is decision time minus latest source event, not receipt/correction time.
        Maximum age must use EVENT_AGE_SECONDS semantics; equality remains fresh.
        The exact rounded quantity stays in the model; the named signal explicitly
        projects it to binary64. Missing/stale values stay null and require fallback.
        """
        model = QuantityMeanModel.model_validate(self)
        request = SignalRequest.model_validate(request)
        age_limit = Quantity.model_validate(maximum_event_age)
        if age_limit.semantics != EVENT_AGE_SECONDS:
            raise ValueError("Incompatible source-event age units or meaning")
        estimate = next((item for item in model.estimates if item.entity_id == request.entity_id), None)
        measurement, availability = _signal_value(model, request, estimate, age_limit)
        return DecisionSignal(
            schema_version=1,
            request=request,
            interpretation="quantity_estimate",
            measurement=measurement,
            availability=availability,
            maximum_event_age=age_limit,
            provenance=SignalProvenance(
                source_id=model.training.snapshot.source_id,
                source_version=model.training.snapshot.source_version,
                source_fingerprint=model.training.snapshot.fingerprint,
                model_fingerprint=model.fingerprint,
                producer="coleman.observations.quantity-mean",
                producer_version="1",
                model_version="1",
                training_cutoff=model.training.snapshot.information_cutoff,
                frozen_at=model.frozen_at,
            ),
        )

    def verify_signal(self, signal: DecisionSignal) -> DecisionSignal:
        """Recompute a serialized mean signal against this frozen model and its declared freshness rule.

        Generic DecisionSignal readers validate typed metadata, not arbitrary
        producer execution. This explicit verifier checks the whole mean output,
        source/model provenance and status without importing code by fingerprint.
        Missing policy or any changed value/context raises ValueError; no I/O occurs.
        """
        signal = DecisionSignal.model_validate(signal)
        if signal.maximum_event_age is None:
            raise ValueError("Mean signal is missing its freshness policy")
        expected = self.signal(signal.request, maximum_event_age=signal.maximum_event_age)
        if signal != expected:
            raise ValueError("Signal disagrees with its frozen model and eligible training evidence")
        return signal


def _signal_value(
    model: QuantityMeanModel, request: SignalRequest, estimate: MeanEstimate | None, age_limit: Quantity
) -> tuple[Measurement, Literal["available", "missing", "stale"]]:
    """Keep historical-event freshness and real zeros distinct from missing data."""
    status: Literal["unknown", "unavailable", "observed"] = "unknown"
    availability: Literal["available", "missing", "stale"] = "missing"
    value: float | None = None
    if estimate is not None:
        elapsed = request.decided_at - estimate.latest_event_at
        age = Fraction(elapsed.days * 86400 + elapsed.seconds) + Fraction(elapsed.microseconds, 10**6)
        if age > Fraction(age_limit.value):
            status, availability = "unavailable", "stale"
        else:
            status, availability, value = "observed", "available", float(estimate.quantity.value)
    semantics = model.training.input_semantics
    return Measurement(
        name=model.training.measurement_name + "_historical_mean_estimate",
        value=value,
        unit=semantics.unit,
        meaning=semantics.meaning + ":historical-mean-ceil-1e-9:binary64-v1",
        status=status,
        direction="none",
        minimum=0,
    ), availability


def fit_quantity_mean(training: TrainingSet, *, frozen_at: Timestamp) -> QuantityMeanModel:
    """Freeze a deterministic simple mean at a caller-declared instant after eligible training.

    No clock, provider, RNG or filesystem is consulted. Invalid/unavailable labels
    and future freezes fail the typed contract; no confidence interval, calibrated
    probability or independence claim is computed.
    """
    training = TrainingSet.model_validate(training)
    return QuantityMeanModel(
        schema_version=1, training=training, frozen_at=frozen_at, estimates=_mean_estimates(training)
    )
