"""Independent point-in-time, lineage, training and signal invariants for owned CI observations."""

import ast
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.evaluation.protected import TemporalObservation, TemporalPlan
from coleman.observations import (
    EVENT_AGE_SECONDS,
    DecisionSignal,
    ObservationHistory,
    ObservationRevision,
    ObservationSnapshot,
    OfflineObservationProvider,
    QuantityMeanModel,
    SignalRequest,
    fit_quantity_mean,
    prepare_training,
    publish_model,
    publish_signal,
)
from coleman.results.domain import Measurement
from coleman.selection import Quantity, UnitSemantics

ROOT = Path(__file__).parents[2]
SECONDS = UnitSemantics(unit="s", meaning="test-runner-occupancy-v1")


def instant(value):
    return datetime.fromisoformat(value)


def changed(model, **values):
    return type(model).model_validate({**model.model_dump(), **values})


def quantity(value, semantics=EVENT_AGE_SECONDS):
    return Quantity(value=Decimal(str(value)), semantics=semantics)


def revision(
    *,
    observation="build-2:test-a",
    entity="test-a",
    revision_id="r1",
    supersedes=None,
    event="2026-01-02T10:00:00Z",
    available="2026-01-02T10:00:02Z",
    received=None,
    value=0.0,
    name="ci_failure",
    unit="1",
    meaning="binary-failing-test-indicator-v1",
    group="build-2",
):
    return ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation.model_validate(
            {
                "observation_id": observation,
                "group_id": group,
                "revision_id": revision_id,
                "event_at": event,
                "available_at": available,
            }
        ),
        entity_id=entity,
        received_at=instant(received or available),
        supersedes=supersedes,
        measurement=Measurement(
            name=name, value=value, unit=unit, meaning=meaning, direction="none", status="observed"
        ),
    )


def history(*rows):
    return ObservationHistory(schema_version=1, source_id="owned-ci-fixture", source_version="1", revisions=rows)


def request(entity="test-a", **overrides):
    return SignalRequest.model_validate(
        {
            "schema_version": 1,
            "decision_id": "build-2",
            "entity_id": entity,
            "decided_at": "2026-01-02T10:00:00Z",
            "information_cutoff": "2026-01-02T10:00:00Z",
            "target_start": "2026-01-02T10:00:00Z",
            "target_end": "2026-01-02T10:01:00Z",
            **overrides,
        }
    )


def duration_row(observation, value, event, entity="test-a", **overrides):
    return revision(
        observation=observation,
        entity=entity,
        value=value,
        event=event,
        available=event,
        name="ci_duration",
        unit=SECONDS.unit,
        meaning=SECONDS.meaning,
        group=observation.split(":")[0],
        **overrides,
    )


@pytest.fixture
def source():
    return history(
        duration_row("build-0:test-a", 1.0, "2025-12-29T10:00:00Z"),
        duration_row("build-1:test-a", 3.0, "2025-12-30T10:00:00Z"),
        duration_row("build-1:test-b", 0.0, "2025-12-30T10:00:00Z", entity="test-b"),
    )


@pytest.fixture
def plan():
    return TemporalPlan.model_validate(
        {
            "train_end": "2025-12-31T23:59:59Z",
            "validation_end": "2026-01-01T23:59:59Z",
            "test_end": "2026-01-03T23:59:59Z",
        }
    )


@pytest.fixture
def training(source, plan):
    return prepare_training(
        source,
        plan,
        information_cutoff=instant("2026-01-02T10:00:00Z"),
        measurement_name="ci_duration",
        input_semantics=SECONDS,
    )


@pytest.fixture
def model(training):
    return fit_quantity_mean(training, frozen_at=instant("2026-01-01T00:00:00Z"))


def test_issue_as_of_values_and_future_correction_invariance():
    first = revision()
    correction = revision(revision_id="r2", supersedes="r1", available="2026-01-02T10:05:00Z", value=1.0)
    original = history(first)
    corrected = history(first, correction)
    assert corrected.as_of(instant("2026-01-02T10:00:01Z")).observations == ()
    early = corrected.as_of(instant("2026-01-02T10:00:03Z"))
    assert early.observations == (first,)
    assert early.observations[0].measurement.value == 0
    assert original.as_of(early.information_cutoff).fingerprint == early.fingerprint
    assert corrected.as_of(instant("2026-01-02T10:05:00Z")).observations == (correction,)
    changed_future = history(first, changed(correction, measurement=changed(correction.measurement, value=7.0)))
    assert changed_future.as_of(early.information_cutoff).fingerprint == early.fingerprint
    assert original.fingerprint != corrected.fingerprint


def test_receipt_delay_out_of_order_arrival_and_explicit_ties():
    first = revision(received="2026-01-02T10:07:00Z", revision_id="z")
    correction = revision(
        revision_id="a",
        supersedes="z",
        available="2026-01-02T10:05:00Z",
        received="2026-01-02T10:05:00Z",
        value=1.0,
    )
    source = history(correction, first)
    assert source.as_of(instant("2026-01-02T10:00:03Z")).observations == ()
    assert source.as_of(instant("2026-01-02T10:06:00Z")).observations == (correction,)
    assert source.as_of(instant("2026-01-02T10:08:00Z")).observations == (correction,)
    root = changed(first, received_at=first.metadata.available_at)
    tied = changed(correction, metadata=changed(correction.metadata, available_at=root.metadata.available_at))
    assert history(root, tied).as_of(tied.received_at).observations == (tied,)
    assert history(root, tied).fingerprint == history(tied, root, root).fingerprint


def test_lineage_conflicts_missing_parents_forks_cycles_and_semantics():
    root = revision()
    child = revision(revision_id="r2", supersedes="r1", value=1.0)
    forks = revision(revision_id="r3", supersedes="r1", value=2.0)
    cyclic = changed(child, supersedes="r3")
    back = changed(forks, supersedes="r2")
    invalid_sets = (
        (root, changed(root, measurement=changed(root.measurement, value=1.0))),
        (root, changed(root, received_at=instant("2026-01-02T10:00:03Z"))),
        (child,),
        (root, child, forks),
        (root, cyclic, back),
        (root, changed(child, entity_id="other")),
        (root, changed(child, metadata=changed(child.metadata, group_id="different"))),
        (root, changed(child, measurement=changed(child.measurement, unit="ms"))),
        (root, changed(child, measurement=changed(child.measurement, meaning="other-v1"))),
    )
    for rows in invalid_sets:
        with pytest.raises(ValueError):
            history(*rows)
    with pytest.raises(ValueError, match="itself"):
        changed(root, supersedes="r1")
    early_receipt = instant("2026-01-02T10:00:01Z")
    with pytest.raises(ValueError, match="receipt"):
        changed(root, received_at=early_receipt)


def test_snapshot_rejects_future_and_duplicate_observations():
    row = revision()
    cutoff = instant("2026-01-02T10:00:01Z")
    with pytest.raises(ValueError, match="unavailable"):
        ObservationSnapshot(
            schema_version=1, source_id="ci", source_version="1", information_cutoff=cutoff, observations=(row,)
        )
    with pytest.raises(ValueError, match="one revision"):
        ObservationSnapshot(
            schema_version=1,
            source_id="ci",
            source_version="1",
            information_cutoff=row.received_at,
            observations=(row, row),
        )
    with pytest.raises(ValueError):
        history(*((row,) * 4097))


def test_timezone_normalization_and_naive_rejection():
    row = revision()
    offset = changed(row, metadata=changed(row.metadata, event_at=instant("2026-01-02T07:00:00-03:00")))
    assert history(offset).fingerprint == history(row).fingerprint
    assert history(row).as_of(instant("2026-01-02T07:00:03-03:00")) == history(row).as_of(
        instant("2026-01-02T10:00:03Z")
    )
    naive = datetime(2026, 1, 2, 10, 0, 2)
    with pytest.raises(ValueError):
        changed(row, received_at=naive)
    with pytest.raises(ValueError):
        request(information_cutoff="2026-01-02T10:00:00")


def test_training_freeze_ignores_validation_and_late_corrections(source, plan, model):
    base = source.revisions[0]
    future = changed(
        base,
        metadata=changed(base.metadata, revision_id="r2", available_at=instant("2026-01-02T10:05:00Z")),
        received_at=instant("2026-01-02T10:05:00Z"),
        supersedes="r1",
        measurement=changed(base.measurement, value=99.0),
    )
    validation = duration_row("validation:test-a", 10000.0, "2026-01-01T10:00:00Z")
    extended = changed(source, revisions=(*source.revisions, future, validation))
    trained = prepare_training(
        extended,
        plan,
        information_cutoff=instant("2026-01-02T12:00:00Z"),
        measurement_name="ci_duration",
        input_semantics=SECONDS,
    )
    frozen = fit_quantity_mean(trained, frozen_at=model.frozen_at)
    assert trained.snapshot.information_cutoff == plan.train_end
    assert frozen.fingerprint == model.fingerprint
    assert frozen.estimates[0].quantity.value == 2
    assert frozen.estimates[0].observations == 2
    assert (
        frozen.signal(request(), maximum_event_age=quantity(1000000)).fingerprint
        == model.signal(request(), maximum_event_age=quantity(1000000)).fingerprint
    )


def test_missing_stale_zero_and_signed_uncalibrated_scores(model):
    fresh = model.signal(request("test-b"), maximum_event_age=quantity(1000000))
    assert fresh.measurement.value == 0
    assert fresh.measurement.status == "observed"
    missing = model.signal(request("unknown"), maximum_event_age=quantity(1000000))
    assert missing.measurement.value is None
    assert missing.availability == "missing"
    stale = model.signal(request("test-b"), maximum_event_age=quantity(0))
    assert stale.measurement.value is None
    assert stale.availability == "stale"
    elapsed = request().decided_at - model.estimates[0].latest_event_at
    boundary = elapsed.days * 86400 + elapsed.seconds
    assert model.signal(request(), maximum_event_age=quantity(boundary)).availability == "available"
    assert model.signal(request(), maximum_event_age=quantity(boundary - 1)).availability == "stale"
    priority = changed(
        fresh,
        interpretation="priority_score",
        measurement=Measurement(
            name="ci_priority",
            value=-2.0,
            unit="1",
            meaning="uncalibrated-ranking-score-v1",
            direction="maximize",
            status="observed",
        ),
        provenance=changed(fresh.provenance, producer="ci.fixture.priority"),
    )
    assert priority.measurement.value == -2
    unavailable = changed(
        fresh,
        availability="not_applicable",
        measurement=changed(fresh.measurement, value=None, status="not_applicable"),
    )
    assert unavailable.measurement.value is None
    with pytest.raises(ValueError, match="disagree"):
        changed(missing, availability="available")


def test_unavailable_negative_and_wrong_unit_training_fail(source, plan, training):
    base = source.revisions[0]
    invalid = (
        changed(base.measurement, value=None, status="unavailable"),
        changed(base.measurement, value=-1.0),
        changed(base.measurement, unit="ms"),
        changed(base.measurement, value=1e13),
    )
    for measurement in invalid:
        rows = (changed(base, measurement=measurement), *source.revisions[1:])
        altered = changed(source, revisions=rows)
        with pytest.raises(ValueError):
            prepare_training(
                altered,
                plan,
                information_cutoff=plan.train_end,
                measurement_name="ci_duration",
                input_semantics=SECONDS,
            )
    snapshot = changed(training.snapshot, information_cutoff=instant("2026-01-02T00:00:00Z"))
    with pytest.raises(ValueError, match="training boundary"):
        changed(training, snapshot=snapshot)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_boolean_measurements_fail(value):
    with pytest.raises(ValueError):
        revision(value=value)


def test_model_tampering_future_freeze_horizon_and_age_units_fail(model):
    estimate = model.estimates[0]
    altered = changed(estimate, quantity=quantity(9, estimate.quantity.semantics))
    with pytest.raises(ValueError, match="evidence"):
        changed(model, estimates=(altered, *model.estimates[1:]))
    too_early = instant("2025-12-01T00:00:00Z")
    with pytest.raises(ValueError, match="training cutoff"):
        changed(model, frozen_at=too_early)
    before = request(information_cutoff="2025-12-31T23:59:59Z")
    age = quantity(1000000)
    with pytest.raises(ValueError, match="not frozen"):
        model.signal(before, maximum_event_age=age)
    with pytest.raises(ValueError, match="cutoff"):
        request(information_cutoff="2026-01-02T10:00:01Z")
    with pytest.raises(ValueError, match="target"):
        request(target_end="2026-01-02T09:59:59Z")
    wrong_unit = quantity(0, UnitSemantics(unit="ms", meaning=EVENT_AGE_SECONDS.meaning))
    valid_request = request()
    with pytest.raises(ValueError, match="units"):
        model.signal(valid_request, maximum_event_age=wrong_unit)


def test_input_permutation_decimal_context_and_source_identity(source, plan, model):
    for rows in itertools.permutations(source.revisions):
        permuted = changed(source, revisions=rows)
        assert permuted.fingerprint == source.fingerprint
        training = prepare_training(
            permuted, plan, information_cutoff=plan.train_end, measurement_name="ci_duration", input_semantics=SECONDS
        )
        with localcontext() as context:
            context.prec = 3
            assert fit_quantity_mean(training, frozen_at=model.frozen_at).fingerprint == model.fingerprint
    another = changed(source, source_version="2")
    assert another.as_of(plan.train_end).fingerprint != source.as_of(plan.train_end).fingerprint


def test_roundtrips_strict_versions_and_provider_boundary(source, training, model):
    signal = model.signal(request(), maximum_event_age=quantity(1000000))
    for value in (source, training.snapshot, training, model, signal):
        assert type(value).from_json(value.to_json()) == value
        for version in (0, 2, True, None):
            with pytest.raises(ValueError):
                changed(value, schema_version=version)
    assert OfflineObservationProvider(source).load_history() == source
    with pytest.raises(ValueError):
        ObservationHistory.from_json('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError):
        ObservationHistory.from_json('{"scenario":"legacy","ttf":0}')
    record = signal.to_domain_record(run_id="native", execution_id="1", evaluation_fingerprint=model.fingerprint)
    assert record.available_at == signal.request.decided_at
    assert record.learning_reward is None
    assert record.measurements == (signal.measurement,)
    assert record.provenance.source_revision == source.source_version


def test_atomic_idempotence_and_interruption(source, model, tmp_path, monkeypatch):
    signal = model.signal(request(), maximum_event_age=quantity(1000000))
    writer = ArtifactWriter(tmp_path)
    saved_model = publish_model(model, writer)
    assert QuantityMeanModel.from_json(saved_model.read_bytes()) == model
    saved = publish_signal(signal, writer)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: publish_signal(signal, writer), range(8)))
    assert paths == [saved] * 8
    assert DecisionSignal.from_json(saved.read_bytes()) == signal
    failed = ArtifactWriter(tmp_path / "failed")

    def interrupted(*_):
        raise OSError("before immutable link")

    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(OSError, match="immutable link"):
        publish_signal(signal, failed)
    assert not list((tmp_path / "failed").rglob("*.json"))
    assert not list((tmp_path / "failed").rglob("*.tmp"))


def test_domain_import_boundaries():
    for path in (ROOT / "coleman/observations").glob("*.py"):
        modules = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Import):
                modules.extend(item.name for item in node.names)
        assert not any(name.startswith(("coleman.cli", "coleman.transports", "mcp")) for name in modules)
    script = "import sys,coleman.observations; assert 'coleman.cli' not in sys.modules; assert 'mcp' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, timeout=30)


def test_independent_mean_rounding_and_no_future_group_peek(plan):
    rows = tuple(
        duration_row(f"build-{i}:test-a", value, f"2025-12-{27 + i:02d}T10:00:00Z")
        for i, value in enumerate((0.0, 1.0, 0.0))
    )
    initial = history(*rows)
    training = prepare_training(
        initial, plan, information_cutoff=plan.train_end, measurement_name="ci_duration", input_semantics=SECONDS
    )
    with localcontext() as context:
        context.prec = 3
        fitted = fit_quantity_mean(training, frozen_at=instant("2026-01-01T00:00:00Z"))
    assert fitted.estimates[0].quantity.value == Decimal("0.333333334")
    assert fitted.estimates[0].observations == 3
    # Future group membership cannot rewrite a past feature. The complete later
    # evaluation still must enforce the existing protected whole-fold group gate.
    future = duration_row("build-0:future", 999.0, "2026-01-02T10:00:00Z")
    extended = history(*rows, future)
    later = prepare_training(
        extended, plan, information_cutoff=plan.test_end, measurement_name="ci_duration", input_semantics=SECONDS
    )
    assert later.fingerprint == training.fingerprint
    metadata = tuple(row.effective_metadata for row in extended.revisions)
    with pytest.raises(ValueError, match="group spans"):
        plan.split(metadata)


def test_temporal_purging_retains_explicit_exclusion_without_labels(source, plan):
    first = source.revisions[0]
    overlap = changed(first, metadata=changed(first.metadata, target_end=instant("2026-01-01T00:00:00Z")))
    altered = changed(source, revisions=(overlap, *source.revisions[1:]))
    selected = prepare_training(
        altered, plan, information_cutoff=plan.train_end, measurement_name="ci_duration", input_semantics=SECONDS
    )
    assert len(selected.snapshot.observations) == 2
    assert len(selected.exclusions) == 1
    assert selected.exclusions[0].reason == "overlapping_target"
    assert selected.exclusions[0].observation.observation_id == first.metadata.observation_id


def test_versioned_golden_fixtures(source, model):
    fixtures = Path(__file__).parent / "fixtures"
    signal = model.signal(request(), maximum_event_age=quantity(1000000))
    for filename, value in (("history-v1.json", source), ("model-v1.json", model), ("signal-v1.json", signal)):
        payload = (fixtures / filename).read_text()
        assert value.to_json() + "\n" == payload
        assert type(value).from_json(payload) == value


def test_signal_specification_freshness_and_explicit_model_verification(model):
    age = quantity(1000000)
    signal = model.signal(request(), maximum_event_age=age)
    assert signal.maximum_event_age == age
    assert model.verify_signal(DecisionSignal.from_json(signal.to_json())) == signal
    stale = model.signal(request(), maximum_event_age=quantity(0))
    assert stale.provenance.model_fingerprint == signal.provenance.model_fingerprint
    assert stale.logical_id != signal.logical_id
    assert stale.specification_fingerprint != signal.specification_fingerprint
    changed_value = changed(signal, measurement=changed(signal.measurement, value=9.0))
    assert changed_value.logical_id == signal.logical_id
    assert changed_value.specification_fingerprint == signal.specification_fingerprint
    assert changed_value.fingerprint != signal.fingerprint
    with pytest.raises(ValueError, match="evidence"):
        model.verify_signal(changed_value)
    missing_policy = changed(signal, maximum_event_age=None)
    with pytest.raises(ValueError, match="freshness policy"):
        model.verify_signal(missing_policy)
    record = signal.to_domain_record(run_id="native", execution_id="1", evaluation_fingerprint=model.fingerprint)
    assert record.provenance.spec_fingerprint == signal.specification_fingerprint


def test_revision_edges_reject_missing_parents_and_backward_availability():
    root = revision()
    child = revision(revision_id="r2", supersedes="missing")
    backwards = changed(
        child, supersedes="r1", metadata=changed(child.metadata, available_at=instant("2026-01-02T10:00:01Z"))
    )
    for row in (child, backwards):
        with pytest.raises(ValueError):
            history(root, row)


def test_serialized_training_rejects_ineligible_rows_and_false_exclusion_audits(training):
    row = training.snapshot.observations[0]
    overlap = changed(row, metadata=changed(row.metadata, target_end=instant("2026-01-01T00:00:00Z")))
    snapshot = changed(training.snapshot, observations=(overlap, *training.snapshot.observations[1:]))
    with pytest.raises(ValueError, match="Ineligible"):
        changed(training, snapshot=snapshot)
    decision = training.plan.decide(row.effective_metadata)
    with pytest.raises(ValueError, match="contradictory"):
        changed(training, exclusions=(decision,))
    late = changed(
        row.effective_metadata,
        observation_id="late",
        group_id="late",
        available_at=instant("2026-01-01T00:00:00Z"),
    )
    exclusion = training.plan.decide(late)
    assert exclusion.reason == "outcome_not_available"
    with pytest.raises(ValueError, match="unavailable"):
        changed(training, exclusions=(exclusion,))
    absent_eligible = changed(row.effective_metadata, observation_id="absent", group_id="absent")
    bogus = training.plan.decide(absent_eligible)
    with pytest.raises(ValueError, match="exclusion reason"):
        changed(training, exclusions=(bogus,))


def test_serialized_signal_rejects_future_fit_and_zero_age_wrong_units(model, tmp_path):
    signal = model.signal(request(), maximum_event_age=quantity(1000000))
    wrong_age = quantity(0, UnitSemantics(unit="ms", meaning=EVENT_AGE_SECONDS.meaning))
    forged = signal.model_copy(update={"maximum_event_age": wrong_age})
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(ValueError, match="units"):
        publish_signal(forged, writer)
    future_provenance = {**signal.provenance.model_dump(), "training_cutoff": instant("2026-01-02T10:00:00Z")}
    with pytest.raises(ValueError, match="training cutoff"):
        changed(signal, provenance=future_provenance)
    assert not list(tmp_path.rglob("*.json"))


def test_latest_unavailable_revision_does_not_revive_an_obsolete_zero():
    first = revision(value=0.0)
    later = revision(revision_id="r2", supersedes="r1", available="2026-01-02T10:05:00Z")
    later = changed(later, measurement=changed(later.measurement, value=None, status="unavailable"))
    source = history(first, later)
    early = source.as_of(instant("2026-01-02T10:00:03Z")).observations[0]
    current = source.as_of(instant("2026-01-02T10:05:00Z")).observations[0]
    assert early.measurement.value == 0
    assert early.measurement.status == "observed"
    assert current == later
    assert current.measurement.value is None
    assert current.measurement.status == "unavailable"


def test_empty_training_is_missing_and_binary64_ceiling_is_explicit(plan):
    empty = prepare_training(
        history(), plan, information_cutoff=plan.train_end, measurement_name="ci_duration", input_semantics=SECONDS
    )
    model = fit_quantity_mean(empty, frozen_at=instant("2026-01-01T00:00:00Z"))
    assert model.estimates == ()
    assert model.signal(request(), maximum_event_age=quantity(1000000)).availability == "missing"
    source = history(duration_row("build-0:test-a", 0.1, "2025-12-29T10:00:00Z"))
    training = prepare_training(
        source, plan, information_cutoff=plan.train_end, measurement_name="ci_duration", input_semantics=SECONDS
    )
    decimal_model = fit_quantity_mean(training, frozen_at=model.frozen_at)
    assert decimal_model.estimates[0].quantity.value == Decimal("0.100000001")


def test_early_information_cutoff_purges_targets_beyond_actual_training_information(plan):
    row = duration_row("build-0:test-a", 7.0, "2025-12-29T10:00:00Z")
    row = changed(row, metadata=changed(row.metadata, target_end=instant("2025-12-31T12:00:00Z")))
    source = history(row)
    cutoff = instant("2025-12-30T00:00:00Z")
    training = prepare_training(
        source, plan, information_cutoff=cutoff, measurement_name="ci_duration", input_semantics=SECONDS
    )
    assert training.snapshot.information_cutoff == cutoff
    assert training.snapshot.observations == ()
    assert len(training.exclusions) == 1
    assert training.exclusions[0].reason == "overlapping_target"
    model = fit_quantity_mean(training, frozen_at=cutoff)
    assert model.estimates == ()
    unsafe = changed(training.snapshot, observations=(row,))
    with pytest.raises(ValueError, match="Ineligible"):
        changed(training, snapshot=unsafe, exclusions=())


def test_changed_cutoff_or_horizon_conflicts_with_an_existing_logical_signal(model, tmp_path):
    from coleman.results.domain_artifacts import DomainArtifacts

    signal = model.signal(request(), maximum_event_age=quantity(1000000))
    same_value = model.signal(
        request(information_cutoff="2026-01-02T09:59:59Z", target_end="2026-01-02T10:02:00Z"),
        maximum_event_age=quantity(1000000),
    )
    assert same_value.logical_id == signal.logical_id
    assert same_value.measurement == signal.measurement
    assert same_value.specification_fingerprint != signal.specification_fingerprint
    first = signal.to_domain_record(run_id="native", execution_id="1", evaluation_fingerprint=model.fingerprint)
    second = same_value.to_domain_record(run_id="native", execution_id="1", evaluation_fingerprint=model.fingerprint)
    artifacts = DomainArtifacts(tmp_path)
    artifacts.write(first)
    with pytest.raises(ValueError, match="Conflicting immutable"):
        artifacts.write(second)


def test_signal_publication_rejects_conflicting_values_for_one_logical_decision(model, tmp_path):
    signal = model.signal(request(), maximum_event_age=quantity(1000000))
    conflicting = changed(signal, measurement=changed(signal.measurement, value=9.0))
    assert conflicting.logical_id == signal.logical_id
    writer = ArtifactWriter(tmp_path)
    path = publish_signal(signal, writer)
    with pytest.raises(ValueError, match="Conflicting immutable"):
        publish_signal(conflicting, writer)
    assert DecisionSignal.from_json(path.read_bytes()) == signal
    assert len(list(tmp_path.rglob("decision-signals/*.json"))) == 1


@pytest.mark.parametrize("field", ["value", "minimum"])
@pytest.mark.parametrize("reverse", [False, True])
def test_signed_zero_duplicate_revisions_conflict_by_canonical_content(field, reverse):
    root = revision(value=0.0)
    root = changed(root, measurement=changed(root.measurement, minimum=0.0))
    signed = changed(root, measurement=changed(root.measurement, **{field: -0.0}))
    assert root == signed  # Numerical equality alone cannot establish identical serialized input.
    assert root.to_json() != signed.to_json()
    rows = (signed, root) if reverse else (root, signed)
    with pytest.raises(ValueError, match="Conflicting observation"):
        history(*rows)
    assert history(root, root).fingerprint == history(root).fingerprint
    assert history(signed, signed).fingerprint == history(signed).fingerprint


@pytest.mark.parametrize("field", ["value", "minimum"])
def test_mean_verifier_rejects_signed_zero_payload_drift(model, field):
    signal = model.signal(request("test-b"), maximum_event_age=quantity(1000000))
    signed = changed(signal, measurement=changed(signal.measurement, **{field: -0.0}))
    assert signed == signal
    assert signed.fingerprint != signal.fingerprint
    with pytest.raises(ValueError, match="evidence"):
        model.verify_signal(signed)
