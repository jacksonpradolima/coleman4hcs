"""Bounded source checks and independently verified native CI fixture artifacts."""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.contracts import CorpusManifest
from coleman.corpus.fixtures import (
    CIReferenceFixture,
    FixtureCase,
    ParallelFixtureCase,
    ParallelReference,
    SerialFixtureCase,
    fixture_bytes,
    parallel_reference,
)
from coleman.corpus.reference import TinyCIReference, tiny_ci_reference
from coleman.pilot.contracts import RATE, SECONDS
from coleman.results.domain import DomainRecord, Identifier, Measurement, ResultProvenance
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity


def _reference(fixture: CIReferenceFixture, case: FixtureCase) -> TinyCIReference | ParallelReference:
    """Verify hand-declared expected values against a separately implemented exact reference."""
    calculated = (
        tiny_ci_reference(fixture.build, case.ranking, case.budget)
        if isinstance(case, SerialFixtureCase)
        else parallel_reference(fixture.build)
    )
    if calculated != case.expected:
        raise ValueError("Hand reference disagrees with independent exact calculation")
    return calculated


def _measure(
    name,
    value,
    unit,
    meaning,
    *,
    status: Literal["observed", "unknown", "unavailable", "not_applicable"] = "observed",
    direction: Literal["minimize", "maximize", "none"] = "none",
) -> Measurement:
    """Explicitly project exact references to finite binary64 for compatibility checks only."""
    return Measurement(
        name="corpus_" + name,
        value=None if value is None else float(value),
        unit=unit,
        meaning=meaning + ":binary64-projection-v1",
        direction=direction,
        status=status,
        minimum=0.0,
    )


def _reference_measurements(reference: TinyCIReference | ParallelReference) -> tuple[Measurement, ...]:
    """Keep exact source answers authoritative and explicit undefined/censored states in sidecars."""
    ratios = [("failing_test_recall", reference.recall)]
    quantities = [("runner_work", reference.runner_work)]
    if isinstance(reference, TinyCIReference):
        ratios += [("napfd", reference.napfd), ("apfdc", reference.apfdc)]
        quantities += [("full_suite_work", reference.full_suite_work)]
    else:
        quantities += [("execution_elapsed", reference.execution_elapsed)]
    result = [
        _measure(
            name,
            ratio.value,
            ratio.unit,
            ratio.meaning,
            status="observed" if ratio.value is not None else "not_applicable",
        )
        for name, ratio in ratios
    ]
    result += [
        _measure("total_failing_tests", reference.total_failing_tests, "1", "full-universe-failing-test-count-v1"),
        _measure("detected_failing_tests", reference.detected_failing_tests, "1", "detected-failing-test-count-v1"),
    ]
    result += [
        _measure(name, quantity.value, quantity.semantics.unit, quantity.semantics.meaning)
        for name, quantity in quantities
    ]
    first = reference.first_failure_elapsed
    result.append(
        _measure(
            "first_failure_elapsed",
            first.value if first is not None else None,
            "s",
            "execution-elapsed-excluding-queue-setup-overhead-v1",
            status="observed" if first is not None else "unavailable",
        )
    )
    return tuple(result)


def _record(
    manifest: CorpusManifest,
    fixture: CIReferenceFixture,
    case: FixtureCase,
    measurements: tuple[Measurement, ...],
    producer: str,
) -> DomainRecord:
    """Bind observations to a common frozen source/protocol and source outcome availability."""
    if manifest.source.sha256 is None:
        raise ValueError("Verified source digest is required")
    return DomainRecord(
        schema_version=1,
        namespace="benchmark.corpus.ci",
        run_id=manifest.fingerprint,
        execution_id=case.kind,
        decision_id=case.case_id,
        data_fingerprint=manifest.source.sha256,
        available_at=fixture.build.outcomes_available_at,
        provenance=ResultProvenance(
            producer=producer,
            producer_version="1",
            spec_fingerprint=manifest.fingerprint,
            evaluation_fingerprint=manifest.fingerprint,
        ),
        measurements=measurements,
    )


def _native_measurements(
    fixture: CIReferenceFixture, case: SerialFixtureCase, expected: tuple[Measurement, ...]
) -> tuple[Measurement, ...]:
    """Exercise the real legacy native metric without altering its existing row conventions.

    New named sidecars explicitly mark undefined ratios/censored detection rather
    than overwriting native fallback values. Zero-work failing suites cannot be
    checked through the legacy APFDc implementation and are rejected explicitly.
    """
    from coleman.evaluation import NAPFDVerdictMetric

    tests = {test.test_id: test for test in fixture.build.tests}
    if all(test.duration.value == 0 for test in tests.values()) and any(
        test.outcome == "failed" for test in tests.values()
    ):
        raise ValueError("Native fixture metric does not support a zero-work failing suite")
    metric = NAPFDVerdictMetric()
    metric.update_budget("fixed_time", float(case.budget.value))
    metric.update_available_time(float(case.budget.value))
    metric.evaluate(
        [
            {
                "Name": name,
                "Duration": float(tests[name].duration.value),
                "Verdict": int(tests[name].outcome == "failed"),
            }
            for name in case.ranking
        ]
    )
    if tuple(metric.scheduled_testcases) != case.expected.selected_ids:
        raise ValueError("Native scheduled identities disagree with independent reference")
    known_failures = sum(test.outcome == "failed" for test in tests.values())
    values = {
        "corpus_total_failing_tests": metric.detected_failures + metric.undetected_failures,
        "corpus_detected_failing_tests": metric.detected_failures,
        "corpus_failing_test_recall": metric.recall if known_failures else None,
        "corpus_napfd": metric.fitness if known_failures else None,
        "corpus_apfdc": metric.cost if known_failures else None,
        "corpus_runner_work": sum(float(tests[name].duration.value) for name in metric.scheduled_testcases),
        "corpus_full_suite_work": sum(float(test.duration.value) for test in tests.values()),
        "corpus_first_failure_elapsed": metric.ttf_duration if metric.detection_ranks else None,
    }
    return _project_actual_values(expected, values)


def _parallel_measurements(
    fixture: CIReferenceFixture, case: ParallelFixtureCase, expected: tuple[Measurement, ...]
) -> tuple[Measurement, ...]:
    """Check real production event replay against the complete simultaneous closed-form reference."""
    from coleman.pilot.replay import replay_events

    actual = replay_events(fixture.build, case.events, Quantity(value=Decimal(0), semantics=RATE))
    values = {
        "corpus_total_failing_tests": actual.failing_tests,
        "corpus_detected_failing_tests": actual.detected_failing_tests,
        "corpus_failing_test_recall": actual.failing_test_recall,
        "corpus_runner_work": float(actual.runner_work.value),
        "corpus_execution_elapsed": float(actual.execution_elapsed.value),
        "corpus_first_failure_elapsed": float(actual.first_failure_elapsed.value)
        if actual.first_failure_elapsed is not None
        else None,
    }
    return _project_actual_values(expected, values)


def _project_actual_values(
    expected: tuple[Measurement, ...], values: dict[str, float | None]
) -> tuple[Measurement, ...]:
    """Derive actual availability from actual measurements, never copy a reference's missingness."""
    projected = []
    for item in expected:
        value = values[item.name]
        status = "observed"
        if value is None:
            status = "unavailable" if item.name == "corpus_first_failure_elapsed" else "not_applicable"
        projected.append(Measurement.model_validate({**item.model_dump(), "value": value, "status": status}))
    return tuple(projected)


class FixtureObservation(Immutable):
    """Actual native measurements for one tiny known-answer case; not an empirical policy trial."""

    case_id: Identifier
    record: DomainRecord


class CorpusFixtureReport(VersionedContract):
    """Revalidated functional fixture evidence, explicitly insufficient for generalization.

    Exact source answers and bytes are embedded/rechecked on read. No host clock,
    CLI state, provider connection or timing claim enters mathematical identity.
    Existing native metrics are untouched; projected sidecars have distinct names.
    """

    manifest: CorpusManifest
    fixture: CIReferenceFixture
    observations: tuple[FixtureObservation, ...] = Field(min_length=1, max_length=8)
    evidence_scope: Literal["tiny_synthetic_correctness_only"] = "tiny_synthetic_correctness_only"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Reject missing/duplicate cases, altered sources and incompatible/incorrect measurements."""
        _source_checks(self.manifest, fixture_bytes(self.fixture))
        _fixture_bindings(self.manifest, self.fixture)
        by_id = {case.case_id: case for case in self.fixture.cases}
        ids = [observation.case_id for observation in self.observations]
        if sorted(ids) != sorted(by_id):
            raise ValueError("Every fixture case must have exactly one native observation")
        for observation in self.observations:
            case = by_id[observation.case_id]
            expected = _record(
                self.manifest,
                self.fixture,
                case,
                _reference_measurements(_reference(self.fixture, case)),
                "coleman.corpus.exact-reference",
            )
            _compare_records(expected, observation.record)
        object.__setattr__(self, "observations", tuple(sorted(self.observations, key=lambda item: item.case_id)))
        return self


def _compare_records(expected: DomainRecord, actual: DomainRecord) -> None:
    """Check explicit projection units/protocol/status and absolute tolerance 10^-12 only."""
    if (
        expected.logical_key != actual.logical_key
        or expected.available_at != actual.available_at
        or {m.name for m in expected.measurements} != {m.name for m in actual.measurements}
        or actual.provenance.spec_fingerprint != expected.provenance.spec_fingerprint
        or actual.provenance.producer != "coleman.corpus.native-verifier"
        or actual.provenance.producer_version != "1"
        or actual.provenance.source_revision != expected.provenance.source_revision
        or actual.learning_reward is not None
    ):
        raise ValueError("Fixture record context or measurement set differs")
    for item in expected.measurements:
        expected.require_compatible(actual, item.name)
        observed = actual.measurement(item.name)
        if item.status != observed.status:
            raise ValueError("Fixture availability state differs")
        if item.value is not None and (
            observed.value is None
            or not math.isclose(
                item.value,
                observed.value,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Native fixture measurement differs from independent reference")


def _source_checks(manifest: CorpusManifest, payload: bytes) -> None:
    """Require permission to embed source bytes, supported adapters and matching size/digest."""
    manifest.require_ready()
    if manifest.license.redistribution != "permitted":
        raise ValueError("Embedded fixture reports require permitted source redistribution")
    if manifest.adapter.name != "ci-fixture-json" or manifest.adapter.version != "1":
        raise ValueError("Corpus adapter is not implemented by the phase-one fixture validator")
    if manifest.evaluation.purpose != "correctness_fixture" or manifest.outcomes.duration != SECONDS:
        raise ValueError("Fixture validator requires correctness purpose and explicit occupancy-second units")
    if (
        len(payload) > min(manifest.resources.maximum_source_bytes, 131072)
        or len(payload) != manifest.source.size_bytes
    ):
        raise ValueError("Source byte limit or declared size mismatch")
    if "sha256:" + hashlib.sha256(payload).hexdigest() != manifest.source.sha256:
        raise ValueError("Source SHA-256 mismatch")


def _fixture_bindings(manifest: CorpusManifest, fixture: CIReferenceFixture) -> None:
    """Verify adapter semantics, actual row conservation and declared executed dimensions."""
    rows = len(fixture.build.tests)
    if (
        rows > manifest.resources.maximum_rows
        or manifest.conversion.input_rows != rows
        or manifest.conversion.output_rows != rows
        or manifest.conversion.exclusions
    ):
        raise ValueError("Fixture row limits or conversion counts disagree with source")
    if manifest.benchmark_id != fixture.fixture_id or manifest.layer != "synthetic":
        raise ValueError("Fixture identity or synthetic source classification differs")
    _fixture_time_and_outcomes(manifest)
    if manifest.evaluation.budget_mode != "fixed_runner_seconds" or manifest.evaluation.seeds:
        raise ValueError("Deterministic fixture validator requires runner seconds and no stochastic seeds")
    if manifest.evaluation.temporal_plan is not None:
        raise ValueError("Correctness fixtures do not implement a protected temporal experiment")
    budgets = {
        case.budget.value if isinstance(case, SerialFixtureCase) else case.expected.runner_work.value
        for case in fixture.cases
    }
    if budgets != {budget.value for budget in manifest.evaluation.budgets}:
        raise ValueError("Declared budgets differ from fixture execution")
    metrics = {
        measurement.name for case in fixture.cases for measurement in _reference_measurements(_reference(fixture, case))
    }
    if metrics != set(manifest.evaluation.metric_ids):
        raise ValueError("Declared metrics differ from fixture observations")


def _fixture_time_and_outcomes(manifest: CorpusManifest) -> None:
    """Reject ignored or misleading identity, availability and verdict declarations."""
    if (
        manifest.time.origin != "constructed"
        or manifest.time.timezone != "UTC"
        or manifest.time.event_field != "build.decided_at"
        or manifest.time.outcome_available_field != "build.outcomes_available_at"
        or manifest.time.feature_available_field != "build.decided_at"
    ):
        raise ValueError("Fixture requires its explicit constructed timestamp/availability fields")
    if manifest.identity.observation_fields != (
        "build.build_id",
        "build.tests.test_id",
    ) or manifest.identity.group_fields != ("build.build_id",):
        raise ValueError("Fixture observation identity must retain build and logical test")
    mapping = {(item.source_value, item.verdict) for item in manifest.outcomes.mapping}
    if (
        mapping != {("passed", 0), ("failed", 1)}
        or manifest.outcomes.unit != "failing_test"
        or manifest.outcomes.unique_fault_mapping is not None
        or manifest.outcomes.unknown_policy != "reject"
    ):
        raise ValueError("Fixture requires complete binary failing-test outcomes without an invented fault map")


def validate_fixture_source(manifest: CorpusManifest, payload: bytes) -> CorpusFixtureReport:
    """Verify exact source bytes/metadata and native fixture correctness without any provider I/O.

    Unknown versions, noncanonical source JSON, unsupported sources, bad licensing,
    resource caps, false hand references or native discrepancies raise ValueError.
    This does not execute a policy experiment, fetch a dataset or admit real CI.
    """
    manifest = CorpusManifest.model_validate(manifest)
    _source_checks(manifest, payload)
    try:
        fixture = CIReferenceFixture.from_json(payload)
    except RecursionError as exc:
        raise ValueError("Fixture JSON exceeds parser depth limits") from exc
    if payload != fixture_bytes(fixture):
        raise ValueError("Fixture source must use the canonical v1 JSON bytes")
    _fixture_bindings(manifest, fixture)
    observations = []
    for case in fixture.cases:
        expected = _reference_measurements(_reference(fixture, case))
        actual = (
            _native_measurements(fixture, case, expected)
            if isinstance(case, SerialFixtureCase)
            else _parallel_measurements(fixture, case, expected)
        )
        observations.append(
            FixtureObservation(
                case_id=case.case_id, record=_record(manifest, fixture, case, actual, "coleman.corpus.native-verifier")
            )
        )
    return CorpusFixtureReport(schema_version=1, manifest=manifest, fixture=fixture, observations=tuple(observations))


def publish_fixture_report(report: CorpusFixtureReport, writer: ArtifactWriter) -> Path:
    """Publish verified deterministic evidence using existing immutable atomic artifacts.

    Identical concurrent retries are idempotent; conflicts/I/O failures propagate.
    Interrupted writes cannot expose a partly completed report. The existing local
    hard-link/fsync and power-loss limits remain unchanged. No new storage exists.
    """
    report = CorpusFixtureReport.model_validate(report)
    path = writer.run_root / "corpus-validations" / f"{report.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, report.to_json() + "\n")
    return path
