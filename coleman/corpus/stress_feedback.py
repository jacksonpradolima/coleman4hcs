"""Bounded complete/disjoint point-in-time feedback pages, separate from synthetic truth."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import Field, model_validator

from coleman.corpus.stress_contracts import StressCase
from coleman.corpus.stress_replay import schedule_build
from coleman.corpus.stress_world import StressObservation, generate_build
from coleman.evaluation.protected.temporal import TemporalObservation
from coleman.observations.contracts import ObservationHistory, ObservationRevision, ObservationSnapshot
from coleman.results.domain import Fingerprint, Measurement, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable


class StressDecisionInput(Immutable):
    """Predecision identity and signed context only; no current duration/label/latent risk."""

    name: str
    display_name: str
    context_score: Annotated[int, Field(strict=True, ge=-1, le=1)] | None


def decision_inputs(case: StressCase, build: int) -> tuple[StressDecisionInput, ...]:
    """Project only information declared before decision; no history or hidden truth is returned."""
    return tuple(
        StressDecisionInput(name=row.execution_name, display_name=row.display_name, context_score=row.context_score)
        for row in generate_build(case, build)
    )


def _interval_seconds(case: StressCase) -> int:
    """Reserve complete serial occupancy plus reporting/receipt using only declared factor bounds."""
    copies = 3 if case.factors.variants == "stable" else 1
    maximum_seconds = {"constant": 1, "heavy_tail": 128, "failure_correlated": 16, "changing": 16}
    return case.factors.test_count * copies * maximum_seconds[case.factors.duration] + 3661


def decision_time(case: StressCase, build: int) -> datetime:
    """Return a constructed UTC cutoff, spaced by declared bounds rather than future observations.

    The interval exceeds the maximum complete serial occupancy plus one hour
    reporting and one minute receipt. It depends only on declared size/variant/
    duration factors, never seed, realized outcomes or future duration samples.
    Up to 32 intervals support late-cutoff queries beyond the <=16-build case.
    Large cases may span years of constructed clock time; no observed CI span is
    inferred. Invalid cases or nonintegral/out-of-bounds indices reject.
    """
    case = StressCase.model_validate(case)
    if type(build) is not int or not 0 <= build <= 32:
        raise ValueError("Constructed decision index must be within [0, 32]")
    return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=build * _interval_seconds(case))


class StressFeedbackPage(VersionedContract):
    """A page of complete lineages, preserving the existing <=4,096 revision bound.

    Pages partition emitted logical observations in canonical execution-name
    order, at most 2,048 roots per page (two revisions in corrected mode). Empty
    selection still emits one empty page. A page never certifies full history:
    consumers must collect indices [0,page_count), verify membership digest and
    counts, then use each history.as_of at the same cutoff. Revision visibility
    uses both source availability and first receipt, not event/build time.
    """

    case_fingerprint: Fingerprint
    build: Annotated[int, Field(strict=True, ge=0, le=15)]
    page_index: Annotated[int, Field(strict=True, ge=0, le=73)]
    page_count: Annotated[int, Field(strict=True, ge=1, le=74)]
    logical_observations: Annotated[int, Field(strict=True, ge=0, le=150000)]
    membership_digest: Fingerprint
    observation_scope: Literal["selected-modeled-execution", "counterfactual-full-suite"]
    history: ObservationHistory

    @model_validator(mode="after")
    def _page(self) -> Self:
        """Bind page dimensions and whole-lineage membership; complete set verification is separate."""
        roots = [r for r in self.history.revisions if r.supersedes is None]
        count = max(1, (self.logical_observations + 2047) // 2048)
        expected = min(2048, self.logical_observations - self.page_index * 2048)
        if self.page_count != count or self.page_index >= count or len(roots) != expected:
            raise ValueError("Stress feedback page dimensions disagree")
        if (
            self.history.source_id != self.case_fingerprint
            or self.history.source_version != "owned-ci-stress-feedback-v1"
        ):
            raise ValueError("Stress feedback source/case binding disagrees")
        for row in self.history.revisions:
            _check_report_measurement(row.measurement)
            _check_report_time(row)
            if row.metadata.observation_id != f"b{self.build:02}:{row.entity_id}":
                raise ValueError("Stress feedback logical identity disagrees")
            if row.metadata.group_id != self.case_fingerprint:
                raise ValueError("Stress feedback dependence group disagrees")
        return self


def _check_report_measurement(measurement: Measurement) -> None:
    """Reject a foreign measure or nonbinary reported outcome before exposing a feedback view.

    Generic ObservationHistory permits other quantities; this producer's v1
    channel specifically contains observed binary reports or unknown/null.
    A structurally valid generic measurement does not establish these semantics.
    """
    definition = (measurement.name, measurement.unit, measurement.meaning, measurement.direction, measurement.minimum)
    if definition != ("synthetic_reported_nonpass", "1", "constructed-binary-report-v1", "none", 0.0):
        raise ValueError("Stress feedback requires its declared binary report measurement semantics")
    if measurement.status not in {"observed", "unknown"}:
        raise ValueError("Stress feedback report must be observed binary or unknown/null")
    if measurement.status == "observed" and measurement.value not in {0.0, 1.0}:
        raise ValueError("Stress feedback observed report must be binary zero or one")


def _check_report_time(revision: ObservationRevision) -> None:
    """Require modeled completion before reporting and the declared synthetic first-receipt offset."""
    target_end = revision.metadata.target_end
    if target_end is None or revision.metadata.available_at - target_end < timedelta(hours=1):
        raise ValueError("Stress feedback must follow modeled completion plus reporting latency")
    if revision.received_at - revision.metadata.available_at != timedelta(minutes=1):
        raise ValueError("Stress feedback first receipt must retain the constructed one-minute offset")


class _FeedbackClock(NamedTuple):
    """Validated constructed clock context, shared without recomputing case fingerprints per revision."""

    event: datetime
    interval: timedelta
    mode: str
    group: str


def _revisions(clock: _FeedbackClock, row: StressObservation, completed: int) -> Iterator[ObservationRevision]:
    """Preserve incorrect initial reports and late corrections without rewriting the past."""
    mode, event = clock.mode, clock.event
    target_end = event + timedelta(seconds=completed)
    available = target_end + timedelta(hours=1) + (2 * clock.interval if mode == "delayed" else timedelta(0))
    observation = f"b{row.build:02}:{row.execution_name}"
    reported = 1 - row.verdict if mode == "corrected" else row.verdict
    root = ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation(
            observation_id=observation,
            revision_id=observation + ":r0",
            group_id=clock.group,
            event_at=event,
            available_at=available,
            target_end=target_end,
        ),
        entity_id=row.execution_name,
        received_at=available + timedelta(minutes=1),
        supersedes=None,
        measurement=Measurement(
            name="synthetic_reported_nonpass",
            unit="1",
            meaning="constructed-binary-report-v1",
            direction="none",
            minimum=0.0,
            value=None if mode == "missing" else float(reported),
            status="unknown" if mode == "missing" else "observed",
        ),
    )
    yield root
    if mode != "corrected":
        return
    corrected_at = target_end + timedelta(hours=1) + 3 * clock.interval
    correction = ObservationRevision.model_validate(
        {
            **root.model_dump(),
            "metadata": {
                **root.metadata.model_dump(),
                "revision_id": observation + ":r1",
                "available_at": corrected_at,
            },
            "received_at": corrected_at + timedelta(minutes=1),
            "supersedes": root.metadata.revision_id,
            "measurement": {**root.measurement.model_dump(), "value": float(row.verdict)},
        }
    )
    yield correction


def feedback_pages(
    case: StressCase,
    build: int,
    ranking: tuple[str, ...] | None = None,
) -> Iterator[StressFeedbackPage]:
    """Emit bounded pages after declared prefix selection; no future value enters a decision view.

    Full reports every truth row; censored reports selected executions only.
    At severe budget full reports use an explicitly counterfactual full-suite
    schedule, not a budget-feasible native learner. Availability follows each
    modeled completion; delayed/corrected revisions add two/three predeclared
    decision intervals without inspecting future realizations.
    Delayed/missing/corrected channels apply to selected executions, leaving
    unselected labels absent (unavailable, not passing). A caller may score a
    descriptive ranking but must not infer an eligible native learner from this
    helper. Inputs/versions/ranking are revalidated before pages are yielded.
    No network, filesystem, transport, policy fit or native execution occurs.
    """
    case = StressCase.model_validate(case)
    observation_case = case
    counterfactual = case.factors.feedback == "full" and case.factors.budget != "full"
    if counterfactual:
        observation_case = StressCase.model_validate(
            {**case.model_dump(), "factors": {**case.factors.model_dump(), "budget": "full"}}
        )
    selected = {
        event.name: event.completed_seconds for event in schedule_build(observation_case, build, ranking).events
    }
    rows = tuple(
        sorted(
            (
                row
                for row in generate_build(case, build)
                if case.factors.feedback == "full" or row.execution_name in selected
            ),
            key=lambda row: row.execution_name,
        )
    )
    membership = fingerprint([row.execution_name for row in rows])
    count = max(1, (len(rows) + 2047) // 2048)
    group = case.fingerprint
    clock = _FeedbackClock(
        decision_time(case, build), timedelta(seconds=_interval_seconds(case)), case.factors.feedback, group
    )
    for page in range(count):
        revisions = tuple(
            revision
            for row in rows[page * 2048 : (page + 1) * 2048]
            for revision in _revisions(clock, row, selected[row.execution_name])
        )
        yield StressFeedbackPage(
            schema_version=1,
            case_fingerprint=group,
            build=build,
            page_index=page,
            page_count=count,
            logical_observations=len(rows),
            membership_digest=membership,
            observation_scope="counterfactual-full-suite" if counterfactual else "selected-modeled-execution",
            history=ObservationHistory(
                schema_version=1,
                source_id=group,
                source_version="owned-ci-stress-feedback-v1",
                revisions=revisions,
            ),
        )


def visible_feedback(pages: tuple[StressFeedbackPage, ...], cutoff: datetime) -> tuple[ObservationSnapshot, ...]:
    """Verify a complete disjoint page set, then reuse point-in-time selection per bounded page.

    Reordered transport delivery is normalized by page index; missing, repeated,
    conflicting or shuffled membership pages reject. This verifies consistency,
    not correspondence to unseen generator truth. No large history is squeezed
    through the existing bounded ObservationSnapshot contract.
    """
    if len(pages) > 74:
        raise ValueError("Stress feedback exceeds its 74-page resource bound")
    checked = sorted((StressFeedbackPage.model_validate(page) for page in pages), key=lambda p: p.page_index)
    if not checked or [p.page_index for p in checked] != list(range(checked[0].page_count)):
        raise ValueError("Stress feedback requires every distinct page exactly once")
    first = checked[0]
    binding = (
        first.case_fingerprint,
        first.build,
        first.page_count,
        first.logical_observations,
        first.membership_digest,
        first.observation_scope,
    )
    names = []
    for page in checked:
        if (
            page.case_fingerprint,
            page.build,
            page.page_count,
            page.logical_observations,
            page.membership_digest,
            page.observation_scope,
        ) != binding:
            raise ValueError("Stress feedback pages have conflicting bindings")
        names.extend(sorted(row.entity_id for row in page.history.revisions if row.supersedes is None))
    if len(set(names)) != len(names) or names != sorted(names) or fingerprint(names) != first.membership_digest:
        raise ValueError("Stress feedback membership is missing, overlapping or out of order")
    return tuple(page.history.as_of(cutoff) for page in checked)
