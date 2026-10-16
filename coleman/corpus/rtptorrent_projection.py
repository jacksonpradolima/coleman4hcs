"""Pure deterministic RTPTorrent source-job projection; no files, clocks, transports or native execution."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import NamedTuple, cast

import polars as pl

from coleman.corpus.rtptorrent_contracts import REPORTED_SECONDS, ClassExecution, RTPTorrentDataset
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLRecipe
from coleman.results.domain import fingerprint
from coleman.selection.contracts import Quantity

_OBSERVATIONS = {
    "execution_id": pl.String,
    "replay_name": pl.String,
    "project": pl.String,
    "job_id": pl.Int64,
    "source_position": pl.Int64,
    "test_class": pl.String,
    "modeled_round": pl.Int64,
    "duration_decimal": pl.String,
    "duration_status": pl.String,
    "duration_unit": pl.String,
    "duration_meaning": pl.String,
    "method_count": pl.Int64,
    "method_failures": pl.Int64,
    "method_errors": pl.Int64,
    "method_skipped": pl.Int64,
    "class_outcome": pl.String,
    "replay_exclusion": pl.String,
}
_JOBS = {
    "project": pl.String,
    "job_id": pl.Int64,
    "logical_build_id": pl.String,
    "modeled_round": pl.Int64,
    "order_basis": pl.String,
    "event_at": pl.String,
    "outcome_available_at": pl.String,
    "source_executions": pl.Int64,
    "reported_failed_class_executions": pl.Int64,
    "unknown_class_outcomes": pl.Int64,
    "replay_executions": pl.Int64,
    "replay_status": pl.String,
}
_NATIVE = {
    "BuildId": pl.Int64,
    "Name": pl.String,
    "Duration": pl.Float64,
    "CalcPrio": pl.Int64,
    "LastRun": pl.String,
    "Verdict": pl.Int64,
    "SourceExecutionId": pl.String,
}


class RTPTorrentTables(NamedTuple):
    """Typed table bundle at the serialization boundary; source DTOs remain authoritative."""

    observations: pl.DataFrame
    jobs: pl.DataFrame
    native: pl.DataFrame
    logical_digest: str
    exclusions: Counter[str]


class _ProjectedJob(NamedTuple):
    """One source job with lossless analytical rows and explicitly eligible native rows."""

    observations: list[dict[str, object]]
    native: list[dict[str, object]]
    exclusions: Counter[str]


def _exclusion(case: ClassExecution, selected: bool) -> str | None:
    """Assign one primary reason; None certifies a selected known verdict and observed duration, including zero."""
    if not selected:
        return "outside_jobs"
    if case.outcome == "not_run":
        return "not_run"
    if case.outcome == "unknown":
        return "unknown_outcome"
    if case.duration is None:
        return "unavailable_duration"
    return None


def _observation(
    recipe: RTPTorrentETLRecipe,
    case: ClassExecution,
    round_id: int | None,
    name: str,
    execution: str,
    reason: str | None,
) -> dict[str, object]:
    """Serialize exact quantities/counts at the analytical Polars boundary; never fill null with zero."""
    return {
        "execution_id": execution,
        "replay_name": name,
        "project": recipe.conversion.project,
        "job_id": case.job_id,
        "source_position": case.position,
        "test_class": case.test_class,
        "modeled_round": round_id,
        "duration_decimal": None if case.duration is None else str(case.duration.value),
        "duration_status": case.duration_status,
        "duration_unit": REPORTED_SECONDS.unit,
        "duration_meaning": REPORTED_SECONDS.meaning,
        "method_count": case.count,
        "method_failures": case.failures,
        "method_errors": case.errors,
        "method_skipped": case.skipped,
        "class_outcome": case.outcome,
        "replay_exclusion": reason,
    }


def _native_record(case: ClassExecution, round_id: int, name: str, execution: str) -> dict[str, object]:
    """Project a certified eligible class to existing native seconds/failed-class semantics.

    Caller first requires _exclusion(case, True) is None, which certifies the
    non-null Quantity. Decimal seconds remain exact analytically; this boundary
    explicitly converts them to the existing native binary64 format. Original
    execution identity remains joinable; LastRun has no fabricated source clock.
    """
    duration = cast(Quantity, case.duration)
    return {
        "BuildId": round_id,
        "Name": name,
        "Duration": float(duration.value),
        "CalcPrio": 0,
        "LastRun": "",
        "Verdict": int(case.outcome == "failed"),
        "SourceExecutionId": execution,
    }


def _project_job(recipe: RTPTorrentETLRecipe, cases: list[ClassExecution], round_id: int | None) -> _ProjectedJob:
    """Assign label-occurrences before eligibility, preserving repeated classes and all raw source positions."""
    observations: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    for case in cases:
        name = fingerprint(
            [
                recipe.test_identity,
                recipe.conversion.project,
                case.test_class,
                occurrences[case.test_class],
            ]
        )
        occurrences[case.test_class] += 1
        execution = case.execution_id(recipe.conversion.project)
        reason = _exclusion(case, round_id is not None)
        observations.append(_observation(recipe, case, round_id, name, execution, reason))
        if reason is not None:
            exclusions[reason] += 1
        else:
            native.append(_native_record(case, cast(int, round_id), name, execution))
    return _ProjectedJob(observations, native, exclusions)


def _job_record(
    recipe: RTPTorrentETLRecipe, cases: list[ClassExecution], round_id: int | None, eligible: int
) -> dict[str, object]:
    """Retain source-job labels and absent logical-build/clock evidence, including fully excluded jobs."""
    status = "outside_replay_slice" if round_id is None else "no_compatible_reported_cases"
    return {
        "project": recipe.conversion.project,
        "job_id": cases[0].job_id,
        "logical_build_id": None,
        "modeled_round": round_id,
        "order_basis": recipe.order,
        "event_at": None,
        "outcome_available_at": None,
        "source_executions": len(cases),
        "reported_failed_class_executions": sum(case.outcome == "failed" for case in cases),
        "unknown_class_outcomes": sum(case.outcome == "unknown" for case in cases),
        "replay_executions": eligible,
        "replay_status": "eligible" if eligible else status,
    }


def project_rtptorrent(dataset: RTPTorrentDataset, recipe: RTPTorrentETLRecipe) -> RTPTorrentTables:
    """Keep the complete canonical raw history; project only declared jobs into explicit simulated rounds.

    Pure projection uses validated immutable source DTOs; it does not certify
    their cached-byte correspondence. The ETL entry point verifies that first.
    Source-job labels are never merged into logical builds or assigned observed
    timestamps. Unknown jobs/contradictory conversion specs raise ValueError.
    Local dictionaries exist only as grouping state and typed Polars row inputs.
    """
    dataset = RTPTorrentDataset.model_validate(dataset)
    recipe = RTPTorrentETLRecipe.model_validate(recipe)
    if dataset.spec != recipe.conversion:
        raise ValueError("RTPTorrent projection conversion identity disagrees")
    groups: dict[int, list[ClassExecution]] = defaultdict(list)
    for case in dataset.executions:
        groups[case.job_id].append(case)
    if set(recipe.replay_job_ids) - groups.keys():
        raise ValueError("RTPTorrent replay selects unknown source job labels")
    rounds = {job: index for index, job in enumerate(recipe.replay_job_ids, 1)}
    observations, jobs, native = [], [], []
    exclusions: Counter[str] = Counter()
    for job, cases in sorted(groups.items()):
        round_id = rounds.get(job)
        projected = _project_job(recipe, cases, round_id)
        observations.extend(projected.observations)
        native.extend(projected.native)
        jobs.append(_job_record(recipe, cases, round_id, len(projected.native)))
        exclusions.update(projected.exclusions)
    frames = (
        pl.DataFrame(observations, schema=_OBSERVATIONS),
        pl.DataFrame(jobs, schema=_JOBS),
        pl.DataFrame(native, schema=_NATIVE),
    )
    return RTPTorrentTables(*frames, fingerprint([frame.to_dicts() for frame in frames]), exclusions)
