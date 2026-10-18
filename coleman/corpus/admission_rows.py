"""Normalize already verified observations for bounded descriptive admission profiles.

Only the existing source validators authorize these reads. Native source values
and historical adapters remain unchanged; analytical labels never become policy
features or invented chronology. Source-specific column names stay in the cards.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from coleman.corpus.admission_contracts import EvidencePayload, ObservationProfile
from coleman.corpus.admission_profiles import ProfileTable, profile_existing_native, profile_table
from coleman.corpus.admission_sources import source_description
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.lrts_contracts import MEANINGS
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.profiles import NativeSourceProfile
from coleman.corpus.rtptorrent_contracts import REPORTED_SECONDS
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_contracts import StressReceipt
from coleman.corpus.stress_world import generate_build
from coleman.pilot.contracts import SECONDS
from coleman.results.domain import canonical_json
from coleman.selection.contracts import UnitSemantics


def _groups(frame: pl.DataFrame, names: tuple[str, ...]) -> pl.Series:
    """Serialize complete group dimensions canonically; punctuation cannot collapse identities."""
    return pl.Series("group_id", [canonical_json(list(row)) for row in frame.select(names).iter_rows()])


def _labels(frame: pl.DataFrame, name: str) -> pl.Series:
    """Project known source outcomes only; errors/skips/unknowns remain unavailable for binary description."""
    return frame.select(
        pl.when(pl.col(name) == "passed")
        .then(0)
        .when(pl.col(name) == "failed")
        .then(1)
        .otherwise(None)
        .cast(pl.Int64)
        .alias("verdict")
    )["verdict"]


def _historical_table(payload: EvidencePayload, root: Path) -> ProfileTable:
    """Read fixed analytical roles after actual source/file verification, without native round pooling."""
    description = source_description(payload)
    frame = pl.read_parquet(root / "observations.parquet")
    if frame.height > 2_000_000:
        raise ValueError("Admission source profile exceeds two million retained observations")
    if isinstance(payload, (LRTSETLReceipt, LRTSStageReceipt)):
        group_fields = description.identity.group_fields
        groups = _groups(frame, group_fields)
        duration = frame["duration"].cast(pl.Float64)
        verdict = _labels(frame, "outcome")
        semantics = UnitSemantics(unit="s", meaning=MEANINGS["lrts_case_duration"])
    elif isinstance(payload, RTPTorrentETLReceipt):
        groups = _groups(frame, ("project", "job_id"))
        duration = frame["duration_decimal"].cast(pl.Float64)
        verdict = _labels(frame, "class_outcome")
        semantics = REPORTED_SECONDS
    elif isinstance(payload, HCSReceipt):
        groups = _groups(frame, ("project", "pipeline_id", "job_id"))
        duration = frame["duration_decimal"].cast(pl.Float64)
        verdict = frame["reported_nonpass"].cast(pl.Int64)
        semantics = payload.recipe.duration_semantics
    else:
        raise ValueError("Unsupported historical admission payload")
    normalized = pl.DataFrame(
        {
            "observation_id": frame["execution_id"],
            "group_id": groups,
            "duration": duration,
            "verdict": verdict,
        }
    )
    return ProfileTable(normalized, description.identity, semantics)


def _controlled_table(payload: ControlledDataset) -> ProfileTable:
    """Describe role observations in harness nanoseconds; partial/error pairs remain visible.

    Integer counters stay exact in the embedded source payload. This profile's
    duration projection is explicitly binary64 and is never used as runner work.
    A test error is not an assertion failure. No process or filesystem is used.
    """
    context = payload.spec.context
    rows = payload.observations
    frame = pl.DataFrame(
        {
            "observation_id": [row.execution_id(context) for row in rows],
            "group_id": [canonical_json([context.project, context.fault_id, row.trial_id]) for row in rows],
            "duration": [row.elapsed.value for row in rows],
            "verdict": [
                0 if row.outcome == "passed" else 1 if row.outcome == "assertion_failed" else None for row in rows
            ],
        },
        schema={"observation_id": pl.String, "group_id": pl.String, "duration": pl.Float64, "verdict": pl.Int64},
    )
    return ProfileTable(
        frame,
        source_description(payload).identity,
        UnitSemantics(unit="ns", meaning="observed-test-harness-monotonic-elapsed-v1"),
    )


def _stress_table(payload: StressReceipt) -> ProfileTable:
    """Regenerate one bounded case's constructed rows; seed/copy counts are never independent units.

    The profile bound is two million rows, deliberately narrower than the stress
    generator's largest sixteen-build case. Check the receipt count before any
    expansion. All default eight-build covering/adversarial cases fit this bound.
    """
    if payload.files[-1].rows > 2_000_000:
        raise ValueError("Stress case exceeds the admission profile's two-million-row bound")
    frames = []
    case_id = payload.case.fingerprint
    for build in range(payload.case.builds):
        rows = generate_build(payload.case, build)
        frames.append(
            pl.DataFrame(
                {
                    "observation_id": [canonical_json([case_id, row.build, row.test_id, row.variant]) for row in rows],
                    "group_id": [canonical_json([case_id, build])] * len(rows),
                    "duration": [row.duration_seconds for row in rows],
                    "verdict": [row.verdict for row in rows],
                },
                schema={
                    "observation_id": pl.String,
                    "group_id": pl.String,
                    "duration": pl.Float64,
                    "verdict": pl.Int64,
                },
            )
        )
    return ProfileTable(pl.concat(frames), source_description(payload).identity, SECONDS)


def describe_verified_source(
    payload: EvidencePayload, root: Path, *, constructed_history: bool = False
) -> ObservationProfile:
    """Profile a previously verified source snapshot; callers must revalidate bytes before publishing.

    Native profiles retain their original units and category counts. Other
    sources use their actual analytical roles or embedded typed observations.
    This performs bounded local reads/regeneration, never acquisition, subject
    execution, CI chronology reconstruction, candidate training or admission.
    Source/schema/Polars/I/O errors propagate without publishing completion.
    """
    if isinstance(payload, NativeSourceProfile):
        return profile_existing_native(payload)
    if isinstance(payload, ControlledDataset):
        table = _controlled_table(payload)
    elif isinstance(payload, StressReceipt):
        table = _stress_table(payload)
    else:
        table = _historical_table(payload, root)
    return profile_table(table._replace(constructed_history=constructed_history))
