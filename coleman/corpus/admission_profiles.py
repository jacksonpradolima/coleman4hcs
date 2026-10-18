"""Bounded descriptive profiles reusing the native inverse-empirical-CDF recipe.

Normalized tables are an internal analytical boundary, never decision features.
Reported decimal seconds and integer harness ticks remain exact in their source
receipts; these distribution summaries explicitly project them to binary64.
No missing clock, test continuity, unique fault or physical saving is inferred.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import polars as pl

from coleman.corpus.admission_contracts import ObservationProfile
from coleman.corpus.contracts import ObservationIdentity
from coleman.corpus.profiles import NativeSourceProfile, _numbers
from coleman.results.domain import Measurement
from coleman.selection.contracts import UnitSemantics


class ProfileTable(NamedTuple):
    """Internal normalized observations, with explicit source identity/group and duration units.

    Required columns are observation_id, group_id, duration and verdict; verdict
    is binary or null, never a calibrated probability. Source-specific verifiers
    establish all input bytes before constructing this table. A group names one
    source execution group, not an independent experimental unit.
    """

    frame: pl.DataFrame
    identity: ObservationIdentity
    duration: UnitSemantics
    constructed_history: bool = False


def _measurement(
    name: str,
    value: float | int | None,
    meaning: str,
    unit: str = "1",
    *,
    missing: Literal["unavailable", "not_applicable"] = "unavailable",
) -> Measurement:
    """Declare observed zero versus unavailable null; descriptive quantities have no objective direction."""
    return Measurement(
        name=name,
        value=None if value is None else float(value),
        unit=unit,
        meaning=meaning,
        direction="none",
        minimum=0,
        status="observed" if value is not None else missing,
    )


def _profile_schema(frame: pl.DataFrame) -> None:
    """Require a bounded normalized schema before descriptive expressions run; booleans never become durations."""
    if frame.height > 2_000_000 or set(frame.columns) != {"observation_id", "group_id", "duration", "verdict"}:
        raise ValueError("Admission profile requires four normalized columns within its row bound")
    if frame.schema["observation_id"] != pl.String or frame.schema["group_id"] != pl.String:
        raise ValueError("Admission observation/group identifiers require strings")
    if frame.schema["verdict"] != pl.Int64:
        raise ValueError("Admission verdicts require explicit binary Int64 or null")
    if frame.schema["duration"] not in {pl.Float64, pl.Float32, pl.Int64}:
        raise ValueError("Admission durations require an explicit numeric projection, never booleans or strings")


def profile_table(table: ProfileTable) -> ObservationProfile:
    """Describe at most two million normalized rows without imputing unavailable measurements.

    Reuses the existing profiler's inverse-CDF quantiles and signed/nonfinite
    categories. Its binary64 projection is named explicitly, especially for
    nanosecond counters above 2**53. Future clocks and ordering assumptions are
    not supplied, so chronological churn and CI span stay unavailable. Invalid
    columns, verdicts, bounds or group identities raise ValueError; no I/O occurs.
    """
    frame = table.frame
    _profile_schema(frame)
    if frame.filter(pl.col("verdict").is_not_null() & ~pl.col("verdict").is_in([0, 1])).height:
        raise ValueError("Admission profile contains a nonbinary verdict")
    identified = frame.filter(pl.col("observation_id").is_not_null())
    if identified.filter(pl.col("group_id").is_null()).height:
        raise ValueError("Identified admission observations require their complete source group")
    unique = identified["observation_id"].n_unique()
    conflicts = identified.group_by("observation_id").agg(
        pl.struct(pl.exclude("observation_id")).n_unique().alias("distinct_payloads")
    )
    conflicts_count = conflicts.filter(pl.col("distinct_payloads") > 1).height
    failed = frame.filter(pl.col("verdict") == 1).height
    passed = frame.filter(pl.col("verdict") == 0).height
    known = failed + passed
    durations = frame["duration"].cast(pl.Float64, strict=True)
    projected = UnitSemantics(unit=table.duration.unit, meaning="binary64-description:" + table.duration.meaning)
    missing = "not_applicable" if table.constructed_history else "unavailable"
    measurements = (
        _measurement("source_failed_rows", failed, "known-source-binary-one-row-count-v1"),
        _measurement("source_passed_rows", passed, "known-source-binary-zero-row-count-v1"),
        _measurement("source_unknown_outcomes", frame.height - known, "unavailable-source-binary-outcome-rows-v1"),
        _measurement(
            "source_failure_fraction",
            failed / known if known else None,
            "descriptive-known-source-binary-label-fraction-v1",
        ),
        _measurement("source_observed_ci_span", None, "verified-historical-ci-time-span-v1", "s", missing=missing),
        _measurement("source_observed_churn", None, "verified-continuous-logical-test-churn-v1", missing=missing),
    )
    return ObservationProfile(
        rows=frame.height,
        unique_observations=unique,
        duplicate_rows=identified.height - unique,
        missing_identity_rows=frame.height - identified.height,
        conflicting_identities=conflicts_count,
        source_groups=frame["group_id"].drop_nulls().n_unique(),
        identity=table.identity,
        execution_group_fields=table.identity.group_fields,
        duration=_numbers(durations, projected),
        measurements=measurements,
    )


def profile_existing_native(profile: NativeSourceProfile) -> ObservationProfile:
    """Reuse an existing frozen native profile without changing raw units, verdicts or chronology.

    This does not verify source bytes: callers must use profile_native_source
    against the exact registered artifact first. All old profile fields and
    invalid/null verdict counts retain their original meaning.
    """
    known = profile.failed_rows + profile.passed_rows
    missing = "not_applicable" if profile.card.history == "constructed" else "unavailable"
    return ObservationProfile(
        rows=profile.rows,
        unique_observations=profile.unique_observations,
        duplicate_rows=profile.duplicate_identity_rows,
        missing_identity_rows=profile.missing_identity_rows,
        conflicting_identities=profile.conflicting_identities,
        source_groups=profile.execution_groups,
        identity=profile.card.manifest.identity,
        execution_group_fields=("BuildId", "Variant") if "Variant" in profile.columns else ("BuildId",),
        duration=profile.raw_duration,
        measurements=(
            _measurement("source_failed_rows", profile.failed_rows, "raw-native-binary-one-row-count-v1"),
            _measurement("source_passed_rows", profile.passed_rows, "raw-native-binary-zero-row-count-v1"),
            _measurement(
                "source_unknown_outcomes",
                profile.null_verdict_rows + profile.invalid_verdict_rows,
                "null-or-invalid-raw-native-verdict-row-count-v1",
            ),
            _measurement(
                "source_failure_fraction",
                profile.failed_rows / known if known else None,
                "descriptive-known-raw-native-binary-label-fraction-v1",
            ),
            _measurement("source_observed_ci_span", None, "verified-historical-ci-time-span-v1", "s", missing=missing),
            _measurement("source_observed_churn", None, "verified-continuous-logical-test-churn-v1", missing=missing),
        ),
    )
