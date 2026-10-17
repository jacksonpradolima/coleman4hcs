"""Pure sparse HCS projection; original pipelines, attempts, aliases and exclusions remain explicit."""

from __future__ import annotations

import struct
from collections import Counter
from typing import NamedTuple, cast

import polars as pl

from coleman.corpus.hcs_contracts import HCSAccounting, HCSObservation, HCSSlice, HCSVariant, variant_alias
from coleman.results.domain import fingerprint

_NATIVE = {
    "BuildId": pl.Int64,
    "Name": pl.String,
    "Duration": pl.Float64,
    "CalcPrio": pl.Int64,
    "LastRun": pl.String,
    "Verdict": pl.Int64,
    "SourceExecutionId": pl.String,
    "Variant": pl.String,
}
_OBSERVATIONS = {
    "execution_id": pl.String,
    "source_position": pl.Int64,
    "project": pl.String,
    "pipeline_id": pl.Int64,
    "job_id": pl.Int64,
    "commit_id": pl.String,
    "test_name": pl.String,
    "replay_name": pl.String,
    "variant_name": pl.String,
    "variant_alias": pl.String,
    "modeled_round": pl.Int64,
    "duration_lexeme": pl.String,
    "duration_decimal": pl.String,
    "duration_status": pl.String,
    "duration_unit": pl.String,
    "duration_meaning": pl.String,
    "native_duration_seconds": pl.Float64,
    "native_duration_meaning": pl.String,
    "reported_nonpass": pl.Int64,
    "last_run_label": pl.String,
    "job_started_at": pl.String,
    "job_finished_at": pl.String,
    "outcome_available_at": pl.String,
    "feature_available_at": pl.String,
    "disposition": pl.String,
}
_CELLS = {
    "pipeline_id": pl.Int64,
    "modeled_round": pl.Int64,
    "variant_name": pl.String,
    "variant_alias": pl.String,
    "metadata_jobs": pl.Int64,
    "parsed_rows": pl.Int64,
    "replay_rows": pl.Int64,
    "reported_nonpass_rows": pl.Int64,
    "cell_status": pl.String,
    "setup_seconds": pl.Float64,
    "shared_runner_occupancy": pl.Float64,
}


class HCSProjection(NamedTuple):
    """Serialization-boundary tables plus conserved typed counters; no caller dictionaries cross domain APIs."""

    tables: tuple[tuple[str, pl.DataFrame], ...]
    accounting: HCSAccounting
    variants: tuple[HCSVariant, ...]
    logical_digest: str


def _test_name(project: str, label: str) -> str:
    """Use one shared-label assumption across variants/rounds, with original labels retained analytically."""
    return "t_" + fingerprint(["hcs-shared-test-label-v1", project, label]).removeprefix("sha256:")


def _native_seconds(case: HCSObservation) -> float | None:
    """Explicitly round reported seconds to the existing bandit's binary32 cost scale.

    Carry the result exactly in binary64 Parquet/native-loader fields so WTS,
    VTS and the bandit share a cost definition. Original decimals remain in
    analytical evidence. Supported source bounds avoid overflow/underflow to
    zero; no missing quantity is filled. This is a modeled cost, not a new
    physical measurement or a change to any legacy bandit/input artifact.
    """
    return None if case.duration is None else struct.unpack("!f", struct.pack("!f", float(case.duration.value)))[0]


def _dispositions(source: HCSSlice) -> tuple[str, ...]:
    """Assign duplicate/status/attempt/outcome/duration precedence without examining policy performance."""
    seen: dict[tuple[int, int, str], HCSObservation] = {}
    first_attempt: dict[tuple[int, str], int] = {}
    for case in source.observations:
        if case.job.status in {"success", "failed"}:
            key = (case.job.pipeline_id, case.job.variant)
            first_attempt[key] = min(first_attempt.get(key, case.job.job_id), case.job.job_id)
    result = []
    for case in source.observations:
        previous = seen.get(case.key)
        if previous is not None:
            left = previous.model_dump(exclude={"source_position"})
            if left != case.model_dump(exclude={"source_position"}):
                raise ValueError("Conflicting HCS source execution identity")
            result.append("duplicate_rows")
            continue
        seen[case.key] = case
        if case.job.status not in {"success", "failed"}:
            reason = "unsupported_job_rows"
        elif first_attempt[(case.job.pipeline_id, case.job.variant)] != case.job.job_id:
            reason = "other_attempt_rows"
        elif case.verdict is None:
            reason = "unavailable_outcome_rows"
        elif case.duration is None:
            reason = "unavailable_duration_rows"
        else:
            reason = "replay_rows"
        result.append(reason)
    return tuple(result)


def _analytical(source: HCSSlice, case: HCSObservation, round_id: int, reason: str) -> dict[str, object]:
    """Flatten typed evidence at the Polars boundary; absent clocks and durations stay null."""
    job, recipe = case.job, source.recipe
    return {
        "execution_id": case.execution_id(recipe.project),
        "source_position": case.source_position,
        "project": recipe.project,
        "pipeline_id": job.pipeline_id,
        "job_id": job.job_id,
        "commit_id": job.commit_id,
        "test_name": case.test_name,
        "replay_name": _test_name(recipe.project, case.test_name),
        "variant_name": job.variant,
        "variant_alias": variant_alias(job.variant),
        "modeled_round": round_id,
        "duration_lexeme": case.duration_lexeme,
        "duration_decimal": None if case.duration is None else case.duration.model_dump(mode="json")["value"],
        "duration_status": "unavailable" if case.duration is None else "observed",
        "duration_unit": recipe.duration_semantics.unit,
        "duration_meaning": recipe.duration_semantics.meaning,
        "native_duration_seconds": _native_seconds(case),
        "native_duration_meaning": recipe.native_costs,
        "reported_nonpass": case.verdict,
        "last_run_label": case.last_run_label,
        "job_started_at": job.started_at,
        "job_finished_at": job.finished_at,
        "outcome_available_at": None,
        "feature_available_at": None,
        "disposition": reason,
    }


def _native(source: HCSSlice, case: HCSObservation, round_id: int) -> dict[str, object]:
    """Use declared binary32-rounded seconds, exactly represented in existing binary64 scheduling fields."""
    if case.duration is None or case.verdict is None:
        raise ValueError("Native HCS requires observed duration and reported outcome")
    return {
        "BuildId": round_id,
        "Name": _test_name(source.recipe.project, case.test_name),
        "Duration": _native_seconds(case),
        "CalcPrio": 0,
        "LastRun": "",
        "Verdict": case.verdict,
        "SourceExecutionId": case.execution_id(source.recipe.project),
        "Variant": variant_alias(case.job.variant),
    }


def _union(native: pl.DataFrame) -> pl.DataFrame:
    """Define a surrogate whole-system union, never additional observed executions or additive variant work."""
    rows = []
    for part in native.partition_by(["BuildId", "Name"], maintain_order=False):
        rows.append(
            {
                "BuildId": part["BuildId"][0],
                "Name": part["Name"][0],
                "Duration": part["Duration"].max(),
                "CalcPrio": 0,
                "LastRun": "",
                "Verdict": part["Verdict"].max(),
                "SourceExecutionId": fingerprint(sorted(part["SourceExecutionId"].to_list())),
                "Variant": "whole-system-surrogate",
            }
        )
    return pl.DataFrame(rows, schema=_NATIVE).sort("BuildId", "Name")


def _require_stable_cost_sums(native: pl.DataFrame) -> None:
    """Reject dynamic ranges whose nonnegative projected prefix sums can exceed binary64 integer precision.

    Binary32 costs are dyadic rationals. Express them on their finest common
    power-of-two lattice. If the total integer coefficient fits 53 bits, every
    nonnegative partial sum also fits and is exactly representable in binary64,
    regardless of test order or the native summation implementation. This is a
    supported-input condition, not a tolerance or a decimal arithmetic claim.
    V1 rejects unsupported ranges rather than omitting tests at a full budget.
    """
    for part in native.partition_by(["BuildId", "Variant"], maintain_order=False):
        ratios = [float(value).as_integer_ratio() for value in part["Duration"]]
        scale = max(denominator for _, denominator in ratios)
        ticks = sum(numerator * (scale // denominator) for numerator, denominator in ratios)
        if ticks.bit_length() > 53:
            raise ValueError("HCS projected cost range cannot guarantee stable native prefix sums")


def _cells(source: HCSSlice, variants: tuple[HCSVariant, ...], native: pl.DataFrame) -> pl.DataFrame:
    """Keep absent parsed cells visible, distinguishing metadata-only jobs from observed all-pass tests."""
    counts = Counter((c.job.pipeline_id, c.job.variant) for c in source.observations)
    jobs = Counter((j.pipeline_id, j.variant) for j in source.jobs)
    native_groups: dict[tuple[int, str], pl.DataFrame] = {}
    for part in native.partition_by(["BuildId", "Variant"], maintain_order=False):
        native_groups[int(part["BuildId"][0]), str(part["Variant"][0])] = part
    rows = []
    for round_id, pipeline in enumerate(source.recipe.pipeline_ids, 1):
        for variant in variants:
            part = native_groups.get((round_id, variant.alias))
            parsed = counts[pipeline, variant.name]
            rows.append(
                {
                    "pipeline_id": pipeline,
                    "modeled_round": round_id,
                    "variant_name": variant.name,
                    "variant_alias": variant.alias,
                    "metadata_jobs": jobs[pipeline, variant.name],
                    "parsed_rows": parsed,
                    "replay_rows": 0 if part is None else part.height,
                    "reported_nonpass_rows": None if part is None else part["Verdict"].sum(),
                    "cell_status": "eligible" if part is not None else "excluded" if parsed else "absent_parsed_source",
                    "setup_seconds": None,
                    "shared_runner_occupancy": None,
                }
            )
    return pl.DataFrame(rows, schema=_CELLS)


def project_hcs(source: HCSSlice) -> HCSProjection:
    """Project one verified selected slice into WTS and independent-state VTS inputs.

    Full raw byte verification belongs to read_hcs_slice/ETL. Missing cells are
    not filled with passing outcomes or zero setup. Conflicting duplicate keys
    reject; identical duplicate physical rows and other eligible attempts receive
    explicit primary exclusions. Original variant names map bijectively to safe
    aliases; they are never sanitized into possibly colliding legacy labels.
    Native union names cover every native variant test in each round by
    construction. Per-cell suites above 1,000 tests and >64 variants reject.
    No policy, runner, transport, statistics, file writes or source execution.
    """
    source = HCSSlice.model_validate(source)
    reasons = _dispositions(source)
    variants = tuple(
        HCSVariant(name=n, alias=variant_alias(n)) for n in sorted({c.job.variant for c in source.observations})
    )
    if len(variants) > 64 or len({v.alias for v in variants}) != len(variants):
        raise ValueError("HCS variant bounds or digest collision")
    rounds = {pipeline: i for i, pipeline in enumerate(source.recipe.pipeline_ids, 1)}
    analytical, native_rows = [], []
    for case, reason in zip(source.observations, reasons, strict=True):
        round_id = rounds[case.job.pipeline_id]
        analytical.append(_analytical(source, case, round_id, reason))
        if reason == "replay_rows":
            native_rows.append(_native(source, case, round_id))
    observations = pl.DataFrame(analytical, schema=_OBSERVATIONS).sort(
        "pipeline_id", "job_id", "test_name", "source_position"
    )
    native = pl.DataFrame(native_rows, schema=_NATIVE).sort("BuildId", "Variant", "Name")
    union = _union(native)
    if union.height and cast(int, union.group_by("BuildId").len()["len"].max()) > 1000:
        raise ValueError("HCS native suite exceeds 1000 tests")
    _require_stable_cost_sums(native)
    _require_stable_cost_sums(union)
    cells = _cells(source, variants, native)
    prefix = f"datasets/{source.recipe.dataset_id}/"
    tables = [
        ("observations.parquet", observations),
        ("cells.parquet", cells),
        (prefix + "features-engineered.parquet", union),
        (prefix + "data-variants.parquet", native),
    ]
    tables.extend(
        (f"vts/{v.alias}/features-engineered.parquet", native.filter(pl.col("Variant") == v.alias)) for v in variants
    )
    counts = Counter(reasons)
    accounting = HCSAccounting(
        input_rows=source.input_rows,
        outside_slice=source.outside_slice,
        selected_rows=len(source.observations),
        duplicate_rows=counts["duplicate_rows"],
        unsupported_job_rows=counts["unsupported_job_rows"],
        other_attempt_rows=counts["other_attempt_rows"],
        unavailable_outcome_rows=counts["unavailable_outcome_rows"],
        unavailable_duration_rows=counts["unavailable_duration_rows"],
        replay_rows=counts["replay_rows"],
    )
    return HCSProjection(
        tuple(tables), accounting, variants, fingerprint([[path, frame.to_dicts()] for path, frame in tables])
    )
