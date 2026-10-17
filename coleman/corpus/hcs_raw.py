"""Bounded offline reader for publisher GitLab execution CSV and matching original job JSON."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import cast

from coleman.corpus.hcs_contracts import HCS_SECONDS, HCSJob, HCSObservation, HCSRecipe, HCSSlice
from coleman.corpus.materialization import SourceCache
from coleman.corpus.materialization_contracts import MaterializationReceipt, MaterializationSpec
from coleman.results.domain import decode_json
from coleman.selection.contracts import Quantity

_FIELDS = ("job_id", "sha", "pipeline_id", "last_run", "test_name", "verdict", "duration")


def _source_path(package: MaterializationSpec, cache: SourceCache) -> Path:
    """Rehash one raw source before decoding; logical member labels never execute as paths."""
    return cache.member_path(MaterializationReceipt(schema_version=1, spec=package), package.members[0].name)


def _integer(value: object) -> int:
    """Read positive bounded source IDs without booleans, fractional truncation or missing coercion."""
    if type(value) is int:
        number = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9]\d{0,18}", value, flags=re.ASCII):
        number = int(value)
    else:
        raise ValueError("HCS source ID must be a positive integer")
    if not 1 <= number <= 2**63 - 1:
        raise ValueError("HCS source ID exceeds native signed 64-bit bound")
    return number


def _object(value: object) -> dict[str, object]:
    """Narrow a decoded JSON object only at the source parsing boundary."""
    if not isinstance(value, dict):
        raise ValueError("HCS job metadata requires JSON objects")
    return cast(dict[str, object], value)


def _job(value: dict[str, object]) -> HCSJob:
    """Extract only required metadata and prove both pipeline and commit links before use."""
    pipeline, commit = _object(value.get("pipeline")), _object(value.get("commit"))
    sha = commit.get("id")
    if not isinstance(sha, str):
        raise ValueError("HCS job requires source commit metadata")
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None or pipeline.get("sha") != sha:
        raise ValueError("HCS pipeline/commit identity disagrees")
    return HCSJob.model_validate(
        {
            "job_id": _integer(value.get("id")),
            "pipeline_id": _integer(pipeline.get("id")),
            "commit_id": sha,
            "variant": value.get("name"),
            "status": value.get("status"),
            "started_at": value.get("started_at"),
            "finished_at": value.get("finished_at"),
        }
    )


def _jobs(recipe: HCSRecipe, cache: SourceCache) -> dict[int, HCSJob]:
    """Read at most 16,384 original jobs; duplicate source IDs reject even outside the selected slice."""
    with _source_path(recipe.jobs, cache).open("rb") as stream:
        payload = stream.read(64 * 1024**2 + 1)
    if len(payload) > 64 * 1024**2:
        raise ValueError("HCS job JSON exceeds 64 MiB")
    values = decode_json(payload)
    if not isinstance(values, list) or len(values) > 16384:
        raise ValueError("HCS requires at most 16384 source job records")
    selected, seen = {}, set()
    for value in values:
        value = _object(value)
        job_id = _integer(value.get("id"))
        if job_id in seen:
            raise ValueError("Duplicate HCS source job metadata")
        seen.add(job_id)
        pipeline = _object(value.get("pipeline"))
        if _integer(pipeline.get("id")) in recipe.pipeline_ids:
            selected[job_id] = _job(value)
    if {job.pipeline_id for job in selected.values()} != set(recipe.pipeline_ids):
        raise ValueError("Selected HCS pipeline absent from original job metadata")
    commits: dict[int, str] = {}
    for job in selected.values():
        if commits.setdefault(job.pipeline_id, job.commit_id) != job.commit_id:
            raise ValueError("Conflicting HCS commits in one source pipeline")
    return selected


def _observation(row: dict[str, str], position: int, jobs: dict[int, HCSJob]) -> HCSObservation:
    """Join exact raw identities; empty source quantities stay unavailable and invalid numbers reject."""
    job = jobs.get(_integer(row["job_id"]))
    if job is None or (job.pipeline_id, job.commit_id) != (_integer(row["pipeline_id"]), row["sha"]):
        raise ValueError("HCS execution/job/pipeline/commit correspondence failed")
    verdict = row["verdict"]
    if verdict not in {"", "0", "1"}:
        raise ValueError("Unsupported HCS reported nonpass verdict")
    duration = row["duration"]
    if len(duration) > 128:
        raise ValueError("HCS duration lexeme exceeds 128 characters")
    amount = None if duration == "" else Quantity.model_validate({"value": duration, "semantics": HCS_SECONDS})
    if amount is not None and amount.value > 10**12:
        raise ValueError("HCS source duration exceeds 10^12 seconds")
    return HCSObservation(
        source_position=position,
        job=job,
        test_name=row["test_name"],
        last_run_label=row["last_run"],
        duration_lexeme=duration,
        duration=amount,
        verdict=None if verdict == "" else int(verdict),
    )


def read_hcs_slice(recipe: HCSRecipe, cache: SourceCache) -> HCSSlice:
    """Verify full pinned source bytes, stream up to 500,000 CSV rows and retain the frozen selected slice.

    Selected rows are capped at 16,000 and 1,000 per native suite later. Outside
    rows are counted rather than silently admitted or decoded as native cases;
    original bytes remain lossless in the cache. All rows require the exact
    seven-column format, finite field sizes and numeric pipeline identity.
    Missing metadata, wrong joins, oversized/malformed source and invalid selected
    quantities raise ValueError; decoding/I/O errors propagate. No network,
    clocks, subprocesses, CLI or MCP imports occur. Unknown historical source
    omissions and collapsed outcomes cannot be recovered by this reader.
    """
    recipe = HCSRecipe.model_validate(recipe)
    jobs = _jobs(recipe, cache)
    observations = []
    total = outside = 0
    with _source_path(recipe.executions, cache).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";", strict=True)
        if tuple(reader.fieldnames or ()) != _FIELDS:
            raise ValueError("Unsupported HCS execution CSV schema")
        for total, row in enumerate(reader, 1):
            if total > 500000 or None in row or any(v is None or len(v) > 1024 for v in row.values()):
                raise ValueError("HCS execution row/field bounds exceeded")
            if _integer(row["pipeline_id"]) not in recipe.pipeline_ids:
                outside += 1
                continue
            if len(observations) >= 16000:
                raise ValueError("HCS selected execution slice exceeds 16000 rows")
            observations.append(_observation(row, total, jobs))
    return HCSSlice(
        recipe=recipe,
        input_rows=total,
        outside_slice=outside,
        observations=tuple(observations),
        jobs=tuple(sorted(jobs.values(), key=lambda j: j.job_id)),
    )
