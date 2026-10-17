"""Tiny authored historical-format fixtures; no foreign benchmark rows or code are bundled."""

import csv
import io
import json

from coleman.corpus.hcs_contracts import HCSRecipe
from coleman.corpus.materialization import SourceCache, materialize_source
from tests.corpus.test_source_views import source_spec


def hcs_recipe(*, rows_transform=lambda rows: rows, jobs_transform=lambda jobs: jobs):
    """Create sparse variants, an explicit retry, observed zero and two unavailable quantities."""
    jobs = []
    for job_id, pipeline, name, status in [
        (1, 101, "linux.x", "failed"),
        (2, 101, "linux_x", "failed"),
        (3, 101, "linux_x", "success"),
        (4, 202, "linux.x", "success"),
        (5, 202, "windows", "failed"),
        (6, 303, "outside", "success"),
        (7, 202, "slow", "canceled"),
    ]:
        sha = ("a" if pipeline == 101 else "b") * 40
        jobs.append(
            dict(
                id=job_id,
                name=name,
                status=status,
                commit=dict(id=sha),
                pipeline=dict(id=pipeline, sha=sha),
                started_at="2020-01-01T00:00:00Z",
                finished_at="2020-01-01T00:01:00Z",
                unrelated={"do_not_export": "owned synthetic metadata"},
            )
        )
    rows = []
    for job, values in zip(
        jobs,
        [
            ((0, 0), (2, 1), (3, 0)),
            ((1, 0), (2, 0), (3, 1)),
            ((9, 1),) * 3,
            ((0, 0), (4, 0), (2, 1)),
            ((1, 0), ("", 1), (2, "")),
            ((1, 0),),
            ((1, 1),),
        ],
        strict=True,
    ):
        for test, (duration, verdict) in zip(("A", "B", "C"), values, strict=False):
            rows.append(
                dict(
                    job_id=str(job["id"]),
                    sha=job["commit"]["id"],
                    pipeline_id=str(job["pipeline"]["id"]),
                    last_run="2020-01-01 00:00:00",
                    test_name=test,
                    verdict=str(verdict),
                    duration=str(duration),
                )
            )
    rows.append(rows[0].copy())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows_transform(rows))
    csv_bytes = stream.getvalue().encode()
    jobs_bytes = json.dumps(jobs_transform(jobs), sort_keys=True, separators=(",", ":")).encode()
    recipe = HCSRecipe(
        schema_version=1,
        project="owned-hcs",
        dataset_id="owned-hcs",
        executions=source_spec("executions.csv", csv_bytes),
        jobs=source_spec("jobs.json", jobs_bytes),
        pipeline_ids=(101, 202),
        native_costs="additive-binary32-rounded-reported-test-seconds-v1",
    )
    return recipe, csv_bytes, jobs_bytes


def prepare_hcs(tmp_path, **kwargs):
    """Use the actual licensed source cache; mocks never replace reader/projection/storage components."""
    recipe, csv_bytes, jobs_bytes = hcs_recipe(**kwargs)
    cache = SourceCache(tmp_path / "cache")
    materialize_source(recipe.executions, cache, lambda: io.BytesIO(csv_bytes))
    materialize_source(recipe.jobs, cache, lambda: io.BytesIO(jobs_bytes))
    return recipe, cache
