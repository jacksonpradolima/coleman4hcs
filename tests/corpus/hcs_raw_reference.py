"""Independent original-byte reference for the frozen public LibSSH HCS qualification.

This test helper reads source CSV/JSON and native result rows, never converter,
projection, metric or scheduler functions. It deliberately supports only the
published first-sixteen-pipeline protocol, Random/UCB, and ratio budgets 0.5/1.0.
It is a functional accounting reference, not source admission or inference.
"""

import csv
import hashlib
import json
import struct
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import polars as pl

from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.materialization import SourceCache


def _blob(cache, package):
    """Verify original pinned bytes through the shared cache before independent parsing."""
    source = package.source
    assert source.sha256 is not None and source.size_bytes is not None
    return cache.verify_blob(source.sha256, source.size_bytes)


def _test_name(project, label):
    """Implement the documented shared-label identity using only standard-library JSON/SHA."""
    raw = json.dumps(["hcs-shared-test-label-v1", project, label], separators=(",", ":"), ensure_ascii=False)
    return "t_" + hashlib.sha256(raw.encode()).hexdigest()


def _source_cases(cache, receipt):
    """Stream the original CSV; retain only the frozen cohort and original per-job quantities."""
    with _blob(cache, receipt.recipe.jobs).open() as stream:
        jobs = {job["id"]: job for job in json.load(stream)}
    selected, total, pipelines = [], 0, set()
    with _blob(cache, receipt.recipe.executions).open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            total += 1
            pipelines.add(int(row["pipeline_id"]))
            if int(row["pipeline_id"]) in receipt.recipe.pipeline_ids:
                selected.append(row)
    assert total == 278950 and len(selected) == 885
    assert receipt.recipe.pipeline_ids == tuple(sorted(pipelines)[:16])
    first = {}
    for row in selected:
        job = jobs[int(row["job_id"])]
        assert job["pipeline"]["id"] == int(row["pipeline_id"])
        assert job["pipeline"]["sha"] == job["commit"]["id"] == row["sha"]
        key = int(row["pipeline_id"]), job["name"]
        if job["status"] in {"success", "failed"}:
            first[key] = min(first.get(key, job["id"]), job["id"])
    retained = [
        row
        for row in selected
        if int(row["job_id"]) == first[(int(row["pipeline_id"]), jobs[int(row["job_id"])]["name"])]
    ]
    assert len(retained) == 856
    assert len({(row["pipeline_id"], row["job_id"], row["test_name"]) for row in retained}) == 856
    variants = defaultdict(dict)
    union = defaultdict(dict)
    for row in retained:
        pipeline = int(row["pipeline_id"])
        variant = jobs[int(row["job_id"])]["name"]
        name = _test_name(receipt.recipe.project, row["test_name"])
        cost, verdict = Decimal(row["duration"]), int(row["verdict"])
        assert cost.is_finite() and cost >= 0 and verdict in {0, 1}
        variants[pipeline, variant][name] = cost, verdict
        previous = union[pipeline].get(name, (Decimal(0), 0))
        union[pipeline][name] = max(previous[0], cost), max(previous[1], verdict)
    assert len(variants) == 44 and sum(len(cases) for cases in union.values()) == 320
    return variants, union


def _check_row(cases, row):
    """Recompute projected prefixes/denominators; report decimal boundary differences without tolerance."""
    assert row["budget_mode"] == "ratio" and row["experiment"] == 1
    assert row["reward_function"] == "Reward Based on Failures" and row["step"] >= 1
    order = json.loads(row["prioritization_order_top_k"])
    assert len(order) == len(set(order)) == len(cases) and set(order) == cases.keys()
    projected = {name: struct.unpack("!f", struct.pack("!f", float(case[0])))[0] for name, case in cases.items()}
    total = sum(projected.values())
    assert total == row["total_build_duration"]
    original_total = sum((Fraction(case[0]) for case in cases.values()), Fraction(0))
    elapsed, original_elapsed = 0.0, Fraction(0)
    selected, original_selected = [], []
    for name in order:
        elapsed += projected[name]
        original_elapsed += Fraction(cases[name][0])
        if elapsed <= total * row["budget_value"]:
            selected.append(name)
        if original_elapsed <= original_total * Fraction(str(row["budget_value"])):
            original_selected.append(name)
    detected = sum(cases[name][1] for name in selected)
    nonpass = sum(case[1] for case in cases.values())
    assert row["tests_ran"] == len(selected)
    assert row["tests_not_ran"] == len(cases) - len(selected)
    assert row["detected"] == detected and row["missed"] == nonpass - detected
    return selected != original_selected


def verify_libssh_raw_reference(cache_root, completion, runs, working_directory):
    """Verify the declared real default matrix against original sources; return deterministic counts.

    Paths from the installed child are resolved against its actual working
    directory. The frozen source pins/protocol are required; this is not a generic
    reader or a claimed oracle for other sources. Assertions are intentional test
    failures. No network, principal-component mock or source mutation occurs.
    """
    working_directory = Path(working_directory)
    receipt = HCSReceipt.from_json((working_directory / completion).read_bytes())
    assert receipt.recipe.project == "libssh" and len(receipt.recipe.pipeline_ids) == 16
    assert receipt.recipe.executions.source.sha256 == (
        "sha256:ba50a92a4571fec2b520f1155f1f8a8377236d44d95f3d6c683726bd9a2968c6"
    )
    assert receipt.recipe.jobs.source.sha256 == (
        "sha256:19b8652f902c20706fb32b7e9bcf3cb4859c701b6a900cd3954152c563454cd0"
    )
    variants, union = _source_cases(SourceCache(cache_root), receipt)
    aliases = {variant.alias: variant.name for variant in receipt.variants}
    assert len(aliases) == len(set(aliases.values())) == 7
    assert set(aliases.values()) == {variant for _, variant in variants}
    run_results, differences = [], []
    report = dict(
        schema_version=1,
        receipt=receipt.fingerprint,
        method="independent-original-csv-json-binary32-prefix-v1",
        source_rows=278950,
        retained_rows=856,
        other_attempt_rows=29,
        union_rows=320,
        active_cells=44,
        declared_cells=112,
        runs=run_results,
        decimal_boundary_differences=differences,
    )
    assert {run["strategy"] for run in runs} == {"wts", "vts"} and len(runs) == 2
    for run in sorted(runs, key=lambda run: run["strategy"]):
        rows = pl.read_parquet(
            str(working_directory / run["artifacts_dir"] / "results/**/*.parquet"), hive_partitioning=True
        )
        assert set(rows["budget_value"]) == {0.5, 1.0}
        assert set(rows["policy"]) == {"Random", "UCB (C=1.0)"}
        expected_cells = {(pipeline, None) for pipeline in union} if run["strategy"] == "wts" else set()
        expected_cells |= set(variants)
        keys = set()
        for row in rows.iter_rows(named=True):
            if run["strategy"] == "wts":
                assert row["scenario"] == receipt.recipe.dataset_id
                pipeline = receipt.recipe.pipeline_ids[row["step"] - 1]
                variant = None if row["variant"] is None else aliases[row["variant"]]
                cases = union[pipeline] if variant is None else variants[pipeline, variant]
            else:
                assert row["variant"] is None
                variant = aliases[row["scenario"]]
                pipeline = sorted(p for p, name in variants if name == variant)[row["step"] - 1]
                cases = variants[pipeline, variant]
            key = pipeline, variant, row["policy"], row["budget_value"]
            assert key not in keys
            keys.add(key)
            if _check_row(cases, row):
                differences.append(
                    dict(
                        strategy=run["strategy"],
                        pipeline=pipeline,
                        variant=variant,
                        policy=row["policy"],
                        ratio=row["budget_value"],
                    )
                )
        assert keys == {
            (p, v, policy, ratio)
            for p, v in expected_cells
            for policy in ("Random", "UCB (C=1.0)")
            for ratio in (0.5, 1.0)
        }
        run_results.append(dict(strategy=run["strategy"], rows=rows.height, native_run_id=run["run_id"]))
    differences.sort(
        key=lambda row: (row["strategy"], row["pipeline"], row["variant"] or "", row["policy"], row["ratio"])
    )
    return report
