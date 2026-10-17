"""Pure typed LRTS stage correspondence and analytical/native serialization boundaries."""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import polars as pl

from coleman.corpus.lrts_contracts import LRTSDataset
from coleman.corpus.lrts_etl import _BUILDS, _NATIVE, _OBSERVATIONS, _case_record, _exclusion
from coleman.corpus.lrts_etl_contracts import ETLMetadataAccounting
from coleman.corpus.lrts_stage_contracts import LRTSStageRecipe
from coleman.results.domain import canonical_json, fingerprint


class LRTSStageTables(NamedTuple):
    """Explicit fixed-schema Parquet boundary; original source DTOs remain authoritative."""

    observations: pl.DataFrame
    builds: pl.DataFrame
    native: pl.DataFrame
    source_datasets: tuple[str, ...]
    logical_digest: str
    exclusions: Counter[str]
    metadata_accounting: ETLMetadataAccounting


def _common_metadata(dataset: LRTSDataset) -> float:
    """Reject conflicting repeated build fields; retain stage-specific counts/measurements separately."""
    stage_fields = {
        "stage_id",
        "processed_pass_classes",
        "processed_fail_classes",
        "processed_transition_classes",
        "processed_pass_methods",
        "processed_fail_methods",
        "measurements",
    }
    common = []
    for row in dataset.metadata:
        value = row.model_dump(mode="json", exclude=stage_fields)
        value["build_measurements"] = [
            row.measurement(name).model_dump(mode="json")
            for name in ("lrts_build_execution_duration", "lrts_scheduled_epoch_seconds", "lrts_report_duration")
        ]
        common.append(canonical_json(value))
    if len(set(common)) != 1:
        raise ValueError("LRTS stages contradict repeated build metadata")
    scheduled = dataset.metadata[0].measurement("lrts_scheduled_epoch_seconds").value
    if scheduled is None:
        raise ValueError("LRTS stage replay requires a reported scheduled label")
    return scheduled


def _stage_map(dataset: LRTSDataset, recipe: LRTSStageRecipe) -> dict[int, str]:
    """Verify exact metadata stages and the complete raw context/node relation, including empty suites."""
    bindings = [
        b
        for b in recipe.stage_bindings
        if (b.pull_request, b.build_id) == (dataset.spec.pull_request, dataset.spec.build_id)
    ]
    if {b.stage_id for b in bindings} != {m.stage_id for m in dataset.metadata}:
        raise ValueError("LRTS stage bindings disagree with selected metadata stages")
    declared = {(b.enclosing_block_names, node): b.stage_id for b in bindings for node in b.node_ids}
    observed = set()
    result = {}
    for suite in dataset.suites:
        key = (suite.enclosing_block_names, suite.node_id)
        if suite.enclosing_block_names is None or suite.node_id is None or key not in declared:
            raise ValueError("LRTS raw context/node relation disagrees with explicit stage bindings")
        observed.add(key)
        result[suite.position] = declared[(suite.enclosing_block_names, suite.node_id)]
    if observed != declared.keys():
        raise ValueError("LRTS raw context/node relation disagrees with explicit stage bindings")
    return result


class _Rows(NamedTuple):
    """Per-build serialization records, created only at the typed table boundary."""

    observations: list[dict[str, object]]
    native: list[dict[str, object]]
    exclusions: Counter[str]


def _rows(dataset: LRTSDataset, recipe: LRTSStageRecipe, round_id: int) -> _Rows:
    """Assign stage-qualified occurrences before eligibility and preserve every original execution ID."""
    mapping = _stage_map(dataset, recipe)
    observations: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    group = fingerprint(
        ["lrts-source-build-group-v1", dataset.spec.project, dataset.spec.pull_request, dataset.spec.build_id]
    )
    for suite in dataset.suites:
        stage_id = mapping[suite.position]
        for case in suite.cases:
            label = fingerprint(
                [dataset.spec.project, stage_id, suite.name, suite.reported_id, case.class_name, case.method_name]
            )
            name = fingerprint([recipe.test_identity, label, occurrences[label]])
            occurrences[label] += 1
            execution = case.execution_id(
                dataset.spec.project, dataset.spec.pull_request, dataset.spec.build_id, suite.position
            )
            reason = "outside_stage" if stage_id != recipe.native_stage_id else _exclusion(case)
            record = _case_record(dataset, suite, case, round_id, name, execution, reason)
            record.update(stage_id=stage_id, source_build_group=group)
            observations.append(record)
            if reason is not None:
                exclusions[reason] += 1
            else:
                native.append(
                    {
                        "BuildId": round_id,
                        "Name": name,
                        "Duration": case.duration.value,
                        "CalcPrio": 0,
                        "LastRun": "",
                        "Verdict": int(case.outcome == "failed"),
                        "SourceExecutionId": execution,
                    }
                )
    return _Rows(observations, native, exclusions)


def project_lrts_stages(recipe: LRTSStageRecipe, datasets: tuple[LRTSDataset, ...]) -> LRTSStageTables:
    """Project complete verified source DTOs; no acquisition, clocks, writes or runner imports.

    Full raw observations and all processed stage records survive selection of
    one modeled native stage. Counts conserve primary exclusions; reported zero
    seconds and unavailable durations remain distinct. Repeated build-level
    fields must agree across stages. Order uses scheduled labels only, never
    inferred execution/outcome/feature clocks. Invalid mappings/order/budgets
    raise ValueError. Caller must verify enclosing source integrity separately.
    """
    recipe = LRTSStageRecipe.model_validate(recipe)
    source = recipe.source
    if len(datasets) != len(source.builds):
        raise ValueError("LRTS stage selected dataset count disagrees")
    observations, builds, native = [], [], []
    source_datasets = []
    exclusions: Counter[str] = Counter()
    prior = None
    total = 0
    for round_id, (selection, dataset) in enumerate(zip(source.builds, datasets, strict=True), 1):
        if (dataset.spec.project, dataset.spec.pull_request, dataset.spec.build_id, dataset.spec.history) != (
            source.project,
            selection.pull_request,
            selection.build_id,
            source.history,
        ):
            raise ValueError("LRTS stage source identity disagrees with recipe")
        scheduled = _common_metadata(dataset)
        key = (scheduled, selection.pull_request, selection.build_id)
        if prior is not None and key <= prior:
            raise ValueError("LRTS stage source order contradicts scheduled-label declaration")
        prior = key
        total += len(dataset.cases)
        if total > source.maximum_cases:
            raise ValueError("LRTS stage total source-case limit exceeded")
        rows = _rows(dataset, recipe, round_id)
        source_identity = dataset.fingerprint
        source_datasets.append(source_identity)
        observations.extend(rows.observations)
        native.extend(rows.native)
        exclusions.update(rows.exclusions)
        builds.append(
            {
                "round": round_id,
                "project": source.project,
                "pull_request": selection.pull_request,
                "source_build_id": selection.build_id,
                "scheduled_label_epoch_seconds": scheduled,
                "outcome_available_at": None,
                "metadata_json": canonical_json([m.model_dump(mode="json") for m in dataset.metadata]),
                "source_dataset_sha256": source_identity,
                "replay_cases": len(rows.native),
                "replay_status": "eligible" if rows.native else "no_compatible_reported_cases",
                "source_stages": len(dataset.metadata),
                "native_stage_id": recipe.native_stage_id,
                "source_build_group": fingerprint(
                    ["lrts-source-build-group-v1", source.project, selection.pull_request, selection.build_id]
                ),
            }
        )
    frames = (
        pl.DataFrame(observations, schema={**_OBSERVATIONS, "stage_id": pl.String, "source_build_group": pl.String}),
        pl.DataFrame(
            builds,
            schema={
                **_BUILDS,
                "source_stages": pl.Int64,
                "native_stage_id": pl.String,
                "source_build_group": pl.String,
            },
        ),
        pl.DataFrame(native, schema=_NATIVE),
    )
    selected = sum(len(d.metadata) for d in datasets)
    duplicates = sum(d.metadata_duplicates_excluded for d in datasets)
    input_rows = datasets[0].metadata_input_rows
    if any(d.metadata_input_rows != input_rows for d in datasets):
        raise ValueError("LRTS stage datasets disagree on physical metadata input")
    accounting = ETLMetadataAccounting(
        input_rows=input_rows,
        selected_rows=selected,
        duplicates_excluded=duplicates,
        outside_slice=input_rows - selected - duplicates,
    )
    return LRTSStageTables(
        *frames,
        tuple(source_datasets),
        fingerprint([f.to_dicts() for f in frames]),
        exclusions,
        accounting,
    )
