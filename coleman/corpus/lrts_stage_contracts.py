"""Explicit versioned LRTS stage correspondence and bounded single-stage native views."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.lrts_contracts import Label, Name, Positive
from coleman.corpus.lrts_etl_contracts import Count, ETLFile, ETLMetadataAccounting, LRTSETLRecipe
from coleman.results.domain import Fingerprint, canonical_json
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable


class LRTSStageBinding(Immutable):
    """One processed stage bound to an exact raw context and complete node-ID set in one source build.

    Arrays are compared as tuples, never joined or matched by substring. Node
    IDs are local to this PR/build and are not stable test or environment IDs.
    Enclosing block IDs remain separate raw evidence; their array lengths need
    not match the names. Construction does not certify source correspondence.
    """

    pull_request: Label
    build_id: Positive
    stage_id: Name
    enclosing_block_names: Annotated[tuple[Name, ...], Field(min_length=1, max_length=64)]
    node_ids: Annotated[tuple[Name, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _nodes(self) -> Self:
        """Canonicalize set-like node IDs, rejecting duplicates rather than collapsing them."""
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("Duplicate LRTS stage node identity")
        object.__setattr__(self, "node_ids", tuple(sorted(self.node_ids)))
        return self


class LRTSStageRecipe(VersionedContract):
    """Freeze full analytical source and one explicitly selected modeled native stage.

    The embedded v1 recipe retains its source pins/order/units/feedback without
    relaxing legacy single-stage export. All processed stages and raw methods
    remain analytical observations; outside-stage rows have explicit exclusions.
    Label/occurrence correspondence is an assumption, not verified continuity.
    New family/schema identities cannot reinterpret existing receipts. No I/O.
    """

    recipe: Literal["lrts-stage-parquet-replay-v1"] = "lrts-stage-parquet-replay-v1"
    source: LRTSETLRecipe
    dataset_id: Label
    native_stage_id: Name
    stage_bindings: Annotated[tuple[LRTSStageBinding, ...], Field(min_length=1, max_length=1024)]
    test_identity: Literal["stage-suite-class-method-label-occurrence-assumption-v1"] = (
        "stage-suite-class-method-label-occurrence-assumption-v1"
    )

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Require complete declared build coverage and unambiguous stage/context/node mappings."""
        self.source.require_native_feedback()
        builds = {(b.pull_request, b.build_id) for b in self.source.builds}
        groups: dict[tuple[str, int], list[LRTSStageBinding]] = {}
        for binding in self.stage_bindings:
            groups.setdefault((binding.pull_request, binding.build_id), []).append(binding)
        if groups.keys() != builds:
            raise ValueError("Stage bindings must exactly cover selected LRTS builds")
        for bindings in groups.values():
            stages = [b.stage_id for b in bindings]
            contexts = [b.enclosing_block_names for b in bindings]
            nodes = [n for b in bindings for n in b.node_ids]
            if len(set(stages)) != len(stages) or len(set(contexts)) != len(contexts) or len(set(nodes)) != len(nodes):
                raise ValueError("Ambiguous LRTS stage/context/node binding")
            if self.native_stage_id not in stages:
                raise ValueError("Native LRTS stage must be explicitly bound in every build")
        object.__setattr__(
            self,
            "stage_bindings",
            tuple(sorted(self.stage_bindings, key=lambda b: (b.pull_request, b.build_id, b.stage_id))),
        )
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > 1024**2:
            raise ValueError("LRTS stage recipe exceeds 1 MiB canonical metadata limit")
        return self


class LRTSStageEvidence(Immutable):
    """Reproducible admission blockers for these source views, independent of successful conversion.

    Explicit source-stage binding resolves correspondence only within a report.
    Scheduled labels and raw timestamps cannot establish these missing clocks,
    a complete test universe, independent observations or physical work saved.
    Reported environment names do not verify runner OS or machine topology.
    """

    stage_correspondence: Literal["explicit-context-and-node-binding"] = "explicit-context-and-node-binding"
    executor_start: Literal["unavailable"] = "unavailable"
    outcome_availability: Literal["unavailable"] = "unavailable"
    feature_availability: Literal["unavailable"] = "unavailable"
    expected_test_universe: Literal["unavailable"] = "unavailable"
    cross_round_correspondence: Literal["label-occurrence-assumption"] = "label-occurrence-assumption"
    concurrency_topology: Literal["unavailable"] = "unavailable"

    def require_temporal_evaluation(self) -> None:
        """Reject protected temporal evaluation; a successful ETL never fills missing source evidence."""
        raise ValueError(
            "LRTS stage evaluation blocked: executor/outcome/feature availability, universe, continuity and concurrency"
        )


class LRTSStageReceipt(VersionedContract):
    """Last-published completion metadata; source/bytes require explicit offline validation.

    Every source method has exactly one primary disposition. Unknown future
    schemas reject; old single-stage receipts remain with their original reader.
    Stage views share source project/build groups and are never independent
    projects or unique faults. An incomplete export has no completion receipt.
    """

    recipe: LRTSStageRecipe
    producer_sha256: Fingerprint
    parquet_writer: str
    source_datasets: Annotated[tuple[Fingerprint, ...], Field(min_length=1, max_length=16)]
    files: Annotated[tuple[ETLFile, ...], Field(min_length=3, max_length=3)]
    logical_rows_sha256: Fingerprint
    observed_cases: Count
    replay_cases: Count
    excluded_outside_stage: Count
    excluded_not_run: Count
    excluded_unknown_outcome: Count
    excluded_unavailable_duration: Count
    native_builds: Count
    source_stages: Count
    metadata_accounting: ETLMetadataAccounting
    temporal: LRTSStageEvidence = Field(default_factory=LRTSStageEvidence)
    admission: Literal["blocked_source_evidence"] = "blocked_source_evidence"
    evidence: Literal["posthoc-serial-simulation"] = "posthoc-serial-simulation"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Bind three table roles and conserve cases, stages, metadata and native round counts."""
        if self.observed_cases != sum(
            (
                self.replay_cases,
                self.excluded_outside_stage,
                self.excluded_not_run,
                self.excluded_unknown_outcome,
                self.excluded_unavailable_duration,
            )
        ):
            raise ValueError("LRTS stage source/exclusion row conservation failed")
        builds = len(self.recipe.source.builds)
        if len(self.source_datasets) != builds or len(set(self.source_datasets)) != builds:
            raise ValueError("LRTS stage source identities disagree")
        if (
            self.source_stages != len(self.recipe.stage_bindings)
            or self.metadata_accounting.selected_rows != self.source_stages
        ):
            raise ValueError("LRTS stage metadata accounting disagrees")
        if (
            self.native_builds > builds
            or self.native_builds > self.replay_cases
            or ((self.native_builds == 0) != (self.replay_cases == 0))
        ):
            raise ValueError("LRTS stage native round counts disagree")
        if self.observed_cases > self.recipe.source.maximum_cases:
            raise ValueError("LRTS stage source-case limit exceeded")
        expected = {
            "observations.parquet": self.observed_cases,
            "builds.parquet": builds,
            f"datasets/{self.recipe.dataset_id}/features-engineered.parquet": self.replay_cases,
        }
        if len({f.path for f in self.files}) != 3 or {f.path: f.rows for f in self.files} != expected:
            raise ValueError("LRTS stage Parquet roles/row counts disagree")
        return self
