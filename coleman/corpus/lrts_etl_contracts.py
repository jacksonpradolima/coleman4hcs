"""Versioned LRTS Parquet ETL recipes and explicitly modeled native replay contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.archive_selection import ArchiveSelectionSpec
from coleman.corpus.lrts_contracts import Label, Positive
from coleman.corpus.materialization_contracts import MaterializationSpec, safe_member_name
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

Count = Annotated[int, Field(strict=True, ge=0, le=100000)]


class LRTSBuildSelection(Immutable):
    """One pinned source report in a declared replay order; build IDs are local to a PR."""

    pull_request: Label
    build_id: Positive
    archive_member: str


class LRTSETLRecipe(VersionedContract):
    """Freeze selected real reports, metadata and a reproducible post-hoc simulation profile.

    Scheduled labels order modeled rounds, not verified outcome availability.
    Costs are additive reported method seconds; feedback is modeled as complete
    after each round. No modeled quantity becomes actual CI time/work. Stable
    replay names assume matching suite/class/method labels and occurrence order;
    original source execution IDs remain independently preserved. These explicit
    assumptions never admit a source to protected temporal evaluation.
    """

    recipe: Literal["lrts-parquet-serial-replay-v1"] = "lrts-parquet-serial-replay-v1"
    project: Label
    history: Literal["observed", "constructed"]
    metadata: MaterializationSpec
    reports: ArchiveSelectionSpec
    builds: Annotated[tuple[LRTSBuildSelection, ...], Field(min_length=1, max_length=16)]
    order: Literal["scheduled-label-then-pr-build"] = "scheduled-label-then-pr-build"
    costs: Literal["additive-reported-method-seconds"] = "additive-reported-method-seconds"
    feedback: Literal["full-reported-labels-after-modeled-round"] = "full-reported-labels-after-modeled-round"
    test_identity: Literal["suite-class-method-label-occurrence-assumption"] = (
        "suite-class-method-label-occurrence-assumption"
    )
    maximum_cases: Annotated[int, Field(strict=True, ge=1, le=50000)] = 50000

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Require one-to-one source selection and raw metadata; never infer missing archive/member links."""
        if self.metadata.archive != "raw" or self.metadata.source.size_bytes is None:
            raise ValueError("ETL metadata must be one frozen raw CSV")
        if self.metadata.source.size_bytes > 32 * 1024**2:
            raise ValueError("ETL metadata exceeds 32 MiB")
        keys = {(item.pull_request, item.build_id) for item in self.builds}
        names = {item.archive_member for item in self.builds}
        if len(keys) != len(self.builds) or len(names) != len(self.builds):
            raise ValueError("Duplicate ETL source build/member")
        selected = {item.member.name: item for item in self.reports.selections}
        if names != selected.keys():
            raise ValueError("ETL builds must exactly match selected source members")
        for build in self.builds:
            expected = f"{self.project}/PR/{build.pull_request}/testReport_build{build.build_id}.json.zip"
            package = selected[build.archive_member].package
            if build.archive_member != expected or package.archive != "zip" or len(package.members) != 1:
                raise ValueError("ETL requires a correctly bound one-report nested ZIP")
            if package.members[0].name != f"testReport_build{build.build_id}.json":
                raise ValueError("ETL decoded report disagrees with source build")
        return self


class ETLFile(Immutable):
    """Relative output identity, SHA-256 and exact byte/row counts; no file access occurs on read."""

    path: str
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=64 * 1024**2)]
    rows: Count

    @model_validator(mode="after")
    def _path(self) -> Self:
        """Reject unsafe portable relative names before any output filesystem operation."""
        safe_member_name(self.path)
        return self


class ETLMetadataAccounting(Immutable):
    """Conserved physical metadata rows; unselected projects/builds and exact duplicates stay explicit."""

    input_rows: Count
    selected_rows: Count
    duplicates_excluded: Count
    outside_slice: Count

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Prevent unselected metadata and exact duplicates from disappearing in aggregate counts."""
        if self.input_rows != self.selected_rows + self.duplicates_excluded + self.outside_slice:
            raise ValueError("ETL metadata row conservation failed")
        return self


class LRTSETLReceipt(VersionedContract):
    """Completion contract written last after every Parquet file is immutable and verified.

    Reading checks invariants only; validate_lrts_etl verifies source and actual
    files. Partial files/blobs never imply completion. Recipe/schema/writer
    identity forms the export namespace. Native replay remains an explicitly
    modeled post-hoc simulation, not confirmatory historical CI evidence.
    Missing/future schemas reject; no legacy Parquet is reinterpreted/migrated.
    """

    recipe: LRTSETLRecipe
    producer_sha256: Fingerprint
    parquet_writer: str
    source_datasets: Annotated[tuple[Fingerprint, ...], Field(min_length=1, max_length=16)]
    files: Annotated[tuple[ETLFile, ...], Field(min_length=3, max_length=3)]
    logical_rows_sha256: Fingerprint
    observed_cases: Count
    replay_cases: Count
    excluded_not_run: Count
    excluded_unknown_outcome: Count
    excluded_unavailable_duration: Count
    native_builds: Count
    metadata_accounting: ETLMetadataAccounting
    admission: Literal["blocked_source_evidence"] = "blocked_source_evidence"
    evidence: Literal["posthoc-serial-simulation"] = "posthoc-serial-simulation"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Conserve every observed source method and bind all three expected Parquet roles/counts."""
        if self.observed_cases != (
            self.replay_cases
            + self.excluded_not_run
            + self.excluded_unknown_outcome
            + self.excluded_unavailable_duration
        ):
            raise ValueError("ETL source/exclusion row conservation failed")
        expected = {
            "observations.parquet": self.observed_cases,
            "builds.parquet": len(self.recipe.builds),
            f"datasets/{self.recipe.project}/features-engineered.parquet": self.replay_cases,
        }
        if len({item.path for item in self.files}) != 3 or {item.path: item.rows for item in self.files} != expected:
            raise ValueError("ETL Parquet roles/row counts disagree")
        if len(self.source_datasets) != len(self.recipe.builds) or self.native_builds > len(self.recipe.builds):
            raise ValueError("ETL source/build counts disagree")
        if len(set(self.source_datasets)) != len(self.source_datasets):
            raise ValueError("ETL source build fingerprints must be distinct")
        if (self.native_builds == 0) != (self.replay_cases == 0) or self.native_builds > self.replay_cases:
            raise ValueError("ETL runnable rows/builds disagree")
        if self.metadata_accounting.selected_rows != len(self.recipe.builds):
            raise ValueError("ETL requires one selected metadata row per source build")
        if self.observed_cases > self.recipe.maximum_cases:
            raise ValueError("ETL exceeds declared total source-case limit")
        return self
