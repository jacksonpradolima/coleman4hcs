"""Versioned real RTPTorrent Parquet recipes with explicit source-job simulation assumptions."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.corpus.lrts_etl_contracts import Count, ETLFile
from coleman.corpus.materialization_contracts import safe_member_name
from coleman.corpus.rtptorrent_contracts import Positive, RTPTorrentSpec
from coleman.corpus.zip64_contracts import Zip64SelectionSpec
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract


class RTPTorrentETLRecipe(VersionedContract):
    """Bind one complete raw history to a bounded, explicitly modeled source-job replay.

    All canonical raw executions are retained analytically; only declared jobs
    enter native projection. Numeric job order is a simulation convention, never
    verified chronology or a logical CI build mapping. Additive reported class
    seconds are modeled serial costs, not observed parallel latency/occupancy.
    Class-label/within-job-occurrence correspondence is an explicit assumption.
    Native RNFail rewards detections within budget; zero nondetection reward for
    unscheduled cases does not assert a passing verdict or full-label feedback.
    No transport, source execution or scientific admission occurs on model reads.
    """

    recipe: Literal["rtptorrent-parquet-source-job-replay-v1"] = "rtptorrent-parquet-source-job-replay-v1"
    source: Zip64SelectionSpec
    conversion: RTPTorrentSpec
    dataset_id: Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    replay_job_ids: Annotated[tuple[Positive, ...], Field(min_length=1, max_length=16)]
    order: Literal["ascending-job-labels-not-chronology"] = "ascending-job-labels-not-chronology"
    costs: Literal["additive-reported-class-seconds"] = "additive-reported-class-seconds"
    feedback: Literal["budget-censored-native-RNFail-zero-nondetection"]
    test_identity: Literal["class-label-within-job-occurrence-assumption-v1"] = (
        "class-label-within-job-occurrence-assumption-v1"
    )

    @field_validator("dataset_id")
    @classmethod
    def _dataset_name(cls, value: str) -> str:
        """Require one safe dataset directory component before any output path is formed."""
        return safe_member_name(value)

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Require one pinned raw member and the declared numeric order without silent sorting/migration."""
        if len(self.source.selections) != 1:
            raise ValueError("RTPTorrent ETL requires one complete pinned raw-class member")
        package = self.source.selections[0].package
        if self.conversion.materialization.spec != package or self.conversion.member != package.members[0].name:
            raise ValueError("RTPTorrent conversion must identify the selected raw child exactly")
        if tuple(sorted(set(self.replay_job_ids))) != self.replay_job_ids:
            raise ValueError("RTPTorrent replay job labels must be distinct and in declared numeric order")
        return self


class RTPTorrentETLReceipt(VersionedContract):
    """Last-written completion for analytical histories/jobs and a separate native projection.

    Reading validates metadata only; validate_rtptorrent_etl rechecks actual
    source membership, logical rows, counts and physical output bytes. Future
    schemas and unknown producers reject verification rather than reinterpret
    existing artifacts. Use the recorded implementation to verify older producer
    identities; no native/LRTS/filtered-baseline artifact is silently migrated.
    Partial files never imply completion. Counts refer to class executions/jobs,
    not independent observations, logical builds or unique faults.
    """

    recipe: RTPTorrentETLRecipe
    producer_sha256: Fingerprint
    parquet_writer: str
    source_dataset_sha256: Fingerprint
    files: Annotated[tuple[ETLFile, ...], Field(min_length=3, max_length=3)]
    logical_rows_sha256: Fingerprint
    input_rows: Count
    duplicate_rows_excluded: Count
    observed_executions: Count
    source_jobs: Annotated[int, Field(strict=True, ge=1, le=50000)]
    replay_executions: Count
    excluded_outside_jobs: Count
    excluded_not_run: Count
    excluded_unknown_outcome: Count
    excluded_unavailable_duration: Count
    native_rounds: Count
    admission: Literal["blocked_missing_temporal_evidence"] = "blocked_missing_temporal_evidence"
    evidence: Literal["posthoc-source-job-serial-simulation"] = "posthoc-source-job-serial-simulation"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Conserve physical/canonical/projected rows, every primary exclusion and the exact three file roles."""
        if self.input_rows != self.observed_executions + self.duplicate_rows_excluded:
            raise ValueError("RTPTorrent ETL physical row conservation failed")
        if self.observed_executions != (
            self.replay_executions
            + self.excluded_outside_jobs
            + self.excluded_not_run
            + self.excluded_unknown_outcome
            + self.excluded_unavailable_duration
        ):
            raise ValueError("RTPTorrent ETL projection row conservation failed")
        if self.input_rows > self.recipe.conversion.maximum_rows or self.source_jobs > self.observed_executions:
            raise ValueError("RTPTorrent ETL source row/job bounds disagree")
        if self.native_rounds > len(self.recipe.replay_job_ids) or self.source_jobs < len(self.recipe.replay_job_ids):
            raise ValueError("RTPTorrent ETL selected job/round counts disagree")
        if (self.native_rounds == 0) != (self.replay_executions == 0) or self.native_rounds > self.replay_executions:
            raise ValueError("RTPTorrent ETL native executions/rounds disagree")
        expected = {
            "observations.parquet": self.observed_executions,
            "jobs.parquet": self.source_jobs,
            f"datasets/{self.recipe.dataset_id}/features-engineered.parquet": self.replay_executions,
        }
        if len({item.path for item in self.files}) != 3 or {item.path: item.rows for item in self.files} != expected:
            raise ValueError("RTPTorrent ETL Parquet roles/rows disagree")
        return self
