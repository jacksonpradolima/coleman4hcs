"""Versioned historical HCS projections, with pipeline identity and explicit simulation limits."""

from __future__ import annotations

import ntpath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.lrts_contracts import Label, Name, Positive
from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.materialization_contracts import MaterializationSpec
from coleman.results.domain import Fingerprint, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

HCS_SECONDS = UnitSemantics(unit="s", meaning="gitlabci-torrent-reported-test-seconds-v1")
Rows = Annotated[int, Field(strict=True, ge=0, le=500000)]


class HCSRecipe(VersionedContract):
    """Pin two reviewed data files and at most sixteen numeric source-pipeline rounds.

    The CSV is the publisher's parsed execution table, not its lossy engineered
    commit union. Jobs supply exact pipeline/commit/variant correspondence. Time
    units follow the linked extractor's documented seconds interpretation; the
    actual historical generator revision is unverified. Binary one means
    reported nonpass, including statuses collapsed upstream, never unique faults.
    Numeric pipeline order is a modeled order, not availability chronology.
    Native modeled costs explicitly round to the existing bandit's binary32
    precision before deriving budgets; analytical source decimals are unchanged.
    Existing prepared native/legacy artifacts are never migrated or rewritten.
    """

    recipe: Literal["gitlab-hcs-pipeline-parquet-v1"] = "gitlab-hcs-pipeline-parquet-v1"
    project: Label
    dataset_id: Label
    executions: MaterializationSpec
    jobs: MaterializationSpec
    pipeline_ids: Annotated[tuple[Positive, ...], Field(min_length=1, max_length=16)]
    duration_semantics: UnitSemantics = HCS_SECONDS
    native_costs: Literal["additive-binary32-rounded-reported-test-seconds-v1"]
    attempt_rule: Literal["lowest-eligible-parsed-job-id-per-pipeline-variant"] = (
        "lowest-eligible-parsed-job-id-per-pipeline-variant"
    )
    union_rule: Literal["max-reported-seconds-any-reported-nonpass"] = "max-reported-seconds-any-reported-nonpass"
    feedback: Literal["budget-censored-native-RNFail-zero-nondetection"] = (
        "budget-censored-native-RNFail-zero-nondetection"
    )

    @model_validator(mode="after")
    def _source_contract(self) -> Self:
        """Reject unsupported formats, unsafe bounds and ambiguous order before source access."""
        if self.duration_semantics != HCS_SECONDS:
            raise ValueError("Incompatible HCS duration units or meaning")
        if ntpath.isreserved(self.dataset_id):
            raise ValueError("HCS dataset alias is Windows-reserved")
        if tuple(sorted(set(self.pipeline_ids))) != self.pipeline_ids:
            raise ValueError("HCS pipelines require unique ascending numeric IDs")
        for package in (self.executions, self.jobs):
            if package.archive != "raw" or package.members[0].size_bytes > 64 * 1024**2:
                raise ValueError("HCS requires raw data files within 64 MiB each")
        if self.executions.source.sha256 == self.jobs.source.sha256:
            raise ValueError("HCS execution CSV and job JSON must be distinct sources")
        return self


class HCSJob(Immutable):
    """Selected public job metadata; source labels are not parsed configuration features.

    Job timestamp labels remain strings from the publisher. A finished job does
    not establish when parsed labels/features reached a prioritizer. No user,
    credentials, runner addresses or unrelated source fields are exported.
    """

    job_id: Positive
    pipeline_id: Positive
    commit_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{40}$")]
    variant: Name
    status: Name
    started_at: Name | None
    finished_at: Name | None


class HCSObservation(Immutable):
    """One source execution; zero is observed, absent duration/verdict stays null.

    Source position is physical CSV provenance, excluded from logical identity.
    Decimal spelling is retained separately from canonical nonnegative seconds.
    Repeated content at one key can be explicitly excluded; conflicting content
    rejects rather than max-merging runs with different pipeline/job identities.
    """

    source_position: Positive
    job: HCSJob
    test_name: Name
    last_run_label: Annotated[str, Field(strict=True, max_length=1024)]
    duration_lexeme: Annotated[str, Field(strict=True, max_length=128)]
    duration: Quantity | None
    verdict: Annotated[int, Field(strict=True, ge=0, le=1)] | None

    @model_validator(mode="after")
    def _units(self) -> Self:
        """Require reported seconds when observed; negative/nonfinite values fail in Quantity."""
        if self.duration is not None and self.duration.semantics != HCS_SECONDS:
            raise ValueError("Incompatible HCS observation duration")
        expected = (
            None
            if self.duration_lexeme == ""
            else Quantity.model_validate({"value": self.duration_lexeme, "semantics": HCS_SECONDS})
        )
        if self.duration != expected:
            raise ValueError("HCS duration disagrees with its original lexeme")
        return self

    @property
    def key(self) -> tuple[int, int, str]:
        """Return pipeline/job/test identity, independent of values and physical row position."""
        return self.job.pipeline_id, self.job.job_id, self.test_name

    def execution_id(self, project: str) -> str:
        """Bind the public project namespace; repeats/variants remain dependent, never new faults."""
        return fingerprint(["hcs-source-execution-v1", project, *self.key])


class HCSSlice(Immutable):
    """Validated intermediate source slice, not a persisted artifact or proof of cached-byte correspondence.

    Physical positions preserve source order; repeated positions or contradictory
    job/pipeline/commit bindings reject. Raw files remain in the verified cache.
    All public projection boundaries revalidate nested immutable DTOs rather than
    trusting unchecked model construction. No transport or filesystem access.
    """

    recipe: HCSRecipe
    input_rows: Rows
    outside_slice: Rows
    observations: Annotated[tuple[HCSObservation, ...], Field(max_length=16000)]
    jobs: Annotated[tuple[HCSJob, ...], Field(max_length=16384)]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Bind observed rows to exact metadata and conserve source counters before pure projection."""
        if self.input_rows != self.outside_slice + len(self.observations):
            raise ValueError("HCS source-slice row conservation failed")
        jobs = {job.job_id: job for job in self.jobs}
        if len(jobs) != len(self.jobs) or {job.pipeline_id for job in self.jobs} != set(self.recipe.pipeline_ids):
            raise ValueError("HCS source-slice job/pipeline bindings disagree")
        commits: dict[int, str] = {}
        for job in self.jobs:
            if commits.setdefault(job.pipeline_id, job.commit_id) != job.commit_id:
                raise ValueError("Conflicting HCS source-slice pipeline commits")
        positions = [case.source_position for case in self.observations]
        if len(set(positions)) != len(positions) or positions != sorted(positions):
            raise ValueError("HCS physical source positions must be unique and ordered")
        for case in self.observations:
            if case.source_position > self.input_rows or jobs.get(case.job.job_id) != case.job:
                raise ValueError("HCS source-slice observation/metadata binding disagrees")
        return self


class HCSVariant(Immutable):
    """Reversible original name to portable digest alias; no punctuation/case normalization."""

    name: Name
    alias: Annotated[str, Field(pattern=r"^v_[0-9a-f]{64}$", strict=True)]

    @model_validator(mode="after")
    def _alias(self) -> Self:
        """Bind the complete original label; collision checks also apply to the full mapping."""
        if self.alias != variant_alias(self.name):
            raise ValueError("HCS variant alias disagrees with original name")
        return self


def variant_alias(name: str) -> str:
    """Encode a variant label into a portable opaque alias; retain HCSVariant for reverse lookup."""
    return "v_" + fingerprint(["hcs-variant-name-v1", name]).removeprefix("sha256:")


class HCSAccounting(Immutable):
    """Conserve physical CSV rows with one primary disposition; counts are not independent samples."""

    input_rows: Rows
    outside_slice: Rows
    selected_rows: Rows
    duplicate_rows: Rows
    unsupported_job_rows: Rows
    other_attempt_rows: Rows
    unavailable_outcome_rows: Rows
    unavailable_duration_rows: Rows
    replay_rows: Rows

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Conserve selected rows, retaining fully unavailable/omitted source jobs as unknown upstream scope."""
        if self.input_rows != self.outside_slice + self.selected_rows:
            raise ValueError("HCS input row conservation failed")
        if self.selected_rows != sum(
            getattr(self, name)
            for name in (
                "duplicate_rows",
                "unsupported_job_rows",
                "other_attempt_rows",
                "unavailable_outcome_rows",
                "unavailable_duration_rows",
                "replay_rows",
            )
        ):
            raise ValueError("HCS selected row conservation failed")
        return self


class HCSEvidence(Immutable):
    """Explicit missing evidence; successful HCS execution cannot grant temporal or cost admission."""

    outcome_availability: Literal["unavailable"] = "unavailable"
    feature_availability: Literal["unavailable"] = "unavailable"
    test_correspondence: Literal["shared-original-label-assumption"] = "shared-original-label-assumption"
    upstream_exclusion_count: Literal["unavailable"] = "unavailable"
    setup_seconds: None = None
    shared_runner_occupancy: None = None
    concurrency_topology: Literal["unavailable"] = "unavailable"
    independent_projects: Literal["not_established"] = "not_established"

    def require_temporal_evaluation(self) -> None:
        """Reject protected evaluation until source evidence is independently admitted in #309/#310."""
        raise ValueError("HCS temporal evaluation blocked: outcome/feature availability and dependence unverified")


class HCSReceipt(VersionedContract):
    """Last-published immutable completion metadata, verified separately against cached source/file bytes.

    Producer and Parquet writer select the export namespace. Unsupported old or
    future schemas fail closed; old native files keep their own readers. No
    implicit migration, source acquisition, execution or statistics occurs here.
    A partial export never carries a new complete.json. Unknown overhead is null.
    """

    recipe: HCSRecipe
    producer_sha256: Fingerprint
    parquet_writer: Name
    accounting: HCSAccounting
    variants: Annotated[tuple[HCSVariant, ...], Field(max_length=64)]
    files: Annotated[tuple[ETLFile, ...], Field(min_length=4, max_length=68)]
    logical_rows_sha256: Fingerprint
    evidence: HCSEvidence = Field(default_factory=HCSEvidence)
    scope: Literal["posthoc-test-only-serial-HCS-simulation"] = "posthoc-test-only-serial-HCS-simulation"
    admission: Literal["blocked_source_evidence"] = "blocked_source_evidence"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _roles(self) -> Self:
        """Bind all analytical/union/variant files and primary row counts without trusting their bytes."""
        names = [v.name for v in self.variants]
        aliases = [v.alias for v in self.variants]
        if len(names) != len(set(names)) or len(aliases) != len(set(aliases)):
            raise ValueError("Duplicate HCS variant binding")
        if names != sorted(names):
            raise ValueError("HCS variant bindings must be canonical")
        dataset = self.recipe.dataset_id
        expected = {
            "observations.parquet",
            "cells.parquet",
            f"datasets/{dataset}/features-engineered.parquet",
            f"datasets/{dataset}/data-variants.parquet",
            *(f"vts/{v.alias}/features-engineered.parquet" for v in self.variants),
        }
        if len(self.files) != len(expected) or {f.path for f in self.files} != expected:
            raise ValueError("HCS Parquet role set disagrees")
        counts = {f.path: f.rows for f in self.files}
        if counts["observations.parquet"] != self.accounting.selected_rows:
            raise ValueError("HCS observation rows disagree")
        if counts[f"datasets/{dataset}/data-variants.parquet"] != self.accounting.replay_rows:
            raise ValueError("HCS native variant rows disagree")
        if (
            sum(counts[f"vts/{v.alias}/features-engineered.parquet"] for v in self.variants)
            != self.accounting.replay_rows
        ):
            raise ValueError("HCS VTS rows must conserve WTS variant executions")
        if counts["cells.parquet"] != len(self.recipe.pipeline_ids) * len(self.variants):
            raise ValueError("HCS sparse matrix must retain every declared pipeline/variant cell")
        return self
