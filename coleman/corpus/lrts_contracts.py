"""Typed LRTS source slices; reported clocks, durations and method outcomes stay distinct."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.results.domain import Identifier, Measurement, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

Integer = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Positive = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
Label = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
Status = Literal["PASSED", "FIXED", "FAILED", "REGRESSION", "SKIPPED"]
Name = Annotated[Identifier, Field(max_length=1024)]

METADATA_COLUMNS = (
    "project",
    "pr_name",
    "pr_base_branch",
    "build_id",
    "build_duration",
    "build_timestamp",
    "build_result",
    "build_head_sha",
    "ts_duration",
    "passcount",
    "failcount",
    "skipcount",
    "build_date",
    "trunk_sha",
    "trunk_sha_source",
    "stage_id",
    "stage_duration_by_suite_sum",
    "test_suite_duration_s",
    "num_pass_class",
    "num_fail_class",
    "num_trans_class",
    "num_pass_method",
    "num_fail_method",
)
METADATA_MEASURES = (
    ("build_duration", "lrts_build_execution_duration"),
    ("build_timestamp", "lrts_scheduled_epoch_seconds"),
    ("ts_duration", "lrts_report_duration"),
    ("stage_duration_by_suite_sum", "lrts_suite_duration_sum"),
    ("test_suite_duration_s", "lrts_processed_class_duration_sum"),
)
MEANINGS = MappingProxyType(
    {
        "lrts_build_execution_duration": "lrts-source-truncated-build-execution-seconds-binary64-v1",
        "lrts_scheduled_epoch_seconds": "lrts-source-truncated-scheduled-unix-epoch-seconds-binary64-v1",
        "lrts_report_duration": "jenkins-source-report-duration-seconds-binary64-projection-v1",
        "lrts_suite_duration_sum": "lrts-source-suite-duration-sum-seconds-binary64-v1",
        "lrts_processed_class_duration_sum": "lrts-source-processed-class-duration-sum-seconds-binary64-v1",
        "lrts_suite_duration": "jenkins-source-suite-duration-seconds-binary64-projection-v1",
        "lrts_case_duration": "jenkins-source-method-duration-seconds-binary64-projection-v1",
    }
)


def source_measurement(value: Measurement, name: str) -> Measurement:
    """Check exact source units/meaning and binary64 bounds; canonicalize genuine signed zero.

    Source measurements accept observed or unavailable states only. These are
    reported/computed seconds, not exact physical work or verified outcome clocks.
    Values above 10^12 seconds and incompatible declarations raise ValueError.
    This pure validator performs no I/O or inference about missing observations.
    """
    if name not in MEANINGS:
        raise ValueError("Unknown LRTS source measurement")
    if (value.name, value.unit, value.meaning, value.direction, value.minimum) != (
        name,
        "s",
        MEANINGS[name],
        "none",
        0.0,
    ) or value.status not in {"observed", "unavailable"}:
        raise ValueError("Incompatible LRTS source measurement units/meaning/status")
    if value.value is not None and value.value > 10**12:
        raise ValueError("LRTS reported seconds exceed 10^12")
    if value.value == 0:
        return Measurement.model_validate({**value.model_dump(), "value": 0.0})
    return value


class LRTSSpec(VersionedContract):
    """Freeze one metadata/build and raw Jenkins report pair, independently of transport.

    Both receipts pin actual source/member bytes and reviewed data terms. The
    origin label records the producer's member mapping, not independent proof of
    its enclosing archive. No ZIP limit is widened and no download is started.
    Reported scheduled timestamps never substitute for outcome availability.
    """

    recipe: Literal["lrts-source-assessment-v1"] = "lrts-source-assessment-v1"
    metadata_materialization: MaterializationReceipt
    metadata_member: Annotated[Identifier, Field(max_length=1024)]
    report_materialization: MaterializationReceipt
    report_member: Annotated[Identifier, Field(max_length=1024)]
    project: Label
    pull_request: Label
    build_id: Positive
    report_origin_path: Annotated[Identifier, Field(max_length=1024)]
    history: Literal["observed", "constructed"]
    maximum_metadata_rows: Annotated[int, Field(strict=True, ge=1, le=100000)] = 100000
    maximum_line_bytes: Annotated[int, Field(strict=True, ge=128, le=16384)] = 16384
    maximum_suites: Annotated[int, Field(strict=True, ge=1, le=5000)] = 5000
    maximum_cases: Annotated[int, Field(strict=True, ge=1, le=50000)] = 50000

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Require pinned bounded members and a consistent declared project/PR/build origin label."""
        expected = f"{self.project}/PR/{self.pull_request}/testReport_build{self.build_id}.json"
        if self.report_origin_path != expected:
            raise ValueError("LRTS declared report origin disagrees with project/PR/build")
        for receipt, name, maximum in (
            (self.metadata_materialization, self.metadata_member, 32 * 1024**2),
            (self.report_materialization, self.report_member, 16 * 1024**2),
        ):
            members = [member for member in receipt.spec.members if member.name == name]
            if len(members) != 1 or members[0].size_bytes > maximum:
                raise ValueError("LRTS selected member is missing or exceeds its byte limit")
        return self


class LRTSMetadataRow(Immutable):
    """One processed metadata stage; build-level counts may repeat across stages.

    Counts are nonnegative integers, with null meaning unavailable. Processed
    class/method counts retain their distinct definitions. build_date is an
    uninterpreted producer label, not a verified UTC date. No field is a fault ID.
    """

    project: Label
    pull_request: Label
    build_id: Positive
    stage_id: Annotated[Identifier, Field(max_length=1024)]
    base_branch: Annotated[Identifier, Field(max_length=1024)] | None
    build_result: Literal["SUCCESS", "FAILURE", "UNSTABLE", "ABORTED", "NOT_BUILT"] | None
    build_head_sha: Annotated[Identifier, Field(max_length=128)] | None
    build_date: Annotated[Identifier, Field(max_length=128)] | None
    trunk_sha: Annotated[Identifier, Field(max_length=128)] | None
    trunk_sha_source: Annotated[Identifier, Field(max_length=128)] | None
    reported_pass_methods: Integer | None
    reported_fail_methods: Integer | None
    reported_skip_methods: Integer | None
    processed_pass_classes: Integer | None
    processed_fail_classes: Integer | None
    processed_transition_classes: Integer | None
    processed_pass_methods: Integer | None
    processed_fail_methods: Integer | None
    measurements: Annotated[tuple[Measurement, ...], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def _measurements(self) -> Self:
        """Keep all five distinct source quantities, including units for unavailable values."""
        expected = {name for _, name in METADATA_MEASURES}
        if {item.name for item in self.measurements} != expected:
            raise ValueError("LRTS metadata requires exactly its five source measurements")
        values = tuple(
            sorted((source_measurement(item, item.name) for item in self.measurements), key=lambda m: m.name)
        )
        for item in values:
            if (
                item.name in {"lrts_scheduled_epoch_seconds", "lrts_build_execution_duration"}
                and item.value is not None
                and not item.value.is_integer()
            ):
                raise ValueError("LRTS processed scheduling/execution fields are truncated whole seconds")
        object.__setattr__(self, "measurements", values)
        return self

    def measurement(self, name: str) -> Measurement:
        """Return a declared quantity; an unknown measurement name raises KeyError."""
        for item in self.measurements:
            if item.name == name:
                return item
        raise KeyError(name)


class LRTSCase(Immutable):
    """One reported method execution; source position preserves repeated labels and retries.

    PASSED/FIXED are reported passes; FAILED/REGRESSION are reported failures,
    never mapped unique faults. A null status stays unknown unless skipped is
    explicitly true. Duration is reported binary64 seconds; unavailable is null.
    """

    position: Integer
    class_name: Annotated[Identifier, Field(max_length=1024)]
    method_name: Annotated[Identifier, Field(max_length=1024)]
    source_status: Status | None
    skipped: Annotated[bool, Field(strict=True)]
    duration: Measurement

    @model_validator(mode="after")
    def _source(self) -> Self:
        """Reject contradictory skipped/status labels without repairing or inferring outcomes."""
        if self.source_status is not None and (self.source_status == "SKIPPED") != self.skipped:
            raise ValueError("LRTS skipped flag contradicts reported method status")
        object.__setattr__(self, "duration", source_measurement(self.duration, "lrts_case_duration"))
        return self

    @property
    def outcome(self) -> Literal["passed", "failed", "not_run", "unknown"]:
        """Project only explicit source verdicts; absence never becomes a pass."""
        if self.skipped:
            return "not_run"
        if self.source_status is None:
            return "unknown"
        return "failed" if self.source_status in {"FAILED", "REGRESSION"} else "passed"

    def execution_id(self, project: str, pull_request: str, build_id: int, suite_position: int) -> str:
        """Hash source labels/positions, not values; this is not cross-build logical test identity."""
        return fingerprint(
            [
                "lrts-execution-v1",
                project,
                pull_request,
                build_id,
                suite_position,
                self.position,
                self.class_name,
                self.method_name,
            ]
        )


class LRTSSuite(Immutable):
    """A raw suite and its enclosing node identities, without processed-stage alias inference.

    Null block metadata is unavailable; empty tuples mean the report supplies no
    enclosure links. Raw timestamp strings retain their spelling and unverified
    timezone/availability semantics. No sum is interpreted as CI elapsed savings.
    """

    position: Integer
    name: Annotated[Identifier, Field(max_length=1024)]
    reported_id: Annotated[Identifier, Field(max_length=1024)] | None
    node_id: Annotated[Identifier, Field(max_length=1024)] | None
    enclosing_blocks: Annotated[tuple[Name, ...], Field(max_length=64)] | None
    enclosing_block_names: Annotated[tuple[Name, ...], Field(max_length=64)] | None
    raw_timestamp: Annotated[str, Field(strict=True, max_length=128)] | None
    duration: Measurement
    cases: Annotated[tuple[LRTSCase, ...], Field(max_length=50000)]

    @model_validator(mode="after")
    def _source(self) -> Self:
        """Require complete source-position accounting and explicit suite-duration meaning."""
        if tuple(case.position for case in self.cases) != tuple(range(len(self.cases))):
            raise ValueError("LRTS method source positions must be contiguous")
        object.__setattr__(self, "duration", source_measurement(self.duration, "lrts_suite_duration"))
        return self


class LRTSDataset(VersionedContract):
    """A bounded source-correspondence slice, not an admitted historical experiment.

    Only selected build/stage metadata is interpreted; other physical metadata
    rows are counted as outside the selected build. Identical selected metadata
    duplicates are counted/excluded, conflicting ones reject. Raw suites/cases
    are never collapsed. Diagnostics/actions are not used as decision features.
    """

    spec: LRTSSpec
    metadata_input_rows: Integer
    metadata_outside_build: Integer
    metadata_duplicates_excluded: Integer
    metadata: Annotated[tuple[LRTSMetadataRow, ...], Field(min_length=1, max_length=64)]
    report_pass_methods: Integer
    report_fail_methods: Integer
    report_skip_methods: Integer
    report_duration: Measurement
    reported_empty: Annotated[bool, Field(strict=True)]
    suites: Annotated[tuple[LRTSSuite, ...], Field(max_length=5000)]
    feedback_scope: Literal["reported_cases_only"] = "reported_cases_only"
    admission: Literal["blocked_source_evidence"] = "blocked_source_evidence"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _context(self) -> Self:
        """Conserve selected/excluded rows and preserve distinct processed stage identities."""
        if (
            self.metadata_input_rows
            != len(self.metadata) + self.metadata_outside_build + self.metadata_duplicates_excluded
        ):
            raise ValueError("LRTS metadata row conservation failed")
        if self.metadata_input_rows > self.spec.maximum_metadata_rows:
            raise ValueError("LRTS metadata exceeds its row limit")
        if len({row.stage_id for row in self.metadata}) != len(self.metadata):
            raise ValueError("Duplicate LRTS processed stage identity")
        expected = (self.spec.project, self.spec.pull_request, self.spec.build_id)
        if any((row.project, row.pull_request, row.build_id) != expected for row in self.metadata):
            raise ValueError("LRTS metadata disagrees with selected project/PR/build")
        object.__setattr__(self, "metadata", tuple(sorted(self.metadata, key=lambda row: row.stage_id)))
        return self

    @model_validator(mode="after")
    def _reports(self) -> Self:
        """Check reported-case counts, positions and pair contradictions, not complete-suite coverage."""
        if len(self.suites) > self.spec.maximum_suites or len(self.cases) > self.spec.maximum_cases:
            raise ValueError("LRTS report exceeds declared suite/method limits")
        if tuple(suite.position for suite in self.suites) != tuple(range(len(self.suites))):
            raise ValueError("LRTS suite source positions must be contiguous")
        object.__setattr__(self, "report_duration", source_measurement(self.report_duration, "lrts_report_duration"))
        _report_counts(self)
        for row in self.metadata:
            _metadata_report_pair(row, self)
        return self

    @property
    def cases(self) -> tuple[LRTSCase, ...]:
        """Return every raw method execution in source order, without deduplication."""
        return tuple(case for suite in self.suites for case in suite.cases)

    @property
    def repeated_method_labels(self) -> int:
        """Count labels beyond the first class/method occurrence, without asserting independence."""
        return len(self.cases) - len({(case.class_name, case.method_name) for case in self.cases})

    def require_temporal_evaluation(self) -> None:
        """Fail closed on missing outcome availability, raw/processed stage mapping and expected universe."""
        raise ValueError(
            "LRTS evaluation blocked: outcome availability, stage mapping and complete-suite evidence unavailable"
        )


def _report_counts(dataset: LRTSDataset) -> None:
    """Conserve methods, require a consistent Jenkins empty flag and retain unknown classifications."""
    known = Counter(case.outcome for case in dataset.cases)
    declared = (dataset.report_pass_methods, dataset.report_fail_methods, dataset.report_skip_methods)
    if sum(declared) != len(dataset.cases):
        raise ValueError("LRTS declared method counts disagree with reported cases")
    if dataset.reported_empty != (len(dataset.cases) == 0):
        raise ValueError("LRTS reported empty flag contradicts reported method count")
    if any(known[label] > count for label, count in zip(("passed", "failed", "not_run"), declared, strict=True)):
        raise ValueError("LRTS known verdicts exceed reported method counts")


def _metadata_report_pair(row: LRTSMetadataRow, dataset: LRTSDataset) -> None:
    """Reject known count/duration contradictions; processed class sums keep their own definitions."""
    for known, declared in (
        (row.reported_pass_methods, dataset.report_pass_methods),
        (row.reported_fail_methods, dataset.report_fail_methods),
        (row.reported_skip_methods, dataset.report_skip_methods),
    ):
        if known is not None and known != declared:
            raise ValueError("LRTS metadata/report method counts contradict")
    duration = row.measurement("lrts_report_duration").value
    if (
        duration is not None
        and dataset.report_duration.value is not None
        and float(duration).hex() != float(dataset.report_duration.value).hex()
    ):
        raise ValueError("LRTS metadata/report declared durations contradict")
