"""Typed RTPTorrent class-execution slices; missing CI clocks never become timestamps."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.results.domain import Identifier, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

REPORTED_SECONDS = UnitSemantics(unit="s", meaning="rtptorrent-reported-class-elapsed-v1")
Integer = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Positive = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]


class RTPTorrentSpec(VersionedContract):
    """Frozen raw-class-table conversion, distinct from published filtered baseline tables.

    The selected member is a separately pinned data file, not an assertion about
    an unverified enclosing archive. Materialization retains exact URI/version,
    data terms and byte provenance. History is an explicit producer attestation;
    constructed fixtures never become observed historical CI. Limits bound input
    bytes, physical CSV line bytes and rows, not interpreter peak RSS or CPU time.
    No network, filesystem or transport is consulted by these models.
    """

    recipe: Literal["rtptorrent-class-results-v1"] = "rtptorrent-class-results-v1"
    materialization: MaterializationReceipt
    member: Annotated[Identifier, Field(max_length=1024)]
    project: Annotated[Identifier, Field(max_length=256)]
    history: Literal["observed", "constructed"]
    duration_semantics: UnitSemantics = REPORTED_SECONDS
    maximum_rows: Annotated[int, Field(strict=True, ge=1, le=50000)] = 50000
    maximum_line_bytes: Annotated[int, Field(strict=True, ge=128, le=16384)] = 16384

    @model_validator(mode="after")
    def _source(self) -> Self:
        """Require the pinned selected member and explicit RTPTorrent 1.1 format semantics."""
        if self.materialization.spec.source.version != "1.1":
            raise ValueError("RTPTorrent adapter requires explicit source format version 1.1")
        if self.duration_semantics != REPORTED_SECONDS:
            raise ValueError("RTPTorrent duration units/meaning cannot be reinterpreted")
        members = [item for item in self.materialization.spec.members if item.name == self.member]
        if len(members) != 1 or members[0].size_bytes > 16 * 1024**2:
            raise ValueError("RTPTorrent member must be pinned and at most 16 MiB")
        return self


class ClassExecution(Immutable):
    """One reported execution, retaining repeated classes and partial method-count knowledge.

    Duration is reported class elapsed seconds, not observed runner occupancy or
    parallel CI latency. Null means unavailable; zero is observed zero. Negative,
    nonfinite and overprecision durations reject. Counts are nonnegative method
    counts, never unique faults; an exception count is retained separately from
    assertion failures. No unknown/skipped/not-run observation becomes a pass.
    Source position distinguishes executions, not stable cross-job test identity.
    """

    job_id: Positive
    test_class: Annotated[Identifier, Field(max_length=1024)]
    position: Integer
    duration: Quantity | None
    duration_status: Literal["observed", "unavailable"]
    count: Integer | None
    failures: Integer | None
    errors: Integer | None
    skipped: Integer | None

    @model_validator(mode="after")
    def _semantics(self) -> Self:
        """Reject incompatible units and impossible disjoint method counts; never repair source values."""
        if self.duration is not None and (self.duration.semantics != REPORTED_SECONDS or self.duration.value > 10**12):
            raise ValueError("RTPTorrent duration requires reported class seconds within 10^12")
        if (self.duration is None) != (self.duration_status == "unavailable"):
            raise ValueError("Null duration requires explicit unavailable status; zero remains observed")
        known = sum(value for value in (self.failures, self.errors, self.skipped) if value is not None)
        if self.count is not None and known > self.count:
            raise ValueError("Failing/error/skipped method counts exceed total methods")
        return self

    @property
    def outcome(self) -> Literal["failed", "passed", "not_run", "unknown"]:
        """Derive only supported class verdicts; no inference about unique faults or flakiness."""
        if (self.failures or 0) + (self.errors or 0) > 0:
            return "failed"
        if any(value is None for value in (self.count, self.failures, self.errors, self.skipped)):
            return "unknown"
        return "not_run" if self.count == self.skipped else "passed"

    def execution_id(self, project: str) -> str:
        """Hash project/job/class/position labels independently of values and physical CSV row order."""
        return fingerprint(["rtptorrent-execution-v1", project, self.job_id, self.test_class, self.position])


class RTPTorrentDataset(VersionedContract):
    """Canonical execution records with explicit duplicate exclusions and unresolved admission.

    Reading checks invariants, not correspondence with cached source bytes; use
    validate_rtptorrent_dataset before trusting that correspondence. Identical
    repeated rows are counted/excluded explicitly; conflicting same-key rows
    reject. Distinct positions of the same class are retained. Canonical job-ID
    sorting is serialization order, never an observed chronology or independence
    assumption. Legacy native artifacts remain separate and are never migrated.
    """

    spec: RTPTorrentSpec
    input_rows: Integer
    duplicate_rows_excluded: Integer
    executions: Annotated[tuple[ClassExecution, ...], Field(min_length=1, max_length=50000)]
    admission: Literal["blocked_missing_temporal_evidence"] = "blocked_missing_temporal_evidence"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _records(self) -> Self:
        """Conserve physical rows and require unique execution keys without merging repeated classes."""
        if self.input_rows != len(self.executions) + self.duplicate_rows_excluded:
            raise ValueError("RTPTorrent row conservation failed")
        if self.input_rows > self.spec.maximum_rows:
            raise ValueError("RTPTorrent input exceeds declared row limit")
        keys = [(item.job_id, item.test_class, item.position) for item in self.executions]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate converted execution identity")
        object.__setattr__(
            self,
            "executions",
            tuple(sorted(self.executions, key=lambda item: (item.job_id, item.position, item.test_class))),
        )
        return self

    @property
    def job_ids(self) -> tuple[int, ...]:
        """Return canonical source job labels, with no claim that numeric order is chronological."""
        return tuple(sorted({item.job_id for item in self.executions}))

    @property
    def repeated_class_executions(self) -> int:
        """Count executions beyond the first project/job/class occurrence; they remain in the dataset."""
        return len(self.executions) - len({(item.job_id, item.test_class) for item in self.executions})

    @property
    def reported_zero_failure_jobs(self) -> int:
        """Count jobs with only passing/not-run source rows, not independently verified full CI passes."""
        counts = Counter(item.job_id for item in self.executions if item.outcome in {"passed", "not_run"})
        totals = Counter(item.job_id for item in self.executions)
        return sum(counts[job] == total for job, total in totals.items())

    def job(self, job_id: int) -> tuple[ClassExecution, ...]:
        """Select one isolated job without grouping it into a logical build; unknown labels raise KeyError."""
        if type(job_id) is not int or job_id < 1:
            raise ValueError("Job labels must be positive integers")
        rows = tuple(item for item in self.executions if item.job_id == job_id)
        if not rows:
            raise KeyError(job_id)
        return rows

    def require_temporal_evaluation(self) -> None:
        """Fail closed: raw tables supply neither event/outcome clocks nor a complete build/job mapping.

        Build numbers, job IDs, commit dates and serial-duration sums cannot
        substitute for observed outcome availability. This adapter never starts
        a protected historical/confirmatory experiment or silently changes its
        eligibility when rows are sorted, repeated or reserialized.
        """
        raise ValueError(
            "RTPTorrent evaluation blocked: event/outcome availability and logical build mapping unavailable"
        )
