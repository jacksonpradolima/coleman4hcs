"""Versioned original Westermo source records; anonymized labels never become UTC clocks."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.materialization_contracts import MaterializationSpec
from coleman.results.domain import Fingerprint, canonical_json, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, UnitSemantics

Count = Annotated[int, Field(strict=True, ge=0, le=2_000_000)]
SourceID = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
Label = Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
Duration = Annotated[int, Field(strict=True, ge=0, le=10**12)]
SESSION_SECONDS = UnitSemantics(unit="s", meaning="westermo-reported-session-envelope-v1")
TEST_DURATION = UnitSemantics(unit="westermo-oduration-unverified", meaning="westermo-original-test-duration-v1")

Blocker = Literal[
    "test_duration_unit_unverified",
    "outcome_availability_unverified",
    "feature_availability_unverified",
    "expected_test_universe_unverified",
    "physical_execution_topology_unverified",
]
WESTERMO_BLOCKERS: tuple[Blocker, ...] = (
    "test_duration_unit_unverified",
    "outcome_availability_unverified",
    "feature_availability_unverified",
    "expected_test_universe_unverified",
    "physical_execution_topology_unverified",
)


class WestermoSessionKey(Immutable):
    """Author-defined session identity; system copies are dependent members of one source family."""

    jid: SourceID
    system: Label


class WestermoRecipe(VersionedContract):
    """Pin one licensed original TSV and a bounded explicit session selection.

    Shifted dates stay naive source labels. Test duration has an unverified unit,
    so this recipe cannot supply native second budgets or protected evaluation.
    Only original statuses 0/1/2/3 are supported. No source is acquired by a model.
    Unknown versions reject; no legacy/native artifacts are migrated.
    """

    recipe: Literal["westermo-original-session-parquet-v1"] = "westermo-original-session-parquet-v1"
    source: MaterializationSpec
    sessions: Annotated[tuple[WestermoSessionKey, ...], Field(min_length=1, max_length=128)]
    session_duration: UnitSemantics = SESSION_SECONDS
    test_duration: UnitSemantics = TEST_DURATION
    clock: Literal["author-shifted-naive-labels-not-utc"] = "author-shifted-naive-labels-not-utc"
    independent_family: Literal["westermo-original-release"] = "westermo-original-release"

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject incompatible units, duplicate selections and unsupported decoded formats/bounds."""
        if self.session_duration != SESSION_SECONDS or self.test_duration != TEST_DURATION:
            raise ValueError("Incompatible Westermo units or quantity semantics")
        if len(self.source.members) != 1 or self.source.members[0].size_bytes > 128 * 1024**2:
            raise ValueError("Westermo requires one TSV member within 128 MiB")
        keys = [(item.jid, item.system) for item in self.sessions]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate Westermo session selection")
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > 1024**2:
            raise ValueError("Westermo canonical recipe metadata exceeds 1 MiB")
        object.__setattr__(self, "sessions", tuple(sorted(self.sessions, key=lambda item: (item.jid, item.system))))
        return self


class WestermoSession(Immutable):
    """One repeated session header; NULL duration means abnormal/incomplete, never zero.

    Seconds measure the reported session envelope, not additive runner work.
    Release/testware labels do not prove planned membership or feature availability.
    """

    key: WestermoSessionKey
    sessiontype: Literal["N", "I", "Q"]
    rel: Label
    sstarted: Label
    sduration: Duration | None
    fgstate: Label

    @field_validator("sstarted")
    @classmethod
    def _shifted_label(cls, value: str) -> str:
        """Validate original second-resolution calendar syntax without adding a timezone or reversing anonymization."""
        if datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S") != value:
            raise ValueError("Westermo requires a canonical shifted naive date label")
        return value


class WestermoObservation(Immutable):
    """One original outcome ID; invalid/unmappable outcomes are not failing tests.

    Per-test numeric duration is preserved without asserting seconds. Parameters
    retain literal source text (including NULL); this does not impute missingness.
    Physical row position is provenance, excluded from logical execution identity.
    """

    source_position: Annotated[int, Field(strict=True, ge=1, le=2_000_000)]
    session: WestermoSession
    oid: SourceID
    tid: SourceID
    path: Label
    parameters: Label
    result: Annotated[int, Field(strict=True, ge=0, le=3)]
    ostarted: Label
    oduration: Duration

    @field_validator("ostarted")
    @classmethod
    def _shifted_label(cls, value: str) -> str:
        """Require the same naive anonymized source syntax as session dates."""
        return WestermoSession._shifted_label(value)

    @property
    def execution_id(self) -> str:
        """Return a value/position-independent original-release execution key, without I/O."""
        return fingerprint(["westermo-original-outcome-v1", self.oid])


class WestermoCounts(Immutable):
    """Conserve every input row, selected verdict and selected complete/incomplete session."""

    input_rows: Count
    selected_rows: Count
    outside_selection: Count
    passed: Count
    failed: Count
    invalid: Count
    unmappable: Count
    complete_sessions: Count
    incomplete_sessions: Count

    @model_validator(mode="after")
    def _conservation(self) -> Self:
        """Reject inconsistent row and outcome totals; no status is excluded from the analytical view."""
        if self.input_rows != self.selected_rows + self.outside_selection:
            raise ValueError("Westermo input row conservation failed")
        if self.selected_rows != self.passed + self.failed + self.invalid + self.unmappable:
            raise ValueError("Westermo verdict conservation failed")
        return self


class WestermoReceipt(VersionedContract):
    """Completion metadata; actual source/Parquet bytes require offline validation.

    Publication is complete-last and immutable. The logical digest excludes row
    positions; physical digests additionally bind source layout and writer version.
    Version 1 has no migration: retain the recorded package/Arrow implementation
    to revalidate old exports. Neither parsing nor conversion grants admission.
    """

    recipe: WestermoRecipe
    producer_sha256: Fingerprint
    arrow_writer: Label
    counts: WestermoCounts
    files: Annotated[tuple[ETLFile, ...], Field(min_length=2, max_length=2)]
    logical_rows_sha256: Fingerprint
    admission: Literal["blocked_source_evidence"] = "blocked_source_evidence"
    blockers: tuple[Blocker, ...] = WESTERMO_BLOCKERS

    @model_validator(mode="after")
    def _files(self) -> Self:
        """Bind the exact output roles, row counts, selected sessions and immutable blocker set."""
        sessions = len(self.recipe.sessions)
        if self.counts.complete_sessions + self.counts.incomplete_sessions != sessions:
            raise ValueError("Westermo selected session conservation failed")
        if self.counts.selected_rows < sessions:
            raise ValueError("Every selected Westermo session requires an observation")
        if self.counts.selected_rows > 20_000:
            raise ValueError("Westermo selected slice exceeds 20000 observations")
        if tuple((item.path, item.rows) for item in self.files) != (
            ("observations.parquet", self.counts.selected_rows),
            ("sessions.parquet", sessions),
        ):
            raise ValueError("Westermo output file roles or row counts disagree")
        if self.blockers != WESTERMO_BLOCKERS:
            raise ValueError("Westermo source blockers cannot be removed by editing metadata")
        return self
