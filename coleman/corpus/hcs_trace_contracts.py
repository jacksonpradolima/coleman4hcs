"""Original CTest trace evidence, separate from legacy collapsed HCS verdicts.

These records certify internal consistency, not source authenticity, permissions,
predecision membership or historical availability. No source is admitted by a
successful parse. Existing HCS receipts and their readers remain unchanged.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.hcs_contracts import HCSJob
from coleman.results.domain import Fingerprint, Identifier, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

CTEST_ELAPSED = UnitSemantics(unit="s", meaning="ctest-reported-executable-elapsed-seconds-v1")
CTEST_WALL = UnitSemantics(unit="s", meaning="ctest-reported-run-wall-envelope-seconds-v1")
Index = Annotated[int, Field(strict=True, ge=1, le=10000)]
Label = Annotated[Identifier, Field(max_length=1024)]
CTestStatus = Literal["passed", "failed", "timeout", "not_run", "exception", "skipped"]
_RAW_STATUSES: dict[str, CTestStatus] = {
    "Passed": "passed",
    "***Failed": "failed",
    "***Timeout": "timeout",
    "***Not Run": "not_run",
    "***Skipped": "skipped",
}


def ctest_status(raw: str) -> CTestStatus:
    """Map only the documented result-line profile; retain exception text separately.

    Unknown statuses raise ValueError, rather than becoming passing tests. An
    exception is a process-level observation, not necessarily an assertion or
    a unique software fault. This operation performs no I/O or conversion.
    """
    if raw.startswith("***Exception: ") and len(raw) > len("***Exception: "):
        return "exception"
    status = _RAW_STATUSES.get(raw)
    if status is None:
        raise ValueError("Unsupported CTest result status")
    return status


class CTestExecution(Immutable):
    """One original indexed executable result with exact reported decimal seconds.

    Elapsed zero is an observed report value, including for skipped/not-run
    records; it does not assert execution. Timeout, exception, skipped and not-run
    states never become a binary test failure or pass. Physical line position is
    provenance only. Negative/nonfinite quantities and incompatible units reject.
    """

    index: Index
    completion_position: Index
    name: Label
    raw_status: Label
    status: CTestStatus
    elapsed: Quantity
    source_line: Annotated[int, Field(strict=True, ge=1, le=131072)]

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Bind category and unit to the retained raw status without changing old HCS labels."""
        if self.status != ctest_status(self.raw_status) or self.elapsed.semantics != CTEST_ELAPSED:
            raise ValueError("CTest result status or elapsed units disagree with the source profile")
        if self.elapsed.value > 10**12:
            raise ValueError("CTest elapsed seconds exceed 10^12")
        return self

    @property
    def binary_verdict(self) -> Literal[0, 1] | None:
        """Return reported passed/failed executable labels; other process outcomes stay unavailable."""
        if self.status == "passed":
            return 0
        return 1 if self.status == "failed" else None


class CTestRun(Immutable):
    """One trace-local invocation, including interrupted or absent result coverage.

    ``expected_results`` comes from printed N/M counters, never a predecision
    manifest. Null means the trace supplied no denominator. A partial invocation
    may retain all result lines but lack its final summary/wall measurement.
    Invocation order and original result indices are significant, not independent
    trials. Elapsed sums are deliberately not equated with the wall envelope.
    """

    ordinal: Annotated[int, Field(strict=True, ge=1, le=16)]
    state: Literal["complete", "partial"]
    expected_results: Index | None
    executions: Annotated[tuple[CTestExecution, ...], Field(max_length=10000)]
    summary_failed: Annotated[int, Field(strict=True, ge=0, le=10000)] | None
    wall: Quantity | None

    @model_validator(mode="after")
    def _coverage(self) -> Self:
        """Reject duplicate indices/names, contradictory counts and falsely complete interrupted traces."""
        self._identities()
        self._denominator()
        self._summary()
        self._completion()
        return self

    def _identities(self) -> None:
        """Allow parallel completion order while retaining unique physical source identities."""
        indices = [row.index for row in self.executions]
        positions = [row.completion_position for row in self.executions]
        names = [row.name for row in self.executions]
        lines = [row.source_line for row in self.executions]
        if len(set(indices)) != len(indices) or len(set(names)) != len(names) or lines != sorted(set(lines)):
            raise ValueError("Duplicate or reordered CTest result identity")
        if len(set(positions)) != len(positions):
            raise ValueError("Duplicate CTest completion position")

    def _denominator(self) -> None:
        """Require indexed results to fit the source's own printed denominator and seconds units."""
        if self.expected_results is None:
            if self.executions or self.summary_failed is not None:
                raise ValueError("CTest results require a reported denominator")
        elif any(row.completion_position > self.expected_results for row in self.executions):
            raise ValueError("CTest completion position exceeds its reported denominator")
        if self.wall is not None and (self.wall.semantics != CTEST_WALL or self.wall.value > 10**12):
            raise ValueError("Incompatible CTest wall units or bound")

    def _summary(self) -> None:
        """Conserve nonpass process counts without treating skipped executions as failed tests."""
        if self.summary_failed is not None:
            if self.expected_results is None or self.summary_failed > self.expected_results:
                raise ValueError("CTest summary failure count exceeds its denominator")
            failures = sum(row.status not in {"passed", "skipped"} for row in self.executions)
            if self.summary_failed != failures:
                raise ValueError("CTest summary disagrees with reported process outcomes")

    def _completion(self) -> None:
        """Retain partial state until every result, final summary and wall measurement exists."""
        complete = (
            self.expected_results is not None
            and len(self.executions) == self.expected_results
            and sorted(row.completion_position for row in self.executions) == list(range(1, self.expected_results + 1))
            and self.summary_failed is not None
            and self.wall is not None
        )
        if (self.state == "complete") != complete:
            raise ValueError("CTest completion state disagrees with available evidence")


class HCSTraceEvidence(VersionedContract):
    """Bounded raw-log interpretation; digest identifies bytes without redistributing them.

    Job identity is a supplied source declaration, not proven by log contents.
    Repeated invocations remain separate, and source lines retain physical
    provenance. Reformatting a log changes this evidence digest but not logical
    execution IDs. Unknown versions reject; no migration of old collapsed
    verdicts, availability clocks or source-purpose admission is implicit.
    """

    parser: Literal["ctest-result-lines-v1"] = "ctest-result-lines-v1"
    job: HCSJob
    source_sha256: Fingerprint
    source_size_bytes: Annotated[int, Field(strict=True, ge=0, le=2 * 1024**2)]
    runs: Annotated[tuple[CTestRun, ...], Field(max_length=16)]
    outcome_availability: Literal["unverified"] = "unverified"
    predecision_membership: Literal["unverified"] = "unverified"
    admission: Literal["not_granted"] = "not_granted"

    @model_validator(mode="after")
    def _invocations(self) -> Self:
        """Keep invocation order, all-line uniqueness and the global ten-thousand-result bound."""
        if [run.ordinal for run in self.runs] != list(range(1, len(self.runs) + 1)):
            raise ValueError("CTest invocations require consecutive original ordinals")
        lines = [row.source_line for run in self.runs for row in run.executions]
        if len(lines) > 10000 or lines != sorted(set(lines)):
            raise ValueError("CTest global result bound or physical line ordering failed")
        return self

    def execution_id(self, project: str, ordinal: int, index: int) -> str:
        """Return value-independent project/pipeline/job/invocation/index identity.

        The coordinate must exist in this evidence; invalid coordinates and
        empty/untrimmed project namespaces raise ValueError. No I/O occurs.
        """
        _project_namespace(project)
        if type(ordinal) is not int or type(index) is not int:
            raise ValueError("CTest coordinate requires integer invocation and index")
        run = next((run for run in self.runs if run.ordinal == ordinal), None)
        if run is None or not any(row.index == index for row in run.executions):
            raise ValueError("CTest coordinate does not exist")
        return fingerprint(["hcs-ctest-execution-v1", project, self.job.pipeline_id, self.job.job_id, ordinal, index])


def _project_namespace(project: str) -> None:
    """Reject ambiguous project namespaces before generating a logical execution identity."""
    if not isinstance(project, str) or not project or project.strip() != project or "\n" in project or "\r" in project:
        raise ValueError("CTest project requires a nonempty trimmed namespace")
