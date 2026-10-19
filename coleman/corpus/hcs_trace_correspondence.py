"""Exact original CTest/archived-CSV correspondence without silently upgrading old evidence."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.hcs_contracts import HCSObservation, HCSSlice
from coleman.corpus.hcs_trace_contracts import CTestExecution, HCSTraceEvidence
from coleman.results.domain import Fingerprint, fingerprint
from coleman.results.versioned import VersionedContract

Count = Annotated[int, Field(strict=True, ge=0, le=16000)]
SourcePosition = Annotated[int, Field(strict=True, ge=1, le=500000)]


class HCSTraceCorrespondence(VersionedContract):
    """Content-bound correspondence, not authentication or historical source admission.

    Every original source position remains accounted for. Exact repeated CSV
    observations are duplicates, not new trials; conflicting duplicates reject.
    The legacy mapping is explicitly any non-Passed CTest status -> 1, preserving
    its collapsed meaning. New trace categories remain separate. Unknown/future
    versions reject, with no implicit reinterpretation of old HCS artifacts.
    """

    trace_fingerprint: Fingerprint
    slice_fingerprint: Fingerprint
    execution_source_sha256: Fingerprint
    matched_observations: Count
    duplicate_rows: Count
    physical_positions: Annotated[tuple[SourcePosition, ...], Field(max_length=16000)]
    legacy_nonpass_rows: Count
    mapping: Literal["archived-any-nonpassed-ctest-status-v1"] = "archived-any-nonpassed-ctest-status-v1"
    admission: Literal["not_granted"] = "not_granted"

    @model_validator(mode="after")
    def _counts(self) -> Self:
        """Require unique positive physical positions and exact count conservation."""
        if self.physical_positions != tuple(sorted(set(self.physical_positions))):
            raise ValueError("HCS correspondence source positions require unique ascending order")
        if self.matched_observations + self.duplicate_rows != len(self.physical_positions):
            raise ValueError("HCS correspondence row conservation failed")
        if self.legacy_nonpass_rows > self.matched_observations:
            raise ValueError("HCS correspondence nonpass count exceeds unique observations")
        return self


def _same_observation(left: HCSObservation, right: HCSObservation) -> bool:
    """Ignore physical positions only; differing spelling/labels remain conflicting original content."""
    return left.model_dump(exclude={"source_position"}) == right.model_dump(exclude={"source_position"})


def _unique_observations(originals: tuple[HCSObservation, ...]) -> dict[str, HCSObservation]:
    """Retain one exact original value per test; count repeated physical rows separately."""
    unique: dict[str, HCSObservation] = {}
    for row in originals:
        previous = unique.setdefault(row.test_name, row)
        if not _same_observation(previous, row):
            raise ValueError("Conflicting original HCS observations cannot corroborate a trace")
    return unique


def _match_executions(executions: tuple[CTestExecution, ...], unique: dict[str, HCSObservation]) -> None:
    """Compare exact source name coverage, decimal seconds and the retained legacy nonpass meaning."""
    if set(unique) != {row.name for row in executions}:
        raise ValueError("CTest and archived HCS name coverage differ")
    for execution in executions:
        original = unique[execution.name]
        if original.duration is None or original.duration.value != execution.elapsed.value:
            raise ValueError("CTest and archived HCS decimal durations differ")
        if original.verdict != int(execution.status != "passed"):
            raise ValueError("CTest status disagrees with the explicit archived HCS nonpass mapping")


def corroborate_hcs_ctest_trace(trace: HCSTraceEvidence, source: HCSSlice) -> HCSTraceCorrespondence:
    """Compare one complete original invocation with every archived row for that exact job.

    Call ``read_hcs_slice`` against the verified original cache first and parse
    the pinned raw log separately. This pure boundary revalidates DTOs and checks
    exact job/pipeline/commit/variant metadata, name coverage, decimal seconds
    and the explicit legacy nonpass mapping. It does not authenticate supplied
    DTOs or perform network/filesystem access itself.

    Missing/extra results, unknown archived quantities, conflicting duplicates,
    partial traces or repeated invocations raise ValueError. Repeated invocations
    cannot be max-merged into the legacy table's single job/test key. Original
    zero remains zero; skipped/not-run/exception states never become a binary
    pass or assertion-failure observation in the richer trace. No source clock,
    test-universe-before-decision claim or historical admission is synthesized.
    """
    trace = HCSTraceEvidence.model_validate(trace)
    source = HCSSlice.model_validate(source)
    if trace.job not in source.jobs:
        raise ValueError("HCS trace job does not match the supplied source metadata")
    if len(trace.runs) != 1 or trace.runs[0].state != "complete":
        raise ValueError("HCS legacy correspondence requires one complete unambiguous CTest invocation")
    originals = tuple(row for row in source.observations if row.job == trace.job)
    unique = _unique_observations(originals)
    executions = trace.runs[0].executions
    _match_executions(executions, unique)
    return HCSTraceCorrespondence(
        schema_version=1,
        trace_fingerprint=trace.fingerprint,
        slice_fingerprint=fingerprint(source.model_dump(mode="json")),
        execution_source_sha256=source.recipe.executions.members[0].sha256,
        matched_observations=len(unique),
        duplicate_rows=len(originals) - len(unique),
        physical_positions=tuple(row.source_position for row in originals),
        legacy_nonpass_rows=sum(row.status != "passed" for row in executions),
    )
