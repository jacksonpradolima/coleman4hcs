"""Bounded, pure CTest log parsing; no clocks, source acquisition or transport imports.

The closed v1 profile recognizes CTest project headers, indexed result lines,
summary counts and reported wall seconds. It retains partial invocations and
rejects malformed/ambiguous evidence rather than silently completing a job.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import NamedTuple

from coleman.corpus.hcs_contracts import HCSJob
from coleman.corpus.hcs_trace_contracts import (
    CTEST_ELAPSED,
    CTEST_WALL,
    CTestExecution,
    CTestRun,
    HCSTraceEvidence,
    ctest_status,
)
from coleman.selection.contracts import Quantity, UnitSemantics

_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# Scope ASCII to numeric tokens, retaining the existing Unicode whitespace behavior.
_RESULT_PREFIX = re.compile(r"^\s*(?a:\d+/\d+)\s+Test\s+#")
_RESULT = re.compile(r"^\s*(?a:(\d{1,5})/(\d{1,5})) Test\s+#(?a:(\d{1,5})): (.*)$")
_SUMMARY = re.compile(r"^(?a:(100|\d{1,2})% tests passed, (\d{1,5}) tests failed out of (\d{1,5}))\s*$")
_WALL = re.compile(r"^Total Test time \(real\) =\s*(\S+) sec\s*$")


class _ResultParts(NamedTuple):
    """Closed private result-line fields before immutable evidence validation."""

    position: int
    total: int
    index: int
    name: str
    status: str
    seconds: str


def _result_parts(match: re.Match[str]) -> _ResultParts:
    """Split bounded result suffixes without overlapping wildcard regex backtracking."""
    position, total, index, body = match.groups()
    parts = body.rsplit(None, 2)
    if len(parts) != 3 or parts[2] != "sec":
        raise ValueError("Malformed CTest result duration suffix")
    result, seconds, _ = parts
    if result.endswith("Passed"):
        prefix, raw_status = result[:-6], "Passed"
    else:
        prefix, marker, status = result.rpartition("***")
        if not marker:
            raise ValueError("Unsupported CTest result status")
        raw_status = marker + status
    padding = prefix.rstrip()
    name = padding.rstrip(".")
    if len(padding) - len(name) < 2 or not name.endswith((" ", "\t")):
        raise ValueError("Malformed CTest name/padding boundary")
    name = name.strip()
    if not name or len(name) > 1024 or len(raw_status) > 1024:
        raise ValueError("CTest name/status exceeds supported bounds")
    return _ResultParts(int(position), int(total), int(index), name, raw_status, seconds)


def _quantity(text: str, semantics: UnitSemantics) -> Quantity:
    """Parse at most 64 ASCII decimal characters; nonfinite/negative/exponent forms reject."""
    if len(text) > 64 or re.fullmatch(r"\d+(?:\.\d+)?", text, flags=re.ASCII) is None:
        raise ValueError("CTest reported seconds require bounded nonnegative decimal text")
    quantity = Quantity(value=Decimal(text), semantics=semantics)
    if quantity.value > 10**12:
        raise ValueError("CTest reported seconds exceed 10^12")
    return quantity


@dataclass
class _Invocation:
    """Private bounded parser state; no uncontracted mapping crosses a public boundary."""

    ordinal: int
    expected: int | None = None
    rows: list[CTestExecution] = field(default_factory=list)
    indices: set[int] = field(default_factory=set)
    positions: set[int] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    failed: int | None = None
    wall: Quantity | None = None

    def denominator(self, value: int) -> None:
        """Require one consistent positive printed denominator per invocation."""
        if not 1 <= value <= 10000 or (self.expected is not None and self.expected != value):
            raise ValueError("Conflicting or unsupported CTest denominator")
        self.expected = value

    def result(self, match: re.Match[str], line_number: int) -> None:
        """Append an original result before summary; reject indexed aliases and malformed units."""
        if self.failed is not None or self.wall is not None:
            raise ValueError("CTest result appears after invocation summary")
        position, total, index, name, raw_status, seconds = _result_parts(match)
        self.denominator(total)
        if index in self.indices or position in self.positions or name in self.names:
            raise ValueError("Duplicate CTest result index or name")
        self.rows.append(
            CTestExecution(
                index=index,
                completion_position=position,
                name=name,
                raw_status=raw_status,
                status=ctest_status(raw_status),
                elapsed=_quantity(seconds, CTEST_ELAPSED),
                source_line=line_number,
            )
        )
        self.indices.add(index)
        self.positions.add(position)
        self.names.add(name)

    def finish(self) -> CTestRun:
        """Build an immutable invocation; missing summary/wall/results remain explicitly partial."""
        complete = (
            self.expected is not None
            and len(self.rows) == self.expected
            and self.failed is not None
            and self.wall is not None
        )
        return CTestRun(
            ordinal=self.ordinal,
            state="complete" if complete else "partial",
            expected_results=self.expected,
            executions=tuple(self.rows),
            summary_failed=self.failed,
            wall=self.wall,
        )

    def summary(self, match: re.Match[str]) -> None:
        """Require one consistent summary, with CTest half-up rounding and a 99% cap when failures exist."""
        if self.failed is not None or self.wall is not None:
            raise ValueError("Duplicate or reordered CTest summary")
        percentage, failed, total = map(int, match.groups())
        self.denominator(total)
        if failed > total:
            raise ValueError("CTest summary failure count exceeds its denominator")
        expected = (200 * (total - failed) + total) // (2 * total)
        expected = min(expected, 99) if failed else expected
        if percentage != expected:
            raise ValueError("CTest summary percentage disagrees with reported counts")
        self.failed = failed

    def wall_measurement(self, text: str) -> None:
        """Accept bounded wall seconds only after a summary and without a duplicate measurement."""
        if self.failed is None or self.wall is not None:
            raise ValueError("CTest wall measurement requires one preceding summary")
        self.wall = _quantity(text, CTEST_WALL)


def _consume(invocation: _Invocation, line: str, line_number: int) -> bool:
    """Consume one closed-profile line; preserve unrelated build output without interpreting it."""
    if match := _RESULT.fullmatch(line):
        invocation.result(match, line_number)
        return True
    if match := _SUMMARY.fullmatch(line):
        invocation.summary(match)
        return True
    if match := _WALL.fullmatch(line):
        invocation.wall_measurement(match[1])
        return True
    if _RESULT_PREFIX.match(line) or "tests failed out of" in line or line.startswith("Total Test time (real)"):
        raise ValueError("Malformed or unsupported CTest result/summary line")
    return False


def _trace_text(payload: bytes) -> str:
    """Bound original bytes before UTF-8 decoding; do not normalize the source digest."""
    if not isinstance(payload, bytes) or len(payload) > 2 * 1024**2:
        raise ValueError("CTest trace requires at most 2 MiB of bytes")
    if b"\x00" in payload:
        raise ValueError("CTest trace contains NUL bytes")
    return payload.decode("utf-8")


def _checked_line(raw: str, number: int) -> str:
    """Enforce physical line bounds before ignoring ANSI CSI decoration for interpretation."""
    if number > 131072 or len(raw.encode("utf-8")) > 16384:
        raise ValueError("CTest trace line count or byte bound exceeded")
    return _CSI.sub("", raw)


def parse_hcs_ctest_trace(payload: bytes, job: HCSJob) -> HCSTraceEvidence:
    """Interpret a <=2 MiB UTF-8 job log, retaining at most 16 runs / 10,000 results.

    Job metadata is revalidated but not authenticated by this pure operation.
    ANSI CSI decoration is ignored for parsing only; digest/size use original
    bytes. Lines exceeding 16 KiB, NULs, unsupported result forms, duplicate
    indices/names and contradictory summaries raise ValueError. Invalid UTF-8
    raises UnicodeDecodeError. No log contents are included in errors.

    Interruption without a final CTest summary/wall measurement yields a partial
    invocation, never a falsely completed result. Logs with no CTest header yield
    no runs. A printed denominator does not establish predecision membership or
    source availability. No I/O, clock access, admission, repair or old-artifact
    mutation occurs. Repeated invocations retain separate ordinal identities.
    """
    text = _trace_text(payload)
    job = HCSJob.model_validate(job)
    runs: list[CTestRun] = []
    invocation: _Invocation | None = None
    total = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = _checked_line(raw, number)
        if line.startswith("Test project "):
            if invocation is not None:
                runs.append(invocation.finish())
            if len(runs) >= 16:
                raise ValueError("CTest trace invocation bound exceeded")
            invocation = _Invocation(len(runs) + 1)
        elif invocation is not None:
            before = len(invocation.rows)
            _consume(invocation, line, number)
            total += len(invocation.rows) - before
            if total > 10000:
                raise ValueError("CTest trace result bound exceeded")
        elif _RESULT_PREFIX.match(line):
            raise ValueError("CTest result has no preceding project header")
    if invocation is not None:
        runs.append(invocation.finish())
    return HCSTraceEvidence(
        schema_version=1,
        job=job,
        source_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        source_size_bytes=len(payload),
        runs=tuple(runs),
    )


def validate_hcs_ctest_trace(trace: HCSTraceEvidence, payload: bytes) -> None:
    """Reparse actual original bytes and require every retained field to match.

    A receipt's self-declared digest is insufficient. This pure operation checks
    digest, byte count, physical provenance, reported states and all invocations
    using the recorded v1 profile. Mismatch raises ValueError. It authenticates
    neither the supplied job identity nor source permissions; use separately
    verified source metadata and pins. No artifact is written or reinterpreted.
    """
    trace = HCSTraceEvidence.model_validate(trace)
    if trace != parse_hcs_ctest_trace(payload, trace.job):
        raise ValueError("CTest evidence differs from actual original log bytes")
