"""Closed raw Jenkins DTOs used only at the LRTS JSON input boundary."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, JsonValue

from coleman.corpus.lrts_contracts import Integer, Name, Status
from coleman.results.domain import Identifier
from coleman.selection.contracts import Immutable


def _number(value: Any) -> Decimal | int:
    """Require actual JSON numeric tokens; booleans, strings and nonfinite constants reject."""
    if type(value) is int and 0 <= value <= 10**12:
        return value
    if isinstance(value, Decimal) and value.is_finite() and 0 <= value <= 10**12:
        return value
    raise ValueError("Jenkins durations require finite JSON numbers within [0,10^12] seconds or explicit null")


Number = Annotated[Decimal, BeforeValidator(_number)]
Diagnostic = Annotated[str, Field(strict=True, max_length=1024**2)]
Actions = Annotated[tuple[JsonValue, ...], Field(max_length=128)]


class RawCase(Immutable):
    """Known method schema; diagnostics/history counters are validated but never decision features."""

    class_name: Annotated[Identifier, Field(alias="className", max_length=1024)]
    method_name: Annotated[Identifier, Field(alias="name", max_length=1024)]
    duration: Number | None
    status: Status | None
    skipped: Annotated[bool, Field(strict=True)]
    age: Annotated[int, Field(strict=True)] | None = None
    failed_since: Annotated[int, Field(strict=True)] | None = Field(default=None, alias="failedSince")
    error_details: Diagnostic | None = Field(default=None, alias="errorDetails")
    error_stack_trace: Diagnostic | None = Field(default=None, alias="errorStackTrace")
    skipped_message: Diagnostic | None = Field(default=None, alias="skippedMessage")
    stderr: Diagnostic | None = None
    stdout: Diagnostic | None = None
    test_actions: Actions | None = Field(default=None, alias="testActions")


class RawSuite(Immutable):
    """Retain raw node/enclosing identities; unavailable arrays do not become empty arrays."""

    name: Annotated[Identifier, Field(max_length=1024)]
    reported_id: Annotated[Identifier, Field(max_length=1024)] | None = Field(default=None, alias="id")
    node_id: Annotated[Identifier, Field(max_length=1024)] | None = Field(default=None, alias="nodeId")
    enclosing_blocks: Annotated[tuple[Name, ...], Field(max_length=64)] | None = Field(
        default=None, alias="enclosingBlocks"
    )
    enclosing_block_names: Annotated[tuple[Name, ...], Field(max_length=64)] | None = Field(
        default=None, alias="enclosingBlockNames"
    )
    timestamp: Annotated[str, Field(strict=True, max_length=128)] | None = None
    duration: Number | None
    cases: Annotated[tuple[RawCase, ...], Field(max_length=50000)]
    stderr: Diagnostic | None = None
    stdout: Diagnostic | None = None


class RawReport(Immutable):
    """Explicit Jenkins JUnit report layout, distinct from JUnit XML or processed class tables."""

    api_class: Literal["hudson.tasks.junit.TestResult"] = Field(alias="_class")
    duration: Number | None
    empty: Annotated[bool, Field(strict=True)]
    pass_methods: Integer = Field(alias="passCount")
    fail_methods: Integer = Field(alias="failCount")
    skip_methods: Integer = Field(alias="skipCount")
    suites: Annotated[tuple[RawSuite, ...], Field(max_length=5000)]
    test_actions: Actions | None = Field(default=None, alias="testActions")
