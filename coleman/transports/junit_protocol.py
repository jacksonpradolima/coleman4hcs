"""Strict bounded protocol interpretation for the optional JUnit source producer."""

from __future__ import annotations

import base64
from datetime import datetime

from coleman.transports.junit_contracts import (
    JunitCaptureRecipe,
    JunitCase,
    JunitHeader,
    JunitRoleCapture,
    JunitTrailer,
    ProducerElapsed,
)
from coleman.transports.junit_process import ProcessStreams


def parsed_junit_stream(payload: bytes, recipe: JunitCaptureRecipe) -> tuple[JunitHeader, tuple[JunitCase, ...]]:
    """Require one header, all declared cases and one trailer; do not interpret incomplete output.

    LF framing is byte-based so a Unicode line separator inside a JSON diagnostic
    cannot split a record. Duplicate keys, missing/future versions, unknown fields,
    wrong clocks/runtime/universe, repeated/missing cases and absent trailers
    reject with ValueError. The caller first applies the combined stream byte cap.
    No source/process/file/network access occurs and test failures remain outcomes.
    """
    recipe = JunitCaptureRecipe.model_validate(recipe)
    if len(payload) > recipe.limits.maximum_output_bytes or not payload.endswith(b"\n"):
        raise ValueError("JUnit stream is oversized or lacks its final LF")
    lines = payload.split(b"\n")[:-1]
    if not 3 <= len(lines) <= 10002:
        raise ValueError("JUnit stream record count is outside the declared bound")
    header = JunitHeader.from_json(lines[0])
    trailer = JunitTrailer.from_json(lines[-1])
    cases = tuple(JunitCase.from_json(line) for line in lines[1:-1])
    expected_architecture = {"linux/arm64": {"aarch64"}, "linux/amd64": {"amd64", "x86_64"}}
    if (
        header.java_version != recipe.runtime.version
        or header.architecture not in expected_architecture[recipe.runtime.platform]
        or header.timezone != recipe.runtime.timezone
    ):
        raise ValueError("JUnit actual runtime disagrees with the frozen recipe")
    identities = tuple(item.test_id for item in cases)
    if header.test_ids != recipe.test_ids or identities != recipe.test_ids or trailer.case_count != len(cases):
        raise ValueError("JUnit actual discovery/case sequence disagrees with the frozen universe")
    return header, cases


def _completed_protocol(
    payload: bytes, exit_code: int, recipe: JunitCaptureRecipe
) -> tuple[str, JunitHeader | None, tuple[JunitCase, ...]]:
    """Derive an exited process's status and cases from the actual code/bytes, including negative evidence."""
    if exit_code != 0:
        return "process_error", None, ()
    try:
        header, cases = parsed_junit_stream(payload, recipe)
    except (ValueError, RecursionError):
        return "protocol_error", None, ()
    return "collected", header, cases


def role_capture(
    recipe: JunitCaptureRecipe,
    context_fingerprint: str,
    trial_id: str,
    role: str,
    started_at: datetime,
    finished_at: datetime,
    streams: ProcessStreams,
) -> JunitRoleCapture:
    """Retain exact bounded raw bytes and distinguish process/timeout/protocol failures from assertions.

    Failed collection retains no promoted partial case observations; the raw
    packet remains auditable and the other role can still supply observations.
    Incomplete pairs therefore remain visible in generic controlled results.
    The producer timer includes container/compile overhead and is not a test cost.
    """
    status = "collected"
    header = None
    cases: tuple[JunitCase, ...] = ()
    if streams.termination != "exited":
        status = streams.termination
    else:
        status, header, cases = _completed_protocol(streams.stdout, streams.exit_code, recipe)
    return JunitRoleCapture.model_validate(
        {
            "schema_version": 1,
            "context_fingerprint": context_fingerprint,
            "trial_id": trial_id,
            "role": role,
            "status": status,
            "exit_code": streams.exit_code,
            "capture_started_at": started_at,
            "capture_finished_at": finished_at,
            "producer_elapsed": ProducerElapsed(value=streams.elapsed_ns, status="observed"),
            "stdout_base64": base64.b64encode(streams.stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(streams.stderr).decode("ascii"),
            "header": header,
            "cases": cases,
        }
    )


def validate_role_bytes(capture: JunitRoleCapture, recipe: JunitCaptureRecipe) -> None:
    """Reparse actual stored streams; typed metadata alone cannot substitute declared cases for raw evidence.

    Canonical base64, combined byte limits and successful record/runtime bindings
    are rechecked on resume. Exited process/protocol error statuses must agree
    with the actual exit code and protocol, preventing silent observation loss.
    Timeout/output-limit statuses remain producer attestations; raw bytes alone
    cannot establish which parent guard stopped collection.
    Schema errors raise ValueError; no files or source processes are accessed.
    """
    capture = JunitRoleCapture.model_validate(capture)
    recipe = JunitCaptureRecipe.model_validate(recipe)
    if capture.stdout_base64 is None or capture.stderr_base64 is None:
        if capture.status not in {"interrupted", "process_error"}:
            raise ValueError("Stored role claims unavailable streams for a collected process")
        return
    try:
        stdout = base64.b64decode(capture.stdout_base64, validate=True)
        stderr = base64.b64decode(capture.stderr_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid JUnit capture base64") from exc
    if (
        base64.b64encode(stdout).decode("ascii") != capture.stdout_base64
        or base64.b64encode(stderr).decode("ascii") != capture.stderr_base64
        or len(stdout) + len(stderr) > recipe.limits.maximum_output_bytes
    ):
        raise ValueError("Noncanonical or oversized stored JUnit streams")
    if capture.status in {"collected", "protocol_error", "process_error"}:
        if capture.exit_code is None:
            raise ValueError("Stored exited role requires an observed exit code")
        expected = _completed_protocol(stdout, capture.exit_code, recipe)
        if (capture.status, capture.header, capture.cases) != expected:
            raise ValueError("Stored JUnit status or observations disagree with raw streams and exit code")
