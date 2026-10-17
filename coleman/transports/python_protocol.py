"""Bounded Python harness framing and replay validation, independent of Docker and file access."""

from __future__ import annotations

import base64
from typing import NamedTuple

from coleman.results.domain import decode_json
from coleman.transports.junit_contracts import ProducerElapsed
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.junit_source import digest
from coleman.transports.python_contracts import (
    PythonCaptureRecipe,
    PythonCase,
    PythonHarnessPlan,
    PythonHeader,
    PythonRoleCapture,
    PythonTrailer,
)

PREFIX = b"COLEMAN-PYTEST-V1\t"


class PythonRecords(NamedTuple):
    """Validated complete records; absent trailer means an incomplete stream, never successful collection."""

    header: PythonHeader | None
    cases: tuple[PythonCase, ...]
    trailer: PythonTrailer | None


def _header_bindings(header: PythonHeader, recipe: PythonCaptureRecipe, plan: PythonHarnessPlan) -> None:
    """Check actual runtime/discovery against predeclared identities, without inferring independent faults."""
    architectures = {"linux/arm64": {"aarch64"}, "linux/amd64": {"amd64", "x86_64"}}
    expected = (
        plan.context_fingerprint,
        plan.trial_id,
        plan.role,
        digest((plan.to_json() + "\n").encode()),
        recipe.runtime.version,
        recipe.runtime.timezone,
        recipe.runtime.locale,
        recipe.runtime.preferred_encoding,
        "ascii" if recipe.runtime.locale == "C" else "utf-8",
        plan.distributions,
        recipe.test_ids,
    )
    actual = (
        header.context_fingerprint,
        header.trial_id,
        header.role,
        header.plan_sha256,
        header.python_version,
        header.timezone,
        header.locale,
        header.preferred_encoding,
        header.filesystem_encoding,
        header.distributions,
        header.test_ids,
    )
    if actual != expected or header.architecture not in architectures[recipe.runtime.platform]:
        raise ValueError("Python actual runtime/discovery disagrees with the frozen plan")


def parsed_python_stream(payload: bytes, recipe: PythonCaptureRecipe, plan: PythonHarnessPlan) -> PythonRecords:
    """Parse LF-framed records, retaining complete case evidence before an interrupted final line.

    Non-protocol stdout remains in the raw packet and cannot become observations.
    A truncated final frame is not a record. Malformed complete frames, unknown
    versions, duplicate keys, repeated/reordered cases and extra protocol frames
    raise ValueError. This function does not grant completion: the caller must
    additionally check process termination, exit status and trailer consistency.
    Inputs are byte-bounded before JSON decoding; no I/O or subject code runs.
    """
    recipe = PythonCaptureRecipe.model_validate(recipe)
    plan = PythonHarnessPlan.model_validate(plan)
    if len(payload) > recipe.limits.maximum_output_bytes:
        raise ValueError("Python stream exceeds its byte limit")
    frames = [line[len(PREFIX) :] for line in payload.split(b"\n")[:-1] if line.startswith(PREFIX)]
    if not frames:
        return PythonRecords(None, (), None)
    if len(frames) > len(recipe.test_ids) + 2:
        raise ValueError("Python stream exceeds its record count")
    header = PythonHeader.from_json(frames[0])
    _header_bindings(header, recipe, plan)
    cases = []
    trailer = None
    for frame in frames[1:]:
        if trailer is not None:
            raise ValueError("Python protocol frame follows its trailer")
        # Decode with the same duplicate-key/version safeguards as each concrete model.
        data = decode_json(frame, exact_decimals=True)
        if isinstance(data, dict) and data.get("record_type") == "trailer":
            trailer = PythonTrailer.model_validate(data)
        else:
            case = PythonCase.model_validate(data)
            index = len(cases)
            if index >= len(recipe.test_ids) or case.test_id != recipe.test_ids[index]:
                raise ValueError("Python case sequence disagrees with the frozen universe")
            cases.append(case)
    if trailer is not None and trailer.cases != len(cases):
        raise ValueError("Python trailer disagrees with actual completed cases")
    if trailer is not None and payload.rsplit(b"\n", 1)[-1].startswith(PREFIX):
        raise ValueError("Truncated Python protocol frame follows its trailer")
    return PythonRecords(header, tuple(cases), trailer)


def python_role_capture(
    recipe: PythonCaptureRecipe, plan: PythonHarnessPlan, streams: ProcessStreams
) -> PythonRoleCapture:
    """Derive bounded capture state from actual streams, preserving assertions, errors and physical termination.

    A valid incomplete stream may retain completed cases; missing cases stay
    unobserved. Malformed protocol retains raw bytes but promotes no cases.
    Producer elapsed includes startup/dependency installation and is never a
    scheduling cost. A normal pytest failure is still a collected result.
    """
    records = PythonRecords(None, (), None)
    diagnostics = ""
    try:
        records = parsed_python_stream(streams.stdout, recipe, plan)
    except (ValueError, RecursionError) as exc:
        status = "protocol_error"
        diagnostics = str(exc)[:4096]
    else:
        if streams.termination != "exited":
            status = streams.termination
        elif streams.exit_code != 0:
            status = "process_error"
        elif (
            records.header is None
            or records.trailer is None
            or len(records.cases) != len(recipe.test_ids)
            or records.trailer.pytest_exit_code not in {0, 1}
            or (records.trailer.pytest_exit_code == 1)
            != any(case.outcome in {"assertion_failed", "test_error"} for case in records.cases)
        ):
            status = "protocol_error"
            diagnostics = "Exited Python harness lacks consistent completion evidence"
        else:
            status = "collected"
    return PythonRoleCapture.model_validate(
        {
            "schema_version": 1,
            "context_fingerprint": plan.context_fingerprint,
            "trial_id": plan.trial_id,
            "role": plan.role,
            "status": status,
            "termination": streams.termination,
            "exit_code": streams.exit_code,
            "producer_elapsed": ProducerElapsed(value=streams.elapsed_ns, status="observed"),
            "stdout_base64": base64.b64encode(streams.stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(streams.stderr).decode("ascii"),
            "header": records.header,
            "cases": records.cases,
            "trailer": records.trailer,
            "diagnostics": diagnostics,
        }
    )


def validate_python_role_bytes(
    capture: PythonRoleCapture, recipe: PythonCaptureRecipe, plan: PythonHarnessPlan
) -> None:
    """Rebuild stored outcomes from raw bytes and frozen plan; reject coherent-hash metadata rewrites.

    Canonical base64, combined byte cap, physical limit/deadline and all normalized
    case fields are rechecked. Started/uncommitted evidence remains unavailable.
    These are consistency checks, not cryptographic authentication of execution.
    Invalid artifacts raise ValueError; this never starts Docker or accesses files.
    """
    capture = PythonRoleCapture.model_validate(capture)
    recipe = PythonCaptureRecipe.model_validate(recipe)
    plan = PythonHarnessPlan.model_validate(plan)
    if (capture.context_fingerprint, capture.trial_id, capture.role) != (
        plan.context_fingerprint,
        plan.trial_id,
        plan.role,
    ):
        raise ValueError("Python capture identity disagrees with its frozen plan")
    stdout, stderr = _decoded_streams(capture, recipe.limits.maximum_output_bytes)
    size = len(stdout) + len(stderr)
    if capture.termination is None:
        if stdout or stderr or capture.header is not None or capture.cases or capture.trailer is not None:
            raise ValueError("Unobserved Python process cannot contain execution evidence")
        return
    if capture.termination == "output_limit" and size != recipe.limits.maximum_output_bytes:
        raise ValueError("Python output-limit evidence must fill the complete bounded prefix")
    if capture.termination == "timeout" and (
        capture.producer_elapsed.value is None or capture.producer_elapsed.value < recipe.limits.timeout_seconds * 10**9
    ):
        raise ValueError("Python timeout elapsed must reach the declared deadline")
    if capture.exit_code is None or capture.producer_elapsed.value is None:
        raise ValueError("Attempted Python process lacks exit/elapsed evidence")
    expected = python_role_capture(
        recipe,
        plan,
        ProcessStreams(stdout, stderr, capture.exit_code, capture.producer_elapsed.value, capture.termination),
    )
    if capture != expected:
        raise ValueError("Stored Python status/cases disagree with raw streams, plan or process evidence")


def _decoded_streams(capture: PythonRoleCapture, maximum_bytes: int) -> tuple[bytes, bytes]:
    """Decode exact canonical base64 under a combined raw-byte bound; never infer absent process evidence."""
    try:
        stdout = base64.b64decode(capture.stdout_base64, validate=True)
        stderr = base64.b64decode(capture.stderr_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid Python capture base64") from exc
    if (
        base64.b64encode(stdout).decode("ascii") != capture.stdout_base64
        or base64.b64encode(stderr).decode("ascii") != capture.stderr_base64
        or len(stdout) + len(stderr) > maximum_bytes
    ):
        raise ValueError("Noncanonical or oversized stored Python streams")
    return stdout, stderr
