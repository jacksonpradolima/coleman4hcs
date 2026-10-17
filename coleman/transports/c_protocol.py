"""Bounded C harness framing and replay validation, independent of Docker and file access."""

from __future__ import annotations

import base64
from typing import NamedTuple

from coleman.results.domain import decode_json
from coleman.transports.c_contracts import (
    CCaptureRecipe,
    CCase,
    CHarnessPlan,
    CHeader,
    CRoleCapture,
    CTrailer,
)
from coleman.transports.junit_contracts import ProducerElapsed
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.junit_source import digest

PREFIX = b"COLEMAN-C-V1\t"


class CRecords(NamedTuple):
    """Validated complete records; absent trailer means an incomplete stream, never successful collection."""

    header: CHeader | None
    cases: tuple[CCase, ...]
    trailer: CTrailer | None


def _header_bindings(header: CHeader, recipe: CCaptureRecipe, plan: CHarnessPlan) -> None:
    """Check actual runtime/discovery against predeclared identities, without inferring independent faults."""
    architectures = {"linux/arm64": {"aarch64"}, "linux/amd64": {"amd64", "x86_64"}}
    expected = (
        plan.context_fingerprint,
        plan.trial_id,
        plan.role,
        digest((plan.to_json() + "\n").encode()),
        recipe.runtime.version,
        recipe.runtime.target,
        recipe.runtime.python_version,
        recipe.runtime.libc_version,
        recipe.runtime.timezone,
        recipe.runtime.locale,
        recipe.test_ids,
    )
    actual = (
        header.context_fingerprint,
        header.trial_id,
        header.role,
        header.plan_sha256,
        header.compiler_version,
        header.compiler_target,
        header.python_version,
        header.libc_version,
        header.timezone,
        header.locale,
        header.test_ids,
    )
    if actual != expected or header.architecture not in architectures[recipe.runtime.platform]:
        raise ValueError("C actual compiler/runtime/discovery disagrees with frozen plan")


def parsed_c_stream(payload: bytes, recipe: CCaptureRecipe, plan: CHarnessPlan) -> CRecords:
    """Parse LF-framed records, retaining complete case evidence before an interrupted final line.

    Non-protocol stdout remains in the raw packet and cannot become observations.
    A truncated final frame is not a record. Malformed complete frames, unknown
    versions, duplicate keys, repeated/reordered cases and extra protocol frames
    raise ValueError. This function does not grant completion: the caller must
    additionally check process termination, exit status and trailer consistency.
    Inputs are byte-bounded before JSON decoding; no I/O or subject code runs.
    """
    recipe = CCaptureRecipe.model_validate(recipe)
    plan = CHarnessPlan.model_validate(plan)
    if plan.recipe != recipe:
        raise ValueError("C bootstrap recipe disagrees with parser recipe")
    if len(payload) > recipe.limits.maximum_output_bytes:
        raise ValueError("C stream exceeds its byte limit")
    frames = [line[len(PREFIX) :] for line in payload.split(b"\n")[:-1] if line.startswith(PREFIX)]
    if not frames:
        return CRecords(None, (), None)
    if len(frames) > len(recipe.test_ids) + 2:
        raise ValueError("C stream exceeds its record count")
    header = CHeader.from_json(frames[0])
    _header_bindings(header, recipe, plan)
    cases, trailer = _case_frames(frames[1:], recipe)
    if trailer is not None and trailer.cases != len(cases):
        raise ValueError("C trailer disagrees with actual completed cases")
    if trailer is not None and payload.rsplit(b"\n", 1)[-1].startswith(PREFIX):
        raise ValueError("Truncated C protocol frame follows its trailer")
    return CRecords(header, cases, trailer)


def _case_frames(frames: list[bytes], recipe: CCaptureRecipe) -> tuple[tuple[CCase, ...], CTrailer | None]:
    """Parse already byte/count-bounded case/trailer frames, preserving the declared case order and schemas."""
    cases = []
    trailer = None
    for frame in frames:
        if trailer is not None:
            raise ValueError("C protocol frame follows its trailer")
        # Decode with the same duplicate-key/version safeguards as each concrete model.
        data = decode_json(frame, exact_decimals=True)
        if isinstance(data, dict) and data.get("record_type") == "trailer":
            trailer = CTrailer.model_validate(data)
        else:
            case = CCase.model_validate(data)
            index = len(cases)
            if index >= len(recipe.test_ids) or case.test_id != recipe.test_ids[index]:
                raise ValueError("C case sequence disagrees with the frozen universe")
            expected = recipe.cases[index].expected_stdout
            if (
                case.termination == "exited"
                and case.exit_code == 0
                and (case.outcome == "passed") != (case.stdout == expected and case.stderr == "")
            ):
                raise ValueError("C normalized outcome disagrees with the independent output oracle")
            cases.append(case)
    return tuple(cases), trailer


def c_role_capture(recipe: CCaptureRecipe, plan: CHarnessPlan, streams: ProcessStreams) -> CRoleCapture:
    """Derive bounded capture state from actual streams, preserving assertions, errors and physical termination.

    A valid incomplete stream may retain completed cases; missing cases stay
    unobserved. Malformed protocol retains raw bytes but promotes no cases.
    Producer elapsed includes startup/dependency installation and is never a
    scheduling cost. A normal oracle failure is still a collected result.
    """
    records = CRecords(None, (), None)
    diagnostics = ""
    try:
        records = parsed_c_stream(streams.stdout, recipe, plan)
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
            or records.trailer.oracle_exit_code not in {0, 1}
            or (records.trailer.oracle_exit_code == 1)
            != any(case.outcome in {"assertion_failed", "test_error", "timeout"} for case in records.cases)
        ):
            status = "protocol_error"
            diagnostics = "Exited C harness lacks consistent completion evidence"
        else:
            status = "collected"
    return CRoleCapture.model_validate(
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


def validate_c_role_bytes(capture: CRoleCapture, recipe: CCaptureRecipe, plan: CHarnessPlan) -> None:
    """Rebuild stored outcomes from raw bytes and frozen plan; reject coherent-hash metadata rewrites.

    Canonical base64, combined byte cap, physical limit/deadline and all normalized
    case fields are rechecked. Started/uncommitted evidence remains unavailable.
    These are consistency checks, not cryptographic authentication of execution.
    Invalid artifacts raise ValueError; this never starts Docker or accesses files.
    """
    capture = CRoleCapture.model_validate(capture)
    recipe = CCaptureRecipe.model_validate(recipe)
    plan = CHarnessPlan.model_validate(plan)
    if (capture.context_fingerprint, capture.trial_id, capture.role) != (
        plan.context_fingerprint,
        plan.trial_id,
        plan.role,
    ):
        raise ValueError("C capture identity disagrees with its frozen plan")
    stdout, stderr = _decoded_streams(capture, recipe.limits.maximum_output_bytes)
    size = len(stdout) + len(stderr)
    if capture.termination is None:
        if stdout or stderr or capture.header is not None or capture.cases or capture.trailer is not None:
            raise ValueError("Unobserved C process cannot contain execution evidence")
        return
    if capture.termination == "output_limit" and size != recipe.limits.maximum_output_bytes:
        raise ValueError("C output-limit evidence must fill the complete bounded prefix")
    if capture.termination == "timeout" and (
        capture.producer_elapsed.value is None or capture.producer_elapsed.value < recipe.limits.timeout_seconds * 10**9
    ):
        raise ValueError("C timeout elapsed must reach the declared deadline")
    if capture.exit_code is None or capture.producer_elapsed.value is None:
        raise ValueError("Attempted C process lacks exit/elapsed evidence")
    expected = c_role_capture(
        recipe,
        plan,
        ProcessStreams(stdout, stderr, capture.exit_code, capture.producer_elapsed.value, capture.termination),
    )
    if capture != expected:
        raise ValueError("Stored C status/cases disagree with raw streams, plan or process evidence")


def _decoded_streams(capture: CRoleCapture, maximum_bytes: int) -> tuple[bytes, bytes]:
    """Decode exact canonical base64 under a combined raw-byte bound; never infer absent process evidence."""
    try:
        stdout = base64.b64decode(capture.stdout_base64, validate=True)
        stderr = base64.b64decode(capture.stderr_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid C capture base64") from exc
    if (
        base64.b64encode(stdout).decode("ascii") != capture.stdout_base64
        or base64.b64encode(stderr).decode("ascii") != capture.stderr_base64
        or len(stdout) + len(stderr) > maximum_bytes
    ):
        raise ValueError("Noncanonical or oversized stored C streams")
    return stdout, stderr
