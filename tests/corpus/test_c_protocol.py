"""Owned C protocol messages; no compiler, external input or container is accessed."""

import base64
import json

import pytest

from coleman.transports.c_contracts import CCaptureRecipe, CHarnessPlan, CInputCase, CRoleCapture, CRuntimeImage
from coleman.transports.c_protocol import PREFIX, c_role_capture, parsed_c_stream, validate_c_role_bytes
from coleman.transports.junit_process import ProcessStreams
from tests.corpus.test_c_contracts import owned_case, owned_header
from tests.corpus.test_python_protocol import owned_view
from tests.corpus.test_source_views import changed, digest


def owned_recipe():
    """Declare a synthetic metadata-only C function pair and one known-answer input."""
    return CCaptureRecipe(
        schema_version=1,
        recipe_id="owned-c-schema-only",
        project="owned-ci",
        fault_id="owned-assertion",
        dependence_group="owned-single-fixture",
        buggy=owned_view("subject.c", "buggy-owned"),
        fixed=owned_view("subject.c", "fixed-owned"),
        fixture=owned_view("support.c", "fixed-owned"),
        source_member="subject.c",
        support_units=("support.c",),
        buggy_function="bad",
        fixed_function="good",
        cases=(CInputCase(test_id=owned_case().test_id, value=0),),
        runtime=CRuntimeImage(
            version="owned-runtime",
            target="owned-target",
            python_version="owned",
            libc_version="owned",
            platform="linux/arm64",
            image_digest=digest(b"owned image"),
            emulated=False,
        ),
        harness_sha256=digest(b"owned harness"),
        collector_sha256=digest(b"owned collector"),
        limitations=("Owned schema fixture only; not execution evidence",),
    )


def owned_plan():
    """Freeze metadata-only producer bootstrap without accessing its mount paths."""
    h = owned_header()
    return CHarnessPlan(
        schema_version=1,
        context_fingerprint=h.context_fingerprint,
        trial_id=h.trial_id,
        role=h.role,
        recipe=owned_recipe(),
    )


def owned_records():
    """Bind authored frames to exact canonical bootstrap bytes."""
    h = changed(owned_header(), plan_sha256=digest((owned_plan().to_json() + "\n").encode()))
    return [
        h.model_dump(mode="json"),
        owned_case().model_dump(mode="json"),
        dict(schema_version=1, record_type="trailer", cases=1, oracle_exit_code=0),
    ]


def wire(rows):
    """Frame supplied owned records; Unicode separators remain JSON content."""
    return b"".join(PREFIX + json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)


def test_complete_roundtrip_and_raw_revalidation_preserve_observed_zero():
    spec, plan = owned_recipe(), owned_plan()
    streams = ProcessStreams(wire(owned_records()), b"owned stderr", 0, 0, "exited")
    capture = c_role_capture(spec, plan, streams)
    assert capture.status == "collected"
    assert capture.cases[0].elapsed.value == 0
    assert capture.producer_elapsed.value == 0
    assert CRoleCapture.from_json(capture.to_json()) == capture
    validate_c_role_bytes(capture, spec, plan)


@pytest.mark.parametrize("noise", [b"", b"pip installation log\n", "owned\u2028separator\n".encode()])
def test_nonprotocol_noise_is_not_an_observation(noise):
    spec, plan = owned_recipe(), owned_plan()
    raw = noise + wire(owned_records()) + b"unframed final output"
    capture = c_role_capture(spec, plan, ProcessStreams(raw, b"", 0, 1, "exited"))
    assert capture.status == "collected"
    assert base64.b64decode(capture.stdout_base64) == raw
    assert capture.cases == (owned_case(),)


@pytest.mark.parametrize("last", [b"", PREFIX + b'{"schema_version":1,', PREFIX + b'{"schema_version":999}'])
def test_timeout_retains_only_completed_cases_and_no_truncated_record(last):
    spec, plan = owned_recipe(), owned_plan()
    raw = wire(owned_records()[:2]) + last
    capture = c_role_capture(spec, plan, ProcessStreams(raw, b"", -9, spec.limits.timeout_seconds * 10**9, "timeout"))
    assert capture.status == "timeout"
    assert capture.termination == "timeout"
    assert capture.trailer is None
    assert capture.cases == (owned_case(),)
    validate_c_role_bytes(capture, spec, plan)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[0].update(python_version="other"),
        lambda rows: rows[0].update(plan_sha256=digest(b"other plan")),
        lambda rows: rows[0].update(locale="C.UTF-8"),
        lambda rows: rows[0].update(compiler_target="other"),
        lambda rows: rows[0].update(architecture="x86_64"),
        lambda rows: rows[0].update(test_ids=[]),
        lambda rows: rows[1].update(schema_version=2),
        lambda rows: rows[1].update(test_id="tests/test_owned.py::test_extra"),
        lambda rows: rows[1]["elapsed"].update(unit="s"),
        lambda rows: rows[1]["elapsed"].update(value=None, status="unavailable"),
        lambda rows: rows[2].update(cases=0),
        lambda rows: rows.append(rows[1]),
        lambda rows: rows.insert(2, rows[1]),
    ],
)
def test_malformed_runtime_units_identity_order_or_future_frames_never_promote_cases(mutate):
    rows = owned_records()
    mutate(rows)
    capture = c_role_capture(owned_recipe(), owned_plan(), ProcessStreams(wire(rows), b"", 0, 1, "exited"))
    assert capture.status == "protocol_error"
    assert capture.cases == ()
    validate_c_role_bytes(capture, owned_recipe(), owned_plan())


def test_malformed_protocol_preserves_physical_timeout_instead_of_claiming_exit():
    raw = PREFIX + b'{"schema_version":999}\n'
    spec = owned_recipe()
    capture = c_role_capture(
        spec, owned_plan(), ProcessStreams(raw, b"", -9, spec.limits.timeout_seconds * 10**9, "timeout")
    )
    assert capture.status == "protocol_error"
    assert capture.termination == "timeout"
    validate_c_role_bytes(capture, spec, owned_plan())


@pytest.mark.parametrize("exit_code", [0, 1, -9])
def test_absent_footer_or_failed_process_never_claims_collected(exit_code):
    capture = c_role_capture(
        owned_recipe(), owned_plan(), ProcessStreams(wire(owned_records()[:2]), b"", exit_code, 1, "exited")
    )
    assert capture.status == ("protocol_error" if exit_code == 0 else "process_error")
    assert capture.trailer is None
    validate_c_role_bytes(capture, owned_recipe(), owned_plan())


def test_duplicate_keys_reject_and_failed_test_remains_valid_completed_collection():
    raw = wire(owned_records()).replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1', 1)
    spec, plan = owned_recipe(), owned_plan()
    with pytest.raises(ValueError):
        parsed_c_stream(raw, spec, plan)
    rows = owned_records()
    rows[1].update(outcome="assertion_failed", stdout="owned assertion")
    rows[2]["oracle_exit_code"] = 1
    capture = c_role_capture(owned_recipe(), owned_plan(), ProcessStreams(wire(rows), b"", 0, 1, "exited"))
    assert capture.status == "collected"
    assert capture.cases[0].outcome == "assertion_failed"
    assert capture.cases[0].stdout == "owned assertion"


def test_rewritten_normalized_observations_reject_even_when_contract_is_coherent():
    spec, plan = owned_recipe(), owned_plan()
    capture = c_role_capture(owned_recipe(), owned_plan(), ProcessStreams(wire(owned_records()), b"", 0, 1, "exited"))
    rewritten = changed(capture, cases=(changed(capture.cases[0], stderr="rewritten"),))
    with pytest.raises(ValueError, match="disagree with raw"):
        validate_c_role_bytes(rewritten, spec, plan)
    other = changed(plan, trial_id="another-trial")
    with pytest.raises(ValueError, match="identity"):
        validate_c_role_bytes(capture, spec, other)


@pytest.mark.parametrize("status,elapsed,raw", [("timeout", 0, b""), ("output_limit", 1, b"too short")])
def test_parent_guard_claims_require_deadline_or_full_prefix(status, elapsed, raw):
    spec, plan = owned_recipe(), owned_plan()
    capture = c_role_capture(spec, plan, ProcessStreams(raw, b"", -9, elapsed, status))
    with pytest.raises(ValueError, match="deadline|complete bounded prefix"):
        validate_c_role_bytes(capture, spec, plan)


def test_declaration_order_canonical_but_record_order_is_significant():
    spec = owned_recipe()
    other = CInputCase(test_id="second", value=1)
    pair = changed(spec, cases=(*spec.cases, other))
    assert changed(pair, cases=tuple(reversed(pair.cases))).fingerprint == pair.fingerprint
    rows = owned_records()
    reordered = wire([rows[1], rows[0], rows[2]])
    plan = owned_plan()
    with pytest.raises(ValueError):
        parsed_c_stream(reordered, spec, plan)


def test_truncated_extra_protocol_after_footer_never_acquires_completion():
    raw = wire(owned_records()) + PREFIX + b'{"schema_version":1,'
    capture = c_role_capture(owned_recipe(), owned_plan(), ProcessStreams(raw, b"", 0, 1, "exited"))
    assert capture.status == "protocol_error"
    assert capture.cases == ()


@pytest.mark.parametrize("payload", ["%%%", "YR=="])
def test_invalid_and_noncanonical_base64_cannot_rewrite_raw_evidence(payload):
    spec, plan = owned_recipe(), owned_plan()
    capture = c_role_capture(spec, plan, ProcessStreams(wire(owned_records()), b"", 0, 0, "exited"))
    altered = changed(capture, stdout_base64=payload)
    with pytest.raises(ValueError, match="base64|Noncanonical"):
        validate_c_role_bytes(altered, spec, plan)


def test_combined_raw_stream_bound_is_revalidated_on_replay():
    spec, plan = owned_recipe(), owned_plan()
    capture = c_role_capture(spec, plan, ProcessStreams(wire(owned_records()), b"", 0, 0, "exited"))
    oversized = base64.b64encode(b"x" * (spec.limits.maximum_output_bytes + 1)).decode("ascii")
    altered = changed(capture, stdout_base64=oversized)
    with pytest.raises(ValueError, match="oversized"):
        validate_c_role_bytes(altered, spec, plan)


@pytest.mark.parametrize("outcome,stdout", [("passed", "0\n"), ("assertion_failed", "1\n")])
def test_even_coherent_raw_frames_must_satisfy_the_predeclared_independent_oracle(outcome, stdout):
    rows = owned_records()
    rows[1].update(outcome=outcome, stdout=stdout)
    rows[2]["oracle_exit_code"] = int(outcome != "passed")
    actual = c_role_capture(owned_recipe(), owned_plan(), ProcessStreams(wire(rows), b"", 0, 0, "exited"))
    assert actual.status == "protocol_error"
    assert not actual.cases
    validate_c_role_bytes(actual, owned_recipe(), owned_plan())
