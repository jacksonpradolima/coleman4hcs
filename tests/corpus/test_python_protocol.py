"""Owned Python protocol fixtures; every runtime label is synthetic and no foreign subject executes."""

import base64
import json

import pytest

from coleman.corpus.source_views import SourceViewSpec, source_view_bytes
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.python_contracts import (
    DistributionVersion,
    PythonCaptureRecipe,
    PythonHarnessPlan,
    PythonRoleCapture,
)
from coleman.transports.python_protocol import (
    PREFIX,
    parsed_python_stream,
    python_role_capture,
    validate_python_role_bytes,
)
from tests.corpus.test_python_contracts import owned_case, owned_header
from tests.corpus.test_source_views import changed, digest, source_spec


def owned_view(name, version):
    """Build a schema-only Python source view, never executable upstream code."""
    raw = source_spec(name, b"Owned schema-only source fixture.\n")
    raw = changed(raw, source=changed(raw.source, version=version))
    payload = source_view_bytes(((name, b"Owned schema-only source fixture.\n"),))
    output = source_spec(name + "-view", payload, "zip", [raw.members[0].model_dump()])
    output = changed(output, source=changed(output.source, version=version))
    return SourceViewSpec(schema_version=1, inputs=(raw,), output=output)


def owned_recipe():
    """Declare synthetic schemas with no valid installable wheel or actual foreign revision."""
    package = source_spec("owned_pytest-1-py3-none-any.whl", b"Owned schema-only wheel bytes")
    package = changed(
        package,
        source=changed(package.source, uri="urn:coleman:owned_pytest-1-py3-none-any.whl", version="owned"),
    )
    metadata = source_spec("owned_pytest-1.dist-info/METADATA", b"owned metadata").members[0]
    wheel = source_spec("owned_pytest-1.dist-info/WHEEL", b"owned wheel").members[0]
    return PythonCaptureRecipe.model_validate(
        {
            "schema_version": 1,
            "recipe_id": "owned-python-schema-only",
            "project": "owned-ci",
            "fault_id": "owned-assertion",
            "fault_origin": "synthetic",
            "dependence_group": "owned-single-fixture",
            "buggy": owned_view("owned/__init__.py", "buggy-owned"),
            "fixed": owned_view("owned/__init__.py", "fixed-owned"),
            "fixture": owned_view(owned_case().test_id.split("::")[0], "fixed-owned"),
            "wheels": [
                {
                    "distribution": {"name": "pytest", "version": "owned"},
                    "package": package,
                    "members": [metadata, wheel],
                    "entry_count": 2,
                    "decoded_bytes": metadata.size_bytes + wheel.size_bytes,
                    "components": [
                        {
                            "component": "owned-schema-only",
                            "scope": "Owned metadata, not a licensed foreign distribution",
                            "license": changed(
                                package.license, redistribution="recipe_only", terms_sha256=metadata.sha256
                            ),
                            "notice_members": [metadata.name],
                            "additional_notices": [],
                            "attribution": "declared_upstream",
                            "citation": "Owned synthetic schema fixture only",
                        }
                    ],
                }
            ],
            "runtime": {
                "version": "owned-runtime",
                "platform": "linux/arm64",
                "image_digest": digest(b"Owned schema-only runtime"),
                "emulated": False,
                "locale": "C",
                "preferred_encoding": "ASCII",
                "provided_distributions": [{"name": "pip", "version": "owned"}],
            },
            "production_module": "owned",
            "test_ids": [owned_case().test_id],
            "harness_sha256": digest(b"Owned schema-only harness"),
            "limitations": ["Owned schema-only fixture; not installable, not source execution evidence"],
        }
    )


def owned_plan():
    """Freeze schema-only bootstrap roots and identities, without touching those paths."""
    return PythonHarnessPlan(
        schema_version=1,
        context_fingerprint=owned_header().context_fingerprint,
        trial_id=owned_header().trial_id,
        role="buggy",
        fixture_root="/fixture",
        production_root="/source",
        dependency_root="/tmp/dependencies",
        wheel_root="/wheels",
        wheels=("owned_pytest-1-py3-none-any.whl",),
        distributions=(*owned_header().distributions, DistributionVersion(name="pip", version="owned")),
        test_ids=owned_recipe().test_ids,
    )


def owned_records():
    """Return protocol messages bound to the exact synthetic bootstrap wire bytes."""
    header = changed(
        owned_header(),
        architecture="aarch64",
        distributions=owned_plan().distributions,
        plan_sha256=digest((owned_plan().to_json() + "\n").encode()),
    )
    return [
        header.model_dump(mode="json"),
        owned_case().model_dump(mode="json"),
        {"schema_version": 1, "record_type": "trailer", "cases": 1, "pytest_exit_code": 0},
    ]


def wire(rows):
    """Encode explicitly framed messages; Unicode separators are content, not line framing."""
    return b"".join(PREFIX + json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)


def test_complete_roundtrip_and_raw_revalidation_preserve_observed_zero():
    spec, plan = owned_recipe(), owned_plan()
    streams = ProcessStreams(wire(owned_records()), b"owned stderr", 0, 0, "exited")
    capture = python_role_capture(spec, plan, streams)
    assert capture.status == "collected"
    assert capture.cases[0].elapsed.value == 0
    assert capture.producer_elapsed.value == 0
    assert PythonRoleCapture.from_json(capture.to_json()) == capture
    validate_python_role_bytes(capture, spec, plan)


@pytest.mark.parametrize("noise", [b"", b"pip installation log\n", "owned\u2028separator\n".encode()])
def test_nonprotocol_noise_is_not_an_observation(noise):
    spec, plan = owned_recipe(), owned_plan()
    raw = noise + wire(owned_records()) + b"unframed final output"
    capture = python_role_capture(spec, plan, ProcessStreams(raw, b"", 0, 1, "exited"))
    assert capture.status == "collected"
    assert base64.b64decode(capture.stdout_base64) == raw
    assert capture.cases == (owned_case(),)


@pytest.mark.parametrize("last", [b"", PREFIX + b'{"schema_version":1,', PREFIX + b'{"schema_version":999}'])
def test_timeout_retains_only_completed_cases_and_no_truncated_record(last):
    spec, plan = owned_recipe(), owned_plan()
    raw = wire(owned_records()[:2]) + last
    capture = python_role_capture(
        spec, plan, ProcessStreams(raw, b"", -9, spec.limits.timeout_seconds * 10**9, "timeout")
    )
    assert capture.status == "timeout"
    assert capture.termination == "timeout"
    assert capture.trailer is None
    assert capture.cases == (owned_case(),)
    validate_python_role_bytes(capture, spec, plan)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[0].update(python_version="other"),
        lambda rows: rows[0].update(plan_sha256=digest(b"other plan")),
        lambda rows: rows[0].update(locale="C.UTF-8"),
        lambda rows: rows[0].update(filesystem_encoding="utf-8"),
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
    capture = python_role_capture(owned_recipe(), owned_plan(), ProcessStreams(wire(rows), b"", 0, 1, "exited"))
    assert capture.status == "protocol_error"
    assert capture.cases == ()
    validate_python_role_bytes(capture, owned_recipe(), owned_plan())


def test_malformed_protocol_preserves_physical_timeout_instead_of_claiming_exit():
    raw = PREFIX + b'{"schema_version":999}\n'
    spec = owned_recipe()
    capture = python_role_capture(
        spec, owned_plan(), ProcessStreams(raw, b"", -9, spec.limits.timeout_seconds * 10**9, "timeout")
    )
    assert capture.status == "protocol_error"
    assert capture.termination == "timeout"
    validate_python_role_bytes(capture, spec, owned_plan())


@pytest.mark.parametrize("exit_code", [0, 1, -9])
def test_absent_footer_or_failed_process_never_claims_collected(exit_code):
    capture = python_role_capture(
        owned_recipe(), owned_plan(), ProcessStreams(wire(owned_records()[:2]), b"", exit_code, 1, "exited")
    )
    assert capture.status == ("protocol_error" if exit_code == 0 else "process_error")
    assert capture.trailer is None
    validate_python_role_bytes(capture, owned_recipe(), owned_plan())


def test_duplicate_keys_reject_and_failed_test_remains_valid_completed_collection():
    raw = wire(owned_records()).replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1', 1)
    with pytest.raises(ValueError):
        parsed_python_stream(raw, owned_recipe(), owned_plan())
    rows = owned_records()
    rows[1].update(outcome="assertion_failed", error_type="builtins.AssertionError", diagnostics="owned\u2028assertion")
    rows[2]["pytest_exit_code"] = 1
    capture = python_role_capture(owned_recipe(), owned_plan(), ProcessStreams(wire(rows), b"", 0, 1, "exited"))
    assert capture.status == "collected"
    assert capture.cases[0].outcome == "assertion_failed"
    assert "\u2028" in capture.cases[0].diagnostics


def test_rewritten_normalized_observations_reject_even_when_contract_is_coherent():
    capture = python_role_capture(
        owned_recipe(), owned_plan(), ProcessStreams(wire(owned_records()), b"", 0, 1, "exited")
    )
    rewritten = changed(capture, cases=(changed(capture.cases[0], diagnostics="rewritten"),))
    with pytest.raises(ValueError, match="disagree with raw"):
        validate_python_role_bytes(rewritten, owned_recipe(), owned_plan())
    with pytest.raises(ValueError, match="identity"):
        validate_python_role_bytes(capture, owned_recipe(), changed(owned_plan(), trial_id="another-trial"))


@pytest.mark.parametrize("status,elapsed,raw", [("timeout", 0, b""), ("output_limit", 1, b"too short")])
def test_parent_guard_claims_require_deadline_or_full_prefix(status, elapsed, raw):
    capture = python_role_capture(owned_recipe(), owned_plan(), ProcessStreams(raw, b"", -9, elapsed, status))
    with pytest.raises(ValueError, match="deadline|complete bounded prefix"):
        validate_python_role_bytes(capture, owned_recipe(), owned_plan())


def test_record_order_is_significant_but_declaration_order_is_canonical():
    plan = owned_plan()
    reordered = changed(plan, distributions=tuple(reversed(plan.distributions)))
    assert reordered.fingerprint == plan.fingerprint
    spec = owned_recipe()
    reordered_spec = changed(spec, wheels=(changed(spec.wheels[0], members=tuple(reversed(spec.wheels[0].members))),))
    assert reordered_spec.fingerprint == spec.fingerprint
    rows = owned_records()
    with pytest.raises(ValueError):
        parsed_python_stream(wire([rows[1], rows[0], rows[2]]), spec, plan)


def test_truncated_extra_protocol_after_footer_never_acquires_completion():
    raw = wire(owned_records()) + PREFIX + b'{"schema_version":1,'
    capture = python_role_capture(owned_recipe(), owned_plan(), ProcessStreams(raw, b"", 0, 1, "exited"))
    assert capture.status == "protocol_error"
    assert capture.cases == ()
