"""Owned JUnit protocol fixtures retain negative evidence without executing foreign source."""

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coleman.corpus.source_views import SourceViewSpec, source_view_bytes
from coleman.transports.junit_contracts import JunitCaptureRecipe, JunitCase, JunitRoleCapture, ProducerElapsed
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.junit_protocol import parsed_junit_stream, role_capture, validate_role_bytes
from tests.corpus.test_source_views import changed, digest, source_spec


def view(label, prefix, version):
    name = prefix + "/owned/CheckCase.java"
    payload = b"Owned schema-only source fixture.\n"
    raw = source_spec(name, payload)
    raw = changed(raw, source=changed(raw.source, version=version))
    output = source_spec(label, source_view_bytes(((name, payload),)), "zip", [raw.members[0].model_dump()])
    output = changed(output, source=changed(output.source, version=version))
    return SourceViewSpec(schema_version=1, inputs=(raw,), output=output)


def recipe():
    return JunitCaptureRecipe.model_validate(
        {
            "schema_version": 1,
            "recipe_id": "owned-junit-schema-v1",
            "project": "owned-ci",
            "fault_id": "owned-case",
            "fault_origin": "synthetic",
            "dependence_group": "owned-ci-single-fixture",
            "buggy": view("buggy", "src/java", "buggy-v1"),
            "fixed": view("fixed", "src/java", "fixed-v1"),
            "fixture": view("fixture", "src/test", "fixed-v1"),
            "dependencies": [source_spec("junit.jar", b"Owned schema-only dependency, not a real JAR")],
            "runtime": {
                "version": "11.0.32.1",
                "platform": "linux/arm64",
                "image_digest": digest(b"owned schema-only image"),
                "emulated": False,
                "timezone": "UTC",
            },
            "production_prefix": "src/java",
            "fixture_prefix": "src/test",
            "test_ids": ["owned.CheckCase::test_ci"],
            "harness_sha256": digest(b"owned schema-only harness"),
            "limitations": ["Schema-only fixture; no foreign runtime, source or independent faults claimed"],
        }
    )


def records():
    return [
        {
            "schema_version": 1,
            "record_type": "header",
            "java_version": "11.0.32.1",
            "architecture": "aarch64",
            "timezone": "UTC",
            "capture_clock": "producer-truncated-utc-microsecond-v1",
            "test_ids": ["owned.CheckCase::test_ci"],
            "excluded_classes": ["owned.Example"],
        },
        {
            "schema_version": 1,
            "record_type": "case",
            "test_id": "owned.CheckCase::test_ci",
            "tests_run": 1,
            "failures": 1,
            "errors": 0,
            "elapsed_ns": 0,
            "capture_started_at": "2026-01-01T00:00:00Z",
            "capture_finished_at": "2026-01-01T00:00:00Z",
            "diagnostics": "Owned diagnostic\u2028retains its Unicode separator",
            "diagnostics_truncated": False,
        },
        {"schema_version": 1, "record_type": "trailer", "case_count": 1},
    ]


def encoded(rows):
    return b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)


def test_protocol_runtime_unicode_framing_counters_and_packet_roundtrip():
    spec = recipe()
    raw = encoded(records())
    header, cases = parsed_junit_stream(raw, spec)
    assert header.test_ids == spec.test_ids
    assert cases[0].outcome == "assertion_failed"
    assert cases[0].elapsed_ns == 0
    assert "\u2028" in cases[0].diagnostics
    now = datetime(2026, 1, 1, tzinfo=UTC)
    streams = ProcessStreams(raw, b"owned compiler diagnostic", 0, 0, "exited")
    packet = role_capture(spec, digest(b"context"), "owned-trial", "buggy", now, now, streams)
    assert packet.status == "collected"
    assert packet.producer_elapsed.value == 0
    assert packet.producer_elapsed.meaning != "observed-test-harness-monotonic-elapsed-v1"
    assert JunitRoleCapture.from_json(packet.to_json()) == packet
    fixtures = Path(__file__).with_name("fixtures")
    assert packet.to_json() + "\n" == (fixtures / "junit-role-owned-v1.json").read_text()
    assert spec.to_json() + "\n" == (fixtures / "junit-recipe-owned-v1.json").read_text()
    validate_role_bytes(packet, spec)
    altered_case = changed(packet.cases[0], failures=0)
    altered = changed(packet, cases=(altered_case,))
    with pytest.raises(ValueError, match="disagree with raw streams"):
        validate_role_bytes(altered, spec)


@pytest.mark.parametrize(
    "change",
    [
        {"java_version": "other"},
        {"architecture": "amd64"},
        {"timezone": "other"},
        {"test_ids": ["owned.CheckCase::different"]},
        {"schema_version": 2},
        {"extra": "unknown"},
    ],
)
def test_bad_header_and_future_versions_reject(change):
    rows = records()
    rows[0].update(change)
    raw = encoded(rows)
    spec = recipe()
    with pytest.raises(ValueError):
        parsed_junit_stream(raw, spec)


@pytest.mark.parametrize(
    "change",
    [
        {"elapsed_ns": -1},
        {"elapsed_ns": float("inf")},
        {"elapsed_ns": True},
        {"failures": None},
        {"tests_run": 0},
        {"schema_version": 2},
    ],
)
def test_invalid_case_quantities_and_future_versions_reject(change):
    data = {**records()[1], **change}
    payload = json.dumps(data)
    with pytest.raises(ValueError):
        JunitCase.from_json(payload)


def test_missing_final_marker_duplicates_and_incompatible_case_universe_reject():
    spec = recipe()
    rows = records()
    candidates = [
        encoded(rows)[:-1],
        encoded(rows[:-1]),
        encoded([rows[0], rows[1], rows[1], rows[2]]),
        encoded([rows[0], {**rows[1], "test_id": "owned.CheckCase::different"}, rows[2]]),
        encoded([rows[0], rows[1], {**rows[2], "case_count": 2}]),
    ]
    for raw in candidates:
        with pytest.raises(ValueError):
            parsed_junit_stream(raw, spec)


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"errors": 1}, "test_error"),
        ({"failures": 0}, "passed"),
        ({"diagnostics_truncated": True}, "infrastructure_error"),
    ],
)
def test_test_error_pass_and_truncation_remain_distinct(updates, expected):
    data = {**records()[1], **updates}
    assert JunitCase.model_validate(data).outcome == expected


@pytest.mark.parametrize(
    "termination,exit_code,stdout,status",
    [
        ("timeout", -9, b"partial output", "timeout"),
        ("output_limit", -9, b"clipped", "output_limit"),
        ("exited", 2, b"compile failed", "process_error"),
        ("exited", 0, b"not JSON\n", "protocol_error"),
    ],
)
def test_incomplete_process_does_not_promote_partial_cases(termination, exit_code, stdout, status):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    spec = recipe()
    streams = ProcessStreams(stdout, b"raw stderr", exit_code, 1, termination)
    packet = role_capture(spec, digest(b"context"), "owned-trial", "buggy", now, now, streams)
    assert packet.status == status
    assert packet.cases == ()
    assert packet.stdout_base64 is not None
    assert base64.b64decode(packet.stdout_base64) == stdout
    validate_role_bytes(packet, spec)


def test_producer_counter_has_explicit_units_and_missingness():
    assert ProducerElapsed(value=None, status="unavailable").value is None
    for data in (
        {"value": None, "status": "observed"},
        {"value": 0, "status": "unavailable"},
        {"value": -1, "status": "observed"},
        {"value": 0, "status": "observed", "unit": "s"},
    ):
        with pytest.raises(ValueError):
            ProducerElapsed.model_validate(data)
