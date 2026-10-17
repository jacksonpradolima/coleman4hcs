"""Owned Python wire schemas and a real tiny harness workflow; no foreign subject is executed here."""

import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coleman.corpus.controlled_contracts import ControlledElapsed
from coleman.transports.junit_contracts import ProducerElapsed
from coleman.transports.python_contracts import (
    DistributionVersion,
    PythonCaptureReceipt,
    PythonCase,
    PythonHarnessPlan,
    PythonHeader,
    PythonRoleCapture,
    PythonRoleReference,
    PythonTrailer,
)

ROOT = Path(__file__).resolve().parents[2]


def owned_case():
    """Return schema-only zero ticks; placeholder identities do not claim actual execution."""
    return PythonCase(
        schema_version=1,
        record_type="case",
        test_id="tests/test_owned.py::test_pass",
        outcome="passed",
        elapsed=ControlledElapsed(value=0, status="observed"),
        capture_started_at=datetime(2020, 1, 1, tzinfo=UTC),
        capture_finished_at=datetime(2020, 1, 1, tzinfo=UTC),
        error_type=None,
        diagnostics="",
    )


def owned_header():
    """Declare owned schema metadata with no installed subject or foreign source input."""
    return PythonHeader(
        schema_version=1,
        record_type="header",
        context_fingerprint="sha256:" + "0" * 64,
        trial_id="owned-schema-only",
        role="buggy",
        plan_sha256="sha256:" + "1" * 64,
        python_version="owned-runtime",
        architecture="owned",
        timezone="UTC",
        locale="C",
        preferred_encoding="ASCII",
        filesystem_encoding="ascii",
        utf8_mode=0,
        capture_clock="producer-truncated-utc-microsecond-v1",
        distributions=(DistributionVersion(name="pytest", version="owned"),),
        test_ids=(owned_case().test_id,),
    )


def owned_role():
    """Construct schema-only completion; separate protocol validation must verify actual raw bytes."""
    header = owned_header()
    return PythonRoleCapture(
        schema_version=1,
        context_fingerprint=header.context_fingerprint,
        trial_id=header.trial_id,
        role=header.role,
        status="collected",
        termination="exited",
        exit_code=0,
        producer_elapsed=ProducerElapsed(value=0, status="observed"),
        stdout_base64="",
        stderr_base64="",
        header=header,
        cases=(owned_case(),),
        trailer=PythonTrailer(schema_version=1, record_type="trailer", cases=1, pytest_exit_code=0),
        diagnostics="Owned schema only; not an execution attestation",
    )


def owned_receipt():
    """Build schema-only references; these placeholder digests do not attest actual source execution."""
    roles = tuple(
        PythonRoleReference(
            role=role,
            packet_sha256="sha256:" + "0" * 64,
            status="collected",
            producer_elapsed=ProducerElapsed(value=0, status="observed"),
        )
        for role in ("buggy", "fixed")
    )
    return PythonCaptureReceipt(
        schema_version=1,
        plan_fingerprint="sha256:" + "1" * 64,
        context_fingerprint="sha256:" + "2" * 64,
        dataset_fingerprint="sha256:" + "3" * 64,
        roles=roles,
        status="collected",
    )


@pytest.mark.parametrize(
    "factory,fixture",
    [
        (owned_case, "python-case-owned-v1.json"),
        (owned_role, "python-role-owned-v1.json"),
        (owned_receipt, "python-receipt-owned-v1.json"),
    ],
)
def test_owned_versioned_goldens_and_roundtrip(factory, fixture):
    record = factory()
    assert type(record).from_json(record.to_json()) == record
    path = Path(__file__).with_name("fixtures") / fixture
    assert path.read_text() == record.to_json() + "\n"


@pytest.mark.parametrize("factory", [owned_case, owned_header, owned_role, owned_receipt])
@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_old_missing_future_and_coerced_versions_reject(factory, version):
    record = factory()
    data = record.model_dump(mode="json")
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    payload = json.dumps(data)
    with pytest.raises(ValueError):
        type(record).from_json(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"unit": "s"},
        {"meaning": "CI-runner-work-v1"},
        {"value": -1},
        {"value": True},
        {"value": 0.0},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": None},
        {"value": None, "status": "unavailable"},
        {"value": None, "status": "not_applicable"},
    ],
)
def test_completed_case_units_ticks_and_missingness_reject(update):
    data = owned_case().model_dump(mode="json")
    data["elapsed"].update(update)
    with pytest.raises(ValueError):
        PythonCase.model_validate(data)


def test_observed_zero_unavailable_null_and_valid_negative_process_exit_are_distinct():
    assert owned_case().elapsed.value == 0
    data = owned_role().model_dump(mode="json")
    data.update(status="timeout", termination="timeout", exit_code=-9, header=None, cases=[], trailer=None)
    timed = PythonRoleCapture.model_validate(data)
    assert timed.exit_code == -9
    assert timed.producer_elapsed.value == 0
    data.update(
        status="interrupted",
        termination=None,
        exit_code=None,
        producer_elapsed={"value": None, "status": "unavailable"},
    )
    interrupted = PythonRoleCapture.model_validate(data)
    assert interrupted.producer_elapsed.value is None
    assert interrupted.producer_elapsed.status == "unavailable"
    assert interrupted != timed
    data["producer_elapsed"] = {"value": 0, "status": "observed"}
    with pytest.raises(ValueError, match="cannot invent"):
        PythonRoleCapture.model_validate(data)


@pytest.mark.parametrize("value", [False, 1, 0.0, "0"])
def test_utf8_mode_does_not_coerce_flags(value):
    data = owned_header().model_dump(mode="json")
    data["utf8_mode"] = value
    with pytest.raises(ValueError):
        PythonHeader.model_validate(data)


@pytest.mark.parametrize("fraction", ["1234567", "123456000"])
def test_submicrosecond_capture_clocks_never_truncate_silently(fraction):
    data = owned_case().model_dump(mode="json")
    data["capture_started_at"] = "2020-01-01T00:00:00." + fraction + "Z"
    with pytest.raises(ValueError, match="six fractional"):
        PythonCase.model_validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(cases=data["cases"] * 2),
        lambda data: data.update(header=None),
        lambda data: data.update(status="protocol_error", header=None),
        lambda data: data["trailer"].update(cases=0),
        lambda data: data["header"].update(test_ids=[]),
        lambda data: data["trailer"].update(pytest_exit_code=1),
        lambda data: data.update(status="timeout", termination=None),
        lambda data: data.update(status="output_limit", termination="timeout"),
        lambda data: data.update(status="process_error", termination=None),
        lambda data: data["cases"][0].update(error_type="builtins.ValueError"),
        lambda data: data["cases"][0].update(capture_finished_at="2019-01-01T00:00:00Z"),
        lambda data: data["cases"][0].update(capture_started_at=True),
        lambda data: data["header"].update(test_ids=data["header"]["test_ids"] * 2),
        lambda data: data["header"].update(distributions=data["header"]["distributions"] * 2),
    ],
)
def test_contradictory_completion_exception_clock_or_missingness_is_not_a_valid_record(mutation):
    data = owned_role().model_dump(mode="json")
    mutation(data)
    with pytest.raises(ValueError):
        PythonRoleCapture.model_validate(data)


def test_receipt_conserves_distinct_roles_and_completion_status():
    data = owned_receipt().model_dump(mode="json")
    data["roles"][1]["role"] = "buggy"
    with pytest.raises(ValueError, match="distinct source roles"):
        PythonCaptureReceipt.model_validate(data)
    data = owned_receipt().model_dump(mode="json")
    data["status"] = "incomplete"
    with pytest.raises(ValueError, match="completion disagrees"):
        PythonCaptureReceipt.model_validate(data)


def test_actual_owned_harness_preserves_assertions_errors_skips_and_elapsed(tmp_path):
    fixture = tmp_path / "fixture"
    tests = fixture / "tests"
    tests.mkdir(parents=True)
    (tests / "test_owned.py").write_text(
        "import pytest\n"
        "def test_pass():\n    assert 2 + 2 == 4\n"
        "def test_assert():\n    assert 2 + 2 == 5\n"
        'def test_error():\n    raise ValueError("owned deliberate error")\n'
        'def test_skip():\n    pytest.skip("owned deliberate skip")\n'
    )
    plan = PythonHarnessPlan(
        schema_version=1,
        context_fingerprint="sha256:" + "0" * 64,
        trial_id="owned-harness-component",
        role="buggy",
        fixture_root=str(fixture),
        production_root=str(tmp_path / "production"),
        dependency_root=str(tmp_path / "dependencies"),
        wheel_root=str(tmp_path / "wheels"),
        wheels=(),
        distributions=(DistributionVersion(name="pytest", version=importlib.metadata.version("pytest")),),
        test_ids=tuple("tests/test_owned.py::test_" + name for name in ("pass", "assert", "error", "skip")),
    )
    path = tmp_path / "plan.json"
    path.write_text(plan.to_json() + "\n")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(
        LC_ALL="C",
        PYTHONUTF8="0",
        PYTHONCOERCECLOCALE="0",
        PYTHONIOENCODING="utf-8",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        TZ="UTC",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "coleman/corpus/resources/ColemanPytestCapture.py.source"), str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=True,
        timeout=60,
    )
    rows = [
        json.loads(line.split(b"\t", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith(b"COLEMAN-PYTEST-V1\t")
    ]
    header = PythonHeader.model_validate(rows[0])
    cases = [PythonCase.model_validate(row) for row in rows[1:-1]]
    trailer = PythonTrailer.model_validate(rows[-1])
    assert set(header.test_ids) == set(plan.test_ids)
    assert {case.test_id for case in cases} == set(plan.test_ids)
    assert trailer.cases == 4
    assert trailer.pytest_exit_code == 1
    assert {case.outcome for case in cases} == {"passed", "assertion_failed", "test_error", "skipped"}
    assert all(case.elapsed.status == "observed" for case in cases)
    assert all(case.elapsed.unit == "ns" for case in cases)
    assert {case.error_type for case in cases if case.error_type} == {"builtins.AssertionError", "builtins.ValueError"}
