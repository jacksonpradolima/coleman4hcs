"""Owned C schema fixtures; placeholder digests are not source execution attestations."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coleman.corpus.controlled_contracts import ControlledElapsed
from coleman.transports.c_contracts import (
    CCaptureReceipt,
    CCase,
    CHeader,
    CInputCase,
    CRoleCapture,
    CRoleReference,
    CTrailer,
)
from coleman.transports.junit_contracts import ProducerElapsed
from tests.corpus.test_source_views import changed


def owned_case():
    """Construct schema-only observed zero; no foreign code is executed."""
    return CCase(
        schema_version=1,
        record_type="case",
        test_id="owned-zero",
        outcome="passed",
        termination="exited",
        exit_code=0,
        stdout="1\n",
        stderr="",
        elapsed=ControlledElapsed(value=0, status="observed"),
        capture_started_at=datetime(2020, 1, 1, tzinfo=UTC),
        capture_finished_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def owned_header():
    """Declare synthetic ABI/version labels with no actual runtime execution claim."""
    return CHeader(
        schema_version=1,
        record_type="header",
        context_fingerprint="sha256:" + "0" * 64,
        trial_id="owned-schema-only",
        role="buggy",
        plan_sha256="sha256:" + "1" * 64,
        compiler_version="owned-runtime",
        compiler_target="owned-target",
        python_version="owned",
        libc_version="owned",
        architecture="aarch64",
        unsigned_bits=32,
        timezone="UTC",
        locale="C",
        executable_sha256="sha256:" + "2" * 64,
        test_ids=(owned_case().test_id,),
    )


def owned_role():
    """Build only a schema fixture; the separate raw verifier must establish byte correspondence."""
    h = owned_header()
    return CRoleCapture(
        schema_version=1,
        context_fingerprint=h.context_fingerprint,
        trial_id=h.trial_id,
        role=h.role,
        status="collected",
        termination="exited",
        exit_code=0,
        producer_elapsed=ProducerElapsed(value=0, status="observed"),
        stdout_base64="",
        stderr_base64="",
        header=h,
        cases=(owned_case(),),
        trailer=CTrailer(schema_version=1, record_type="trailer", cases=1, oracle_exit_code=0),
        diagnostics="Owned schema only",
    )


def owned_receipt():
    """Reference placeholder owned packets without asserting actual execution."""
    roles = tuple(
        CRoleReference(
            role=r,
            packet_sha256="sha256:" + "0" * 64,
            status="collected",
            producer_elapsed=ProducerElapsed(value=0, status="observed"),
        )
        for r in ("buggy", "fixed")
    )
    return CCaptureReceipt(
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
        (owned_case, "c-case-owned-v1.json"),
        (owned_role, "c-role-owned-v1.json"),
        (owned_receipt, "c-receipt-owned-v1.json"),
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
    model = type(record)
    with pytest.raises(ValueError):
        model.from_json(payload)


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
        CCase.model_validate(data)


def test_observed_zero_unavailable_null_and_valid_negative_process_exit_are_distinct():
    assert owned_case().elapsed.value == 0
    data = owned_role().model_dump(mode="json")
    data.update(status="timeout", termination="timeout", exit_code=-9, header=None, cases=[], trailer=None)
    timed = CRoleCapture.model_validate(data)
    assert timed.exit_code == -9
    assert timed.producer_elapsed.value == 0
    data.update(
        status="interrupted",
        termination=None,
        exit_code=None,
        producer_elapsed={"value": None, "status": "unavailable"},
    )
    interrupted = CRoleCapture.model_validate(data)
    assert interrupted.producer_elapsed.value is None
    assert interrupted.producer_elapsed.status == "unavailable"
    assert interrupted != timed
    data["producer_elapsed"] = {"value": 0, "status": "observed"}
    with pytest.raises(ValueError, match="cannot invent"):
        CRoleCapture.model_validate(data)


@pytest.mark.parametrize("value", [False, 1, 0.0, "0"])
def test_uint32_abi_does_not_coerce_flags(value):
    data = owned_header().model_dump(mode="json")
    data["unsigned_bits"] = value
    with pytest.raises(ValueError):
        CHeader.model_validate(data)


@pytest.mark.parametrize("fraction", ["1234567", "123456000"])
def test_submicrosecond_capture_clocks_never_truncate_silently(fraction):
    data = owned_case().model_dump(mode="json")
    data["capture_started_at"] = "2020-01-01T00:00:00." + fraction + "Z"
    with pytest.raises(ValueError, match="six fractional"):
        CCase.model_validate(data)


def test_receipt_conserves_distinct_roles_and_completion_status():
    data = owned_receipt().model_dump(mode="json")
    data["roles"][1]["role"] = "buggy"
    with pytest.raises(ValueError, match="distinct source roles"):
        CCaptureReceipt.model_validate(data)
    data = owned_receipt().model_dump(mode="json")
    data["status"] = "incomplete"
    with pytest.raises(ValueError, match="completion disagrees"):
        CCaptureReceipt.model_validate(data)


@pytest.mark.parametrize("value", [0, 1, 2, 2**31, 2**32 - 2, *[i * 15485863 % (2**32 - 1) for i in range(64)]])
def test_uint32_oracle_metamorphic_successor(value):
    assert int(CInputCase(test_id="owned", value=value).expected_stdout) == value + 1
    assert CInputCase(test_id="owned", value=value).expected_stdout.endswith("\n")


@pytest.mark.parametrize("value", [-1, 2**32, True, 0.0, None, float("nan"), float("inf")])
def test_oracle_rejects_non_uint32_inputs(value):
    with pytest.raises(ValueError):
        CInputCase(test_id="owned", value=value)


def test_uint32_max_uses_explicit_rejection_not_wrapping_or_missingness():
    assert (
        CInputCase(test_id="owned", value=2**32 - 1).expected_stdout
        == "data value is too large to perform arithmetic safely.\n"
    )


def test_negative_posix_child_signal_is_valid_test_error():
    case = changed(owned_case(), exit_code=-9, outcome="test_error")
    assert CCase.from_json(case.to_json()).exit_code == -9


@pytest.mark.parametrize(
    "update",
    [
        {"stdout": "\u2028"},
        {"stdout": "x" * 65536, "stderr": "x"},
        {"termination": "output_limit", "outcome": "test_error", "exit_code": 0},
        {"termination": "timeout", "outcome": "timeout", "exit_code": -9},
        {"termination": "exited", "outcome": "passed", "exit_code": -9},
        {"capture_finished_at": "2019-01-01T00:00:00Z"},
    ],
)
def test_case_guard_and_physical_outcome_contradictions_reject(update):
    data = owned_case().model_dump(mode="json")
    data.update(update)
    with pytest.raises(ValueError):
        CCase.model_validate(data)


def test_timeout_and_output_limit_preserve_observed_physical_exit_zero():
    from coleman.corpus.controlled_contracts import ControlledElapsed

    timeout = changed(
        owned_case(),
        termination="timeout",
        outcome="timeout",
        exit_code=0,
        elapsed=ControlledElapsed(value=5 * 10**9, status="observed"),
    )
    limited = changed(owned_case(), termination="output_limit", outcome="test_error", exit_code=0, stdout="x" * 65536)
    assert CCase.from_json(timeout.to_json()) == timeout
    assert CCase.from_json(limited.to_json()) == limited


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing_header",
        "missing_completion",
        "trailer_count",
        "universe",
        "oracle",
        "missing_exit",
        "termination",
        "unstarted_exit",
    ],
)
def test_contradictory_role_packet_states_reject_at_the_contract_boundary(kind):
    data = owned_role().model_dump(mode="json")
    if kind == "duplicate":
        data["cases"] *= 2
    elif kind == "missing_header":
        data.update(status="process_error", header=None)
    elif kind == "missing_completion":
        data["trailer"] = None
    elif kind == "trailer_count":
        data["trailer"]["cases"] = 0
    elif kind == "universe":
        data["header"]["test_ids"] = ["other"]
    elif kind == "oracle":
        data["trailer"]["oracle_exit_code"] = 1
    elif kind == "missing_exit":
        data.update(status="protocol_error", exit_code=None)
    elif kind == "termination":
        data.update(status="timeout", termination="exited")
    else:
        data.update(status="process_error", termination=None, producer_elapsed={"status": "unavailable", "value": None})
    with pytest.raises(ValueError):
        CRoleCapture.model_validate(data)
