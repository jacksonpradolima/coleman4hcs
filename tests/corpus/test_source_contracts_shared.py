"""Version and unit invariants apply to both concrete source protocols, without changing either wire schema."""

import json
from pathlib import Path

import pytest

from tests.corpus import test_c_contracts as c
from tests.corpus import test_python_contracts as python


@pytest.mark.parametrize(
    "factory,fixture",
    [
        (python.owned_case, "python-case-owned-v1.json"),
        (python.owned_role, "python-role-owned-v1.json"),
        (python.owned_receipt, "python-receipt-owned-v1.json"),
        (c.owned_case, "c-case-owned-v1.json"),
        (c.owned_role, "c-role-owned-v1.json"),
        (c.owned_receipt, "c-receipt-owned-v1.json"),
    ],
)
def test_owned_versioned_goldens_and_roundtrip(factory, fixture):
    record = factory()
    assert type(record).from_json(record.to_json()) == record
    path = Path(__file__).with_name("fixtures") / fixture
    assert path.read_text() == record.to_json() + "\n"


@pytest.mark.parametrize(
    "factory",
    [
        getattr(provider, name)
        for provider in (c, python)
        for name in ("owned_case", "owned_header", "owned_role", "owned_receipt")
    ],
)
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


@pytest.mark.parametrize("factory", [c.owned_case, python.owned_case])
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
def test_completed_case_units_ticks_and_missingness_reject(update, factory):
    record = factory()
    model = type(record)
    data = record.model_dump(mode="json")
    data["elapsed"].update(update)
    with pytest.raises(ValueError):
        model.model_validate(data)
