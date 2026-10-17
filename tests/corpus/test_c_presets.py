"""Frozen C source/terms pins and owned recipe invariants; normal tests perform no downloads."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.transports.c import load_c_recipe
from coleman.transports.c_contracts import CCaptureRecipe, CHarnessPlan
from coleman.transports.c_source import c_collector_bytes, c_harness_bytes, prepare_c_inputs
from tests.corpus.test_c_capture import prepared_inputs
from tests.corpus.test_c_protocol import owned_plan, owned_recipe
from tests.corpus.test_source_views import changed, digest


def test_source_function_selectors_preserve_byte_identity_without_inventing_real_faults():
    recipe = load_c_recipe()
    assert recipe.harness_sha256 == digest(c_harness_bytes())
    assert recipe.collector_sha256 == digest(c_collector_bytes())
    assert recipe.fault_origin == "synthetic"
    assert recipe.scratch_policy == "private-executable-tmpfs-v1"
    assert len(recipe.cases) == 6
    assert recipe.cases[-1].value == 2**32 - 1
    assert recipe.buggy.output.members == recipe.fixed.output.members
    assert recipe.buggy.output.source.sha256 == recipe.fixed.output.source.sha256
    assert recipe.buggy.output.source.version != recipe.fixed.output.source.version
    for view in (recipe.buggy, recipe.fixed, recipe.fixture):
        assert view.output.license.redistribution == "recipe_only"
        assert view.output.license.license_id == "CC0-1.0 / US public domain"
        assert any(member.name == "LICENSE" for member in view.output.members)
    assert CCaptureRecipe.from_json(recipe.to_json()) == recipe
    with pytest.raises(ValueError, match="Unknown C source preset"):
        load_c_recipe("../outside")


@pytest.mark.parametrize(
    "factory,name", [(owned_recipe, "c-recipe-owned-v1.json"), (owned_plan, "c-plan-owned-v1.json")]
)
def test_owned_versioned_recipe_plan_goldens(factory, name):
    value = factory()
    payload = (Path(__file__).with_name("fixtures") / name).read_bytes()
    assert value.to_json() + "\n" == payload.decode()
    assert type(value).from_json(payload) == value


@pytest.mark.parametrize("factory", [owned_recipe, owned_plan])
@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_old_missing_future_or_coerced_schema_rejects(factory, version):
    record = factory()
    payload = record.model_dump(mode="json")
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version
    encoded = json.dumps(payload)
    model = type(record)
    with pytest.raises(ValueError):
        model.from_json(encoded)


@pytest.mark.parametrize(
    "update",
    [
        {"source_member": "../subject.c"},
        {"source_member": "subject.py"},
        {"source_member": "missing.c"},
        {"buggy_function": "bad(); system(1)"},
        {"fixed_function": "bad"},
        {"support_units": ["../support.c"]},
        {"support_units": ["support.h"]},
        {"support_units": ["missing.c"]},
        {"support_units": ["support.c", "support.c"]},
        {"fault_origin": "real"},
        {"oracle": "subject-output-as-reference"},
        {"scratch_policy": "host-executable-directory"},
    ],
)
def test_unsafe_or_unsupported_recipe_inputs_reject_before_execution(update):
    value = owned_recipe().model_dump(mode="json")
    value.update(update)
    with pytest.raises(ValueError):
        CCaptureRecipe.model_validate(value)


def test_duplicate_cases_and_ambiguous_role_versions_reject():
    value = owned_recipe()
    with pytest.raises(ValueError, match="Duplicate C input case"):
        changed(value, cases=value.cases * 2)
    with pytest.raises(ValueError, match="distinguish function"):
        changed(value, buggy=value.fixed)


def test_legacy_controlled_dataset_keeps_original_bytes_and_semantics():
    path = Path(__file__).with_name("fixtures") / "controlled-owned-pair-v1.json"
    original = path.read_bytes()
    value = ControlledDataset.from_json(original)
    assert value.to_json() + "\n" == original.decode()
    assert value.history == "constructed"
    assert value.admission == "not_evaluated"
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["harness_sha256", "collector_sha256"])
def test_changed_authored_implementation_rejects_before_preparing_source(tmp_path, field):
    recipe, cache, environment = prepared_inputs(tmp_path)
    other = changed(recipe, **{field: digest(b"changed implementation")})
    with pytest.raises(ValueError, match="observer/collector"):
        prepare_c_inputs(other, cache, environment)


def test_nonboolean_fetch_and_invalid_environment_fail_closed(tmp_path):
    recipe, cache, environment = prepared_inputs(tmp_path)
    invalid_fetch = cast(Any, 1)
    with pytest.raises(ValueError, match="boolean"):
        prepare_c_inputs(recipe, cache, environment, fetch=invalid_fetch)
    from coleman.transports.c_source import owned_c_package

    invalid = owned_c_package(cache, "invalid-environment.json", b"{}", "owned-invalid-v1")
    with pytest.raises(ValueError):
        prepare_c_inputs(recipe, cache, invalid)


def test_new_contract_can_not_cross_bind_a_different_bootstrap_recipe():
    from coleman.transports.c_protocol import parsed_c_stream

    recipe = changed(owned_recipe(), fault_id="other")
    plan = CHarnessPlan.model_validate(owned_plan())
    with pytest.raises(ValueError, match="bootstrap recipe"):
        parsed_c_stream(b"", recipe, plan)


def test_bootstrap_byte_budget_is_checked_before_any_execution(monkeypatch):
    from coleman.transports.c_source import c_harness_plan

    plan = owned_plan()
    monkeypatch.setattr(CHarnessPlan, "MAXIMUM_BYTES", 1)
    with pytest.raises(ValueError, match="bootstrap exceeds"):
        c_harness_plan(plan.recipe, plan.context_fingerprint, plan.trial_id, plan.role)
