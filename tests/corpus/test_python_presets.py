"""Frozen pre-execution Python metadata, legacy controlled artifacts and typed recipe invariants."""

import json
from pathlib import Path

import pytest

from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.transports.python import PRESETS, load_python_recipe
from coleman.transports.python_contracts import PythonCaptureRecipe, PythonHarnessPlan
from coleman.transports.python_source import python_harness_bytes
from tests.corpus.test_python_protocol import owned_plan, owned_recipe
from tests.corpus.test_source_views import changed, digest


@pytest.mark.parametrize("preset", PRESETS)
def test_public_presets_preserve_pre_execution_source_fixture_units_and_terms(preset):
    spec = load_python_recipe(preset)
    assert spec.harness_sha256 == digest(python_harness_bytes())
    assert spec.test_ids == ("tests/test_chinese.py::test_chinese",)
    assert len(spec.buggy.output.members) == 6
    assert len(spec.fixed.output.members) == 6
    assert len(spec.fixture.output.members) == 4
    assert len(spec.wheels) == 8
    assert PythonCaptureRecipe.from_json(spec.to_json()) == spec
    toolbox = next(wheel for wheel in spec.wheels if wheel.distribution.name == "python-toolbox")
    assert toolbox.entry_count == 420
    assert len(toolbox.package.members) == 1
    assert toolbox.installation_scripts == ("python_toolbox-1.3.2.data/scripts/_test_python_toolbox.py",)
    assert len(toolbox.components) == 6
    assert any(item.attribution == "source_similarity_inferred" for item in toolbox.components)
    assert all(item.license.redistribution == "recipe_only" for item in toolbox.components)
    assert {item.version for item in spec.runtime.provided_distributions if item.name == "setuptools"} == {"45.1.0"}
    assert "setuptools" not in {wheel.distribution.name for wheel in spec.wheels}


def test_locale_contexts_share_source_and_dependence_without_becoming_independent_faults():
    ascii_spec, utf8_spec = (load_python_recipe(preset) for preset in PRESETS)
    assert ascii_spec.buggy == utf8_spec.buggy
    assert ascii_spec.fixed == utf8_spec.fixed
    assert ascii_spec.fixture == utf8_spec.fixture
    assert ascii_spec.wheels == utf8_spec.wheels
    assert ascii_spec.dependence_group == utf8_spec.dependence_group
    assert ascii_spec.fingerprint != utf8_spec.fingerprint
    assert (ascii_spec.runtime.locale, utf8_spec.runtime.locale) == ("C", "C.UTF-8")
    with pytest.raises(ValueError, match="Unknown Python source preset"):
        load_python_recipe("../outside")


@pytest.mark.parametrize(
    "factory,name", [(owned_recipe, "python-recipe-owned-v1.json"), (owned_plan, "python-plan-owned-v1.json")]
)
def test_versioned_owned_recipe_and_bootstrap_goldens(factory, name):
    record = factory()
    path = Path(__file__).with_name("fixtures") / name
    assert record.to_json() + "\n" == path.read_text()
    assert type(record).from_json(path.read_bytes()) == record


@pytest.mark.parametrize("model,factory", [(PythonCaptureRecipe, owned_recipe), (PythonHarnessPlan, owned_plan)])
@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_missing_old_future_and_coerced_recipe_versions_reject(model, factory, version):
    data = factory().model_dump(mode="json")
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    with pytest.raises(ValueError):
        model.from_json(json.dumps(data))


def test_legacy_controlled_artifact_keeps_original_bytes_schema_and_missingness():
    path = Path(__file__).with_name("fixtures") / "controlled-owned-pair-v1.json"
    original = path.read_bytes()
    dataset = ControlledDataset.from_json(original)
    assert dataset.to_json() + "\n" == original.decode()
    assert dataset.history == "constructed"
    assert dataset.admission == "not_evaluated"
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["buggy", "fixture"])
def test_identical_source_roles_or_changed_fixture_revision_reject(field):
    spec = owned_recipe()
    other = spec.fixed if field == "buggy" else spec.buggy
    with pytest.raises(ValueError, match="distinct revisions|fixed revision"):
        changed(spec, **{field: other})


def test_universe_duplicates_undeclared_files_and_dependency_collisions_reject():
    spec = owned_recipe()
    with pytest.raises(ValueError, match="Duplicate declared"):
        changed(spec, test_ids=spec.test_ids * 2)
    with pytest.raises(ValueError, match="outside the declared"):
        changed(spec, test_ids=("tests/other.py::test_missing",))
    with pytest.raises(ValueError, match="distinct distributions"):
        changed(spec, wheels=spec.wheels * 2)
    provided = (*spec.runtime.provided_distributions, spec.wheels[0].distribution)
    with pytest.raises(ValueError, match="distinct distributions"):
        changed(spec, runtime=changed(spec.runtime, provided_distributions=provided))
