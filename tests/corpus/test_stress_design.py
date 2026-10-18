"""Independent static interaction enumeration and mutation guards."""

import itertools
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from coleman.corpus.stress_contracts import StressCase, StressDesign, StressFactors, factor_levels
from coleman.corpus.stress_design import adversarial_cases, certify_design, covering_design
from tests.corpus.stress_support import case

FIXTURES = Path(__file__).parent / "fixtures"


def test_complete_irredundant_pair_certificate_and_versioned_golden():
    design = covering_design()
    axes = factor_levels()
    # Independent enumeration by factor names, not production pair encoding.
    for (left, a), (right, b) in itertools.combinations(axes, 2):
        observed = {(r.model_dump()[left], r.model_dump()[right]) for r in design.rows}
        assert observed == set(itertools.product(a, b))
    assert len(design.rows) == 32
    assert len(certify_design(design.rows)) == 32
    golden = (FIXTURES / "stress-design-v1.json").read_text().strip()
    assert StressDesign.from_json(golden) == design
    assert design.to_json() == golden
    assert covering_design().fingerprint == design.fingerprint
    assert design.minimality == "inclusion-minimal-not-global-minimum"


def test_every_row_has_private_witness_and_deletion_is_detected():
    design = covering_design()
    for i in range(len(design.rows)):
        with pytest.raises(ValueError, match="omits"):
            certify_design(design.rows[:i] + design.rows[i + 1 :])
    with pytest.raises(ValueError, match="redundant"):
        certify_design(design.rows + design.rows[:1])
    with pytest.raises(ValueError, match="omits"):
        certify_design(())
    oversized = (StressFactors(),) * 257
    with pytest.raises(ValueError, match="256-row"):
        certify_design(oversized)


@pytest.mark.parametrize("version", [None, 0, 2, 999, True, "1"])
def test_unknown_and_legacy_versions_do_not_migrate(version):
    value = case().model_dump(mode="json")
    value["schema_version"] = version
    payload = json.dumps(value)
    with pytest.raises(ValidationError):
        StressCase.from_json(payload)


@pytest.mark.parametrize("field,value", [("seed", -1), ("seed", 2**32), ("seed", True), ("builds", 1), ("builds", 17)])
def test_case_resource_bounds(field, value):
    payload = {**case().model_dump(), field: value}
    with pytest.raises(ValidationError):
        StressCase.model_validate(payload)


def test_typed_factor_bounds_and_canonical_input_order():
    original = case(context="adversarial", size="large")
    value = original.model_dump(mode="json")
    value["factors"] = dict(reversed(tuple(value["factors"].items())))
    assert StressCase.model_validate(value).fingerprint == original.fingerprint
    assert StressCase.from_json(original.to_json()) == original
    for field, invalid in [("size", 50000), ("feedback", "instant"), ("workers", 8), ("extra", "value")]:
        with pytest.raises(ValidationError):
            StressFactors.model_validate({field: invalid})
    with pytest.raises(ValueError, match="Duplicate JSON"):
        StressCase.from_json('{"schema_version":1,"schema_version":2}')


def test_named_adverse_sequences_are_not_extra_independent_samples():
    named = dict(adversarial_cases())
    assert len(named) == 7
    assert named["large-heavy-tail"].factors.test_count == 50000
    assert all(c.evidence == "constructed-dependent-mechanism-fixture" for c in named.values())
    assert named["delayed-abrupt-drift"].factors.feedback == "delayed"


@pytest.mark.parametrize(
    "feedback,budget,eligible",
    [
        ("full", "full", True),
        ("full", "severe", False),
        ("censored", "severe", True),
        ("delayed", "full", False),
        ("missing", "full", False),
        ("corrected", "full", False),
    ],
)
def test_native_eligibility_is_explicit(feedback, budget, eligible):
    value = case(feedback=feedback, budget=budget)
    if eligible:
        value.require_native()
    else:
        with pytest.raises(ValueError, match="Native RNFail|Full-label"):
            value.require_native()


@pytest.mark.parametrize("churn", ["additions", "removals"])
@pytest.mark.parametrize("builds,minimum,eligible", [(4, 2, True), (8, 1, False), (16, 1, False)])
def test_native_requires_positive_full_suite_work_from_declared_population(churn, builds, minimum, eligible):
    value = case(churn=churn, builds=builds)
    assert value.minimum_active_tests == minimum
    if eligible:
        value.require_native()
    else:
        with pytest.raises(ValueError, match="positive full-suite work"):
            value.require_native()
