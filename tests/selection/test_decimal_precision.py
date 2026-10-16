"""Decimal-context underflow cannot bypass declared finite precision or amplify serialization."""

from decimal import Decimal, Inexact, Rounded, Subnormal, Underflow, localcontext

import pytest
from pydantic import TypeAdapter

from coleman.allocation.quantities import ExactPayoff, NonnegativeMoment, Probability, SignedPayoff
from coleman.application.contracts import Parameters, Submit
from coleman.results.domain import Finite, decode_json
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Quantity, UnitSemantics, validate_decimal_places

UNITS = UnitSemantics(unit="s", meaning="modeled-ci-serial-work-v1")


@pytest.mark.parametrize("kind", [Quantity, SignedPayoff, ExactPayoff, NonnegativeMoment])
@pytest.mark.parametrize("text", ["1e-1000100", "1e-999999999"])
def test_compact_underflow_is_rejected_before_quantity_serialization(kind, text):
    with pytest.raises(ValueError):
        kind.model_validate({"value": text, "semantics": UNITS})


def test_negative_signed_underflow_is_not_rounded_to_zero():
    with pytest.raises(ValueError):
        SignedPayoff.model_validate({"value": "-1e-999999999", "semantics": UNITS})


@pytest.mark.parametrize(
    "kind,places", [(Quantity, 12), (SignedPayoff, 12), (ExactPayoff, 48), (NonnegativeMoment, 48)]
)
def test_declared_boundary_trailing_zeros_and_valid_signed_values_are_preserved(kind, places):
    value = "1e-" + str(places)
    model = kind.model_validate({"value": value, "semantics": UNITS})
    assert model.value == Decimal(value)
    assert model.model_dump(mode="json")["value"] == "0." + "0" * (places - 1) + "1"
    extended = kind.model_validate({"value": "1000e-" + str(places + 3), "semantics": UNITS})
    assert extended.model_dump(mode="json") == model.model_dump(mode="json")
    with pytest.raises(ValueError):
        kind.model_validate({"value": "1e-" + str(places + 1), "semantics": UNITS})
    if kind in (SignedPayoff, ExactPayoff):
        negative = kind.model_validate({"value": "-1.25", "semantics": UNITS})
        assert negative.value == Decimal("-1.25")
        assert negative.model_dump(mode="json")["value"] == "-1.25"


@pytest.mark.parametrize("text", ["0e-999999999", "-0e999999999"])
def test_zero_remains_zero_without_expanding_exponents(text):
    model = Quantity.model_validate({"value": text, "semantics": UNITS})
    assert model.value == 0
    assert model.model_dump(mode="json")["value"] == "0"


def test_allocation_probabilities_and_application_overrides_are_guarded():
    probability = TypeAdapter(Probability)
    with pytest.raises(ValueError):
        probability.validate_python("1e-999999999")
    with pytest.raises(ValueError):
        Parameters.model_validate({"budget": "1e-999999999"})
    with pytest.raises(ValueError):
        Submit.from_json(
            '{"schema_version":1,"request_id":"ci-request","spec_id":"ci-fixture","parameters":{"budget":1e-999999999}}'
        )
    assert TypeAdapter(Probability).validate_python("0.25") == Decimal("0.25")
    assert Parameters.model_validate({"budget": "0.25"}).model_dump(mode="json")["budget"] == "0.25"
    assert Parameters().budget is None


def test_precision_is_independent_of_decimal_context_without_mutating_it():
    results = []
    for precision, minimum_exponent in [(28, -999999), (2, -4)]:
        with localcontext() as context:
            context.prec = precision
            context.Emin = minimum_exponent
            context.clear_flags()
            for signal in (Underflow, Subnormal, Inexact, Rounded):
                context.traps[signal] = True
            with pytest.raises(ValueError):
                Quantity.model_validate({"value": "1e-50", "semantics": UNITS})
            model = Quantity.model_validate({"value": "1.123456789012", "semantics": UNITS})
            results.append(model.model_dump(mode="json"))
            assert context.prec == precision
            assert context.Emin == minimum_exponent
            assert not any(context.flags.values())
    assert results[0] == results[1]


@pytest.mark.parametrize("places", [-1, True])
def test_invalid_precision_configuration_is_rejected(places):
    value = Decimal(1)
    with pytest.raises(ValueError):
        validate_decimal_places(value, places=places)


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_shared_precision_guard_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        validate_decimal_places(value)


class DecimalEnvelope(VersionedContract):
    """Exercise the actual artifact reader with both exact and binary64 field semantics."""

    quantity: Quantity
    binary: Finite


def test_versioned_reader_checks_original_numeric_literal_before_underflow():
    with pytest.raises(ValueError):
        DecimalEnvelope.from_json(
            '{"schema_version":1,"binary":0.1,"quantity":'
            '{"value":1e-999999999,"semantics":{"unit":"s","meaning":"ci-work"}}}'
        )
    numeric = DecimalEnvelope.from_json(
        '{"schema_version":1,"binary":0.1,"quantity":{"value":0.125,"semantics":{"unit":"s","meaning":"ci-work"}}}'
    )
    quoted = DecimalEnvelope.from_json(
        '{"schema_version":1,"binary":0.1,"quantity":{"value":"0.125","semantics":{"unit":"s","meaning":"ci-work"}}}'
    )
    assert numeric.quantity.value == Decimal("0.125")
    assert type(numeric.binary) is float
    assert numeric.binary == 0.1
    assert numeric.to_json() == quoted.to_json()
    assert numeric.fingerprint == quoted.fingerprint
    assert type(decode_json("0.125")) is float
    assert decode_json("0.125", exact_decimals=True) == Decimal("0.125")
    with pytest.raises(ValueError, match="Duplicate"):
        decode_json('{"value":0.1,"value":0.2}', exact_decimals=True)


def test_exact_rational_literals_are_bounded_before_denominator_expansion():
    from fractions import Fraction

    from coleman.benchmarks.contracts import ExactFraction

    adapter = TypeAdapter(ExactFraction)
    for value in (
        "1e-999999999",
        "1e999999999",
        "-1e-999999999",
        Decimal("1e-999999999"),
        "1" * 4097,
        1 << 4096,
        Fraction(1, 1 << 4096),
    ):
        with pytest.raises(ValueError):
            adapter.validate_python(value)
    assert adapter.validate_python("1e-48") == Fraction(1, 10**48)
    assert adapter.validate_python("-1.25") == Fraction(-5, 4)
    assert adapter.validate_python("3/7") == Fraction(3, 7)
    assert adapter.validate_python("0e-999999999") == 0
    for invalid in (Decimal("NaN"), Decimal("Infinity"), "bad-e", object(), 0.5, True, str(1 << 4096)):
        with pytest.raises(ValueError):
            adapter.validate_python(invalid)
    for valid in (Fraction(1 << 4095, (1 << 4095) + 1), "1e-512", "9" * 512 + "e512"):
        parsed = adapter.validate_python(valid)
        assert adapter.validate_json(adapter.dump_json(parsed)) == parsed
