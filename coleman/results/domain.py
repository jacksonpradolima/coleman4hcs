"""Immutable, transport-independent version-1 measurement contracts.

Units and meanings are opaque, case-sensitive identifiers: no conversion or
statistical inference is implicit. Native scheduling rows are a separate legacy
contract and cannot be interpreted as these records without explicit provenance.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(strict=True, min_length=1, pattern=r"^\S+(?:[^\r\n]*\S)?$")]
Fingerprint = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]

# Names belonging to MonitorCollector's legacy scheduling contract. Extensions
# use their own measurement names, never impersonating a native output column.
RESERVED_METRICS = frozenset(
    [
        "scenario",
        "experiment",
        "step",
        "execution_id",
        "worker_id",
        "parallel_mode",
        "policy",
        "reward_function",
        "budget_mode",
        "budget_value",
        "total_build_duration",
        "prioritization_time",
        "process_memory_rss_mib",
        "process_memory_peak_rss_mib",
        "process_cpu_utilization_percent",
        "process_cpu_time_seconds",
        "wall_time_seconds",
        "detected",
        "missed",
        "tests_ran",
        "tests_not_ran",
        "ttf",
        "ttf_duration",
        "time_reduction",
        "fitness",
        "cost",
        "rewards",
        "avg_precision",
        "prioritization_order",
        "prioritization_order_hash",
        "prioritization_order_top_k",
        "variant",
    ]
)


def canonical_json(payload: Any) -> str:
    """Encode JSON values with sorted keys, UTF-8 and finite numbers only.

    Non-string object keys and non-JSON Python objects raise ``ValueError`` or
    ``TypeError``. Array order is significant; no rounding or ``default=str``
    coercion occurs. This is COLEMAN JSON v1, not RFC 8785 cross-language JCS.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise ValueError("Canonical JSON object keys must be strings")
            canonical_json(value)
    elif isinstance(payload, list):
        for value in payload:
            canonical_json(value)
    elif payload is not None and not isinstance(payload, str | bool | int | float):
        raise TypeError("Canonical input must contain only JSON values")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def decode_json(payload: str | bytes) -> Any:
    """Parse JSON without silently overwriting duplicate object keys.

    Syntax/duplicate-key failures raise ``ValueError``. Callers validate schema,
    types and finite numeric constraints after decoding; no files are accessed.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject ambiguous duplicate keys at every object depth."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=unique_object)


def fingerprint(payload: Any) -> str:
    """Hash canonical JSON input using identified SHA-256; no filesystem access."""
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class _Contract(BaseModel):  # pylint: disable=too-few-public-methods
    """Strict frozen base; use validated constructors, never ``model_construct``."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class Measurement(_Contract):  # pylint: disable=too-few-public-methods
    """One signed raw observation in an explicit unit and versioned meaning.

    ``value`` is required, even when null. Observed values are finite and may be
    zero or negative. Unknown, unavailable and not-applicable states require
    null; omission means the measurement was not declared at all. ``minimum``
    optionally declares a physical lower bound in the same unit. No implicit
    constraint is imposed on custom units. A reward is never a measurement.
    """

    name: Identifier
    value: Finite | None
    unit: Identifier
    meaning: Identifier
    direction: Literal["minimize", "maximize", "none"]
    status: Literal["observed", "unknown", "unavailable", "not_applicable"]
    minimum: Finite | None = None

    @model_validator(mode="after")
    def _check_observation(self) -> Self:
        """Enforce availability, reserved-name and optional lower-bound invariants."""
        if self.name in RESERVED_METRICS:
            raise ValueError(f"Reserved native metric: {self.name}")
        if (self.status == "observed") != (self.value is not None):
            raise ValueError("Observed requires a value; other statuses require explicit null")
        if self.value is not None and self.minimum is not None and self.value < self.minimum:
            raise ValueError("Measurement is below its declared minimum")
        return self


class LearningReward(_Contract):  # pylint: disable=too-few-public-methods
    """Finite transformed learning signal, separate from raw outcomes.

    ``transform_id`` identifies a versioned transformation; ``unit`` declares
    its output scale. A positive reward cannot erase a negative raw outcome.
    """

    value: Finite
    unit: Identifier
    transform_id: Identifier


class ResultProvenance(_Contract):  # pylint: disable=too-few-public-methods
    """Producer and canonical spec/evaluation fingerprints for reproducible joins.

    Hash the actual resolved spec used, and the declared evaluation context
    (metric definition, protocol and dataset interpretation). These identifiers
    record provenance, not proof that observations are statistically independent.
    ``source_revision`` is optional and is never collected from the environment.
    """

    producer: Identifier
    producer_version: Identifier
    spec_fingerprint: Fingerprint
    evaluation_fingerprint: Fingerprint
    source_revision: Identifier | None = None


class DomainRecord(_Contract):
    """Versioned immutable measurement bundle with a stable logical decision key.

    Identity is (namespace, run_id, execution_id, decision_id), excluding schema
    version and content. Execution IDs must identify the same logical execution
    after resume; never include worker IDs or random retry suffixes. Decision IDs
    distinguish policy/reward/build/variant when applicable. ``available_at`` is
    outcome availability, not ingestion time, normalized to UTC. Naive times,
    duplicate measurement names and versions other than integer 1 are rejected.
    Tuple measurements are sorted by name for deterministic serialization.
    """

    schema_version: Annotated[int, Field(strict=True)]
    namespace: Identifier
    run_id: Identifier
    execution_id: Identifier
    decision_id: Identifier
    data_fingerprint: Fingerprint
    available_at: AwareDatetime
    provenance: ResultProvenance
    measurements: tuple[Measurement, ...]
    learning_reward: LearningReward | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        """Fail closed for unsupported versions, including older unversioned data."""
        if value != 1:
            raise ValueError(f"Unsupported domain schema version: {value}")
        return value

    @field_validator("available_at", mode="before")
    @classmethod
    def _explicit_timestamp(cls, value: Any) -> Any:
        """Reject epoch numbers whose scale/timezone would otherwise be inferred."""
        if isinstance(value, str):
            # ISO parsing excludes Pydantic's numeric-string epoch inference.
            # AwareDatetime subsequently rejects date-only/naive ISO datetimes.
            return datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise ValueError("Availability must be an explicitly zoned timestamp")
        return value

    @field_validator("available_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """Normalize explicit offsets to UTC without inferring a timezone."""
        return value.astimezone(UTC)

    @field_validator("measurements")
    @classmethod
    def _sort_measurements(cls, value: tuple[Measurement, ...]) -> tuple[Measurement, ...]:
        """Reject ambiguous bundles and canonicalize declaration order."""
        names = [item.name for item in value]
        if not names or len(names) != len(set(names)):
            raise ValueError("Measurements must be nonempty with unique names")
        return tuple(sorted(value, key=lambda item: item.name))

    @field_validator("namespace")
    @classmethod
    def _check_namespace(cls, value: str) -> str:
        """Reserve the core namespace to prevent impersonating native records."""
        if value == "coleman" or value.startswith("coleman."):
            raise ValueError("Reserved native namespace")
        return value

    @property
    def logical_key(self) -> tuple[str, str, str, str]:
        """Return a retry-stable key; content conflicts at this key are errors."""
        return self.namespace, self.run_id, self.execution_id, self.decision_id

    @property
    def identity_digest(self) -> str:
        """Return SHA-256 of the logical key, independent of measurement values."""
        return fingerprint(list(self.logical_key))

    @property
    def digest(self) -> str:
        """Return identified SHA-256 of canonical serialized record content."""
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialize v1 deterministically; absent optional fields stay absent.

        Explicit optional nulls remain null. The wire format uses an array of
        named measurements; tuples prevent mutable dictionaries in core APIs.
        Raises validation errors for non-JSON/non-finite values.
        """
        validated = type(self).model_validate(self)
        return canonical_json(validated.model_dump(mode="json", exclude_unset=True))

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read v1 only; unknown fields, duplicate keys and unknown versions fail.

        Legacy native rows must be read with their existing APIs. No automatic
        migration invents missing units, timestamps or provenance.
        """
        raw = decode_json(payload)
        if not isinstance(raw, dict):
            raise ValueError("Domain record must be a JSON object")
        return cls.model_validate(raw)

    def measurement(self, name: str) -> Measurement:
        """Query a declared measurement by name; omission raises ``KeyError``."""
        for item in self.measurements:
            if item.name == name:
                return item
        raise KeyError(name)

    def require_comparable(self, other: DomainRecord, name: str) -> None:
        """Require compatible metadata and observed values; perform no inference.

        Unknown/unavailable outcomes fail explicitly. Use ``require_compatible``
        to validate only metadata when reporting incomplete evidence.
        """
        self.require_compatible(other, name)
        if self.measurement(name).status != "observed" or other.measurement(name).status != "observed":
            raise ValueError("Comparison requires observed values")

    def require_compatible(self, other: DomainRecord, name: str) -> None:
        """Reject incompatible units, meanings, objectives, data or evaluations.

        This validates metadata only, without calculating a ranking, probability
        or statistical comparison. Missing values are allowed for metadata validation.
        Different producers/specs may be compared within the same evaluation.
        """
        left, right = self.measurement(name), other.measurement(name)

        def metadata(item: Measurement) -> tuple[str, str, str, float | None]:
            """Extract explicit semantics, without coercing or comparing values."""
            return item.unit, item.meaning, item.direction, item.minimum

        if (
            self.schema_version != other.schema_version
            or self.namespace != other.namespace
            or self.data_fingerprint != other.data_fingerprint
            or self.provenance.evaluation_fingerprint != other.provenance.evaluation_fingerprint
            or metadata(left) != metadata(right)
        ):
            raise ValueError("Incompatible measurement comparison")
