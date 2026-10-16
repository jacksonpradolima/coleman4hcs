"""Versioned transport-neutral commands, receipts and bounded portable results."""

from __future__ import annotations

import base64
import hashlib
import json
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, WrapValidator, field_serializer, model_validator

from coleman.allocation.results import AllocationResult
from coleman.results.domain import Fingerprint, Finite, decode_json
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import (
    Immutable,
    UnitSemantics,
    decimal_text,
    validate_decimal_places,
    validate_decimal_schema,
)
from coleman.selection.results import CoverageResult

SafeId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", strict=True)]
ErrorCode = Literal[
    "invalid_input", "unknown_spec", "not_found", "forbidden", "expired", "resource_limit", "unavailable_result"
]
JobState = Literal["queued", "running", "completed", "failed", "cancelled", "expired"]


class ApplicationContract(VersionedContract):
    """Preserve JSON decimal literals before scientific quantity validation.

    The shared decoder first rejects duplicate keys; a bounded second parse keeps
    exact decimal literals rather than rounding them through binary64. Only this
    new application contract changes reading; existing native readers retain their
    compatibility behavior. No filesystem or execution side effects occur.
    """

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Reject ambiguous JSON and preserve exact scientific decimal input values."""
        decode_json(payload)
        return cls.model_validate(json.loads(payload, parse_float=Decimal))


class ApplicationError(Exception):
    """Stable public failure; details never contain worker exceptions or server paths."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Store a machine-readable code and safe human explanation."""
        super().__init__(message)
        self.code = code


class Caller(Immutable):
    """Trusted host-supplied identity, never accepted from an application command."""

    subject: SafeId


class Limits(Immutable):
    """Requested caps, reduced by server policy; elapsed seconds include queue time.

    Work is native solver examinations or native experiment build/agent executions.
    Memory is estimated solver working bytes plus sampled own-executable worker peak RSS, not a
    hard allocation sandbox. Output includes all worker files and portable JSON.
    """

    deadline_seconds: Annotated[Finite, Field(gt=0, le=300)] = 30.0
    max_work: Annotated[int, Field(strict=True, ge=0, le=100000)] = 10000
    max_candidates: Annotated[int, Field(strict=True, ge=0, le=1000)] = 100
    max_memory_bytes: Annotated[int, Field(strict=True, ge=1048576, le=2147483648)] = 536870912
    max_output_bytes: Annotated[int, Field(strict=True, ge=1024, le=2097152)] = 262144


class ServerPolicy(Immutable):
    """Trusted finite aggregate reservations; one worker per active job, no process pools.

    Retained jobs include terminal/idempotency tombstones until service shutdown.
    Expiry deletes bytes, not tombstones; bounded admission then fails closed.
    Worker RSS/output sampling can overshoot between 20 ms samples; this is a
    local trusted-code resource guard, not protection against hostile plugins.
    """

    caps: Limits = Limits()
    max_running: Annotated[int, Field(strict=True, ge=1, le=4)] = 1
    max_queued: Annotated[int, Field(strict=True, ge=0, le=64)] = 8
    max_jobs: Annotated[int, Field(strict=True, ge=1, le=128)] = 64
    max_retained_bytes: Annotated[int, Field(strict=True, ge=1024, le=268435456)] = 16777216
    max_active_memory_bytes: Annotated[int, Field(strict=True, ge=1048576, le=8589934592)] = 2147483648
    retention_seconds: Annotated[Finite, Field(gt=0, le=86400)] = 3600.0


class Capability(Immutable):
    """Public registered configuration identity and supported override names."""

    spec_id: SafeId
    kind: Literal["coverage", "allocation", "experiment"]
    configuration_fingerprint: Fingerprint
    parameters: tuple[Literal["budget"], ...]
    budget_semantics: UnitSemantics | None = None


class Discovery(ApplicationContract):
    """Finite capability discovery without configuration paths or executable details."""

    transport_contract: Literal["coleman-application-v1"] = "coleman-application-v1"
    specifications: Annotated[tuple[Capability, ...], Field(max_length=32)]
    policy: ServerPolicy
    error_codes: tuple[ErrorCode, ...] = (
        "invalid_input",
        "unknown_spec",
        "not_found",
        "forbidden",
        "expired",
        "resource_limit",
        "unavailable_result",
    )


class Parameters(Immutable):
    """Explicit v1 override: coverage budget in the registered cost unit; no arbitrary keys."""

    budget: (
        Annotated[
            Decimal,
            Field(ge=0, le=1000000000000, max_digits=25, decimal_places=12),
            AfterValidator(validate_decimal_places),
            WrapValidator(validate_decimal_schema),
        ]
        | None
    ) = None

    @field_serializer("budget")
    def _budget(self, value: Decimal | None) -> str | None:
        """Preserve exact declared decimal units through JSON and request identity."""
        return None if value is None else decimal_text(value)


class Submit(ApplicationContract):
    """Idempotent request scoped by trusted caller and immutable registered configuration."""

    request_id: SafeId
    spec_id: SafeId
    parameters: Parameters = Parameters()
    limits: Limits = Limits()


class Handle(ApplicationContract):
    """Content-addressed local receipt; neither this digest nor job_id grants access."""

    job_id: SafeId
    digest: Fingerprint


class JobView(ApplicationContract):
    """Job state snapshot; handle exists only after successful immutable publication.

    Cancellation/expiry expose no partial result. Native work-limited results may
    be completed application jobs with explicitly non-optimal native status.
    """

    job_id: SafeId
    request_id: SafeId
    spec_id: SafeId
    state: JobState
    limits: Limits
    handle: Handle | None = None
    error: ErrorCode | None = None

    @model_validator(mode="after")
    def _receipt(self) -> Self:
        """Reject contradictory completion, ownership identity and failure/expiry receipts."""
        if self.state == "completed" and self.handle is None:
            raise ValueError("Completed jobs require a result handle")
        if self.handle is not None and (
            self.state not in {"completed", "expired"} or self.handle.job_id != self.job_id
        ):
            raise ValueError("Result handle does not match completed job identity")
        if self.state == "failed" and self.error not in {"resource_limit", "unavailable_result"}:
            raise ValueError("Failed jobs require a stable worker error")
        if self.state == "expired" and self.error != "expired":
            raise ValueError("Expired jobs require the expiry error")
        if self.state not in {"failed", "expired"} and self.error is not None:
            raise ValueError("Nonfailed jobs cannot claim a failure error")
        return self


class PortableFile(Immutable):
    """Opaque native artifact bytes, with integrity and safe logical name; never extracted.

    Legacy Parquet values keep their existing meanings and units. Packaging does
    not migrate them into domain measurements or assert scientific completeness.
    """

    name: Annotated[str, Field(max_length=256, strict=True)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=2097152)]
    content_base64: Annotated[str, Field(max_length=2796204, strict=True)]

    @model_validator(mode="after")
    def _integrity(self) -> Self:
        """Reject traversal, noncanonical base64 and modified bytes before accepting a receipt."""
        path = PurePosixPath(self.name)
        if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in self.name:
            raise ValueError("Unsafe logical artifact name")
        if str(path) != self.name:
            raise ValueError("Noncanonical logical artifact name")
        content = base64.b64decode(self.content_base64, validate=True)
        if base64.b64encode(content).decode("ascii") != self.content_base64:
            raise ValueError("Noncanonical base64")
        if len(content) != self.size_bytes or hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError("Artifact integrity mismatch")
        return self


class PortableNativeProvenance(ApplicationContract):
    """Explicit derivative of native provenance without ambient server filesystem context.

    The source digest binds the original local artifact for host-side audit; it
    does not authenticate a producer. Only runtime facts are copied. Redaction
    is explicit and versioned; the existing native artifact is never modified.
    """

    artifact_type: Literal["portable-native-provenance-v1"] = "portable-native-provenance-v1"
    producer: Literal["coleman.api.run"] = "coleman.api.run"
    producer_version: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    source_sha256: Fingerprint
    python_version: Annotated[str, Field(strict=True, min_length=1, max_length=512)]
    platform: Annotated[str, Field(strict=True, min_length=1, max_length=512)]
    redacted_fields: tuple[Literal["cwd"], Literal["git"], Literal["uv_lock_hash"]] = ("cwd", "git", "uv_lock_hash")


class ExperimentResult(ApplicationContract):
    """Native api.run receipt with original artifact bytes and stable resolved run identity.

    No metrics are inferred from an empty RunResult.metrics. Files include native
    result/specification bytes and an explicitly versioned, redacted derivative
    of provenance. Original native provenance remains unchanged locally and is
    never exported by execution. Imported files retain their declared contracts.
    Runtime timestamps and measurements may differ between equivalent experiments.
    """

    run_id: Annotated[str, Field(pattern=r"^[0-9a-f]{12}$", strict=True)]
    work_units: Annotated[int, Field(strict=True, ge=1, le=100000)]
    files: Annotated[tuple[PortableFile, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _files(self) -> Self:
        """Bound aggregate bytes and require unique sorted names for deterministic packaging."""
        names = tuple(file.name for file in self.files)
        if names != tuple(sorted(set(names))) or sum(file.size_bytes for file in self.files) > 2097152:
            raise ValueError("Invalid native artifact collection")
        return self


class ResultBundle(ApplicationContract):
    """Explicit portable v1 result with approved-source identity and native proof validation.

    Import verifies integrity/math, never authenticates the original producer or
    re-executes experiments. Imported records are rebound to the importing caller.
    Old native artifacts remain readable by their original APIs; unknown bundle
    versions fail closed without silent migration. Digests cover canonical bytes.
    """

    spec_id: SafeId
    configuration_fingerprint: Fingerprint
    kind: Literal["coverage", "allocation", "experiment"]
    result: CoverageResult | AllocationResult | ExperimentResult

    @model_validator(mode="after")
    def _kind(self) -> Self:
        """Prevent a native result from masquerading as another capability."""
        expected = {"coverage": CoverageResult, "allocation": AllocationResult, "experiment": ExperimentResult}
        if not isinstance(self.result, expected[self.kind]):
            raise ValueError("Result kind mismatch")
        return self
