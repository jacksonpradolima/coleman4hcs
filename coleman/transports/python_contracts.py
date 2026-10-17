"""Typed optional Python/pytest capture records; the controlled domain stays transport independent."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BeforeValidator, Field, model_validator

from coleman.corpus.controlled_contracts import ControlledContext, ControlledElapsed, Label, Role
from coleman.corpus.materialization_contracts import (
    MaterializationReceipt,
    safe_member_name,
)
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.transports.junit_contracts import ProducerElapsed
from coleman.transports.python_recipe_contracts import (
    DistributionVersion as DistributionVersion,
)
from coleman.transports.python_recipe_contracts import (
    PythonCaptureLimits as PythonCaptureLimits,
)
from coleman.transports.python_recipe_contracts import (
    PythonCaptureRecipe as PythonCaptureRecipe,
)
from coleman.transports.python_recipe_contracts import (
    PythonComponentTerms as PythonComponentTerms,
)
from coleman.transports.python_recipe_contracts import (
    PythonRuntimeImage as PythonRuntimeImage,
)
from coleman.transports.python_recipe_contracts import (
    PythonTest,
)
from coleman.transports.python_recipe_contracts import (
    PythonWheel as PythonWheel,
)

Count = Annotated[int, Field(strict=True, ge=0, le=256)]


def _instant(value):
    """Reject ambiguous or sub-microsecond wire clocks before datetime could discard their precision."""
    if (
        isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value, flags=re.ASCII) is None
    ):
        raise ValueError("Python capture instants require UTC Z and at most six fractional digits")
    if not isinstance(value, str | datetime):
        raise ValueError("Python capture instant must be an explicitly zoned value")
    return value


CaptureInstant = Annotated[Timestamp, BeforeValidator(_instant)]


class PythonHeader(VersionedContract):
    """Actual interpreter/locale/distribution facts and discovered selected IDs, emitted before test cases."""

    record_type: Literal["header"]
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    plan_sha256: Fingerprint
    python_version: Label
    architecture: Label
    timezone: Label
    locale: Label
    preferred_encoding: Label
    filesystem_encoding: Label
    utf8_mode: Annotated[int, Field(strict=True, ge=0, le=0)]
    capture_clock: Literal["producer-truncated-utc-microsecond-v1"]
    distributions: Annotated[tuple[DistributionVersion, ...], Field(min_length=1, max_length=32)]
    test_ids: Annotated[tuple[PythonTest, ...], Field(max_length=256)]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        """Reject ambiguous discovery/version evidence and sort declaration order without changing observations."""
        if len(self.test_ids) != len(set(self.test_ids)) or len(self.distributions) != len(
            {item.name for item in self.distributions}
        ):
            raise ValueError("Duplicate Python runtime discovery")
        object.__setattr__(self, "test_ids", tuple(sorted(self.test_ids)))
        object.__setattr__(self, "distributions", tuple(sorted(self.distributions, key=lambda item: item.name)))
        return self


class PythonHarnessPlan(VersionedContract):
    """Producer-written bounded bootstrap input, not an operator shell command or a domain execution API.

    Production paths are fixed container mounts; owned harness tests may supply
    separate temporary roots. The outer producer validates/stages every path.
    Dependency storage is an explicit private-runtime allocation policy, not a
    caller-selected predictable directory. Wheel filenames/test members remain
    portable and relative; constructing this model never creates directories.
    """

    MAXIMUM_BYTES: ClassVar[int] = 1024**2
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    fixture_root: Label
    production_root: Label
    dependency_storage: Literal["private-runtime-tempdir-v1"]
    wheel_root: Label
    wheels: Annotated[tuple[Label, ...], Field(max_length=16)]
    distributions: Annotated[tuple[DistributionVersion, ...], Field(min_length=1, max_length=32)]
    test_ids: Annotated[tuple[PythonTest, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def _members(self) -> Self:
        """Reject duplicate/escaped wheel names and logical cases before writing bootstrap input."""
        if len(self.wheels) != len(set(self.wheels)) or len(self.test_ids) != len(set(self.test_ids)):
            raise ValueError("Duplicate Python harness input")
        for name in self.wheels:
            safe_member_name(name)
            if "/" in name or not name.endswith("none-any.whl"):
                raise ValueError("Python harness requires relative pure-wheel filenames")
        for test_id in self.test_ids:
            safe_member_name(test_id.split("::", 1)[0])
        if len(self.distributions) != len({item.name for item in self.distributions}):
            raise ValueError("Duplicate Python harness distribution")
        object.__setattr__(self, "test_ids", tuple(sorted(self.test_ids)))
        object.__setattr__(self, "wheels", tuple(sorted(self.wheels)))
        object.__setattr__(self, "distributions", tuple(sorted(self.distributions, key=lambda item: item.name)))
        return self


class PythonCase(VersionedContract):
    """One completed pytest protocol interval, including setup/call/teardown; elapsed ticks are not CI costs."""

    record_type: Literal["case"]
    test_id: PythonTest
    outcome: Literal["passed", "assertion_failed", "test_error", "skipped"]
    elapsed: ControlledElapsed
    capture_started_at: CaptureInstant
    capture_finished_at: CaptureInstant
    error_type: Label | None
    diagnostics: Annotated[str, Field(strict=True, max_length=1024**2)]

    @model_validator(mode="after")
    def _observation(self) -> Self:
        """Require an actual completed counter, ordered clocks and honest exception presence; zero remains observed."""
        if self.elapsed.status != "observed":
            raise ValueError("Completed Python case requires observed elapsed ticks")
        if self.capture_finished_at < self.capture_started_at:
            raise ValueError("Python capture finishes before it starts")
        if (self.error_type is not None) != (self.outcome in {"assertion_failed", "test_error"}):
            raise ValueError("Python case exception identity disagrees with its outcome")
        return self


class PythonTrailer(VersionedContract):
    """Explicit pytest completion; tests-failed is valid collection, not a process-level success claim."""

    record_type: Literal["trailer"]
    cases: Count
    pytest_exit_code: Annotated[int, Field(strict=True, ge=0, le=5)]


class PythonRoleCapture(VersionedContract):
    """Bounded raw evidence and parsed cases for one physical role attempt, with explicit process missingness.

    Negative POSIX exit codes are valid. Interrupted ownership has no recovered
    elapsed counter or cases. Process errors may occur before spawning a child;
    a genuinely observed zero counter must never become unavailable/null.
    """

    MAXIMUM_BYTES: ClassVar[int] = 12 * 1024**2
    kind: Literal["bounded-local-python-pytest-capture-v1"] = "bounded-local-python-pytest-capture-v1"
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    status: Literal["collected", "timeout", "output_limit", "process_error", "protocol_error", "interrupted"]
    termination: Literal["exited", "timeout", "output_limit"] | None
    exit_code: Annotated[int, Field(strict=True, ge=-255, le=255)] | None
    producer_elapsed: ProducerElapsed
    stdout_base64: Annotated[str, Field(strict=True, max_length=6 * 1024**2)]
    stderr_base64: Annotated[str, Field(strict=True, max_length=6 * 1024**2)]
    header: PythonHeader | None
    cases: Annotated[tuple[PythonCase, ...], Field(max_length=256)]
    trailer: PythonTrailer | None
    diagnostics: Annotated[str, Field(strict=True, max_length=4096)]

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Reject fake completion, duplicate cases and observed counters invented after interruption."""
        if len(self.cases) != len({case.test_id for case in self.cases}):
            raise ValueError("Duplicate Python role case")
        if self.status == "collected" and (
            self.header is None
            or self.trailer is None
            or self.producer_elapsed.status != "observed"
            or self.exit_code != 0
            or self.termination != "exited"
        ):
            raise ValueError("Collected Python role requires protocol completion and elapsed evidence")
        return self

    @model_validator(mode="after")
    def _records(self) -> Self:
        """Conserve completed case counts and pytest outcome status without imputing absent observations."""
        if (self.cases or self.trailer is not None) and self.header is None:
            raise ValueError("Python case/trailer evidence requires its runtime header")
        if self.trailer is not None and self.trailer.cases != len(self.cases):
            raise ValueError("Python trailer count disagrees with actual cases")
        if self.status == "collected" and self.header is not None and self.trailer is not None:
            if self.trailer.pytest_exit_code not in {0, 1} or set(self.header.test_ids) != {
                case.test_id for case in self.cases
            }:
                raise ValueError("Collected Python role cannot omit discovered tests")
            failed = any(case.outcome in {"assertion_failed", "test_error"} for case in self.cases)
            if failed != (self.trailer.pytest_exit_code == 1):
                raise ValueError("Python pytest exit status disagrees with observed outcomes")
        return self

    @model_validator(mode="after")
    def _process(self) -> Self:
        """Bind physical process termination/exit/elapsed evidence, including valid negative exit codes."""
        if self.status in {"timeout", "output_limit", "protocol_error"} and (
            self.producer_elapsed.status != "observed" or self.exit_code is None or self.termination is None
        ):
            raise ValueError("Attempted Python process requires observed elapsed/exit evidence")
        if self.status in {"timeout", "output_limit"} and self.termination != self.status:
            raise ValueError("Python termination disagrees with its collection status")
        if (
            self.status == "process_error"
            and self.termination is None
            and (self.exit_code is not None or self.producer_elapsed.status != "unavailable")
        ):
            raise ValueError("Unstarted Python process cannot invent exit or elapsed evidence")
        return self

    @model_validator(mode="after")
    def _interrupted(self) -> Self:
        """Uncommitted ownership records cannot invent recovered outcomes, bytes or elapsed ticks."""
        if self.status == "interrupted" and (
            self.cases
            or self.header is not None
            or self.trailer is not None
            or self.producer_elapsed.status != "unavailable"
            or self.exit_code is not None
            or self.termination is not None
            or self.stdout_base64
            or self.stderr_base64
        ):
            raise ValueError("Interrupted Python role cannot invent case or elapsed evidence")
        return self


class PythonRoleStart(VersionedContract):
    """Durable ownership written before spawning; it is neither a completed capture nor a learning checkpoint."""

    MAXIMUM_BYTES: ClassVar[int] = 8192
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    container_name: Annotated[str, Field(strict=True, pattern=r"^coleman-python-[0-9a-f]{24}$")]


class PythonEnvironment(VersionedContract):
    """Filtered Docker/host facts; excludes tags, contexts, credentials, hostnames and unrelated containers.

    Image size is Docker storage bytes, not peak process memory. The kernel is
    an execution condition, not a property guaranteed by the pinned image.
    """

    kind: Literal["filtered-local-python-runtime-v1"] = "filtered-local-python-runtime-v1"
    host_os: Label
    host_architecture: Label
    host_release: Label
    producer_python_version: Label
    producer_package_version: Label
    docker_client_version: Label
    docker_server_version: Label
    docker_server_architecture: Literal["arm64", "aarch64", "amd64", "x86_64"]
    container_kernel: Label
    image_os: Literal["linux"]
    image_architecture: Literal["arm64", "amd64"]
    image_size_bytes: Annotated[int, Field(strict=True, ge=0, le=64 * 1024**3)]
    image_config_digest: Fingerprint


class PythonCapturePlan(VersionedContract):
    """Durable pre-execution binding, written before any subject process can start.

    Resume rebuilds context from the frozen recipe, cached inputs and recorded
    environment. A changed installed producer requires a new output generation.
    The plan is neither completed evidence nor a native policy checkpoint.
    """

    MAXIMUM_BYTES: ClassVar[int] = 8 * 1024**2
    recipe: PythonCaptureRecipe
    trial_id: Label
    producer_sha256: Fingerprint
    environment: MaterializationReceipt
    context: ControlledContext


class PythonRoleReference(Immutable):
    """Reference a separately validated raw packet; its elapsed remains producer-process nanoseconds."""

    role: Role
    packet_sha256: Fingerprint
    status: Literal["collected", "timeout", "output_limit", "process_error", "protocol_error", "interrupted"]
    producer_elapsed: ProducerElapsed


class PythonCaptureReceipt(VersionedContract):
    """Deterministic capture index; incomplete evidence cannot acquire a completion marker.

    Every referenced dataset/raw packet must still be validated against bytes.
    Collected means protocol completeness, including negative test outcomes;
    admission, historical validity and generalization remain separate questions.
    """

    MAXIMUM_BYTES: ClassVar[int] = 16384
    plan_fingerprint: Fingerprint
    context_fingerprint: Fingerprint
    dataset_fingerprint: Fingerprint
    roles: Annotated[tuple[PythonRoleReference, ...], Field(min_length=2, max_length=2)]
    status: Literal["collected", "incomplete"]
    history: Literal["constructed"] = "constructed"
    admission: Literal["not_evaluated"] = "not_evaluated"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Require both distinct roles and completion exactly when both actual packets were collected."""
        if {item.role for item in self.roles} != {"buggy", "fixed"}:
            raise ValueError("Python receipt requires both distinct source roles")
        if (self.status == "collected") != all(item.status == "collected" for item in self.roles):
            raise ValueError("Python receipt completion disagrees with source role states")
        object.__setattr__(self, "roles", tuple(sorted(self.roles, key=lambda item: item.role)))
        return self
