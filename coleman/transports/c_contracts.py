"""Versioned optional C function capture; generic controlled records retain their original semantics."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.controlled_contracts import ControlledContext, ControlledElapsed, Label, Role
from coleman.corpus.materialization_contracts import MaterializationReceipt, safe_member_name
from coleman.corpus.source_views import SourceViewSpec
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.transports.junit_contracts import CaptureInstant, JunitCaptureLimits, ProducerElapsed

CIdentifier = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=128)]
Unsigned32 = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
Count = Annotated[int, Field(strict=True, ge=0, le=256)]


class CInputCase(Immutable):
    """One declared decimal uint32 input; mathematical expected output is independent of subject execution.

    The v1 checked-increment oracle expects n+1 below UINT32_MAX and an explicit
    rejection message at the maximum. This domain of inputs excludes signed,
    malformed and out-of-range scanf conversions rather than assuming them safe.
    Test labels and replicated executions do not create independent seeded faults.
    """

    test_id: Label
    value: Unsigned32

    @property
    def expected_stdout(self) -> str:
        """Return complete ASCII stdout with LF; observed zero never means a missing observation."""
        if self.value == 2**32 - 1:
            return "data value is too large to perform arithmetic safely.\n"
        return str(self.value + 1) + "\n"


class CRuntimeImage(Immutable):
    """Already present official GCC platform manifest; image identity does not license all its components."""

    implementation: Literal["GCC"] = "GCC"
    version: Label
    platform: Literal["linux/arm64", "linux/amd64"]
    image_digest: Fingerprint
    emulated: Annotated[bool, Field(strict=True)]
    python_version: Label
    libc_version: Label
    target: Label
    timezone: Literal["UTC"] = "UTC"
    locale: Literal["C"] = "C"


class CCaptureRecipe(VersionedContract):
    """Freeze unchanged C functions, common support files and a bounded checked-uint32 increment oracle.

    This optional producer is deliberately limited to void(void) functions using
    stdin/stdout. Role revisions label explicit function-selection interventions,
    not invented upstream commits. It neither runs upstream build scripts nor
    accepts shell/compiler flags. Only synthetic provenance is supported by v1;
    successful collection establishes neither real-fault nor empirical coverage.
    """

    MAXIMUM_BYTES: ClassVar[int] = 4 * 1024**2
    recipe: Literal["c-checked-uint32-function-capture-v1"] = "c-checked-uint32-function-capture-v1"
    recipe_id: Label
    project: Label
    fault_id: Label
    fault_origin: Literal["synthetic"] = "synthetic"
    dependence_group: Label
    buggy: SourceViewSpec
    fixed: SourceViewSpec
    fixture: SourceViewSpec
    source_member: Label
    support_units: Annotated[tuple[Label, ...], Field(min_length=1, max_length=16)]
    buggy_function: CIdentifier
    fixed_function: CIdentifier
    cases: Annotated[tuple[CInputCase, ...], Field(min_length=1, max_length=256)]
    scratch_policy: Literal["private-executable-tmpfs-v1"] = "private-executable-tmpfs-v1"
    oracle: Literal["checked-unsigned32-increment-v1"] = "checked-unsigned32-increment-v1"
    runtime: CRuntimeImage
    harness_sha256: Fingerprint
    collector_sha256: Fingerprint
    limits: JunitCaptureLimits = Field(default_factory=JunitCaptureLimits)
    limitations: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]

    @property
    def test_ids(self) -> tuple[str, ...]:
        """Return canonical logical case identities; physical execution order follows this frozen order."""
        return tuple(case.test_id for case in self.cases)

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject ambiguous selectors, source names, duplicate cases and unsupported source/fixture declarations."""
        safe_member_name(self.source_member)
        if not self.source_member.endswith(".c"):
            raise ValueError("C source must be a pinned C translation unit")
        if self.buggy.output.source.version == self.fixed.output.source.version:
            raise ValueError("C role revisions must explicitly distinguish function selectors")
        if self.buggy_function == self.fixed_function:
            raise ValueError("C role functions must be distinct")
        for view in (self.buggy, self.fixed):
            if self.source_member not in {m.name for m in view.output.members}:
                raise ValueError("C source member is outside its pinned role view")
        fixture = {m.name for m in self.fixture.output.members}
        for name in self.support_units:
            safe_member_name(name)
            if not name.endswith(".c") or name not in fixture:
                raise ValueError("C support unit is outside the pinned common fixture")
        if len(set(self.support_units)) != len(self.support_units):
            raise ValueError("Duplicate C support translation unit")
        if len(self.test_ids) != len(set(self.test_ids)):
            raise ValueError("Duplicate C input case identity")
        object.__setattr__(self, "cases", tuple(sorted(self.cases, key=lambda c: c.test_id)))
        object.__setattr__(self, "support_units", tuple(sorted(self.support_units)))
        return self


class CHarnessPlan(VersionedContract):
    """Producer-written input with fixed mount roots; shell commands and external build scripts are unsupported."""

    MAXIMUM_BYTES: ClassVar[int] = 1024**2
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    recipe: CCaptureRecipe


class CHeader(VersionedContract):
    """Actual compiler/interpreter/ABI conditions and successful build digest, emitted before observations."""

    record_type: Literal["header"]
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    plan_sha256: Fingerprint
    compiler_version: Label
    compiler_target: Label
    python_version: Label
    libc_version: Label
    architecture: Label
    unsigned_bits: Annotated[int, Field(strict=True, ge=32, le=32)]
    timezone: Literal["UTC"]
    locale: Literal["C"]
    executable_sha256: Fingerprint
    test_ids: Annotated[tuple[Label, ...], Field(max_length=256)]


class CCase(VersionedContract):
    """Actual child interval in integer monotonic ns, with full bounded streams and an independent oracle.

    Exit codes include valid negative POSIX signals. A nonzero child exit is a
    test error, never an assertion inferred from exit alone. Completed normal
    children with wrong bytes fail the explicit output assertion. These ticks
    include local process startup and are not removable CI scheduling costs.
    """

    record_type: Literal["case"]
    test_id: Label
    outcome: Literal["passed", "assertion_failed", "test_error", "timeout"]
    termination: Literal["exited", "timeout", "output_limit"]
    elapsed: ControlledElapsed
    capture_started_at: CaptureInstant
    capture_finished_at: CaptureInstant
    exit_code: Annotated[int, Field(strict=True, ge=-255, le=255)]
    stdout: Annotated[str, Field(strict=True, max_length=65536)]
    stderr: Annotated[str, Field(strict=True, max_length=65536)]

    @model_validator(mode="after")
    def _observation(self) -> Self:
        """Require genuine observed ticks and ordered UTC clocks; no completed case can impute missingness."""
        if self.elapsed.status != "observed" or self.capture_finished_at < self.capture_started_at:
            raise ValueError("Completed C case requires ordered clocks and observed elapsed")
        if self.termination == "exited" and ((self.exit_code != 0) != (self.outcome == "test_error")):
            raise ValueError("C child physical exit disagrees with its outcome")
        if (self.termination == "timeout") != (self.outcome == "timeout"):
            raise ValueError("C child deadline disagrees with its outcome")
        if self.termination == "output_limit" and self.outcome != "test_error":
            raise ValueError("C output limit cannot become an assertion or pass")
        try:
            size = len(self.stdout.encode("ascii")) + len(self.stderr.encode("ascii"))
        except UnicodeEncodeError as exc:
            raise ValueError("C v1 child output must remain exact ASCII") from exc
        if size > 65536 or (self.termination == "output_limit" and size != 65536):
            raise ValueError("C case output disagrees with its combined 64KiB guard")
        if self.termination == "timeout" and (self.elapsed.value is None or self.elapsed.value < 5 * 10**9):
            raise ValueError("C case timeout must reach its five-second deadline")
        return self


class CTrailer(VersionedContract):
    """Explicit complete oracle evaluation; negative comparisons remain valid collected results."""

    record_type: Literal["trailer"]
    cases: Count
    oracle_exit_code: Annotated[int, Field(strict=True, ge=0, le=1)]


class CRoleCapture(VersionedContract):
    """Bounded raw evidence and parsed cases for one physical role attempt, with explicit process missingness.

    Negative POSIX exit codes are valid. Interrupted ownership has no recovered
    elapsed counter or cases. Process errors may occur before spawning a child;
    a genuinely observed zero counter must never become unavailable/null.
    """

    MAXIMUM_BYTES: ClassVar[int] = 12 * 1024**2
    kind: Literal["bounded-local-c-oracle-capture-v1"] = "bounded-local-c-oracle-capture-v1"
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    status: Literal["collected", "timeout", "output_limit", "process_error", "protocol_error", "interrupted"]
    termination: Literal["exited", "timeout", "output_limit"] | None
    exit_code: Annotated[int, Field(strict=True, ge=-255, le=255)] | None
    producer_elapsed: ProducerElapsed
    stdout_base64: Annotated[str, Field(strict=True, max_length=6 * 1024**2)]
    stderr_base64: Annotated[str, Field(strict=True, max_length=6 * 1024**2)]
    header: CHeader | None
    cases: Annotated[tuple[CCase, ...], Field(max_length=256)]
    trailer: CTrailer | None
    diagnostics: Annotated[str, Field(strict=True, max_length=4096)]

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Reject fake completion, duplicate cases and observed counters invented after interruption."""
        if len(self.cases) != len({case.test_id for case in self.cases}):
            raise ValueError("Duplicate C role case")
        if self.status == "collected" and (
            self.header is None
            or self.trailer is None
            or self.producer_elapsed.status != "observed"
            or self.exit_code != 0
            or self.termination != "exited"
        ):
            raise ValueError("Collected C role requires protocol completion and elapsed evidence")
        return self

    @model_validator(mode="after")
    def _records(self) -> Self:
        """Conserve completed case counts and oracle outcome status without imputing absent observations."""
        if (self.cases or self.trailer is not None) and self.header is None:
            raise ValueError("C case/trailer evidence requires its runtime header")
        if self.trailer is not None and self.trailer.cases != len(self.cases):
            raise ValueError("C trailer count disagrees with actual cases")
        if self.status == "collected" and self.header is not None and self.trailer is not None:
            if self.trailer.oracle_exit_code not in {0, 1} or set(self.header.test_ids) != {
                case.test_id for case in self.cases
            }:
                raise ValueError("Collected C role cannot omit discovered tests")
            failed = any(case.outcome in {"assertion_failed", "test_error", "timeout"} for case in self.cases)
            if failed != (self.trailer.oracle_exit_code == 1):
                raise ValueError("C oracle exit status disagrees with observed outcomes")
        return self

    @model_validator(mode="after")
    def _process(self) -> Self:
        """Bind physical process termination/exit/elapsed evidence, including valid negative exit codes."""
        if self.status in {"timeout", "output_limit", "protocol_error"} and (
            self.producer_elapsed.status != "observed" or self.exit_code is None or self.termination is None
        ):
            raise ValueError("Attempted C process requires observed elapsed/exit evidence")
        if self.status in {"timeout", "output_limit"} and self.termination != self.status:
            raise ValueError("C termination disagrees with its collection status")
        if (
            self.status == "process_error"
            and self.termination is None
            and (self.exit_code is not None or self.producer_elapsed.status != "unavailable")
        ):
            raise ValueError("Unstarted C process cannot invent exit or elapsed evidence")
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
            raise ValueError("Interrupted C role cannot invent case or elapsed evidence")
        return self


class CRoleStart(VersionedContract):
    """Durable ownership written before spawning; it is neither a completed capture nor a learning checkpoint."""

    MAXIMUM_BYTES: ClassVar[int] = 8192
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Role
    container_name: Annotated[str, Field(strict=True, pattern=r"^coleman-c-[0-9a-f]{24}$")]


class CEnvironment(VersionedContract):
    """Filtered Docker/host facts; excludes tags, contexts, credentials, hostnames and unrelated containers.

    Image size is Docker storage bytes, not peak process memory. The kernel is
    an execution condition, not a property guaranteed by the pinned image.
    """

    kind: Literal["filtered-local-c-runtime-v1"] = "filtered-local-c-runtime-v1"
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


class CCapturePlan(VersionedContract):
    """Durable pre-execution binding, written before any subject process can start.

    Resume rebuilds context from the frozen recipe, cached inputs and recorded
    environment. A changed installed producer requires a new output generation.
    The plan is neither completed evidence nor a native policy checkpoint.
    """

    MAXIMUM_BYTES: ClassVar[int] = 8 * 1024**2
    recipe: CCaptureRecipe
    trial_id: Label
    producer_sha256: Fingerprint
    environment: MaterializationReceipt
    context: ControlledContext


class CRoleReference(Immutable):
    """Reference a separately validated raw packet; its elapsed remains producer-process nanoseconds."""

    role: Role
    packet_sha256: Fingerprint
    status: Literal["collected", "timeout", "output_limit", "process_error", "protocol_error", "interrupted"]
    producer_elapsed: ProducerElapsed


class CCaptureReceipt(VersionedContract):
    """Deterministic capture index; incomplete evidence cannot acquire a completion marker.

    Every referenced dataset/raw packet must still be validated against bytes.
    Collected means protocol completeness, including negative test outcomes;
    admission, historical validity and generalization remain separate questions.
    """

    MAXIMUM_BYTES: ClassVar[int] = 16384
    plan_fingerprint: Fingerprint
    context_fingerprint: Fingerprint
    dataset_fingerprint: Fingerprint
    roles: Annotated[tuple[CRoleReference, ...], Field(min_length=2, max_length=2)]
    status: Literal["collected", "incomplete"]
    history: Literal["constructed"] = "constructed"
    admission: Literal["not_evaluated"] = "not_evaluated"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Require both distinct roles and completion exactly when both actual packets were collected."""
        if {item.role for item in self.roles} != {"buggy", "fixed"}:
            raise ValueError("C receipt requires both distinct source roles")
        if (self.status == "collected") != all(item.status == "collected" for item in self.roles):
            raise ValueError("C receipt completion disagrees with source role states")
        object.__setattr__(self, "roles", tuple(sorted(self.roles, key=lambda item: item.role)))
        return self
