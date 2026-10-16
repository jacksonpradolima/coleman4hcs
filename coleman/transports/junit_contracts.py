"""Typed optional JUnit3 source-capture transport; no domain module imports this adapter."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BeforeValidator, Field, model_validator

from coleman.corpus.controlled_contracts import Label, Ticks
from coleman.corpus.materialization_contracts import MaterializationSpec, safe_member_name
from coleman.corpus.source_views import SourceViewSpec
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

JavaClass = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$")]
JavaTest = Annotated[
    str,
    Field(
        strict=True,
        max_length=1024,
        pattern=r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*::[A-Za-z_$][A-Za-z0-9_$]*$",
    ),
]
Count = Annotated[int, Field(strict=True, ge=0, le=10000)]


def _capture_instant(value):
    """Reject sub-microsecond or ambiguous string instants before Python datetime could truncate them."""
    if (
        isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value, flags=re.ASCII) is None
    ):
        raise ValueError("JUnit capture instants require UTC Z and at most six fractional digits")
    if not isinstance(value, str | datetime):
        raise ValueError("JUnit capture instant must be an explicitly zoned value")
    return value


CaptureInstant = Annotated[Timestamp, BeforeValidator(_capture_instant)]


class JunitCaptureLimits(Immutable):
    """Per-role monotonic seconds, combined stdout/stderr bytes and container memory bytes.

    Network is disabled during execution; two CPUs, 64 PIDs and a 128-MiB tmpfs
    are fixed by this v1 producer. These are execution guards, not benchmark costs.
    Captures remain trusted-code experiments, not a hostile-code security sandbox.
    """

    timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=120)] = 60
    maximum_output_bytes: Annotated[int, Field(strict=True, ge=1024, le=4 * 1024**2)] = 1024**2
    memory_bytes: Annotated[int, Field(strict=True, ge=128 * 1024**2, le=2 * 1024**3)] = 1024**3


class JunitRuntimeImage(Immutable):
    """Pinned public Temurin platform image and expected actual JVM facts; no executable commands."""

    implementation: Literal["Temurin"] = "Temurin"
    version: Label
    platform: Literal["linux/arm64", "linux/amd64"]
    image_digest: Fingerprint
    emulated: Annotated[bool, Field(strict=True)]
    timezone: Label


class JunitCaptureRecipe(VersionedContract):
    """Freeze reviewed source views, a common test universe and a pinned Linux Java image.

    No URI, command or source code is executed by this model. Java/JUnit3 is an
    optional producer boundary; generic controlled observation models stay free
    of it. Source revisions label one focal fix, not universal correctness. All
    same-test negative outcomes remain evidence and no unique-fault map is inferred.
    """

    recipe: Literal["junit3-fixed-tree-capture-v1"] = "junit3-fixed-tree-capture-v1"
    recipe_id: Label
    project: Label
    fault_id: Label
    fault_origin: Literal["real", "synthetic"]
    dependence_group: Label
    buggy: SourceViewSpec
    fixed: SourceViewSpec
    fixture: SourceViewSpec
    dependencies: Annotated[tuple[MaterializationSpec, ...], Field(min_length=1, max_length=16)]
    runtime: JunitRuntimeImage
    production_prefix: Label
    fixture_prefix: Label
    test_ids: Annotated[tuple[JavaTest, ...], Field(min_length=1, max_length=10000)]
    harness_sha256: Fingerprint
    limits: JunitCaptureLimits = Field(default_factory=JunitCaptureLimits)
    limitations: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject incompatible runtime, revisions, classpaths and unsafe/empty source prefixes."""
        if self.buggy.output.source.version == self.fixed.output.source.version:
            raise ValueError("JUnit source roles require distinct revisions")
        if self.fixture.output.source.version != self.fixed.output.source.version:
            raise ValueError("V1 common JUnit fixture must come from the fixed revision")
        for prefix in (self.production_prefix, self.fixture_prefix):
            safe_member_name(prefix)
        for view, prefix in (
            (self.buggy, self.production_prefix),
            (self.fixed, self.production_prefix),
            (self.fixture, self.fixture_prefix),
        ):
            if not any(
                item.name.startswith(prefix + "/") and item.name.endswith(".java") for item in view.output.members
            ):
                raise ValueError("JUnit view contains no Java file under its declared prefix")
        return self

    @model_validator(mode="after")
    def _dependencies(self) -> Self:
        """Require distinct bounded raw JARs and a canonical test universe."""
        if len(self.test_ids) != len(set(self.test_ids)):
            raise ValueError("Duplicate declared JUnit test ID")
        if len({item.package_id for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("Duplicate JUnit dependency package identity")
        if any(item.archive != "raw" or not item.members[0].name.endswith(".jar") for item in self.dependencies):
            raise ValueError("JUnit classpath requires pinned raw JAR packages")
        if (
            any(item.source.size_bytes is None or item.source.size_bytes > 32 * 1024**2 for item in self.dependencies)
            or sum(item.source.size_bytes or 0 for item in self.dependencies) > 64 * 1024**2
        ):
            raise ValueError("JUnit dependencies exceed per-file or total byte bounds")
        object.__setattr__(self, "test_ids", tuple(sorted(self.test_ids)))
        object.__setattr__(self, "dependencies", tuple(sorted(self.dependencies, key=lambda item: item.package_id)))
        return self


class JunitHeader(VersionedContract):
    """Actual runtime/discovery facts emitted before cases; collection is not observed CI history."""

    record_type: Literal["header"]
    java_version: Label
    architecture: Literal["aarch64", "amd64", "x86_64"]
    timezone: Label
    capture_clock: Literal["producer-truncated-utc-microsecond-v1"]
    test_ids: Annotated[tuple[JavaTest, ...], Field(min_length=1, max_length=10000)]
    excluded_classes: Annotated[tuple[JavaClass, ...], Field(max_length=128)]

    @model_validator(mode="after")
    def _universe(self) -> Self:
        """Reject duplicate runtime discoveries and canonicalize declaration order only."""
        if len(self.test_ids) != len(set(self.test_ids)) or len(self.excluded_classes) != len(
            set(self.excluded_classes)
        ):
            raise ValueError("Duplicate JUnit runtime discovery")
        object.__setattr__(self, "test_ids", tuple(sorted(self.test_ids)))
        object.__setattr__(self, "excluded_classes", tuple(sorted(self.excluded_classes)))
        return self


class JunitCase(VersionedContract):
    """One observed JUnit3 case, including setup/teardown in a monotonic harness counter.

    Truncated diagnostics force an infrastructure outcome rather than a successful
    reproduction claim. Test errors are distinct from assertion failures. Counter
    values are not CI scheduling costs and no full-suite green state is inferred.
    """

    record_type: Literal["case"]
    test_id: JavaTest
    tests_run: Annotated[int, Field(strict=True, ge=1, le=1)]
    failures: Count
    errors: Count
    elapsed_ns: Ticks
    capture_started_at: CaptureInstant
    capture_finished_at: CaptureInstant
    diagnostics: Annotated[str, Field(strict=True, max_length=1024**2)]
    diagnostics_truncated: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def _clock(self) -> Self:
        """Reject reversed capture instants without equating UTC and monotonic clocks."""
        if self.capture_finished_at < self.capture_started_at:
            raise ValueError("JUnit capture finishes before it starts")
        return self

    @model_validator(mode="after")
    def _terminal_counters(self) -> Self:
        """Require at most one terminal failure/error for the single executed JUnit3 case."""
        if self.failures + self.errors > self.tests_run:
            raise ValueError("JUnit terminal counters cannot exceed tests_run")
        return self

    @property
    def outcome(self) -> Literal["passed", "assertion_failed", "test_error", "infrastructure_error"]:
        """Classify actual counters; incomplete diagnostics cannot certify a fault reproduction."""
        if self.diagnostics_truncated:
            return "infrastructure_error"
        if self.errors:
            return "test_error"
        return "assertion_failed" if self.failures else "passed"


class JunitTrailer(VersionedContract):
    """Explicit process-level case completion; failures remain successful evidence collection."""

    record_type: Literal["trailer"]
    case_count: Count


class ProducerElapsed(Immutable):
    """Exact producer-process monotonic nanoseconds, including local container startup/observation overhead.

    This is distinct from per-test harness ticks, runner occupancy and CI latency.
    Interrupted recovery has unavailable null rather than an invented zero.
    """

    value: Ticks | None
    unit: Literal["ns"] = "ns"
    meaning: Literal["observed-producer-process-monotonic-elapsed-v1"] = (
        "observed-producer-process-monotonic-elapsed-v1"
    )
    status: Literal["observed", "unavailable"]

    @model_validator(mode="after")
    def _value(self) -> Self:
        """Reject observed null and unavailable numeric values; perform no clock access."""
        if (self.value is not None) != (self.status == "observed"):
            raise ValueError("Producer elapsed value/status disagree")
        return self


class JunitRoleCapture(VersionedContract):
    """Bounded raw role capture, separate from normalized observations and scientific admission.

    Base64 stores exact bounded stream bytes, including invalid/truncated output.
    A started but uncommitted role recovers as interrupted with no invented cases.
    Same-trial replay never silently reruns an interrupted role. A fresh collection
    requires a new trial ID and remains in the same declared dependence group.
    """

    MAXIMUM_BYTES: ClassVar[int] = 16 * 1024**2

    context_fingerprint: Fingerprint
    trial_id: Label
    role: Literal["buggy", "fixed"]
    status: Literal["collected", "process_error", "timeout", "output_limit", "protocol_error", "interrupted"]
    exit_code: Annotated[int, Field(strict=True, ge=-255, le=255)] | None
    capture_started_at: CaptureInstant
    capture_finished_at: CaptureInstant
    producer_elapsed: ProducerElapsed
    stdout_base64: Annotated[str, Field(strict=True, max_length=6 * 1024**2)] | None
    stderr_base64: Annotated[str, Field(strict=True, max_length=6 * 1024**2)] | None
    problem: Label | None = None
    header: JunitHeader | None
    cases: Annotated[tuple[JunitCase, ...], Field(max_length=10000)]

    @model_validator(mode="after")
    def _capture(self) -> Self:
        """Require collected captures to have a successful process, header and matching unique universe."""
        if self.capture_finished_at < self.capture_started_at:
            raise ValueError("Role capture finishes before it starts")
        if (self.stdout_base64 is None) != (self.stderr_base64 is None):
            raise ValueError("Role stream availability must agree")
        if self.status == "interrupted" and (
            self.exit_code is not None
            or self.stdout_base64 is not None
            or self.header is not None
            or self.producer_elapsed.status != "unavailable"
        ):
            raise ValueError("Interrupted role has unavailable streams, runtime facts and elapsed counter")
        if self.stdout_base64 is None and self.status not in {"interrupted", "process_error"}:
            raise ValueError("Role status requires observed stream bytes")
        if self.status == "collected":
            if self.exit_code != 0 or self.header is None:
                raise ValueError("Collected JUnit capture requires process success and runtime header")
            ids = tuple(item.test_id for item in self.cases)
            if len(ids) != len(set(ids)) or set(ids) != set(self.header.test_ids):
                raise ValueError("Collected JUnit cases disagree with declared runtime discovery")
        return self

    @model_validator(mode="after")
    def _incomplete(self) -> Self:
        """Keep incomplete raw packets separate from validated case/runtime observations."""
        if self.status != "collected":
            if self.cases:
                raise ValueError("Incomplete JUnit captures cannot promote partial/invalid streams into observations")
            if self.header is not None:
                raise ValueError("Incomplete JUnit captures cannot claim a validated runtime header")
        return self

    @model_validator(mode="after")
    def _process_facts(self) -> Self:
        """Bind raw-stream availability to observed process facts, including explicit unavailable counters.

        Missing streams cannot retain a known exit code or measured parent time.
        Observed streams require both. This does not prove whether an interrupted
        or failed process started; it only prohibits contradictory saved evidence.
        """
        if self.stdout_base64 is None:
            if self.exit_code is not None or self.producer_elapsed.status != "unavailable":
                raise ValueError("Unavailable role streams require unknown exit code and unavailable elapsed")
        elif self.exit_code is None or self.producer_elapsed.status != "observed":
            raise ValueError("Observed role streams require exit code and observed elapsed")
        return self


class JunitRoleStart(VersionedContract):
    """Immutable pre-execution acknowledgment, never a completed capture or experiment checkpoint.

    After process interruption this marker supplies identity and the collection
    start instant only. It cannot certify test outcomes or elapsed time.
    """

    phase: Literal["source-capture-started"] = "source-capture-started"
    context_fingerprint: Fingerprint
    trial_id: Label
    role: Literal["buggy", "fixed"]
    capture_started_at: CaptureInstant


class JunitEnvironment(VersionedContract):
    """Filtered local runtime facts; excludes image tags, containers, contexts, credentials and hostnames.

    Image size is observed Docker image storage bytes, not process memory or a
    source-cache quota. The kernel belongs to the execution environment, not the
    immutable image. These fields establish conditions, never native performance.
    """

    kind: Literal["filtered-local-junit-runtime-v1"] = "filtered-local-junit-runtime-v1"
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
