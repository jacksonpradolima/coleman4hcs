"""Controlled fault observations: paired functional evidence, never inferred CI chronology."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import Fingerprint, Identifier, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

Label = Annotated[Identifier, Field(max_length=1024)]
Ticks = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Role = Literal["buggy", "fixed"]
Outcome = Literal[
    "passed", "assertion_failed", "test_error", "skipped", "timeout", "cancelled", "infrastructure_error", "unavailable"
]


class ControlledInput(Immutable):
    """A reviewed source input receipt; identity is its descriptive name, never an executable path.

    Receipts prove local byte correspondence only when validated against a cache.
    They do not prove that a process consumed those bytes. Opaque archives can be
    materialized as raw members without broadening extraction support.
    """

    name: Label
    kind: Literal["buggy_source", "fixed_source", "test_fixture", "harness", "dependency"]
    materialization: MaterializationReceipt


class ControlledRuntime(Immutable):
    """Producer-attested execution conditions; commands are descriptions and are never run here.

    Image digest identifies a platform-specific container manifest, not a blanket
    license for its components. Dependency receipts belong in the context inputs.
    Setup/intervention descriptions must include derived build metadata and test
    transplants. Environment identity is not measured performance or isolation proof.
    """

    implementation: Label
    version: Label
    platform: Label
    execution_model: Literal["container", "host", "constructed"]
    image_digest: Fingerprint | None
    emulated: Annotated[bool, Field(strict=True)]
    timezone: Label
    setup: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]
    commands: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]
    interventions: Annotated[tuple[Label, ...], Field(max_length=64)]

    @model_validator(mode="after")
    def _image(self) -> Self:
        """Require an image identity exactly for container executions; do not invent one for host fixtures."""
        if (self.image_digest is not None) != (self.execution_model == "container"):
            raise ValueError("Container runtime requires image digest; host/constructed runtime has none")
        return self


class ControlledContext(VersionedContract):
    """Freeze one fault, revisions, common test universe, inputs and execution recipe.

    ``fault_id`` is a curated subject label, not a verified unique-fault mapping.
    Test fixtures are identical in both roles, including any declared transplant.
    Source/input revisions are explicit labels and no external imports are performed.
    Repeated trials retain the same dependence group; seeds never create new bugs.
    """

    recipe: Literal["controlled-fault-observations-v1"] = "controlled-fault-observations-v1"
    project: Label
    fault_id: Label
    language: Label
    fault_origin: Literal["real", "synthetic"]
    buggy_revision: Label
    fixed_revision: Label
    fixture_revision: Label
    dependence_group: Label
    test_ids: Annotated[tuple[Label, ...], Field(min_length=1, max_length=10000)]
    inputs: Annotated[tuple[ControlledInput, ...], Field(min_length=4, max_length=64)]
    runtime: ControlledRuntime
    limitations: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _identities(self) -> Self:
        """Require distinct revisions, unique inputs/tests and all four source/fixture/harness roles."""
        if self.buggy_revision == self.fixed_revision:
            raise ValueError("Controlled pair requires distinct buggy/fixed revisions")
        if len(set(self.test_ids)) != len(self.test_ids):
            raise ValueError("Duplicate controlled test identity")
        if len({item.name for item in self.inputs}) != len(self.inputs):
            raise ValueError("Duplicate controlled input name")
        for kind in ("buggy_source", "fixed_source", "test_fixture", "harness"):
            if sum(item.kind == kind for item in self.inputs) != 1:
                raise ValueError("Controlled context requires exactly one input of each source/fixture/harness kind")
        revisions = {
            "buggy_source": self.buggy_revision,
            "fixed_source": self.fixed_revision,
            "test_fixture": self.fixture_revision,
        }
        if any(
            item.materialization.spec.source.version != revisions[item.kind]
            for item in self.inputs
            if item.kind in revisions
        ):
            raise ValueError("Controlled input source version disagrees with declared revision")
        object.__setattr__(self, "test_ids", tuple(sorted(self.test_ids)))
        object.__setattr__(self, "inputs", tuple(sorted(self.inputs, key=lambda item: item.name)))
        return self


class ControlledElapsed(Immutable):
    """Nonnegative monotonic elapsed ticks in nanoseconds; null is explicitly unavailable/inapplicable.

    Integer nanoseconds retain exact counter resolution, not physical accuracy.
    No conversion to runner work, per-test removable cost or CI savings is allowed.
    Genuine zero is observed. Negative, fractional, bool and nonfinite values reject.
    """

    value: Ticks | None
    unit: Literal["ns"] = "ns"
    meaning: Literal["observed-test-harness-monotonic-elapsed-v1"] = "observed-test-harness-monotonic-elapsed-v1"
    status: Literal["observed", "unavailable", "not_applicable"]

    @model_validator(mode="after")
    def _missingness(self) -> Self:
        """Keep observed zero separate from an absent elapsed counter without imputing a value."""
        if (self.value is not None) != (self.status == "observed"):
            raise ValueError("Controlled elapsed value/status disagree")
        return self


class ControlledObservation(Immutable):
    """One role/test/trial observation, including aborted and non-assertion outcomes.

    Trial IDs identify actual executions, not independent faults. Capture instants
    are collection clocks, never historical CI events or verified feature clocks.
    A timeout can have elapsed ticks; it cannot become an assertion failure. Raw
    diagnostics remain in separate pinned report-package members, outside decision
    features. A non-null digest must match such a member during conversion.
    """

    trial_id: Label
    role: Role
    test_id: Label
    outcome: Outcome
    elapsed: ControlledElapsed
    capture_started_at: Timestamp
    capture_finished_at: Timestamp
    diagnostics_sha256: Fingerprint | None

    @model_validator(mode="after")
    def _clock(self) -> Self:
        """Reject reversed capture clocks; do not equate UTC interval with monotonic elapsed ticks."""
        if self.capture_finished_at < self.capture_started_at:
            raise ValueError("Controlled capture finishes before it starts")
        return self

    @property
    def logical_key(self) -> tuple[str, str, str]:
        """Return role/test/trial identity; context adds project/fault/runtime/input identity."""
        return self.trial_id, self.test_id, self.role

    def execution_id(self, context: ControlledContext) -> str:
        """Hash stable context and logical key, excluding observations and capture values."""
        return fingerprint(["controlled-execution-v1", context.fingerprint, *self.logical_key])


class ControlledReport(VersionedContract):
    """Bounded producer observation interchange, bound to an exact frozen context.

    Duplicate physical observations remain visible for adapter accounting. Conflict
    detection happens during conversion; this is not an executable test driver.
    Missing, legacy and future schema versions reject without migration.
    """

    context_fingerprint: Fingerprint
    observations: Annotated[tuple[ControlledObservation, ...], Field(max_length=20000)]


class ControlledSpec(VersionedContract):
    """Bind a frozen context to a reviewed, pinned producer report in an external source cache."""

    context: ControlledContext
    report_materialization: MaterializationReceipt
    report_member: Label

    @model_validator(mode="after")
    def _member(self) -> Self:
        """Require exactly one declared report member with an uncompressed 8-MiB ceiling."""
        members = [item for item in self.report_materialization.spec.members if item.name == self.report_member]
        if len(members) != 1 or members[0].size_bytes > 8 * 1024**2:
            raise ValueError("Controlled report member missing or exceeds 8 MiB")
        return self


class ControlledPair(Immutable):
    """Derived same-test/same-trial comparison; absent roles and non-assertion outcomes are inconclusive."""

    trial_id: Label
    test_id: Label
    buggy: Outcome | None
    fixed: Outcome | None

    @property
    def classification(self) -> str:
        """Return expected, both-pass, both-fail, reversed or inconclusive without dropping negatives."""
        classifications: dict[tuple[Outcome | None, Outcome | None], str] = {
            ("assertion_failed", "passed"): "expected",
            ("passed", "passed"): "both_pass",
            ("assertion_failed", "assertion_failed"): "both_fail",
            ("passed", "assertion_failed"): "reversed",
        }
        return classifications.get((self.buggy, self.fixed), "inconclusive")


class ControlledDataset(VersionedContract):
    """Canonical functional observations, separate from historical CI and scientific admission.

    Producer attestations plus byte correspondence do not verify execution truth,
    a complete original suite, independent fault counts or protected efficacy.
    New collections have new trial IDs; retries of the same collection retain IDs.
    The complete source report remains pinned even after duplicate exclusion.
    """

    spec: ControlledSpec
    input_rows: Annotated[int, Field(strict=True, ge=0, le=20000)]
    duplicates_excluded: Annotated[int, Field(strict=True, ge=0, le=20000)]
    observations: Annotated[tuple[ControlledObservation, ...], Field(max_length=20000)]
    history: Literal["constructed"] = "constructed"
    scope: Literal["controlled-functional-observations-only"] = "controlled-functional-observations-only"
    admission: Literal["not_evaluated"] = "not_evaluated"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _rows(self) -> Self:
        """Conserve physical rows, prohibit duplicate keys and reject tests outside the declared universe."""
        if self.input_rows != len(self.observations) + self.duplicates_excluded:
            raise ValueError("Controlled observation conservation failed")
        if len({item.logical_key for item in self.observations}) != len(self.observations):
            raise ValueError("Duplicate controlled execution identity")
        test_ids = set(self.spec.context.test_ids)
        if any(item.test_id not in test_ids for item in self.observations):
            raise ValueError("Controlled observation test outside declared universe")
        object.__setattr__(self, "observations", tuple(sorted(self.observations, key=lambda item: item.logical_key)))
        return self

    @model_validator(mode="after")
    def _diagnostics(self) -> Self:
        """Bind optional diagnostic digests to separate declared report members on conversion and every read.

        This pure check validates declared membership only. Conversion/publication
        separately verifies actual package/member bytes through the source cache.
        """
        diagnostic_digests = {
            item.sha256
            for item in self.spec.report_materialization.spec.members
            if item.name != self.spec.report_member
        }
        if any(
            item.diagnostics_sha256 is not None and item.diagnostics_sha256 not in diagnostic_digests
            for item in self.observations
        ):
            raise ValueError("Controlled diagnostic digest must reference a validated report package member")
        return self

    @property
    def pairs(self) -> tuple[ControlledPair, ...]:
        """Compare only explicitly collected test/trial groups; never invent missing observations."""
        groups: dict[tuple[str, str], dict[str, Outcome]] = {}
        for item in self.observations:
            groups.setdefault((item.trial_id, item.test_id), {})[item.role] = item.outcome
        return tuple(
            ControlledPair(trial_id=trial, test_id=test, buggy=roles.get("buggy"), fixed=roles.get("fixed"))
            for (trial, test), roles in sorted(groups.items())
        )

    def require_temporal_evaluation(self) -> None:
        """Reject historical evaluation: capture clocks do not supply CI chronology or predecision features."""
        raise ValueError("Controlled observations are not historical CI; temporal evaluation is unavailable")
