"""Versioned optional Python source/fixture/wheel recipes; validation never executes or acquires inputs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.contracts import LicenseReview
from coleman.corpus.controlled_contracts import Label
from coleman.corpus.materialization_contracts import (
    MaterializationSpec,
    PackageMember,
    safe_member_name,
)
from coleman.corpus.source_views import SourceViewSpec
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.transports.junit_contracts import JunitCaptureLimits

DistributionName = Annotated[str, Field(strict=True, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=128)]
Count = Annotated[int, Field(strict=True, ge=0, le=256)]
PythonTest = Annotated[
    str,
    Field(
        strict=True,
        max_length=1024,
        pattern=r"^[A-Za-z0-9_/.-]+\.py::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$",
    ),
]


class DistributionVersion(Immutable):
    """One actual or frozen distribution version; versions are labels, never independent observations."""

    name: DistributionName
    version: Label


class PythonCaptureLimits(JunitCaptureLimits):
    """Reuse the established seconds/bytes/container guards without treating their values as CI costs."""


class PythonRuntimeImage(Immutable):
    """Pinned official CPython image and explicit locale; no pull, process or shell command is executed here.

    The image owns its bundled setup tools. Their versions are separately attested;
    image identity does not license every component under one umbrella license.
    UTF-8 coercion/mode are disabled in both locale profiles by the v1 producer.
    """

    implementation: Literal["CPython"] = "CPython"
    version: Label
    platform: Literal["linux/arm64", "linux/amd64"]
    image_digest: Fingerprint
    emulated: Annotated[bool, Field(strict=True)]
    timezone: Literal["UTC"] = "UTC"
    locale: Literal["C", "C.UTF-8"]
    preferred_encoding: Label
    provided_distributions: Annotated[tuple[DistributionVersion, ...], Field(min_length=1, max_length=16)]

    @model_validator(mode="after")
    def _provided(self) -> Self:
        """Reject ambiguous runtime distributions and canonicalize declaration order only."""
        names = [item.name for item in self.provided_distributions]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate Python runtime distribution")
        object.__setattr__(
            self, "provided_distributions", tuple(sorted(self.provided_distributions, key=lambda x: x.name))
        )
        return self


class PythonComponentTerms(Immutable):
    """Component notices, separate from a distribution's top-level license declaration.

    Original notices stay in the unmodified wheel. An additional public notice
    can be retained alongside it without patching wheel bytes. Source similarity
    is explicitly an attribution inference, not established component provenance.
    """

    component: Label
    scope: Label
    license: LicenseReview
    notice_members: Annotated[tuple[Label, ...], Field(min_length=1, max_length=128)]
    additional_notices: Annotated[tuple[MaterializationSpec, ...], Field(max_length=8)]
    attribution: Literal["declared_upstream", "source_similarity_inferred"]
    citation: Label

    @model_validator(mode="after")
    def _reviewed(self) -> Self:
        """Require reviewed recipe-only terms and unambiguous pinned notice members."""
        if self.license.status != "verified" or self.license.redistribution != "recipe_only":
            raise ValueError("Python wheel component requires reviewed recipe-only terms")
        if len(set(self.notice_members)) != len(self.notice_members):
            raise ValueError("Duplicate Python component notice")
        if any(notice.archive != "raw" for notice in self.additional_notices):
            raise ValueError("Additional Python component notice must remain an unmodified raw file")
        if len(self.additional_notices) != len({notice.package_id for notice in self.additional_notices}):
            raise ValueError("Duplicate additional Python component notice")
        object.__setattr__(self, "notice_members", tuple(sorted(self.notice_members)))
        object.__setattr__(
            self, "additional_notices", tuple(sorted(self.additional_notices, key=lambda x: x.package_id))
        )
        return self


class PythonWheel(Immutable):
    """An independently reviewed pure-Python wheel, retained whole as an opaque raw package.

    The enclosing SHA-256 binds every original byte. Selected metadata/notices
    have independent member pins; remaining bodies are not separately decoded or
    certified. Existing bounded ZIP selection validates every entry's path/type
    and local metadata, without expanding core complete-package limits. Labels
    are checked against actual METADATA. Compiled wheels/builds are unsupported.
    """

    distribution: DistributionVersion
    package: MaterializationSpec
    members: Annotated[tuple[PackageMember, ...], Field(min_length=1, max_length=128)]
    entry_count: Annotated[int, Field(strict=True, ge=1, le=1024)]
    decoded_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)]
    components: Annotated[tuple[PythonComponentTerms, ...], Field(min_length=1, max_length=32)]
    installation_scripts: Annotated[tuple[Label, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def _package(self) -> Self:
        """Bound whole-wheel entries/decoded bytes and independently pin selected metadata/notices."""
        names = [member.name for member in self.members]
        if (
            self.package.archive != "raw"
            or not self.package.source.uri.endswith("none-any.whl")
            or not self.package.members[0].name.endswith("none-any.whl")
        ):
            raise ValueError("Python dependencies require pinned pure-Python wheels")
        if (
            sum(name.endswith(".dist-info/METADATA") for name in names) != 1
            or sum(name.endswith(".dist-info/WHEEL") for name in names) != 1
        ):
            raise ValueError("Python wheel requires distinct distribution metadata")
        if self.package.source.size_bytes is None or self.package.source.size_bytes > 8 * 1024**2:
            raise ValueError("Python wheel exceeds its 8-MiB source limit")
        if self.package.source.version != self.distribution.version:
            raise ValueError("Python wheel source version disagrees with its distribution")
        if len(names) != len(set(names)) or sum(member.size_bytes for member in self.members) > 16 * 1024**2:
            raise ValueError("Python wheel members are duplicated or exceed their 16-MiB decoded budget")
        if (
            len(self.members) > self.entry_count
            or sum(member.size_bytes for member in self.members) > self.decoded_bytes
        ):
            raise ValueError("Python selected wheel metadata exceeds declared whole-wheel bounds")
        object.__setattr__(self, "members", tuple(sorted(self.members, key=lambda member: member.name)))
        return self

    @model_validator(mode="after")
    def _components(self) -> Self:
        """Bind every independently reviewed component/script notice to actual retained byte pins."""
        names = {member.name for member in self.members}
        if len(self.components) != len({item.component for item in self.components}):
            raise ValueError("Duplicate Python wheel component terms")
        if len(self.installation_scripts) != len(set(self.installation_scripts)) or any(
            name not in names or ".data/scripts/" not in name or not name.endswith(".py")
            for name in self.installation_scripts
        ):
            raise ValueError("Python wheel installation scripts require explicit distinct original Python member pins")
        if any(not set(component.notice_members) <= set(names) for component in self.components):
            raise ValueError("Python component notice must be independently pinned in its original wheel")
        pins = {member.name: member.sha256 for member in self.members}
        for component in self.components:
            evidence = {pins[name] for name in component.notice_members}
            evidence = evidence | {notice.source.sha256 for notice in component.additional_notices}
            if component.license.terms_sha256 not in evidence:
                raise ValueError(
                    "Python component terms digest must reference retained original or additional evidence"
                )
        object.__setattr__(
            self, "components", tuple(sorted(self.components, key=lambda component: component.component))
        )
        object.__setattr__(self, "installation_scripts", tuple(sorted(self.installation_scripts)))
        return self


class PythonCaptureRecipe(VersionedContract):
    """Freeze a source-only upstream pair and common unmodified fixed-tree pytest fixture.

    This v1 protocol supports simple module/function or class/method node IDs,
    pure wheels and two declared libc locales. It never runs project setup.py or
    arbitrary recipe commands. A curated fix label is not a verified fault map.
    No model validation fetches, stages or executes source bytes.
    """

    MAXIMUM_BYTES: ClassVar[int] = 4 * 1024**2
    recipe: Literal["python-pytest-fixed-tree-capture-v1"] = "python-pytest-fixed-tree-capture-v1"
    recipe_id: Label
    project: Label
    fault_id: Label
    fault_origin: Literal["real", "synthetic"]
    dependence_group: Label
    buggy: SourceViewSpec
    fixed: SourceViewSpec
    fixture: SourceViewSpec
    wheels: Annotated[tuple[PythonWheel, ...], Field(min_length=1, max_length=16)]
    runtime: PythonRuntimeImage
    production_module: Annotated[str, Field(strict=True, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=128)]
    test_ids: Annotated[tuple[PythonTest, ...], Field(min_length=1, max_length=256)]
    harness_sha256: Fingerprint
    limits: PythonCaptureLimits = Field(default_factory=PythonCaptureLimits)
    limitations: Annotated[tuple[Label, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject changed fixture provenance, unsafe/undeclared tests, duplicate wheels or weakened byte budgets."""
        if self.buggy.output.source.version == self.fixed.output.source.version:
            raise ValueError("Python source roles require distinct revisions")
        if self.fixture.output.source.version != self.fixed.output.source.version:
            raise ValueError("V1 common Python fixture must come from the fixed revision")
        if len(self.test_ids) != len(set(self.test_ids)):
            raise ValueError("Duplicate declared Python test ID")
        fixture_names = {item.name for item in self.fixture.output.members}
        for test_id in self.test_ids:
            name = test_id.split("::", 1)[0]
            safe_member_name(name)
            if name not in fixture_names:
                raise ValueError("Python test is outside the declared common fixture")
        for view in (self.buggy, self.fixed):
            if not any(
                member.name == self.production_module + ".py"
                or (member.name.startswith(self.production_module + "/") and member.name.endswith(".py"))
                for member in view.output.members
            ):
                raise ValueError("Python source view lacks its declared production module")
        object.__setattr__(self, "test_ids", tuple(sorted(self.test_ids)))
        return self

    @model_validator(mode="after")
    def _dependencies(self) -> Self:
        """Bound distinct whole-wheel inputs before acquisition, preserving provided image distributions."""
        wheels = {item.distribution.name for item in self.wheels}
        provided = {item.name for item in self.runtime.provided_distributions}
        if len(wheels) != len(self.wheels) or wheels & provided or "pytest" not in wheels:
            raise ValueError("Python wheels require distinct distributions including pytest")
        if len({item.package.package_id for item in self.wheels}) != len(self.wheels):
            raise ValueError("Duplicate Python wheel package identity")
        if sum(item.package.source.size_bytes or 0 for item in self.wheels) > 32 * 1024**2:
            raise ValueError("Python wheels exceed their 32-MiB combined source limit")
        if sum(item.decoded_bytes for item in self.wheels) > 64 * 1024**2:
            raise ValueError("Python wheels exceed their 64-MiB decoded limit")
        object.__setattr__(self, "wheels", tuple(sorted(self.wheels, key=lambda item: item.distribution.name)))
        return self

    @model_validator(mode="after")
    def _notices(self) -> Self:
        """Bound separately retained original notices before any source acquisition or runtime access."""
        notices = {}
        for notice in self.component_notices():
            if notice.package_id in notices and notices[notice.package_id] != notice:
                raise ValueError("Conflicting Python additional notice identity")
            if notice.source.size_bytes is None or notice.source.size_bytes > 1024**2:
                raise ValueError("Python additional notice exceeds its 1-MiB byte limit")
            notices[notice.package_id] = notice
        if len(notices) > 16 or sum(notice.source.size_bytes or 0 for notice in notices.values()) > 4 * 1024**2:
            raise ValueError("Python additional notices exceed their count/4-MiB byte limit")
        return self

    def component_notices(self) -> Iterator[MaterializationSpec]:
        """Yield separately retained public notice recipes, without acquisition or deduplication.

        Validation checks conflicting package identities before any caller can
        materialize them. Repeated identical references are permitted and remain
        one source input, never independent component observations.
        """
        for wheel in self.wheels:
            for component in wheel.components:
                yield from component.additional_notices
