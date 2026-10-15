"""Transport-independent, licensed and bounded source-package integrity contracts."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from coleman.corpus.contracts import LicenseReview, SourceArtifact
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

Bytes = Annotated[int, Field(strict=True, ge=0, le=8 * 1024**3)]


def safe_member_name(name: str) -> str:
    """Require a canonical relative portable archive name, without executing it as a filesystem path.

    Absolute, drive, traversal, empty/dot components, backslash, control and
    non-ASCII names reject with ValueError. V1 accepts conservative portable
    names only. Materialized bytes are addressed by digest, never by this name.
    """
    if not 1 <= len(name) <= 1024 or not re.fullmatch(r"[A-Za-z0-9_./ -]+", name):
        raise ValueError("Unsafe or unsupported archive member name")
    if name.startswith("/") or any(
        part in {"", ".", ".."} or part != part.strip() or part.endswith(".") for part in name.split("/")
    ):
        raise ValueError("Archive member name must be a canonical relative path")
    return name


class ByteReader(Protocol):
    """Minimal synchronous source interface; concrete filesystem/HTTP adapters live outside models."""

    def read(self, maximum: int, /) -> bytes:
        """Read at most maximum bytes; empty bytes means EOF and I/O failures propagate."""
        ...


class PackageMember(Immutable):
    """One expected regular member, pinned before acquisition; size is in uncompressed bytes."""

    name: Annotated[str, Field(strict=True)]
    sha256: Fingerprint
    size_bytes: Bytes

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Reject unsafe logical names before any source/cache access."""
        return safe_member_name(value)


class MaterializationLimits(Immutable):
    """Per-operation source-entity, decoded-byte and metadata limits; not a global cache quota or OS sandbox.

    Existing cached blobs can occupy additional disk. A replay may temporarily
    stage one existing member again for verification. No compression-ratio or
    timing value is interpreted as a scientific result.
    """

    maximum_source_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * 1024**3)] = 64 * 1024**2
    maximum_member_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * 1024**3)] = 128 * 1024**2
    maximum_extracted_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * 1024**3)] = 256 * 1024**2
    maximum_archive_entries: Annotated[int, Field(strict=True, ge=1, le=128)] = 128
    maximum_zip_metadata_bytes: Annotated[int, Field(strict=True, ge=22, le=8 * 1024**2)] = 1024**2


class MaterializationSpec(VersionedContract):
    """Frozen integrity recipe with independently reviewed data terms, not evaluation admission.

    Source URI/version/citation/digest/size and specific license evidence are
    required. URI remains descriptive: the core needs an explicit reader factory
    and never opens a URL/import from these fields. Recipe-only sources may be
    cached locally outside repositories; receipts never embed their data.
    V1 supports raw files and bounded single-disk non-ZIP64 ZIP packages only.
    """

    recipe: Literal["bounded-source-package-v1"] = "bounded-source-package-v1"
    package_id: Identifier
    source: SourceArtifact
    license: LicenseReview
    archive: Literal["raw", "zip"]
    members: Annotated[tuple[PackageMember, ...], Field(min_length=1, max_length=128)]
    limits: MaterializationLimits = Field(default_factory=MaterializationLimits)

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        """Fail closed on unknown terms, missing pins, duplicate members and inconsistent budgets."""
        if self.license.status != "verified" or self.license.redistribution not in {"permitted", "recipe_only"}:
            raise ValueError("Source acquisition requires verified data terms permitting local use")
        if self.source.sha256 is None or self.source.size_bytes is None:
            raise ValueError("Source acquisition requires a frozen digest and byte count")
        if self.source.size_bytes > self.limits.maximum_source_bytes:
            raise ValueError("Source exceeds declared source/disk byte limit")
        names = tuple(member.name for member in self.members)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate package member identity")
        if len(names) > self.limits.maximum_archive_entries:
            raise ValueError("Package exceeds declared entry limit")
        if any(member.size_bytes > self.limits.maximum_member_bytes for member in self.members):
            raise ValueError("Package member exceeds declared decoded-byte limit")
        if sum(member.size_bytes for member in self.members) > self.limits.maximum_extracted_bytes:
            raise ValueError("Package exceeds total decoded-byte limit")
        if self.archive == "raw" and (
            len(self.members) != 1
            or self.members[0].sha256 != self.source.sha256
            or self.members[0].size_bytes != self.source.size_bytes
        ):
            raise ValueError("Raw package must declare exactly its unchanged source bytes")
        object.__setattr__(self, "members", tuple(sorted(self.members, key=lambda member: member.name)))
        return self


class MaterializationReceipt(VersionedContract):
    """Deterministic source-package integrity receipt, separate from scientific admission.

    There are no duplicated derived totals or independently editable success
    counters: expected source/member byte identities are in the frozen recipe.
    Reading validates metadata only, not the existence/truth of cached bytes.
    Use validate_materialization for actual offline source/member verification,
    and publish_materialization to verify before immutable publication. Missing,
    legacy and future versions reject; old native/card artifacts are not migrated.
    """

    spec: MaterializationSpec
    scope: Literal["source-integrity-only"] = "source-integrity-only"
    admission: Literal["not_evaluated"] = "not_evaluated"
