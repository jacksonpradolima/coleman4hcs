"""Explicit ZIP64 selected-file integrity contracts; legacy ZIP recipes remain unchanged."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.corpus.materialization_contracts import MaterializationSpec
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable


def zip64_member_name(name: str) -> str:
    """Validate a logical ASCII archive path, including @; never construct an extraction path.

    Unlike legacy package-v1 names, this separate grammar permits project labels
    containing @. Absolute/drive/traversal/empty/dot/backslash/control names and
    trailing spaces/dots reject. No filesystem or network access occurs.
    """
    if not 1 <= len(name) <= 1024 or not re.fullmatch(r"[A-Za-z0-9_./ @-]+", name):
        raise ValueError("Unsafe or unsupported ZIP64 member name")
    if name.startswith("/") or any(
        part in {"", ".", ".."} or part != part.strip() or part.endswith(".") for part in name.split("/")
    ):
        raise ValueError("ZIP64 member name must be a canonical relative path")
    return name


class Zip64Member(Immutable):
    """One selected regular file; SHA-256 and byte count refer to decoded member bytes."""

    name: Annotated[str, Field(strict=True)]
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=64 * 1024**2)]

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Require the explicit ZIP64 name grammar before source/cache access."""
        return zip64_member_name(value)


class Zip64SelectedFile(Immutable):
    """Bind an archive locator to an existing raw-file cache recipe, with an explicit safe local alias."""

    member: Zip64Member
    package: MaterializationSpec

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Require identical decoded bytes; nested archive expansion is outside this bounded profile."""
        if self.package.archive != "raw":
            raise ValueError("ZIP64 selection supports raw selected files only")
        if (self.member.sha256, self.member.size_bytes) != (self.package.source.sha256, self.package.source.size_bytes):
            raise ValueError("ZIP64 selected file and raw package must identify identical bytes")
        return self


class Zip64SelectionSpec(VersionedContract):
    """Freeze a full opaque ZIP64 source and a bounded selected-file view, independently of v1 ZIP recipes.

    Source bytes are bounded by the existing explicit raw materialization limits
    (at most 8 GiB). The central directory is bounded before standard-library
    allocation; selected decoded bytes total at most 64 MiB. Unselected payloads
    are never decoded/certified. Only single-disk stored/deflated forms are in
    scope. Source/version/data terms stay bound to each raw child. This contract
    makes no temporal/admission/subject-code licensing or execution assertion.
    """

    recipe: Literal["bounded-zip64-file-selection-v1"] = "bounded-zip64-file-selection-v1"
    source: MaterializationSpec
    selections: Annotated[tuple[Zip64SelectedFile, ...], Field(min_length=1, max_length=128)]
    maximum_entries: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    maximum_metadata_bytes: Annotated[int, Field(strict=True, ge=22, le=2 * 1024**2)] = 2 * 1024**2

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Conserve byte/name identity and reviewed terms before acquisition; no implicit legacy migration."""
        if self.source.archive != "raw":
            raise ValueError("ZIP64 container must be pinned as opaque raw bytes")
        if len(self.selections) > self.maximum_entries:
            raise ValueError("ZIP64 selection count exceeds declared directory limit")
        names = [item.member.name for item in self.selections]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP64 selected member")
        if sum(item.member.size_bytes for item in self.selections) > 64 * 1024**2:
            raise ValueError("ZIP64 decoded selection exceeds 64 MiB")
        if any(
            item.package.license != self.source.license or item.package.source.version != self.source.source.version
            for item in self.selections
        ):
            raise ValueError("ZIP64 child must retain enclosing data terms/version")
        object.__setattr__(self, "selections", tuple(sorted(self.selections, key=lambda item: item.member.name)))
        return self
