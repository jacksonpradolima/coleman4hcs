"""Explicit selected-member acquisition reusing complete source verification and the v1 cache."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import Annotated, Literal, Self, cast

from pydantic import Field, model_validator

from coleman.corpus.archives import consume_selected_zip_members
from coleman.corpus.materialization import ReaderFactory, SourceCache, _verified_chunks, materialize_source
from coleman.corpus.materialization_contracts import (
    ByteReader,
    MaterializationReceipt,
    MaterializationSpec,
    PackageMember,
)
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable


class SelectedPackage(Immutable):
    """Pin one decoded outer member and its existing raw/ZIP materialization recipe."""

    member: PackageMember
    package: MaterializationSpec

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Require identical bytes before the nested package's own verification/extraction."""
        if (self.member.sha256, self.member.size_bytes) != (self.package.source.sha256, self.package.source.size_bytes):
            raise ValueError("Selected member and nested package source must identify identical bytes")
        return self


class ArchiveSelectionSpec(VersionedContract):
    """Freeze an opaque ZIP source and selected child packages, without changing package-v1 policy.

    Unselected payloads are not decoded. Whole-container digest, every local
    header and selected child digests are verified before successful return.
    Source and child data terms/version must agree. No network occurs unless
    the caller supplies a reader; completed cache blobs remain reusable after
    interruption. Selection is an integrity operation, never corpus admission.
    """

    recipe: Literal["bounded-zip-member-selection-v1"] = "bounded-zip-member-selection-v1"
    source: MaterializationSpec
    selections: Annotated[tuple[SelectedPackage, ...], Field(min_length=1, max_length=128)]
    maximum_entries: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    maximum_metadata_bytes: Annotated[int, Field(strict=True, ge=22, le=2 * 1024**2)] = 2 * 1024**2

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject duplicate names, inconsistent terms/versions and decoded selection above 64 MiB."""
        if self.source.archive != "raw":
            raise ValueError("Selected ZIP container must first be pinned as opaque raw bytes")
        if len({item.member.name for item in self.selections}) != len(self.selections):
            raise ValueError("Duplicate selected archive member")
        if len(self.selections) > self.maximum_entries:
            raise ValueError("Selected member count exceeds declared directory limit")
        if sum(item.member.size_bytes for item in self.selections) > 64 * 1024**2:
            raise ValueError("Selected archive bytes exceed 64 MiB")
        if any(
            item.package.license != self.source.license or item.package.source.version != self.source.source.version
            for item in self.selections
        ):
            raise ValueError("Nested package must retain enclosing data terms/version")
        object.__setattr__(self, "selections", tuple(sorted(self.selections, key=lambda item: item.member.name)))
        return self


def materialize_selected_packages(
    spec: ArchiveSelectionSpec, cache: SourceCache, source: ReaderFactory | None = None
) -> tuple[MaterializationReceipt, ...]:
    """Acquire/reverify one complete container and selected nested packages offline by default.

    Returns receipts only after all selections succeed. Individual immutable
    verified blobs may survive interruption, but no selection completion marker
    is written. Replays repeat container/member correspondence checks even when
    nested blobs exist. Errors propagate; no files outside the existing cache
    are extracted and no learning checkpoint is created.
    """
    spec = ArchiveSelectionSpec.model_validate(spec)
    parent = materialize_source(spec.source, cache, source)
    path = cache.member_path(parent, spec.source.members[0].name)
    packages = {item.member.name: item.package for item in spec.selections}
    receipts: dict[str, MaterializationReceipt] = {}

    def consume(member: PackageMember, stream: ByteReader) -> None:
        """Force actual selected bytes through the existing nested package verifier on every replay."""

        @contextmanager
        def reader():
            """Lend the already bounded member stream; its archive context owns closing."""
            yield stream

        # Force consumption even on a cache hit: cached children cannot prove membership.
        receipts[member.name] = materialize_source(packages[member.name], cache, reader, refresh_source=True)

    consume_selected_zip_members(
        path,
        tuple(item.member for item in spec.selections),
        consume,
        maximum_entries=spec.maximum_entries,
        maximum_metadata_bytes=spec.maximum_metadata_bytes,
    )
    return tuple(receipts[item.member.name] for item in spec.selections)


def validate_selected_packages(spec: ArchiveSelectionSpec, cache: SourceCache) -> None:
    """Verify cached container/child byte correspondence without network or repairing missing blobs."""
    spec = ArchiveSelectionSpec.model_validate(spec)
    path = cache.verify_blob(cast(str, spec.source.source.sha256), cast(int, spec.source.source.size_bytes))

    def verify(member: PackageMember, stream: ByteReader) -> None:
        """Check both physical membership and the independent cached child, without repairing either."""
        deque(_verified_chunks(stream, member.sha256, member.size_bytes), maxlen=0)
        cache.verify_blob(member.sha256, member.size_bytes)

    consume_selected_zip_members(
        path,
        tuple(item.member for item in spec.selections),
        verify,
        maximum_entries=spec.maximum_entries,
        maximum_metadata_bytes=spec.maximum_metadata_bytes,
    )
