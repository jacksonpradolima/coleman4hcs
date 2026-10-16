"""Deterministic source views assembled from individually pinned, licensed raw packages."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.archives import deterministic_zip_bytes
from coleman.corpus.materialization import ReaderFactory, SourceCache, materialize_source, validate_materialization
from coleman.corpus.materialization_contracts import MaterializationReceipt, MaterializationSpec
from coleman.results.versioned import VersionedContract


class SourceViewSpec(VersionedContract):
    """Bind homogeneous reviewed raw inputs to an expected deterministic ZIP source view.

    Each input contains exactly one regular member. Logical names, bytes, source
    versions, terms and the resulting archive are frozen before acquisition.
    At most 128 members, 8 MiB per file and 16 MiB total decoded input are allowed;
    output ZIP bytes cannot exceed 32 MiB. These limits do not expand v1 archive
    support. A view preserves file bytes and never compiles or executes them.
    Terms must match exactly across inputs/output; heterogeneous component terms
    require separately reviewed packages rather than an inferred umbrella license.
    """

    recipe: Literal["deterministic-raw-source-view-v1"] = "deterministic-raw-source-view-v1"
    inputs: Annotated[tuple[MaterializationSpec, ...], Field(min_length=1, max_length=128)]
    output: MaterializationSpec

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Reject unsupported recipes, mixed terms, ambiguous members and weakened byte bounds."""
        if self.output.archive != "zip" or any(item.archive != "raw" for item in self.inputs):
            raise ValueError("Source views require raw inputs and one ZIP output")
        names = [item.members[0].name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source view member")
        if len({item.package_id for item in self.inputs}) != len(self.inputs):
            raise ValueError("Duplicate source view input package identity")
        members = tuple(sorted((item.members[0] for item in self.inputs), key=lambda item: item.name))
        if members != self.output.members:
            raise ValueError("Source view output must preserve every input member without additions/exclusions")
        if any(item.license != self.output.license for item in self.inputs):
            raise ValueError("Source view input/output terms must match exactly")
        if any(item.source.version != self.output.source.version for item in self.inputs):
            raise ValueError("Source view input/output versions disagree")
        object.__setattr__(self, "inputs", tuple(sorted(self.inputs, key=lambda item: item.members[0].name)))
        return self

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Enforce decoded and archive byte ceilings independently of caller-supplied package limits."""
        members = self.output.members
        if (
            any(item.size_bytes > 8 * 1024**2 for item in members)
            or sum(item.size_bytes for item in members) > 16 * 1024**2
        ):
            raise ValueError("Source view decoded inputs exceed their byte budget")
        if self.output.source.size_bytes is None or self.output.source.size_bytes > 32 * 1024**2:
            raise ValueError("Source view ZIP exceeds 32 MiB")
        return self


def source_view_bytes(entries: tuple[tuple[str, bytes], ...]) -> bytes:
    """Encode a bounded sorted regular-file ZIP recipe, without deriving or changing source content.

    All names must be conservative portable materialization member names. Fixed
    DOS epoch, Unix regular-file permissions, stored bytes and empty comments
    remove host/time/compressor variability. Duplicate names, non-bytes, limits
    and unsafe names raise ValueError/TypeError. No files or network are accessed.
    This format is COLEMAN's ZIP recipe, not a claim about upstream archive bytes.
    """
    return deterministic_zip_bytes(entries, maximum_member_bytes=8 * 1024**2, maximum_total_bytes=16 * 1024**2)


def materialize_source_view(
    spec: SourceViewSpec,
    cache: SourceCache,
    reader_for: Callable[[MaterializationSpec], ReaderFactory] | None = None,
) -> MaterializationReceipt:
    """Acquire/verify raw inputs and a deterministic derived ZIP through the existing cache.

    With no factory this operates offline. A supplied factory is explicit, lazy on
    verified raw cache hits and responsible for transport; no URI is opened here.
    Only individually completed blobs survive interruption. The complete returned
    receipt pins the derived view, not an upstream download archive or scientific
    admission. Invalid/mutated bytes, mismatched generation pins and unsupported
    terms fail closed; I/O errors propagate. No subject process is started.
    """
    spec = SourceViewSpec.model_validate(spec)
    entries = []
    for package in spec.inputs:
        source = None if reader_for is None else reader_for(package)
        receipt = materialize_source(package, cache, source)
        validate_materialization(receipt, cache)
        member = package.members[0]
        path = cache.member_path(receipt, member.name)
        with path.open("rb") as stream:
            payload = stream.read(member.size_bytes + 1)
        if len(payload) != member.size_bytes:
            raise ValueError("Source view input changed after verification")
        entries.append((member.name, payload))
    payload = source_view_bytes(tuple(entries))
    return materialize_source(spec.output, cache, lambda: io.BytesIO(payload), refresh_source=True)
