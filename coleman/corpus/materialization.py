"""Content-addressed local source cache and bounded, restartable package integrity operations."""

from __future__ import annotations

import hashlib
import re
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, cast

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archives import consume_zip_members
from coleman.corpus.materialization_contracts import (
    ByteReader,
    MaterializationReceipt,
    MaterializationSpec,
    PackageMember,
)
from coleman.results.versioned import VersionedContract

ReaderFactory = Callable[[], AbstractContextManager[ByteReader]]
_CHUNK_BYTES = 65536


class _CacheFormat(VersionedContract):
    """Strict local cache namespace marker; raw blobs alone confer no source/evaluation admission."""

    kind: Literal["coleman-source-cache-v1"] = "coleman-source-cache-v1"


def validate_source_storage(root: str | Path) -> None:
    """Reject source-content storage physically inside a Git worktree, without creating files.

    Resolve existing symlink parents and inspect only ancestor Git directory/file
    markers; no Git process, marker content or sibling tree is read. Cache bytes
    and raw producer diagnostics need this guard; metadata-only exports retain
    their existing rules. Invalid roots raise ValueError and filesystem failures
    propagate. Trusted local parents are required: future Git initialization or
    concurrent hostile filesystem mutation is outside this preflight's guarantee.
    """
    root = Path(root).resolve()
    if any(parent.name == ".git" or (parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("Source content storage must be outside a repository")


def _verified_chunks(reader: ByteReader, digest: str, size: int) -> Iterator[bytes]:
    """Yield bounded source bytes and verify size/hash before iterator exhaustion permits publication."""
    hasher = hashlib.sha256()
    total = 0
    while True:
        requested = min(_CHUNK_BYTES, size - total + 1)
        chunk = reader.read(requested)
        if not isinstance(chunk, bytes):
            raise TypeError("Source reader must return bytes")
        if len(chunk) > requested:
            raise ValueError("Source reader exceeded requested chunk bound")
        if not chunk:
            break
        total += len(chunk)
        if total > size:
            raise ValueError("Source/member exceeds frozen byte count")
        hasher.update(chunk)
        yield chunk
    if total != size or "sha256:" + hasher.hexdigest() != digest:
        raise ValueError("Source/member checksum or byte count disagrees with frozen recipe")


class SourceCache:
    """Trusted-local v1 binary cache outside Git repositories, using existing immutable artifacts.

    Construction resolves the explicit root and creates/verifies its small format
    marker. Caller owns the root; existing unreferenced files are neither adopted
    as receipts nor deleted. Unknown marker versions reject. Every accessed blob
    is size/hash verified; filenames alone never certify completion. Concurrent
    identical writers converge through hard-link publication; conflicts reject.
    Private/trusted parent directories and the artifact writer's local filesystem
    durability limits apply. This is not a hostile-filesystem sandbox/global quota.
    """

    def __init__(self, root: str | Path) -> None:
        """Validate the physical root before creating a deterministic version marker; I/O failures propagate."""
        self._root = Path(root).resolve()
        validate_source_storage(self.root)
        self._writer = ArtifactWriter(self.root)
        marker = self._contained("cache-format.json")
        if marker.exists():
            with marker.open("rb") as stream:
                payload = stream.read(4097)
            if len(payload) > 4096:
                raise ValueError("Source cache marker exceeds metadata limit")
            _CacheFormat.from_json(payload)
        self._writer.write_immutable(marker, _CacheFormat(schema_version=1).to_json() + "\n")

    @property
    def root(self) -> Path:
        """Return the immutable physical cache root selected and validated at construction."""
        return self._root

    def _contained(self, relative: str) -> Path:
        """Reject existing symlink escapes/files; cooperative writers may only create normal content paths."""
        path = self.root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Source cache path escapes its trusted root")
        return path

    def _blob(self, digest: str) -> Path:
        """Resolve only validated SHA-256 filenames; no source/member path is executed."""
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError("Source cache requires a SHA-256 blob identity")
        return self._contained("blobs/" + digest.removeprefix("sha256:"))

    def verify_blob(self, digest: str, size: int) -> Path:
        """Verify complete cached bytes in bounded chunks; missing/corrupt blobs raise, without repair or network."""
        if type(size) is not int or not 0 <= size <= 8 * 1024**3:
            raise ValueError("Source cache requires a bounded nonnegative integer byte count")
        path = self._blob(digest)
        with path.open("rb") as stream:
            deque(_verified_chunks(stream, digest, size), maxlen=0)
        return path

    def store_blob(self, digest: str, size: int, reader: ByteReader) -> Path:
        """Consume a complete verified stream before atomic publication; caller closes its input.

        This low-level content-addressed operation grants no license/admission.
        Public acquisition must go through materialize_source's frozen terms gate.
        A replay consumes/verifies the supplied stream even if the blob exists.
        Size/hash/iterator/I/O failures never publish a partial blob.
        """
        if type(size) is not int or not 0 <= size <= 8 * 1024**3:
            raise ValueError("Source cache requires a bounded nonnegative integer byte count")
        path = self._blob(digest)
        self._writer.write_immutable_stream(path, _verified_chunks(reader, digest, size), maximum_bytes=size)
        return path

    def member_path(self, receipt: MaterializationReceipt, name: str) -> Path:
        """Return one verified cached member path; names are logical keys, never filesystem paths.

        This verifies the selected blob, not the full archive-to-member relation.
        Use validate_materialization for the complete package proof. Unknown names
        raise KeyError; changed/missing bytes raise ValueError/OSError.
        """
        receipt = MaterializationReceipt.model_validate(receipt)
        for member in receipt.spec.members:
            if member.name == name:
                return self.verify_blob(member.sha256, member.size_bytes)
        raise KeyError(name)


def _source_blob(
    spec: MaterializationSpec,
    cache: SourceCache,
    source: ReaderFactory | None,
    refresh_source: bool,
) -> Path:
    """Reuse only verified pinned bytes; a refresh explicitly verifies the supplied reader even on cache hits."""
    digest = cast(str, spec.source.sha256)
    size = cast(int, spec.source.size_bytes)
    if not refresh_source:
        try:
            return cache.verify_blob(digest, size)
        except FileNotFoundError:
            pass
    if source is None:
        raise ValueError("Pinned source is not cached; supply an explicit source reader")
    with source() as reader:
        return cache.store_blob(digest, size, reader)


def materialize_source(
    spec: MaterializationSpec,
    cache: SourceCache,
    source: ReaderFactory | None = None,
    *,
    refresh_source: bool = False,
) -> MaterializationReceipt:
    """Acquire verified raw/ZIP bytes and members, without scientific admission or implicit network access.

    Frozen terms/pins/budgets validate before reader/cache access. The optional
    reader factory is lazy on a verified cache hit; refresh_source=True forces its
    verification, useful for explicit local files. ZIP source digest is verified
    before metadata/member decoding. All actual member streams are checked even
    on cache hits, so an unrelated cached blob cannot certify archive membership.
    Interrupted operations retain only individually completed blobs; no receipt
    is returned until all declared members are verified. Retry reuses the frozen
    source and repeats bounded member verification. No second learning checkpoint
    or experiment runner is introduced.
    """
    spec = MaterializationSpec.model_validate(spec)
    if type(refresh_source) is not bool:
        raise ValueError("refresh_source must be a boolean")
    path = _source_blob(spec, cache, source, refresh_source)

    def store_member(member: PackageMember, stream: ByteReader) -> None:
        """Verify actual archive contents before immutable member publication, including cached replays."""
        cache.store_blob(member.sha256, member.size_bytes, stream)

    if spec.archive == "zip":
        consume_zip_members(path, spec, store_member)
    return MaterializationReceipt(schema_version=1, spec=spec)


def validate_materialization(receipt: MaterializationReceipt, cache: SourceCache) -> None:
    """Verify complete cached source, archive membership and member blobs offline, without publishing or repairing.

    A receipt is producer metadata, not proof that files still exist. Source and
    decoded members are rehashed, including their relation to the pinned archive.
    Missing/changed bytes and unsupported metadata fail closed. No network reader
    is constructed and no source availability/license declaration is rewritten.
    """
    receipt = MaterializationReceipt.model_validate(receipt)
    spec = receipt.spec
    path = cache.verify_blob(cast(str, spec.source.sha256), cast(int, spec.source.size_bytes))

    def verify_member(member: PackageMember, stream: ByteReader) -> None:
        """Verify actual archive contents and their independently cached copy before accepting membership."""
        deque(_verified_chunks(stream, member.sha256, member.size_bytes), maxlen=0)
        cache.verify_blob(member.sha256, member.size_bytes)

    if spec.archive == "zip":
        consume_zip_members(path, spec, verify_member)


def validate_cache_separation(cache_root: str | Path, output_root: str | Path) -> None:
    """Reject physically overlapping cache/export roots before any acquisition or output writes.

    Resolve both roots, including existing symlink parents, without creating files.
    Equal roots, either ancestor relation or an escaping materialization-family
    symlink raises ValueError. Callers planning
    acquisition plus publication must run this preflight before constructing the
    cache; publication repeats the check. Trusted private parent directories are
    required: this does not prevent concurrent hostile filesystem mutation.
    """
    cache = Path(cache_root).resolve()
    output = Path(output_root).resolve()
    if not (output / "corpus-materializations").resolve().is_relative_to(output):
        raise ValueError("Materialization artifacts must remain within the published artifact tree")
    if cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError("Source cache and published artifact tree must remain separate")


def publish_materialization(receipt: MaterializationReceipt, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Verify actual offline bytes before publishing metadata-only immutable evidence; replay is idempotent.

    Source/member data remains in the external cache, including recipe-only data.
    A failure cannot publish a falsely completed receipt. The same trusted local
    writer/root and process-interruption/power-loss limits apply.
    """
    receipt = MaterializationReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, writer.run_root)
    validate_materialization(receipt, cache)
    path = writer.run_root / "corpus-materializations" / f"{receipt.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, receipt.to_json() + "\n")
    return path
