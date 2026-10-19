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
    """Reject source storage inside a Git worktree without creating files.

    Parameters
    ----------
    root : str or pathlib.Path
        Proposed source/cache/diagnostic location, resolved through existing
        symlink parents before checking ancestor Git markers.

    Raises
    ------
    ValueError
        If the path is Git-backed or includes a reserved Git directory component.
    OSError
        If path resolution or filesystem inspection fails.

    Notes
    -----
    Reads only ancestor marker existence, not marker contents or sibling trees.
    No Git process is launched. Metadata-only exports retain their own rules.
    Trusted parents are required; future Git initialization and hostile concurrent
    filesystem mutation are outside this preflight guarantee.
    """
    root = Path(root).resolve()
    if any(parent.name.casefold() == ".git" or (parent / ".git").exists() for parent in (root, *root.parents)):
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
    """Store content-addressed binary source blobs outside Git repositories.

    Parameters
    ----------
    root : str or pathlib.Path
        Caller-owned trusted local cache root. Construction resolves it, validates
        storage placement and creates/verifies a small v1 format marker.

    Notes
    -----
    Every accessed blob is rechecked by size and SHA-256. Unreferenced files are
    neither adopted as receipts nor deleted; unknown format markers reject.
    Identical concurrent writers converge through immutable publication, while
    conflicting bytes reject. Requires the artifact writer's local filesystem
    operations; process-interruption limits do not guarantee power-loss recovery.
    This is not a hostile-filesystem sandbox or global disk quota.
    """

    def __init__(self, root: str | Path) -> None:
        """Validate a root and create or verify the deterministic cache marker.

        Parameters
        ----------
        root : str or pathlib.Path
            Trusted local root outside Git; existing symlink parents are resolved.

        Raises
        ------
        ValueError
            If placement or the existing marker is invalid, oversized or unsupported.
        OSError
            If marker publication or local filesystem access fails.
        """
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
        """Reject symlink escapes and Git-backed content paths before reading/writing cooperative cache bytes."""
        path = self.root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Source cache path escapes its trusted root")
        validate_source_storage(path)
        return path

    def _blob(self, digest: str) -> Path:
        """Resolve only validated SHA-256 filenames; no source/member path is executed."""
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError("Source cache requires a SHA-256 blob identity")
        return self._contained("blobs/" + digest.removeprefix("sha256:"))

    def verify_blob(self, digest: str, size: int) -> Path:
        """Verify complete cached bytes offline without repair or publication.

        Parameters
        ----------
        digest : str
            Identified lowercase SHA-256: ``sha256:`` plus 64 hexadecimal characters.
        size : int
            Exact expected byte count, from zero through 8 GiB; booleans reject.

        Returns
        -------
        pathlib.Path
            Verified blob path beneath the cache.

        Raises
        ------
        ValueError
            If identity, bound, actual size or checksum is invalid.
        OSError
            If the blob is missing or cannot be read.

        Notes
        -----
        Reads in bounded chunks with no network access. A filename alone does not
        certify contents, archive membership, permission or source admission.
        """
        if type(size) is not int or not 0 <= size <= 8 * 1024**3:
            raise ValueError("Source cache requires a bounded nonnegative integer byte count")
        path = self._blob(digest)
        with path.open("rb") as stream:
            deque(_verified_chunks(stream, digest, size), maxlen=0)
        return path

    def store_blob(self, digest: str, size: int, reader: ByteReader) -> Path:
        """Verify a complete stream before immutable content-addressed publication.

        Parameters
        ----------
        digest : str
            Expected identified SHA-256 of the complete stream.
        size : int
            Exact byte count, from zero through 8 GiB; booleans reject.
        reader : ByteReader
            Caller-owned binary reader honoring requested chunk sizes. The caller
            closes it; even an identical replay consumes and verifies the stream.

        Returns
        -------
        pathlib.Path
            Verified immutable blob path.

        Raises
        ------
        ValueError
            If identity, bounds, size, checksum or existing contents conflict.
        TypeError
            If the reader returns a non-byte value.
        OSError
            If reading or local immutable publication fails.

        Notes
        -----
        Interrupted writes do not publish partial blobs. This low-level operation
        grants no permission/admission; acquisition uses ``materialize_source`` and
        its frozen component-terms gate. Local filesystem durability limits apply.
        """
        if type(size) is not int or not 0 <= size <= 8 * 1024**3:
            raise ValueError("Source cache requires a bounded nonnegative integer byte count")
        path = self._blob(digest)
        self._writer.write_immutable_stream(path, _verified_chunks(reader, digest, size), maximum_bytes=size)
        return path

    def member_path(self, receipt: MaterializationReceipt, name: str) -> Path:
        """Resolve and verify a declared member blob from a receipt.

        Parameters
        ----------
        receipt : MaterializationReceipt
            Validated v1 source/member declaration; metadata alone is not proof.
        name : str
            Exact logical member key, never interpreted as a filesystem path.

        Returns
        -------
        pathlib.Path
            Cache path whose complete selected bytes match the declared size/hash.

        Raises
        ------
        KeyError
            If the declaration does not contain the name.
        ValueError
            If metadata, size or checksum validation fails.
        OSError
            If cached bytes are missing or unreadable.

        Notes
        -----
        Validates only this blob, without writes or network access. Use
        ``validate_materialization`` to verify the whole archive/member relationship.
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
    """Acquire and verify a frozen raw/ZIP package and its declared members.

    Parameters
    ----------
    spec : MaterializationSpec
        Versioned source/member identities, component terms and resource bounds.
    cache : SourceCache
        Validated external binary cache for immutable source/member bytes.
    source : ReaderFactory or None, optional
        Explicit context-managed binary reader factory. Lazy on a verified
        cache hit; required when the source is absent. No implicit HTTP occurs.
    refresh_source : bool, optional
        Force verification of the supplied source even on a cache hit.

    Returns
    -------
    MaterializationReceipt
        Metadata returned only after the whole source and every declared
        selected member have been verified against actual bytes.

    Raises
    ------
    ValueError
        If terms/pins/bounds, source presence, archive structure, checksums or
        the boolean refresh flag fail validation.
    OSError
        If caller-supplied reading or cache publication fails.

    Notes
    -----
    Validates the frozen declaration before reader/cache access. ZIP source
    identity is checked before metadata/member decoding. Actual member streams
    are verified even on cached replay; an unrelated blob cannot prove archive
    membership. Interrupted work leaves only individually completed blobs.
    Retry reuses the frozen source and repeats bounded member validation.
    This does not grant scientific admission or create a second runner/checkpoint.
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
    """Reverify complete cached source and member correspondence offline.

    Parameters
    ----------
    receipt : MaterializationReceipt
        Versioned package declaration to check against current bytes.
    cache : SourceCache
        External source/member cache used by the declaration.

    Raises
    ------
    ValueError
        If metadata, source identity, archive membership or member bytes disagree.
    OSError
        If required cached files are missing or unreadable.

    Notes
    -----
    Rehashes the complete source and decoded selected members and checks their
    independently cached copies. Publishes/repairs nothing, constructs no network
    reader and changes no source-availability or component-terms declaration.
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
    """Reject overlapping physical cache and publication roots before writes.

    Parameters
    ----------
    cache_root : str or pathlib.Path
        Proposed external source cache, resolved through existing symlink parents.
    output_root : str or pathlib.Path
        Proposed metadata publication root, with its own contained artifact family.

    Raises
    ------
    ValueError
        If roots coincide, either contains the other, or the materialization
        artifact family escapes the publication root.
    OSError
        If filesystem resolution fails.

    Notes
    -----
    Creates no files. Run this preflight before cache construction when planning
    acquisition/publication; publication repeats it. Trusted parents are required;
    hostile concurrent filesystem mutation is outside the guarantee.
    """
    cache = Path(cache_root).resolve()
    output = Path(output_root).resolve()
    if not (output / "corpus-materializations").resolve().is_relative_to(output):
        raise ValueError("Materialization artifacts must remain within the published artifact tree")
    if cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError("Source cache and published artifact tree must remain separate")


def publish_materialization(receipt: MaterializationReceipt, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Verify cached bytes and publish an immutable metadata-only receipt.

    Parameters
    ----------
    receipt : MaterializationReceipt
        Frozen source/member declaration, revalidated against actual cached bytes.
    cache : SourceCache
        External binary cache; raw source/member bytes remain here.
    writer : ArtifactWriter
        Trusted local metadata writer with a physically separate publication root.

    Returns
    -------
    pathlib.Path
        Deterministic receipt path under ``corpus-materializations``.

    Raises
    ------
    ValueError
        If roots overlap, metadata/bytes fail validation or published contents conflict.
    OSError
        If offline reading or local immutable publication fails.

    Notes
    -----
    Identical replay is idempotent. No falsely complete receipt is published
    before verification, and raw source data is not exported. The writer's
    trusted-parent and process-interruption/power-loss limits apply.
    """
    receipt = MaterializationReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, writer.run_root)
    validate_materialization(receipt, cache)
    path = writer.run_root / "corpus-materializations" / f"{receipt.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, receipt.to_json() + "\n")
    return path
