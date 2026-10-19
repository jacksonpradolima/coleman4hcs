"""Original released HCS log archives, reusing source/cache/ZIP and CTest evidence engines.

The explicit classic-ZIP profile permits 16,384 entries and ASCII @ labels.
Older package/selection limits and producer identities remain unchanged. No raw
log is exported, no network is opened and no historical admission is granted.
"""

from __future__ import annotations

import hashlib
import zipfile
import zlib
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archives import _validate_deflate, _zip_footer
from coleman.corpus.contracts import SourceArtifact
from coleman.corpus.hcs_contracts import HCSRecipe
from coleman.corpus.hcs_raw import _jobs
from coleman.corpus.hcs_trace_artifacts import (
    HCSLogSpec,
    HCSTraceCompletion,
    hcs_trace_producer,
    publish_hcs_trace_evidence,
    validate_hcs_trace_completion,
)
from coleman.corpus.materialization import (
    SourceCache,
    validate_cache_separation,
    validate_materialization,
    validate_source_storage,
)
from coleman.corpus.materialization_contracts import MaterializationReceipt, MaterializationSpec
from coleman.corpus.zip64_contracts import Zip64Member
from coleman.corpus.zip64_records import Zip64Record, _central_record, zip64_data_offset
from coleman.corpus.zip64_selection import _bind_metadata
from coleman.corpus.zip_metadata import entry_boundaries
from coleman.results.domain import Fingerprint, fingerprint
from coleman.results.versioned import VersionedContract


class HCSArchiveLogSpec(VersionedContract):
    """Pin one original GitLabCI-Torrent log inside its separately licensed full archive.

    The container is an existing raw materialization, not a relaxed legacy ZIP
    recipe. Complete container/member digests are required. Selected logs are
    bounded to 2 MiB; unselected payloads are not decoded or certified. Container
    terms apply to the actual released member, independently of live-provider
    terms. Its basename must also bind pipeline/commit/job during verification.
    """

    profile: Literal["classic-zip-16384-ctest-logs-v1"] = "classic-zip-16384-ctest-logs-v1"
    job_id: Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
    container: MaterializationSpec
    member: Zip64Member

    @model_validator(mode="after")
    def _container(self) -> Self:
        """Keep whole-archive and decoded-member identities distinct before any cache/file access."""
        if self.container.archive != "raw":
            raise ValueError("HCS log archive requires one opaque raw source package")
        if "#" in self.container.source.uri:
            raise ValueError("HCS log archive locator must identify the whole container")
        if self.member.size_bytes > 2 * 1024**2:
            raise ValueError("HCS selected original log exceeds 2 MiB")
        return self


class HCSArchiveTraceReceipt(VersionedContract):
    """Complete-last original-archive link to an unchanged CTest corroboration receipt.

    The embedded child proves local CSV/log correspondence; this wrapper also
    requires actual full-container/member verification. A child alone does not
    certify its enclosing archive. Old direct-log receipts retain their exact
    schema/serialization/producer identity. Unknown versions/producers reject;
    no automatic migration or historical-availability assertion is implied.
    """

    kind: Literal["hcs-archived-ctest-corroboration-v1"] = "hcs-archived-ctest-corroboration-v1"
    source: HCSArchiveLogSpec
    recipe_fingerprint: Fingerprint
    producer_sha256: Fingerprint
    corroboration_path: Annotated[str, Field(strict=True, pattern=r"^hcs-trace/[0-9a-f]{64}/complete\.json$")]
    corroboration: HCSTraceCompletion
    admission: Literal["not_granted"] = "not_granted"


def hcs_archive_trace_producer() -> str:
    """Bind the existing corroboration producer and exact archive-profile/ZIP implementation bytes."""
    names = (
        "hcs_archive_trace.py",
        "archives.py",
        "zip_metadata.py",
        "zip64_records.py",
        "zip64_contracts.py",
        "zip64_selection.py",
    )
    return fingerprint(
        [
            hcs_trace_producer(),
            [[name, hashlib.sha256(files("coleman.corpus").joinpath(name).read_bytes()).hexdigest()] for name in names],
        ]
    )


def _records(path: Path, offset: int, infos: tuple[zipfile.ZipInfo, ...]) -> tuple[Zip64Record, ...]:
    """Reuse bounded central-record validation, including @ names and link/flag/method rejection."""
    records = []
    consumed = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        for _ in infos:
            record, amount = _central_record(stream, 2 * 1024**2 - consumed)
            consumed += amount
            records.append(record)
    result = tuple(records)
    _bind_metadata(infos, result)
    if any(record.version > 20 or record.offset_only_zip64 for record in result):
        raise ValueError("HCS original log profile supports classic ZIP entries only")
    names = [record.name.removesuffix("/") for record in result]
    if len(names) != len(set(names)):
        raise ValueError("HCS log archive contains duplicate or aliased member names")
    return result


def _archive_offsets(path: Path, offset: int, infos: tuple[zipfile.ZipInfo, ...]) -> dict[str, int]:
    """Check every local header against bounded central metadata before any selected payload is decoded."""
    records = _records(path, offset, infos)
    boundaries = entry_boundaries(infos, offset)
    if not boundaries or min(boundaries) != 0:
        raise ValueError("Prefixed/self-extracting HCS archives are unsupported")
    return {record.name: zip64_data_offset(path, record, boundaries[record.offset]) for record in records}


def _selected_payload(
    archive: zipfile.ZipFile,
    path: Path,
    spec: HCSArchiveLogSpec,
    offsets: dict[str, int],
) -> bytes:
    """Read only the exact selected regular member, under its pinned decoded-byte and digest bounds."""
    selected = next((info for info in archive.infolist() if info.filename == spec.member.name), None)
    if selected is None or selected.is_dir() or selected.file_size != spec.member.size_bytes:
        raise ValueError("HCS selected log name/size differs from its archive record")
    if selected.compress_type == zipfile.ZIP_DEFLATED:
        _validate_deflate(path, selected, offsets[selected.filename])
    with archive.open(selected) as stream:
        payload = stream.read(spec.member.size_bytes + 1)
    if len(payload) != spec.member.size_bytes or "sha256:" + hashlib.sha256(payload).hexdigest() != spec.member.sha256:
        raise ValueError("HCS selected original log differs from its byte pins")
    return payload


def read_hcs_archive_log(spec: HCSArchiveLogSpec, cache: SourceCache) -> bytes:
    """Verify an entire cached container, every local header and one <=2 MiB selected log.

    Reuses the existing raw cache, classic ZIP footer and ZIP metadata/deflate
    validators. Bounds of 16,384 entries / 2 MiB central metadata apply before
    standard ZipFile allocation. Every unselected header is checked, but its
    payload is not decoded. No extraction paths, network, subjects or artifacts
    are created. Changed source/member bytes, CRC, unsupported ZIP64/multidisk,
    links, unsafe/aliased names and invalid structure raise ValueError; I/O errors
    propagate. Trusted immutable local parents are required, as for the cache.
    """
    spec = HCSArchiveLogSpec.model_validate(spec)
    parent = MaterializationReceipt(schema_version=1, spec=spec.container)
    validate_materialization(parent, cache)
    path = cache.member_path(parent, spec.container.members[0].name)
    offset = _zip_footer(path, 16384, 2 * 1024**2)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = tuple(archive.infolist())
            offsets = _archive_offsets(path, offset, infos)
            return _selected_payload(archive, path, spec, offsets)
    except (zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ValueError("Invalid selected original HCS log archive") from exc


def load_hcs_archive_log_spec() -> HCSArchiveLogSpec:
    """Read the bundled reviewed original LibSSH archive/member pins, without source/network access.

    Metadata is bounded to 2 MiB and requires the explicit v1 profile. This
    reads package metadata only; fetching/materialization remains an explicit
    caller action. Unknown schemas, oversized or malformed metadata reject.
    """
    resource = files("coleman.corpus.resources").joinpath("libssh-ctest-original-archive-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(2 * 1024**2 + 1)
    if len(payload) > 2 * 1024**2:
        raise ValueError("HCS original archive preset exceeds 2 MiB")
    return HCSArchiveLogSpec.from_json(payload)


def _log_spec(recipe: HCSRecipe, spec: HCSArchiveLogSpec, cache: SourceCache) -> HCSLogSpec:
    """Bind the original member basename to verified pipeline/commit/job metadata and archive terms."""
    job = _jobs(recipe, cache).get(spec.job_id)
    if job is None or spec.member.name.rsplit("/", 1)[-1] != f"{job.pipeline_id}_{job.commit_id}_{job.job_id}.log":
        raise ValueError("HCS archived log filename disagrees with original pipeline/commit/job metadata")
    parent = spec.container.source
    return HCSLogSpec(
        schema_version=1,
        job_id=spec.job_id,
        source=SourceArtifact(
            uri=parent.uri + "#" + spec.member.name,
            version=parent.version,
            citation=parent.citation + "; selected original log " + spec.member.name,
            sha256=spec.member.sha256,
            size_bytes=spec.member.size_bytes,
            availability="verified_local",
        ),
        terms=spec.container.license,
    )


def _destination(writer: ArtifactWriter, cache: SourceCache, recipe: HCSRecipe, spec: HCSArchiveLogSpec) -> Path:
    """Preflight an explicit archive-wrapper namespace, separate from existing direct-log artifacts."""
    validate_cache_separation(cache.root, writer.run_root)
    identity = fingerprint(
        ["hcs-archived-ctest-corroboration-v1", recipe.fingerprint, spec.fingerprint, hcs_archive_trace_producer()]
    )
    path = writer.run_root / "hcs-archive-trace" / identity.removeprefix("sha256:") / "complete.json"
    if path.is_symlink() or not path.resolve().is_relative_to(writer.run_root.resolve()):
        raise ValueError("HCS archive corroboration output escapes its trusted tree")
    validate_source_storage(path)
    return path


def publish_hcs_archive_trace(
    recipe: HCSRecipe,
    spec: HCSArchiveLogSpec,
    cache: SourceCache,
    writer: ArtifactWriter,
) -> Path:
    """Corroborate a licensed released log, then atomically publish its enclosing-source receipt.

    Materialize the full raw container explicitly with the existing cache first.
    Every retry rechecks actual container/member and original CSV/metadata bytes.
    Reuses publish_hcs_trace_evidence unchanged; its artifacts precede this final
    wrapper. Interrupted wrappers remain uncompleted, and identical concurrent
    retries converge without overwriting any child. No raw log is exported and
    no actual availability, predecision universe or scientific admission is
    inferred from copyright/source integrity. Errors propagate without completion.
    """
    recipe, spec = HCSRecipe.model_validate(recipe), HCSArchiveLogSpec.model_validate(spec)
    path = _destination(writer, cache, recipe, spec)
    log_spec = _log_spec(recipe, spec, cache)
    payload = read_hcs_archive_log(spec, cache)
    child = publish_hcs_trace_evidence(recipe, log_spec, payload, cache, writer)
    corroboration = validate_hcs_trace_completion(child, recipe, payload, cache)
    receipt = HCSArchiveTraceReceipt(
        schema_version=1,
        source=spec,
        recipe_fingerprint=recipe.fingerprint,
        producer_sha256=hcs_archive_trace_producer(),
        corroboration_path=child.relative_to(writer.run_root).as_posix(),
        corroboration=corroboration,
    )
    writer.write_immutable(path, receipt.to_json() + "\n")
    return path


def validate_hcs_archive_trace(path: Path, recipe: HCSRecipe, cache: SourceCache) -> HCSArchiveTraceReceipt:
    """Reverify the full archive/member and embedded original-source correspondence offline.

    Reads at most 2 MiB of receipt metadata. Forged/copied namespaces, future
    versions, unknown producers, modified/missing input/output or symlinks reject.
    The child receipt alone is never accepted as proof of its enclosing archive.
    No network, repair, implicit migration, source-data redistribution or writes
    occur. Retain the recorded package for an older producer's verification.
    """
    recipe = HCSRecipe.model_validate(recipe)
    if path.is_symlink():
        raise ValueError("HCS archive receipt cannot be a symlink")
    validate_source_storage(path)
    with path.open("rb") as stream:
        wire = stream.read(2 * 1024**2 + 1)
    if len(wire) > 2 * 1024**2:
        raise ValueError("HCS archive receipt exceeds 2 MiB")
    receipt = HCSArchiveTraceReceipt.from_json(wire)
    if receipt.producer_sha256 != hcs_archive_trace_producer() or receipt.recipe_fingerprint != recipe.fingerprint:
        raise ValueError("HCS archive receipt requires its recorded producer/recipe")
    root = path.parent.parent.parent
    expected = _destination(ArtifactWriter(root), cache, recipe, receipt.source)
    if path.resolve() != expected.resolve():
        raise ValueError("HCS archive receipt differs from its recorded namespace")
    spec = _log_spec(recipe, receipt.source, cache)
    payload = read_hcs_archive_log(receipt.source, cache)
    child = validate_hcs_trace_completion(root / receipt.corroboration_path, recipe, payload, cache)
    if child != receipt.corroboration or child.log_spec != spec:
        raise ValueError("HCS archive child differs from its original container/member evidence")
    return receipt
