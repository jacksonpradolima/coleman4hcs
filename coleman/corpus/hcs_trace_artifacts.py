"""Offline immutable CTest corroboration artifacts; raw logs are never redistributed.

Completion denotes this exact local corroboration operation, not completion of
the CI job, source admission or verified label availability. Actual input bytes
are rechecked on every retry; old HCS outputs are neither changed nor migrated.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.contracts import LicenseReview, SourceArtifact
from coleman.corpus.hcs_contracts import HCSRecipe
from coleman.corpus.hcs_raw import read_hcs_slice
from coleman.corpus.hcs_trace import parse_hcs_ctest_trace
from coleman.corpus.hcs_trace_correspondence import corroborate_hcs_ctest_trace
from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.results.domain import Fingerprint, fingerprint
from coleman.results.versioned import VersionedContract

_TRACE_FILE = "trace.json"
_CORRESPONDENCE_FILE = "correspondence.json"
_COMPLETE_FILE = "complete.json"
_SHA256_PREFIX = "sha256:"


class HCSLogSpec(VersionedContract):
    """A pinned log locator, job identity and separately recorded component terms.

    URI is descriptive, never fetched here. Pending terms permit local
    corroboration only; no raw log is exported or source admitted. Explicitly
    prohibited use rejects. Verified archive terms must not be silently assigned
    to an upstream log. SHA-256/size must identify the complete supplied bytes.
    """

    job_id: Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
    source: SourceArtifact
    terms: LicenseReview

    @model_validator(mode="after")
    def _pins(self) -> Self:
        """Require concrete local pins and reject expressly prohibited source terms."""
        if self.source.sha256 is None or self.source.size_bytes is None or self.source.availability != "verified_local":
            raise ValueError("CTest log requires verified local SHA-256 and size pins")
        if self.source.size_bytes > 2 * 1024**2:
            raise ValueError("CTest log pin exceeds 2 MiB")
        if self.terms.status == "prohibited":
            raise ValueError("CTest log source terms prohibit this corroboration")
        return self


class HCSTraceCompletion(VersionedContract):
    """Complete-last byte receipt for two deterministic corroboration JSON files.

    File hashes alone do not establish correspondence: validation requires the
    actual raw log and both original cached HCS sources. Input source terms stay
    distinct. Unknown producer/version requires its recorded implementation;
    no automatic migration or rewrite occurs. No physical-work or CI-efficacy
    assertion follows from a completed local evidence export.
    """

    kind: Literal["hcs-ctest-local-corroboration-v1"] = "hcs-ctest-local-corroboration-v1"
    recipe_fingerprint: Fingerprint
    log_spec: HCSLogSpec
    producer_sha256: Fingerprint
    files: Annotated[tuple[ETLFile, ...], Field(min_length=2, max_length=2)]
    admission: Literal["not_granted"] = "not_granted"

    @model_validator(mode="after")
    def _files(self) -> Self:
        """Fix artifact roles; a receipt cannot replace or omit either typed evidence file."""
        if tuple(file.path for file in self.files) != (_TRACE_FILE, _CORRESPONDENCE_FILE):
            raise ValueError("CTest corroboration requires exactly the two recorded evidence roles")
        if self.files[1].rows != 1:
            raise ValueError("CTest corroboration contains exactly one source correspondence")
        return self


def hcs_trace_producer() -> str:
    """Hash actual parser/contracts/source/storage bytes to select a reproducible export namespace."""
    modules = {
        "coleman": ("artifacts.py",),
        "coleman.corpus": (
            "hcs_trace.py",
            "hcs_trace_contracts.py",
            "hcs_trace_correspondence.py",
            "hcs_trace_artifacts.py",
            "hcs_raw.py",
            "hcs_contracts.py",
            "contracts.py",
            "lrts_contracts.py",
            "lrts_etl_contracts.py",
            "materialization.py",
            "materialization_contracts.py",
        ),
        "coleman.results": ("domain.py", "versioned.py"),
        "coleman.selection": ("contracts.py",),
    }
    return fingerprint(
        [
            [package + "/" + name, hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()]
            for package, names in sorted(modules.items())
            for name in names
        ]
    )


def _destination(writer: ArtifactWriter, cache: SourceCache, recipe: HCSRecipe, spec: HCSLogSpec) -> Path:
    """Preflight concrete paths and cache separation before any output is written."""
    identity = fingerprint(
        ["hcs-ctest-local-corroboration-v1", recipe.fingerprint, spec.fingerprint, hcs_trace_producer()]
    )
    root = writer.run_root / "hcs-trace" / identity.removeprefix(_SHA256_PREFIX)
    validate_cache_separation(cache.root, writer.run_root)
    for name in (_TRACE_FILE, _CORRESPONDENCE_FILE, _COMPLETE_FILE):
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(writer.run_root.resolve()):
            raise ValueError("CTest corroboration output escapes its trusted tree")
        validate_source_storage(target)
    return root


def _evidence(recipe: HCSRecipe, spec: HCSLogSpec, payload: bytes, cache: SourceCache) -> tuple[str, str, int]:
    """Rehash original log and cached sources before deterministic typed correspondence."""
    if not isinstance(payload, bytes) or len(payload) != spec.source.size_bytes:
        raise ValueError("CTest original log byte count differs from its pin")
    if _SHA256_PREFIX + hashlib.sha256(payload).hexdigest() != spec.source.sha256:
        raise ValueError("CTest original log digest differs from its pin")
    source = read_hcs_slice(recipe, cache)
    job = next((job for job in source.jobs if job.job_id == spec.job_id), None)
    if job is None:
        raise ValueError("CTest pinned job is absent from the selected original HCS metadata")
    trace = parse_hcs_ctest_trace(payload, job)
    correspondence = corroborate_hcs_ctest_trace(trace, source)
    return trace.to_json() + "\n", correspondence.to_json() + "\n", correspondence.matched_observations


def _receipt(recipe: HCSRecipe, spec: HCSLogSpec, evidence: tuple[str, str, int]) -> HCSTraceCompletion:
    """Describe exact UTF-8 bytes; raw logs are not among the output roles."""
    trace, correspondence, count = evidence
    records = tuple(
        ETLFile(
            path=name,
            sha256=_SHA256_PREFIX + hashlib.sha256(text.encode()).hexdigest(),
            size_bytes=len(text.encode()),
            rows=rows,
        )
        for name, text, rows in ((_TRACE_FILE, trace, count), (_CORRESPONDENCE_FILE, correspondence, 1))
    )
    return HCSTraceCompletion(
        schema_version=1,
        recipe_fingerprint=recipe.fingerprint,
        log_spec=spec,
        producer_sha256=hcs_trace_producer(),
        files=records,
    )


def publish_hcs_trace_evidence(
    recipe: HCSRecipe,
    spec: HCSLogSpec,
    payload: bytes,
    cache: SourceCache,
    writer: ArtifactWriter,
) -> Path:
    """Verify and publish two immutable evidence files, then their completion receipt.

    Reuses the existing source reader/cache and atomic ArtifactWriter. Identical
    concurrent retries converge; conflicting bytes raise without replacement.
    Interruption before the final write leaves no completion receipt and can be
    retried. Every retry rechecks actual log and original source bytes. Pending
    log terms remain pending; no raw log, clock or scientific admission is
    published. Paths/cache must be separate trusted trees outside Git. Validation,
    source corruption, unsupported logs and filesystem errors propagate.
    """
    recipe, spec = HCSRecipe.model_validate(recipe), HCSLogSpec.model_validate(spec)
    root = _destination(writer, cache, recipe, spec)
    evidence = _evidence(recipe, spec, payload, cache)
    receipt = _receipt(recipe, spec, evidence)
    for file, text in zip(receipt.files, evidence[:2], strict=True):
        writer.write_immutable(root / file.path, text)
    path = root / _COMPLETE_FILE
    writer.write_immutable(path, receipt.to_json() + "\n")
    return path


def validate_hcs_trace_completion(
    path: Path,
    recipe: HCSRecipe,
    payload: bytes,
    cache: SourceCache,
) -> HCSTraceCompletion:
    """Recompute completion from original input bytes; never trust declared output hashes alone.

    Read only bounded files in the exact recorded namespace. Unknown schema,
    producer, wrong recipe, changed source/output bytes or partial output raises
    ValueError/OSError. No acquisition, writes or automatic migration occurs.
    Preserve old artifacts and their recorded package when deliberately deriving
    a new producer namespace. This verifies local corroboration only.
    """
    recipe = HCSRecipe.model_validate(recipe)
    if path.is_symlink():
        raise ValueError("CTest completion cannot be a symlink")
    validate_source_storage(path)
    with path.open("rb") as stream:
        wire = stream.read(2 * 1024**2 + 1)
    if len(wire) > 2 * 1024**2:
        raise ValueError("CTest completion metadata exceeds 2 MiB")
    receipt = HCSTraceCompletion.from_json(wire)
    if receipt.producer_sha256 != hcs_trace_producer() or receipt.recipe_fingerprint != recipe.fingerprint:
        raise ValueError("CTest completion requires its recorded recipe/producer implementation")
    root = _destination(ArtifactWriter(path.parent.parent.parent), cache, recipe, receipt.log_spec)
    if path.resolve() != (root / _COMPLETE_FILE).resolve():
        raise ValueError("CTest completion differs from its recorded namespace")
    evidence = _evidence(recipe, receipt.log_spec, payload, cache)
    if receipt != _receipt(recipe, receipt.log_spec, evidence):
        raise ValueError("CTest completion metadata differs from original source evidence")
    for file, text in zip(receipt.files, evidence[:2], strict=True):
        with (root / file.path).open("rb") as stream:
            actual = stream.read(file.size_bytes + 1)
        if actual != text.encode():
            raise ValueError("CTest completion output differs from original source evidence")
    return receipt
