"""Offline source admission through existing receipts, registry and immutable artifacts.

This service performs local byte verification and descriptive profiling only.
It never follows source URLs, executes subject code, trains policies or imports
CLI/MCP/provider transports. Source reviews remain explicit attestations; the
service cannot prove legal rights, execution truth or statistical independence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, Protocol

import polars as pl
import pyarrow as pa

from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission_contracts import (
    AdmissionCompletion,
    AdmissionFile,
    CorpusAdmission,
    EvidencePayload,
    ProducerVerification,
    SourceAssessment,
    SourceReview,
)
from coleman.corpus.admission_reality import production_facts
from coleman.corpus.admission_rows import describe_verified_source
from coleman.corpus.admission_semantics import (
    admitted_regimes,
    historical_blockers,
    observed_models,
    qualification,
    retained_rows,
    validate_producer_verification,
)
from coleman.corpus.admission_sources import source_card
from coleman.corpus.controlled import validate_controlled_dataset
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.hcs_contracts import HCSReceipt
from coleman.corpus.hcs_etl import validate_hcs_etl
from coleman.corpus.lrts_etl import validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.lrts_stage_contracts import LRTSStageReceipt
from coleman.corpus.lrts_stage_etl import validate_lrts_stage_etl
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.profiles import NativeSourceProfile, profile_native_source
from coleman.corpus.registry import CorpusRegistry
from coleman.corpus.rtptorrent_etl import validate_rtptorrent_etl
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLReceipt
from coleman.corpus.stress_artifacts import validate_stress
from coleman.corpus.stress_contracts import StressReceipt
from coleman.results.domain import fingerprint

MAXIMUM_REPORT_BYTES = CorpusAdmission.MAXIMUM_BYTES
EvidenceKind = Literal["native_profile", "lrts", "lrts_stage", "rtptorrent", "hcs", "controlled", "stress"]


class ControlledSourceVerifier(Protocol):
    """Explicit offline raw-producer boundary, implemented outside the domain.

    Implementations must verify the original installed SDK/wheel and existing
    source-specific raw-wire parser, without network or subject execution.
    Returned metadata alone is not verification; publication invokes this
    binding again. No stored field can select an import, process or transport.
    """

    def verify(self, payload: ControlledDataset, cache: SourceCache) -> ProducerVerification:
        """Recheck original raw packets/context/normalized observations; raise on any mismatch."""
        ...


@dataclass(frozen=True)
class SourceEvidence:
    """Explicit local verification binding, excluded from portable report identity.

    ``root`` is the original Parquet file for a native profile, or the export
    directory for completion receipts. Controlled observations use their cache;
    root is descriptive only. Cache is required for foreign-source receipts and
    forbidden for native profiles/stress. Paths are trusted local inputs, never
    inferred from a receipt URI. Review must bind the exact payload fingerprint.
    """

    payload: EvidencePayload
    review: SourceReview
    root: Path
    cache: SourceCache | None = None
    controlled_verifier: ControlledSourceVerifier | None = None


def admission_producer() -> str:
    """Bind the authored audit implementation and tabular versions; unknown producers require their package."""
    names = (
        "admission.py",
        "admission_contracts.py",
        "admission_profiles.py",
        "admission_rows.py",
        "admission_reality.py",
        "admission_semantics.py",
        "admission_sources.py",
        "admission_report.py",
        "contracts.py",
        "registry.py",
        "profiles.py",
        "controlled.py",
        "controlled_contracts.py",
        "materialization.py",
    )
    identities = [
        ["coleman.corpus/" + name, hashlib.sha256(files("coleman.corpus").joinpath(name).read_bytes()).hexdigest()]
        for name in sorted(names)
    ]
    for package, name in (
        ("coleman", "artifacts.py"),
        ("coleman.results", "domain.py"),
        ("coleman.results", "versioned.py"),
        ("coleman.selection", "contracts.py"),
        ("coleman.evaluation.protected", "temporal.py"),
    ):
        identities.append(
            [package + "/" + name, hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()]
        )
    return fingerprint([identities, "polars-" + pl.__version__, "pyarrow-" + pa.__version__])


def load_source_evidence(
    kind: EvidenceKind,
    artifact: Path,
    review: SourceReview,
    *,
    root: Path | None = None,
    cache: SourceCache | None = None,
    controlled_verifier: ControlledSourceVerifier | None = None,
) -> SourceEvidence:
    """Read one explicit bounded local contract; this does not certify source bytes.

    Supported families are a fixed allowlist, never dynamic imports. Native
    profiles require an explicit original-Parquet path; other receipts default
    to their containing directory. Missing/future schemas, duplicate JSON keys,
    unsupported families and oversized contracts reject without migration.
    File errors propagate. No source URI is opened and no output is written.
    """
    readers = {
        "native_profile": NativeSourceProfile,
        "lrts": LRTSETLReceipt,
        "lrts_stage": LRTSStageReceipt,
        "rtptorrent": RTPTorrentETLReceipt,
        "hcs": HCSReceipt,
        "controlled": ControlledDataset,
        "stress": StressReceipt,
    }
    if kind not in readers or (kind == "native_profile" and root is None):
        raise ValueError("Unknown admission family or missing original native Parquet path")
    with artifact.open("rb") as stream:
        payload = stream.read(MAXIMUM_REPORT_BYTES + 1)
    if len(payload) > MAXIMUM_REPORT_BYTES:
        raise ValueError("Admission source contract exceeds 16 MiB")
    parsed = readers[kind].from_json(payload)
    return SourceEvidence(
        parsed, SourceReview.model_validate(review), root or artifact.parent, cache, controlled_verifier
    )


def _preflight_source(evidence: SourceEvidence) -> EvidencePayload:
    """Reject incompatible bindings and resource limits before source file reads or expensive regeneration."""
    payload = type(evidence.payload).model_validate(evidence.payload)
    if retained_rows(payload) > 2_000_000:
        raise ValueError("Admission source profile exceeds its two-million-row bound")
    if evidence.controlled_verifier is not None and not isinstance(payload, ControlledDataset):
        raise ValueError("Raw controlled-source verifier is incompatible with this source family")
    review = SourceReview.model_validate(evidence.review)
    if review.payload_fingerprint != payload.fingerprint:
        raise ValueError("Admission review does not bind the exact source payload")
    if len(payload.to_json().encode("utf-8")) > MAXIMUM_REPORT_BYTES:
        raise ValueError("Admission source metadata exceeds its 16-MiB bound")
    return payload


def _raw_producer_proof(evidence: SourceEvidence, payload: EvidencePayload) -> ProducerVerification | None:
    """Invoke only an explicit controlled-source verifier; normalize and bind its portable result."""
    if not isinstance(payload, ControlledDataset) or evidence.controlled_verifier is None:
        return None
    if evidence.cache is None:
        raise ValueError("Original controlled-producer verification requires its source cache")
    proof = ProducerVerification.model_validate(evidence.controlled_verifier.verify(payload, evidence.cache))
    validate_producer_verification(payload, proof)
    return proof


def _verify_source(evidence: SourceEvidence) -> tuple[EvidencePayload, ProducerVerification | None]:
    """Revalidate actual source/cache/output bytes through the existing family-specific verifier."""
    payload = _preflight_source(evidence)
    if isinstance(payload, NativeSourceProfile):
        if evidence.cache is not None or profile_native_source(payload.card, evidence.root) != payload:
            raise ValueError("Native admission profile differs from its actual source bytes")
    elif isinstance(payload, StressReceipt):
        if evidence.cache is not None:
            raise ValueError("Owned stress verification does not consume a foreign-source cache")
        validate_stress(payload, evidence.root)
    else:
        cache = evidence.cache
        if cache is None:
            raise ValueError("Source admission requires the explicit verified offline cache")
        _verify_cached(payload, cache, evidence.root)
    return payload, _raw_producer_proof(evidence, payload)


def _verify_cached(payload: EvidencePayload, cache: SourceCache, root: Path) -> None:
    """Dispatch a fixed source family; metadata cannot select a module, URL or executable."""
    if isinstance(payload, LRTSETLReceipt):
        validate_lrts_etl(payload, cache, root)
    elif isinstance(payload, LRTSStageReceipt):
        validate_lrts_stage_etl(payload, cache, root)
    elif isinstance(payload, RTPTorrentETLReceipt):
        validate_rtptorrent_etl(payload, cache, root)
    elif isinstance(payload, HCSReceipt):
        validate_hcs_etl(payload, cache, root)
    elif isinstance(payload, ControlledDataset):
        validate_controlled_dataset(payload, cache)
    else:
        raise ValueError("Unsupported cached admission source family")


def assess_source(evidence: SourceEvidence) -> SourceAssessment:
    """Verify and describe one source, preserving a narrow purpose and every historical blocker.

    Performs bounded offline cache/file reads and deterministic stress
    regeneration. Native profiling uses its existing bounds; other profiles
    retain at most two million rows. Missing bytes, source/schema/writer changes
    and conflicts raise before any report completion. No files are published,
    fetched, migrated or repaired. Runtime/rights reviews remain attestations.
    """
    payload, proof = _verify_source(evidence)
    profile = describe_verified_source(
        payload, evidence.root, constructed_history=evidence.review.history == "constructed"
    )
    scope, reasons = qualification(payload, profile, evidence.review, proof)
    return SourceAssessment(
        schema_version=1,
        payload=payload,
        review=evidence.review,
        producer_verification=proof,
        card=source_card(payload, profile, evidence.review),
        profile=profile,
        reality=production_facts(payload, evidence.review),
        scope=scope,
        admission_reasons=reasons,
        historical_blockers=historical_blockers(payload),
    )


def audit_corpus(registry: CorpusRegistry, sources: tuple[SourceEvidence, ...]) -> CorpusAdmission:
    """Extend the existing registry with verified slice cards and derive a deterministic coverage view.

    Existing cards cannot be silently changed. Identical source retries collapse;
    conflicting evidence for one logical selection rejects. Unbound registrations
    remain explicit and cannot cover a regime. At most 64 cards/sources are
    supported; legacy registry/fixture/profile artifacts retain their meanings.
    This performs source verification but no output writes or policy execution.
    """
    registry = CorpusRegistry.model_validate(registry)
    if len(sources) > 64:
        raise ValueError("Admission audit exceeds its 64-source bound")
    assessments = tuple(assess_source(source) for source in sources)
    cards = {card.manifest.benchmark_id: card for card in registry.cards}
    for assessment in assessments:
        key, card = assessment.card.manifest.benchmark_id, assessment.card
        if cards.setdefault(key, card) != card:
            raise ValueError("Admission cannot silently replace an existing source card")
    integrated = CorpusRegistry(
        schema_version=1,
        registry_id=registry.registry_id,
        cards=tuple(cards.values()),
        required_regimes=registry.required_regimes,
    )
    assessed = {item.card.manifest.benchmark_id for item in assessments}
    covered = admitted_regimes(assessments)
    return CorpusAdmission(
        schema_version=1,
        producer=admission_producer(),
        registry=integrated,
        assessments=assessments,
        unassessed_sources=tuple(sorted(cards.keys() - assessed)),
        covered_regimes=covered,
        uncovered_regimes=tuple(sorted(set(integrated.required_regimes) - set(covered))),
        observed_source_models=observed_models(assessments),
    )


def validate_corpus_sources(report: CorpusAdmission, sources: tuple[SourceEvidence, ...]) -> None:
    """Rebuild the complete report from actual local source bytes; self-declared digests are insufficient.

    Unknown producers require their original installed package. Missing/extra
    bindings, tampered profiles/cards/facts, unsupported writers or source changes
    reject without rewriting old artifacts or repairing caches. No publication,
    network, subject execution or native experiment occurs.
    """
    report = CorpusAdmission.model_validate(report)
    if report.producer != admission_producer():
        raise ValueError("Unsupported admission producer; use the recorded installed package")
    if audit_corpus(report.registry, sources) != report:
        raise ValueError("Admission report disagrees with actual source verification bindings")


def _target(root: Path, name: str) -> Path:
    """Contain fixed local output roles; trusted parents and existing artifact durability limits apply."""
    path = root / name
    validate_source_storage(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Admission artifact escapes its trusted output tree")
    return path


def publish_corpus_admission(
    report: CorpusAdmission, sources: tuple[SourceEvidence, ...], writer: ArtifactWriter
) -> Path:
    """Reverify all sources and publish JSON/human report before an immutable completion receipt.

    Outputs must stay outside Git because embedded controlled observations may
    be recipe-only. Each report role is bounded to 16 MiB. Identical concurrent
    retries converge through ArtifactWriter; interrupted individual files never
    imply completed publication. Conflicts/I/O/validation errors propagate. No
    cache, source receipt, old result or learning checkpoint is changed.
    """
    from coleman.corpus.admission_report import render_corpus_admission

    validate_source_storage(writer.run_root)
    for source in sources:
        if source.cache is not None:
            validate_cache_separation(source.cache.root, writer.run_root)
    validate_corpus_sources(report, sources)
    root = _target(writer.run_root, "corpus-admission/" + report.fingerprint.removeprefix("sha256:"))
    content = (("admission.json", report.to_json() + "\n"), ("report.md", render_corpus_admission(report)))
    records = []
    for name, text in content:
        payload = text.encode("utf-8")
        if not 1 <= len(payload) <= MAXIMUM_REPORT_BYTES:
            raise ValueError("Admission report role exceeds its 16-MiB byte bound")
        writer.write_immutable(_target(root, name), text)
        records.append(
            AdmissionFile.model_validate(
                {"path": name, "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            )
        )
    completion = AdmissionCompletion(schema_version=1, report_fingerprint=report.fingerprint, files=tuple(records))
    path = _target(root, "complete.json")
    writer.write_immutable(path, completion.to_json() + "\n")
    return path


def validate_admission_completion(path: Path, sources: tuple[SourceEvidence, ...]) -> CorpusAdmission:
    """Verify completion, both fixed report roles and all source bindings offline, without repair.

    Missing/truncated/unknown-version metadata, wrong byte counts/digests,
    inconsistent human output and source/profile conflicts reject. Completion
    itself is bounded to 16 KiB. No self-signed file digest can replace rebuilding
    actual source evidence. The original producer/package is required for older
    producers; no migration or rewriting is performed.
    """
    root = path.parent
    if path != _target(root, "complete.json"):
        raise ValueError("Admission completion requires its fixed complete.json role")
    with path.open("rb") as stream:
        encoded = stream.read(16 * 1024 + 1)
    if len(encoded) > 16 * 1024:
        raise ValueError("Admission completion metadata exceeds 16 KiB")
    completion = AdmissionCompletion.from_json(encoded)
    contents = {}
    for item in completion.files:
        with _target(root, item.path).open("rb") as stream:
            payload = stream.read(item.size_bytes + 1)
        if len(payload) != item.size_bytes or "sha256:" + hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("Admission report role bytes disagree with completion")
        contents[item.path] = payload
    report = CorpusAdmission.from_json(contents["admission.json"])
    validate_corpus_sources(report, sources)
    from coleman.corpus.admission_report import render_corpus_admission

    if (
        completion.report_fingerprint != report.fingerprint
        or contents["report.md"] != render_corpus_admission(report).encode()
    ):
        raise ValueError("Admission completion identity or human report disagrees with source evidence")
    return report
