"""Curated CI source registry; registrations remain distinct from validated evidence."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.contracts import DataCard
from coleman.corpus.validation import CorpusFixtureReport, validate_fixture_source
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable

REQUIRED_REGIMES = (
    "small_volatile_historical_ci",
    "medium_historical_ci",
    "long_running_large_suite_ci",
    "highly_configurable_systems",
    "controlled_java_real_faults",
    "controlled_non_java_real_faults",
    "deterministic_adversarial_simulation",
)
_BUNDLED_FIXTURES = {
    "ci-serial-boundary-v1": "ci-serial-boundary-v1.json",
    "ci-parallel-occupancy-v1": "ci-parallel-occupancy-v1.json",
}


class CorpusRegistry(VersionedContract):
    """Canonical bounded source/card registry, including blocked and planned sources.

    Reordering entries does not change identity. Existing example/context/variant
    files remain separate source artifacts with explicit shared dependence groups;
    their count cannot be interpreted as independent projects or observations.
    """

    registry_id: Identifier
    cards: Annotated[tuple[DataCard, ...], Field(min_length=1, max_length=64)]
    required_regimes: Annotated[tuple[Identifier, ...], Field(min_length=7, max_length=32)]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        """Reject duplicate source identities and weakened/duplicate minimum diversity cells."""
        ids = [card.manifest.benchmark_id for card in self.cards]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate registered source identity")
        if len(self.required_regimes) != len(set(self.required_regimes)) or not set(REQUIRED_REGIMES).issubset(
            self.required_regimes
        ):
            raise ValueError("Registry must retain all declared minimum diversity regimes")
        object.__setattr__(self, "cards", tuple(sorted(self.cards, key=lambda card: card.manifest.benchmark_id)))
        object.__setattr__(self, "required_regimes", tuple(sorted(self.required_regimes)))
        return self

    def card(self, benchmark_id: str) -> DataCard:
        """Return a typed card; unknown source IDs raise KeyError with no provider/file access."""
        for card in self.cards:
            if card.manifest.benchmark_id == benchmark_id:
                return card
        raise KeyError(benchmark_id)


class SourceBlock(Immutable):
    """A visible blocked registration, not a missing run or a worst-ranked candidate."""

    benchmark_id: Identifier
    reasons: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]


class FixtureReadiness(VersionedContract):
    """Initial visible coverage inventory; tiny fixtures never satisfy corpus diversity.

    This phase-one report cannot certify real-source admission, independence,
    operational value or the full #261 benchmark program. All seven minimum
    regimes stay uncovered until dedicated admission/profile evidence exists.
    """

    registry_fingerprint: Fingerprint
    registry: CorpusRegistry
    validated_reports: Annotated[tuple[CorpusFixtureReport, ...], Field(max_length=64)]
    registered_sources: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=64)]
    blocked_sources: Annotated[tuple[SourceBlock, ...], Field(max_length=64)]
    validated_fixture_sources: Annotated[tuple[Identifier, ...], Field(max_length=64)]
    fixture_dependence_groups: Annotated[tuple[Identifier, ...], Field(max_length=64)]
    uncovered_required_regimes: Annotated[tuple[Identifier, ...], Field(min_length=7, max_length=32)]
    evidence_status: Literal["insufficient_diversity"] = "insufficient_diversity"

    @model_validator(mode="after")
    def _inventory(self) -> Self:
        """Reject duplicated, unregistered, contradictory or weakened evidence summaries."""
        for field in (
            "registered_sources",
            "validated_fixture_sources",
            "fixture_dependence_groups",
            "uncovered_required_regimes",
        ):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError("Duplicate readiness identity")
            object.__setattr__(self, field, tuple(sorted(values)))
        blocked = [source.benchmark_id for source in self.blocked_sources]
        if len(blocked) != len(set(blocked)):
            raise ValueError("Duplicate blocked source")
        _readiness_membership(self, set(blocked))
        _readiness_evidence(self)
        object.__setattr__(
            self, "blocked_sources", tuple(sorted(self.blocked_sources, key=lambda source: source.benchmark_id))
        )
        return self


def _readiness_membership(readiness: FixtureReadiness, blocked: set[str]) -> None:
    """Keep source states inside the registry and require honest missing minimum diversity."""
    registered = set(readiness.registered_sources)
    validated = set(readiness.validated_fixture_sources)
    if not (blocked | validated).issubset(registered) or blocked & validated:
        raise ValueError("Readiness source states are unregistered or contradictory")
    if not set(REQUIRED_REGIMES).issubset(readiness.uncovered_required_regimes):
        raise ValueError("Tiny fixtures cannot cover required corpus diversity")
    group_count = len(readiness.fixture_dependence_groups)
    if group_count > len(validated) or (bool(validated) != bool(group_count)):
        raise ValueError("Validated source dependence groups are inconsistent")


def _validated_evidence(
    registry: CorpusRegistry, reports: tuple[CorpusFixtureReport, ...]
) -> tuple[CorpusFixtureReport, ...]:
    """Revalidate each frozen source and deduplicate identical evidence, rejecting conflicts."""
    unique: dict[str, CorpusFixtureReport] = {}
    for raw in reports:
        report = CorpusFixtureReport.model_validate(raw)
        source_id = report.manifest.benchmark_id
        if registry.card(source_id).manifest != report.manifest:
            raise ValueError("Validation report does not match the registered frozen source")
        previous = unique.setdefault(source_id, report)
        if previous != report:
            raise ValueError("Conflicting validation reports for one source")
    return tuple(unique[source_id] for source_id in sorted(unique))


def _readiness_evidence(readiness: FixtureReadiness) -> None:
    """Derive all evidence identities from the embedded registry and verified reports on every read."""
    registry = readiness.registry
    reports = _validated_evidence(registry, readiness.validated_reports)
    registered = tuple(card.manifest.benchmark_id for card in registry.cards)
    blocked = tuple(
        SourceBlock(benchmark_id=card.manifest.benchmark_id, reasons=card.manifest.blocking_reasons)
        for card in registry.cards
        if card.manifest.blocking_reasons
    )
    validated = tuple(report.manifest.benchmark_id for report in reports)
    groups = tuple(sorted({report.manifest.dependence_group for report in reports}))
    if (
        readiness.registry_fingerprint != registry.fingerprint
        or readiness.registered_sources != registered
        or sorted(readiness.blocked_sources, key=lambda source: source.benchmark_id) != list(blocked)
        or readiness.validated_fixture_sources != validated
        or readiness.fixture_dependence_groups != groups
        or readiness.uncovered_required_regimes != registry.required_regimes
    ):
        raise ValueError("Readiness summary disagrees with registered source and validation evidence")
    object.__setattr__(readiness, "validated_reports", reports)


def load_registry() -> CorpusRegistry:
    """Read the bundled versioned metadata only; no source URI/import is executed."""
    return CorpusRegistry.from_json(files("coleman.corpus").joinpath("resources/registry-v1.json").read_bytes())


def bundled_fixture_bytes(benchmark_id: str) -> bytes:
    """Read one fixed packaged fixture by approved ID; reject arbitrary paths/URLs.

    No network, source archive or repository checkout is consulted. Unknown IDs
    raise KeyError. These tiny repository-authored inputs are redistributable;
    blocked historical registrations are not materialized by this function.
    """
    filename = _BUNDLED_FIXTURES[benchmark_id]
    return files("coleman.corpus").joinpath("resources", filename).read_bytes()


def validate_bundled_source(benchmark_id: str, registry: CorpusRegistry | None = None) -> CorpusFixtureReport:
    """Validate a registered bundled source; unresolved real-source metadata fails closed."""
    registry = load_registry() if registry is None else CorpusRegistry.model_validate(registry)
    manifest = registry.card(benchmark_id).manifest
    manifest.require_ready()
    if manifest.license.redistribution != "permitted":
        raise ValueError("Bundled fixtures require explicit redistribution permission")
    return validate_fixture_source(manifest, bundled_fixture_bytes(benchmark_id))


def fixture_readiness(registry: CorpusRegistry, reports: tuple[CorpusFixtureReport, ...] = ()) -> FixtureReadiness:
    """Summarize verified fixture groups and blocked registrations without promoting tiny evidence.

    Supplied reports are revalidated against this exact registry; duplicate source
    attempts do not increase evidence. A changed manifest/source cannot reuse an
    older untouched-evidence label. Independent real-corpus cells remain missing.
    """
    registry = CorpusRegistry.model_validate(registry)
    reports = _validated_evidence(registry, reports)
    validated = {report.manifest.benchmark_id for report in reports}
    groups = {report.manifest.dependence_group for report in reports}
    return FixtureReadiness(
        schema_version=1,
        registry_fingerprint=registry.fingerprint,
        registry=registry,
        validated_reports=reports,
        registered_sources=tuple(card.manifest.benchmark_id for card in registry.cards),
        blocked_sources=tuple(
            SourceBlock(benchmark_id=card.manifest.benchmark_id, reasons=card.manifest.blocking_reasons)
            for card in registry.cards
            if card.manifest.blocking_reasons
        ),
        validated_fixture_sources=tuple(sorted(validated)),
        fixture_dependence_groups=tuple(sorted(groups)),
        uncovered_required_regimes=registry.required_regimes,
    )


def publish_fixture_readiness(readiness: FixtureReadiness, writer: ArtifactWriter) -> Path:
    """Publish a deterministic insufficient-diversity inventory using existing immutable artifacts.

    Same-content retries are idempotent. I/O/conflicts propagate and interrupted
    temporary files never constitute completed inventory artifacts. This metadata
    report is not a cross-corpus empirical comparison or a license grant.
    """
    readiness = FixtureReadiness.model_validate(readiness)
    path = writer.run_root / "corpus-readiness" / f"{readiness.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, readiness.to_json() + "\n")
    return path
