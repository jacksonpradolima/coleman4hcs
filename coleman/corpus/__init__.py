"""Versioned CI corpus metadata and bounded synthetic correctness validation.

Phase-one source registration is not the complete diversified benchmark program.
Source admission, licensing and generalization remain separate evidence claims.
"""

from coleman.corpus.contracts import (
    AdapterReference,
    ConversionSummary,
    CorpusManifest,
    CorpusResources,
    DataCard,
    EvaluationDeclaration,
    ExclusionCount,
    LicenseReview,
    ObservationIdentity,
    OutcomeSemantics,
    SourceArtifact,
    TimeSemantics,
    VerdictMapping,
)
from coleman.corpus.fixtures import (
    CIReferenceFixture,
    ParallelFixtureCase,
    ParallelReference,
    SerialFixtureCase,
    fixture_bytes,
    parallel_reference,
)
from coleman.corpus.profiles import (
    ColumnMissingness,
    CorpusProfileInventory,
    NativeSourceProfile,
    NumericProfile,
    profile_native_source,
    publish_profile_inventory,
    publish_source_profile,
)
from coleman.corpus.reference import RatioReference, TinyCIReference, tiny_ci_reference
from coleman.corpus.registry import (
    REQUIRED_REGIMES,
    CorpusRegistry,
    FixtureReadiness,
    bundled_fixture_bytes,
    fixture_readiness,
    load_registry,
    publish_fixture_readiness,
    validate_bundled_source,
)
from coleman.corpus.validation import CorpusFixtureReport, publish_fixture_report, validate_fixture_source

__all__ = [
    "AdapterReference",
    "CIReferenceFixture",
    "ColumnMissingness",
    "ConversionSummary",
    "CorpusFixtureReport",
    "CorpusManifest",
    "CorpusProfileInventory",
    "CorpusRegistry",
    "CorpusResources",
    "DataCard",
    "EvaluationDeclaration",
    "ExclusionCount",
    "FixtureReadiness",
    "LicenseReview",
    "NativeSourceProfile",
    "NumericProfile",
    "ObservationIdentity",
    "OutcomeSemantics",
    "ParallelFixtureCase",
    "ParallelReference",
    "REQUIRED_REGIMES",
    "RatioReference",
    "SerialFixtureCase",
    "SourceArtifact",
    "TimeSemantics",
    "TinyCIReference",
    "VerdictMapping",
    "bundled_fixture_bytes",
    "fixture_bytes",
    "fixture_readiness",
    "load_registry",
    "parallel_reference",
    "profile_native_source",
    "publish_fixture_readiness",
    "publish_fixture_report",
    "publish_profile_inventory",
    "publish_source_profile",
    "tiny_ci_reference",
    "validate_bundled_source",
    "validate_fixture_source",
]
