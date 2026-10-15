"""Bounded, offline native-Parquet descriptions; profiling never admits a source."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal, Self

import polars as pl
import pyarrow.parquet as pq
from pydantic import Field, field_validator, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.contracts import DataCard
from coleman.corpus.registry import CorpusRegistry
from coleman.results.domain import Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, UnitSemantics

Count = Annotated[int, Field(strict=True, ge=0, le=2_000_000)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_ROWS = 2_000_000
RAW_DURATION = UnitSemantics(unit="source-unit-unverified", meaning="raw-Duration-field-unverified-v1")
GROUP_ROWS = UnitSemantics(unit="1", meaning="raw-rows-per-complete-execution-group-v1")


class NumericProfile(Immutable):
    """Raw numeric distribution in explicit units, retaining null/nonfinite/negative/zero distinctions.

    Quantiles use the inverse empirical CDF: sorted[ceil(p*n)-1], with the
    minimum for p=0. These are descriptive row statistics, never independent
    trials or calibrated probabilities; declared duration semantics are inherited.
    """

    semantics: UnitSemantics
    finite_count: Count
    null_count: Count
    nonfinite_count: Count
    negative_count: Count
    zero_count: Count
    minimum: Finite | None
    median: Finite | None
    p95: Finite | None
    maximum: Finite | None

    @field_validator("minimum", "median", "p95", "maximum")
    @classmethod
    def _canonical_zero(cls, value: float | None) -> float | None:
        """Treat both binary64 zero signs as one statistical value on construction and reads."""
        return 0.0 if value == 0 else value

    @model_validator(mode="after")
    def _distribution(self) -> Self:
        """Validate availability, ordered quantiles and disjoint finite categories."""
        values = (self.minimum, self.median, self.p95, self.maximum)
        if self.negative_count + self.zero_count > self.finite_count:
            raise ValueError("Finite category counts exceed finite observations")
        if self.finite_count == 0:
            if any(value is not None for value in values):
                raise ValueError("Empty distribution requires null quantiles")
        elif any(value is None for value in values) or list(values) != sorted(
            value for value in values if value is not None
        ):
            raise ValueError("Nonempty distribution requires ordered finite quantiles")
        return self

    @property
    def rows(self) -> int:
        """Return the conserved raw row count, including null and nonfinite values."""
        return self.finite_count + self.null_count + self.nonfinite_count


class ColumnMissingness(Immutable):
    """One source column's raw null count; NaN is non-null and counted separately for Duration."""

    name: str
    null_count: Count


class NativeSourceProfile(VersionedContract):
    """Version-one source-bound, descriptive native Parquet artifact.

    The embedded card and original admission blockers are unchanged. Duration
    values retain declared units or explicit unverified source units. Build
    IDs and variant execution groups are logical labels, not independent trials.
    Quantities can be recomputed only with the matching source bytes. Reading
    validates invariants, not the truth of an arbitrary producer's observations.
    Legacy cards remain readable separately; missing/future versions fail closed.
    """

    recipe: Literal["native-parquet-description-v1"] = "native-parquet-description-v1"
    card: DataCard
    columns: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]
    column_missingness: Annotated[tuple[ColumnMissingness, ...], Field(min_length=1, max_length=64)]
    rows: Count
    missing_identity_rows: Count
    unique_observations: Count
    duplicate_identity_rows: Count
    conflicting_identities: Count
    logical_builds: Count
    execution_groups: Count
    missing_group_rows: Count
    failed_rows: Count
    passed_rows: Count
    null_verdict_rows: Count
    invalid_verdict_rows: Count
    raw_duration: NumericProfile
    tests_per_execution_group: NumericProfile
    chronology_status: Literal["not_evaluated"] = "not_evaluated"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _counts(self) -> Self:
        """Conserve raw rows and reject impossible identities or invented duration semantics."""
        _native_card(self.card)
        _column_bindings(self)
        expected_duration = self.card.manifest.outcomes.duration or RAW_DURATION
        if self.raw_duration.semantics != expected_duration or self.tests_per_execution_group.semantics != GROUP_ROWS:
            raise ValueError("Incompatible numeric profile units/meaning")
        if self.raw_duration.rows != self.rows:
            raise ValueError("Duration counts must conserve source rows")
        if self.failed_rows + self.passed_rows + self.null_verdict_rows + self.invalid_verdict_rows != self.rows:
            raise ValueError("Verdict counts must conserve source rows")
        if self.unique_observations + self.duplicate_identity_rows + self.missing_identity_rows != self.rows:
            raise ValueError("Identity counts must conserve source rows")
        if self.conflicting_identities > min(self.unique_observations, self.duplicate_identity_rows):
            raise ValueError("Conflicts require repeated identified observations")
        if not self.logical_builds <= self.execution_groups <= self.rows - self.missing_group_rows:
            raise ValueError("Execution groups must preserve logical build identity")
        groups = self.tests_per_execution_group
        if groups.null_count or groups.nonfinite_count or groups.negative_count or groups.zero_count:
            raise ValueError("Observed execution groups require positive finite raw row counts")
        if self.tests_per_execution_group.rows != self.execution_groups:
            raise ValueError("Group distribution must conserve execution groups")
        return self

    @property
    def duration_median(self) -> Measurement:
        """Expose raw median in declared or explicitly unverified units; null means unavailable."""
        value = self.raw_duration.median
        semantics = self.card.manifest.outcomes.duration
        return Measurement(
            name="corpus_raw_duration_median",
            value=value,
            unit=semantics.unit if semantics is not None else "source-unit-unverified",
            meaning="raw-Duration-inverse-empirical-cdf-v1:" + (semantics.meaning if semantics else "unverified"),
            direction="none",
            status="observed" if value is not None else "unavailable",
        )


def _column_bindings(profile: NativeSourceProfile) -> None:
    """Canonicalize column declarations and conserve null counts shared with numeric/verdict summaries."""
    required = set(profile.card.manifest.identity.observation_fields) | {"Duration", "Verdict"}
    if not required.issubset(profile.columns):
        raise ValueError("Missing native profile columns")
    if len(set(profile.columns)) != len(profile.columns):
        raise ValueError("Duplicate column declaration")
    columns = tuple(sorted(profile.columns))
    missingness = tuple(sorted(profile.column_missingness, key=lambda item: item.name))
    if tuple(item.name for item in missingness) != columns or any(
        item.null_count > profile.rows for item in missingness
    ):
        raise ValueError("Missingness must describe every column within source row bounds")
    counts = {item.name: item.null_count for item in missingness}
    if counts["Duration"] != profile.raw_duration.null_count or counts["Verdict"] != profile.null_verdict_rows:
        raise ValueError("Column null counts disagree with numeric/verdict summaries")
    object.__setattr__(profile, "columns", columns)
    object.__setattr__(profile, "column_missingness", missingness)


def _native_card(card: DataCard) -> None:
    """Restrict this recipe to declared native identities and explicit binary verdicts."""
    manifest = card.manifest
    if (
        manifest.adapter.name != "native-parquet-corpus"
        or manifest.identity.observation_fields not in {("BuildId", "Name"), ("BuildId", "Name", "Variant")}
        or manifest.identity.group_fields != ("BuildId",)
    ):
        raise ValueError("This descriptive recipe requires declared native identity")
    if manifest.outcomes.unit != "failing_test" or {
        (item.source_value, item.verdict) for item in manifest.outcomes.mapping
    } != {("0", 0), ("1", 1)}:
        raise ValueError("Native profile requires explicit binary failing-test verdict mapping")
    if manifest.source.sha256 is None or manifest.source.size_bytes is None:
        raise ValueError("Source digest and byte size are required for profiling")


def _numbers(series: pl.Series, semantics: UnitSemantics) -> NumericProfile:
    """Compute finite binary64 order statistics; retain invalid values as counts without imputation."""
    values = series.cast(pl.Float64, strict=True)
    finite = values.filter(values.is_finite()).sort()
    # Binary64 signed zeros are one statistical value; ties must not retain source order.
    finite = finite.set(finite == 0, 0.0)
    count = len(finite)
    return NumericProfile(
        semantics=semantics,
        finite_count=count,
        null_count=values.null_count(),
        nonfinite_count=len(values) - values.null_count() - count,
        negative_count=int((finite < 0).sum()),
        zero_count=int((finite == 0).sum()),
        minimum=float(finite[0]) if count else None,
        median=float(finite[(count - 1) // 2]) if count else None,
        p95=float(finite[(95 * count + 99) // 100 - 1]) if count else None,
        maximum=float(finite[-1]) if count else None,
    )


def _source_frame(card: DataCard, path: Path) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Hash one bounded local snapshot before decoding; never follow a URI or import an adapter.

    Footer row/column/uncompressed-byte limits precede column decoding. These are
    conservative trusted-local artifact limits, not an OS sandbox for hostile
    native codecs. No file is modified and no third-party bytes are published.
    """
    manifest = card.manifest
    maximum = min(MAX_SOURCE_BYTES, manifest.resources.maximum_source_bytes)
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("Source exceeds profile byte limit")
    if (
        len(payload) != manifest.source.size_bytes
        or "sha256:" + hashlib.sha256(payload).hexdigest() != manifest.source.sha256
    ):
        raise ValueError("Source bytes disagree with frozen digest/size")
    source = pq.ParquetFile(BytesIO(payload))
    metadata = source.metadata
    decoded = sum(metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups))
    if metadata.num_rows > min(MAX_ROWS, manifest.resources.maximum_rows) or decoded > MAX_DECODED_BYTES:
        raise ValueError("Source exceeds profile decoded row/byte limit")
    if len(source.schema_arrow) > 64:
        raise ValueError("Source exceeds profile column limit")
    names = source.schema_arrow.names
    required = set(manifest.identity.observation_fields) | {"Duration", "Verdict"}
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate native profile columns")
    selected = required | ({"Variant"} if "Variant" in names else set())
    frame = pl.DataFrame(source.read())
    for name in sorted(selected):
        dtype = frame.schema[name]
        allowed = dtype.is_integer() or dtype in {pl.String, pl.Null}
        if name in {"Duration", "Verdict"}:
            allowed = dtype.is_numeric() or dtype == pl.Null
        if not allowed:
            raise ValueError(f"Unsupported profile column type: {name}")
    return frame, tuple(names)


def profile_native_source(card: DataCard, path: str | Path) -> NativeSourceProfile:
    """Describe matching local native bytes, including blocked sources, without admission.

    Raises ValueError for unsupported contracts, changed sources or resource/schema
    limits; filesystem/Parquet decoding errors propagate. Source order does not
    affect statistics; retries do not become independent observations. Variant
    identity is retained even when the original card groups by logical build.
    """
    card = DataCard.model_validate(card)
    _native_card(card)
    frame, columns = _source_frame(card, Path(path))
    identity = list(card.manifest.identity.observation_fields)
    identified = frame.drop_nulls(identity)
    unique_count = identified.unique(identity).height
    keys = ["BuildId"] + (["Variant"] if "Variant" in frame.columns else [])
    grouped = frame.drop_nulls(keys)
    counts = grouped.group_by(keys).len()["len"]
    conflicts = identified.group_by(identity).agg(pl.struct("Duration", "Verdict").n_unique().alias("values"))
    verdict = frame["Verdict"]
    if verdict.dtype == pl.Null:
        verdict = verdict.cast(pl.Float64)
    return NativeSourceProfile(
        schema_version=1,
        card=card,
        columns=columns,
        column_missingness=tuple(ColumnMissingness(name=name, null_count=frame[name].null_count()) for name in columns),
        rows=frame.height,
        missing_identity_rows=frame.height - identified.height,
        unique_observations=unique_count,
        duplicate_identity_rows=identified.height - unique_count,
        conflicting_identities=conflicts.filter(pl.col("values") > 1).height,
        logical_builds=grouped["BuildId"].n_unique(),
        execution_groups=len(counts),
        missing_group_rows=frame.height - grouped.height,
        failed_rows=int((verdict == 1).sum()),
        passed_rows=int((verdict == 0).sum()),
        null_verdict_rows=verdict.null_count(),
        invalid_verdict_rows=int((~verdict.is_in([0, 1])).sum()),
        raw_duration=_numbers(frame["Duration"], card.manifest.outcomes.duration or RAW_DURATION),
        tests_per_execution_group=_numbers(counts, GROUP_ROWS),
    )


def publish_source_profile(report: NativeSourceProfile, writer: ArtifactWriter) -> Path:
    """Atomically publish descriptive JSON once; identical concurrent replay is idempotent.

    Uses the existing artifact writer's local hard-link durability contract. An
    interrupted temporary file is not a completed profile; retry recomputes from
    the frozen source. Conflicting content at the same path raises ValueError.
    """
    report = NativeSourceProfile.model_validate(report)
    path = writer.run_root / "corpus-profiles" / f"{report.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, report.to_json() + "\n")
    return path


class CorpusProfileInventory(VersionedContract):
    """Frozen diversity matrix tying descriptive profiles to exact registered cards.

    Rows and execution groups cannot satisfy source admission or diversity gates.
    Existing card coverage declarations remain unchanged; this inventory certifies
    no regime and makes no cross-project independence or policy ranking claim.
    """

    registry: CorpusRegistry
    profiles: Annotated[tuple[NativeSourceProfile, ...], Field(max_length=64)]
    uncovered_required_regimes: tuple[str, ...]
    evidence_status: Literal["insufficient_diversity"] = "insufficient_diversity"

    @model_validator(mode="after")
    def _membership(self) -> Self:
        """Reject conflicting evidence and weakened diversity; collapse identical replay."""
        unique: dict[str, NativeSourceProfile] = {}
        for profile in self.profiles:
            source_id = profile.card.manifest.benchmark_id
            if self.registry.card(source_id) != profile.card:
                raise ValueError("Profile does not match registered card")
            previous = unique.setdefault(source_id, profile)
            if previous != profile:
                raise ValueError("Conflicting profiles for one registered source")
        if tuple(sorted(self.uncovered_required_regimes)) != self.registry.required_regimes:
            raise ValueError("Descriptive profiles cannot establish diversity coverage")
        object.__setattr__(self, "profiles", tuple(unique[key] for key in sorted(unique)))
        object.__setattr__(self, "uncovered_required_regimes", self.registry.required_regimes)
        return self


def publish_profile_inventory(inventory: CorpusProfileInventory, writer: ArtifactWriter) -> Path:
    """Publish the complete deterministic matrix atomically; retries reuse identical bytes."""
    inventory = CorpusProfileInventory.model_validate(inventory)
    path = writer.run_root / "corpus-profile-inventories" / f"{inventory.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, inventory.to_json() + "\n")
    return path
