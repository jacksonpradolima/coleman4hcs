"""Owned synthetic CI parameters; no empirical admission or independence is implied."""

from __future__ import annotations

from typing import Annotated, Literal, Self, get_args

from pydantic import Field, model_validator

from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable


class StressFactors(Immutable):
    """Eleven explicit experimental axes, with all level combinations legal.

    Size counts logical tests before variants/churn. Durations use integer
    seconds, including zero; worker counts describe modeled list scheduling.
    Factor coverage says nothing about temporal sequence or external validity.
    """

    size: Literal["tiny", "medium", "large"] = "tiny"
    failures: Literal["zero", "rare", "moderate", "bursty"] = "moderate"
    duration: Literal["constant", "heavy_tail", "failure_correlated", "changing"] = "constant"
    churn: Literal["stable", "additions", "removals", "renames", "collisions"] = "stable"
    drift: Literal["sudden", "gradual", "recurring", "contextual"] = "sudden"
    flakiness: Literal["independent", "history", "environment"] = "independent"
    feedback: Literal["full", "censored", "delayed", "missing", "corrected"] = "censored"
    budget: Literal["severe", "full"] = "severe"
    context: Literal["informative", "noisy", "missing", "adversarial"] = "informative"
    variants: Literal["stable", "sparse", "changing"] = "stable"
    workers: Literal["serial", "bounded"] = "serial"

    @property
    def test_count(self) -> int:
        """Return the pre-churn logical size, without materializing observations."""
        return {"tiny": 8, "medium": 256, "large": 50000}[self.size]


def factor_levels() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the fixed v1 order and literal levels; never infer from unordered input."""
    return tuple((name, get_args(field.annotation)) for name, field in StressFactors.model_fields.items())


class StressCase(VersionedContract):
    """Bound one on-demand synthetic realization to <=16 builds and 2.4M rows.

    Seeds select deterministic constructed realizations, never independent
    projects. Logical source identity is the complete case fingerprint. Native
    replay eligibility is narrower than ground-truth generation; unsupported
    feedback must not become zeros/passes. Old readers are unchanged; unknown
    case versions fail without implicit migration.
    """

    generator: Literal["owned-ci-stress-v1"] = "owned-ci-stress-v1"
    factors: StressFactors
    seed: Annotated[int, Field(strict=True, ge=0, le=4294967295)] = 11
    builds: Annotated[int, Field(strict=True, ge=2, le=16)] = 8
    evidence: Literal["constructed-dependent-mechanism-fixture"] = "constructed-dependent-mechanism-fixture"

    def require_native(self) -> None:
        """Reject unsupported immediate-reward semantics before I/O or execution.

        Full-label feedback is eligible only at full budget. Censored RNFail
        emits detection rewards, not selected/nonselected passing labels.
        Parallelism is post-hoc modeled replay, not the native serial executor.
        """
        if self.factors.feedback not in {"full", "censored"}:
            raise ValueError("Native RNFail cannot consume delayed, missing or corrected feedback")
        if self.factors.feedback == "full" and self.factors.budget != "full":
            raise ValueError("Full-label native feedback requires full budget")


class StressDesign(VersionedContract):
    """A bounded strength-two certificate, irredundant but not globally minimum.

    Every row must witness a pair absent from all other rows. Validation
    independently enumerates required interactions, never runs the generator.
    Named adversarial sequences supplement, not count toward, this certificate.
    """

    strength: Literal[2] = 2
    minimality: Literal["inclusion-minimal-not-global-minimum"] = "inclusion-minimal-not-global-minimum"
    rows: Annotated[tuple[StressFactors, ...], Field(min_length=1, max_length=256)]
    methodology: Literal["NIST-SP-800-142-static-factor-pairs"] = "NIST-SP-800-142-static-factor-pairs"

    @model_validator(mode="after")
    def _certificate(self) -> Self:
        """Fail on a missing pair, duplicate row or row without a private witness."""
        from coleman.corpus.stress_design import certify_design

        certify_design(self.rows)
        return self


class StressFile(Immutable):
    """One complete canonical JSONL build or native Parquet file, digest-bound."""

    path: Annotated[
        str, Field(strict=True, pattern=r"^(truth/build-[0-9]{2}\.jsonl|datasets/stress/features-engineered\.parquet)$")
    ]
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=1, le=512 * 1024**2)]
    rows: Annotated[int, Field(strict=True, ge=1, le=2400000)]


class StressReceipt(VersionedContract):
    """Complete-last receipt; verify bytes against the recorded generator to trust it.

    Producer and writer fingerprints select namespaces. No old artifact is
    rewritten; retain its installed producer to validate it. Native projection
    contains oracle truth for scoring, never decision-visible delayed labels.
    """

    case: StressCase
    producer: Fingerprint
    parquet_writer: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    files: Annotated[tuple[StressFile, ...], Field(min_length=3, max_length=17)]
    evidence: Literal["synthetic-not-corpus-admission"] = "synthetic-not-corpus-admission"
    setup_seconds: None = None
    observed_wall_seconds: None = None

    @model_validator(mode="after")
    def _roles(self) -> Self:
        """Require all declared builds and exactly one native file, in canonical order."""
        expected = [f"truth/build-{i:02}.jsonl" for i in range(self.case.builds)]
        expected.append("datasets/stress/features-engineered.parquet")
        if [f.path for f in self.files] != expected:
            raise ValueError("Stress receipt roles do not match its case")
        if self.files[-1].rows != sum(f.rows for f in self.files[:-1]):
            raise ValueError("Stress truth/native row conservation failed")
        return self
