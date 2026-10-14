"""Frozen candidates/protocols and order-independent named RNG streams."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coleman.evaluation.protected.temporal import TemporalPlan
from coleman.results.domain import Fingerprint, Finite, Identifier, canonical_json, decode_json, fingerprint


class EvaluationContract(BaseModel):
    """Strict immutable contract base with fail-closed v1 artifact reading.

    Required integer schema version 1 rejects missing, legacy and future versions.
    Serialization revalidates unchecked Pydantic instances before JSON conversion.
    No files are read or written by these models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]

    def to_json(self) -> str:
        """Return deterministic finite JSON, revalidating nested contracts first."""
        validated = type(self).model_validate(self)
        return canonical_json(validated.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Reject duplicate keys and unsupported versions, without migrating artifacts."""
        return cls.model_validate(decode_json(payload))

    @property
    def fingerprint(self) -> str:
        """Hash the complete versioned contract, independent of Python hash/ordering."""
        return fingerprint(json.loads(self.to_json()))


class FrozenCandidate(BaseModel):  # pylint: disable=too-few-public-methods
    """Named candidate with its entire resolved configuration frozen as canonical JSON.

    Include hyperparameters, learned-state fingerprints, preprocessing and feature
    definitions in this object before protected evaluation. Only JSON objects are
    accepted; use ``from_configuration`` at the input boundary. Opaque config bytes
    cannot be mutated through nested dictionaries. A fingerprint is not evidence
    that an undeclared external dependency was frozen.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    candidate_id: Identifier
    configuration_json: str

    @field_validator("configuration_json")
    @classmethod
    def _canonical(cls, value: str) -> str:
        """Canonicalize a JSON object and reject duplicate keys/nonfinite values."""
        payload = decode_json(value)
        if not isinstance(payload, dict):
            raise ValueError("Candidate configuration must be a JSON object")
        return canonical_json(payload)

    @classmethod
    def from_configuration(cls, candidate_id: str, configuration: Any) -> Self:
        """Freeze a complete JSON object supplied at the configuration boundary."""
        return cls(candidate_id=candidate_id, configuration_json=canonical_json(configuration))

    @property
    def configuration_fingerprint(self) -> str:
        """Return the canonical configuration digest used by domain result provenance."""
        return fingerprint(json.loads(self.configuration_json))


class ComparisonHypothesis(BaseModel):  # pylint: disable=too-few-public-methods
    """Predeclared candidate/baseline pair with a unique auditable hypothesis ID."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    hypothesis_id: Identifier
    candidate_id: Identifier
    baseline_id: Identifier

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        """Reject comparing a candidate to itself."""
        if self.candidate_id == self.baseline_id:
            raise ValueError("A comparison needs distinct candidates")
        return self


class EvaluationProtocol(BaseModel):  # pylint: disable=too-few-public-methods
    """Frozen metric, estimand, independence, family error rate and practical threshold.

    ``practical_effect`` is a nonnegative improvement in the metric's own unit.
    Positive improvement favors the candidate (sign reverses for minimization).
    Alpha controls family-wise error; paired Student-t tests assume independent
    units and normal differences. These assumptions must be explicitly declared,
    not inferred from seeds or successful normality tests. Serial/clustered units
    are representable but explicitly unsupported by this v1 inference method.
    ``holdout_id`` must persist across candidates/protocol revisions using the same
    observations; changing it to disguise reuse violates the contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    protocol_id: Identifier
    holdout_id: Identifier
    metric_name: Identifier
    unit: Identifier
    meaning: Identifier
    direction: Literal["minimize", "maximize"]
    independent_unit: Identifier
    dependence: Literal["independent", "serial", "clustered"]
    normal_differences_assumed: Annotated[bool, Field(strict=True)]
    practical_effect: Annotated[Finite, Field(ge=0)]
    alpha: Annotated[Finite, Field(gt=0, lt=1)] = 0.05
    minimum_units: Annotated[int, Field(strict=True, ge=2)] = 20
    method: Literal["paired_t_holm_bonferroni_ci_v1"] = "paired_t_holm_bonferroni_ci_v1"


class FrozenEvaluation(EvaluationContract):
    """Pre-outcome snapshot including full candidates, chronology, metrics and family.

    Candidate and hypothesis ordering is canonicalized, never used as RNG entropy.
    This contract does not execute/tune candidates or replace the native runner.
    All preprocessing/hyperparameters belong in frozen candidate configurations;
    exclusion intervals and grouped point-in-time semantics belong in the plan.
    """

    data_fingerprint: Fingerprint
    plan: TemporalPlan
    protocol: EvaluationProtocol
    candidates: tuple[FrozenCandidate, ...]
    comparison_family: tuple[ComparisonHypothesis, ...]

    @field_validator("candidates")
    @classmethod
    def _candidates(cls, values: tuple[FrozenCandidate, ...]) -> tuple[FrozenCandidate, ...]:
        """Require distinct named configurations and canonicalize declaration order."""
        ids = [candidate.candidate_id for candidate in values]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("Freeze at least two distinct candidates")
        return tuple(sorted(values, key=lambda candidate: candidate.candidate_id))

    @field_validator("comparison_family")
    @classmethod
    def _family(cls, values: tuple[ComparisonHypothesis, ...]) -> tuple[ComparisonHypothesis, ...]:
        """Freeze a nonempty family without duplicate or reversed hypotheses."""
        ids = [item.hypothesis_id for item in values]
        pairs = [frozenset((item.candidate_id, item.baseline_id)) for item in values]
        if not values or len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
            raise ValueError("Comparison family must contain unique hypotheses/pairs")
        return tuple(sorted(values, key=lambda item: item.hypothesis_id))

    @model_validator(mode="after")
    def _references(self) -> Self:
        """Require every predeclared comparison to refer to a frozen candidate."""
        ids = {candidate.candidate_id for candidate in self.candidates}
        if any(item.candidate_id not in ids or item.baseline_id not in ids for item in self.comparison_family):
            raise ValueError("Comparison references an unfrozen candidate")
        return self

    def candidate(self, candidate_id: str) -> FrozenCandidate:
        """Query a frozen candidate; unknown identity raises KeyError."""
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(candidate_id)


def freeze(
    candidates: tuple[FrozenCandidate, ...],
    data_fingerprint: str,
    plan: TemporalPlan,
    protocol: EvaluationProtocol,
    comparison_family: tuple[ComparisonHypothesis, ...],
) -> FrozenEvaluation:
    """Validate and freeze the complete protocol before reading protected outcomes.

    Pure operation; changes to candidate, data, splits, assumptions, practical
    threshold, metric or comparison family change the canonical fingerprint.
    The caller owns pre-outcome timing and must persist the contract before access.
    """
    return FrozenEvaluation(
        schema_version=1,
        candidates=candidates,
        data_fingerprint=data_fingerprint,
        plan=plan,
        protocol=protocol,
        comparison_family=comparison_family,
    )


def named_seed(root_seed: int, candidate_id: str, purpose: str) -> int:
    """Derive 64-bit NumPy/Polars/Python seed entropy from stable, purpose-separated names.

    Reordering candidates/workers has no effect. Nonnegative integer roots and
    nonempty names are required; no global RNG state is modified. Repeated seeds
    on the same observations never increase the independent-unit count.
    """
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("Root seed must be a nonnegative integer")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id.strip()
        or not isinstance(purpose, str)
        or not purpose.strip()
    ):
        raise ValueError("Candidate and purpose names must be nonempty strings")
    return int(fingerprint(["coleman.named-seed.v1", root_seed, candidate_id, purpose]).split(":")[1][:16], 16)
