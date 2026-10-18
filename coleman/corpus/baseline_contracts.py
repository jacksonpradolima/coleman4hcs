"""Frozen comparable CI candidates and decision inputs, independent of execution transports."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.corpus.repeated_contracts import BUDGET_RATIO, Seed
from coleman.evaluation.protected.temporal import TemporalObservation, Timestamp
from coleman.pilot.contracts import SECONDS
from coleman.results.domain import Fingerprint, Identifier, Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity

Method = Literal[
    "seeded_random_v1",
    "original_order_v1",
    "shortest_prior_v1",
    "recent_failure_v1",
    "failure_frequency_v1",
    "recency_duration_v1",
    "ucb_c1_v1",
    "thompson_beta11_v1",
    "discounted_ucb_095_c1_v1",
    "linucb_alpha05_v1",
    "portfolio_random_ucb_v1",
    "unattainable_failure_first_v1",
    "external_dante_prioritization",
]
METHODS: tuple[Method, ...] = (
    "seeded_random_v1",
    "original_order_v1",
    "shortest_prior_v1",
    "recent_failure_v1",
    "failure_frequency_v1",
    "recency_duration_v1",
    "ucb_c1_v1",
    "thompson_beta11_v1",
    "discounted_ucb_095_c1_v1",
    "linucb_alpha05_v1",
    "portfolio_random_ucb_v1",
    "unattainable_failure_first_v1",
    "external_dante_prioritization",
)
FAILURE_MEANING = "binary-test-failure-v1"
ESTIMATE_MEANING = "prior-test-runner-occupancy-estimate-v1"


class BaselineFeature(Immutable):
    """One declared predecision feature/estimate, retaining missingness and its source.

    ``available_at=None`` means that an original availability clock is unknown.
    Such a feature cannot qualify a point-in-time comparison. An assumed-order
    simulation must declare its ordinal availability separately; no date is invented.
    Values and units use the existing strict measurement contract. A feature is
    a producer declaration, not proof that its upstream extraction avoided labels.
    """

    measurement: Measurement
    source_fingerprint: Fingerprint
    available_at: Timestamp | None
    available_before_step: Annotated[int, Field(strict=True, ge=0, le=512)]


class BaselineTest(Immutable):
    """One complete logical test and separate realized versus decision-time quantities.

    Declared scoring occupancy is seconds, never parallel elapsed time. An
    observed-CI projection models additive reported test work; this declaration
    does not turn source durations into measured whole-runner occupancy. Binary
    failures refer to failing tests, not distinct faults. Zero is observed; null
    unknown/unavailable values remain explicit and prevent native execution.
    Estimates use a different meaning and never inherit current measured costs.
    Context names/semantics must be identical across the selected source universe.
    """

    test_id: Identifier
    duration: Measurement
    failure: Measurement
    prior_cost: BaselineFeature | None = None
    context: Annotated[tuple[BaselineFeature, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def _quantities(self) -> Self:
        """Reject incompatible physical units, nonbinary outcomes and ambiguous feature names."""
        if (self.duration.unit, self.duration.meaning) != (SECONDS.unit, SECONDS.meaning):
            raise ValueError("Baseline truth requires test runner occupancy seconds")
        if self.duration.minimum != 0 or self.failure.minimum != 0:
            raise ValueError("Baseline physical duration and binary failure require minimum zero")
        if self.duration.direction != "minimize" or self.failure.direction != "none":
            raise ValueError("Baseline realized duration minimizes work; a binary outcome has no ranking direction")
        if (self.failure.unit, self.failure.meaning) != ("1", FAILURE_MEANING):
            raise ValueError("Baseline outcome requires the explicit binary failing-test unit")
        if self.failure.value is not None and self.failure.value not in (0.0, 1.0):
            raise ValueError("Baseline failure is binary, never a reward or probability")
        self._validate_inputs()
        return self

    def _validate_inputs(self) -> None:
        """Validate independent prior-cost estimates and canonical context definitions."""
        if self.prior_cost is not None:
            measure = self.prior_cost.measurement
            if (measure.unit, measure.meaning, measure.minimum) != ("s", ESTIMATE_MEANING, 0):
                raise ValueError("Prior cost requires separate nonnegative estimated runner seconds")
            if measure.direction != "minimize":
                raise ValueError("Prior cost estimates minimize occupied runner time")
        names = [feature.measurement.name for feature in self.context]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate baseline context feature")
        object.__setattr__(self, "context", tuple(sorted(self.context, key=lambda item: item.measurement.name)))


class BaselineBuild(Immutable):
    """One source build with explicit original clocks and optional meaningful base order.

    ``step`` is a frozen replay ordinal, not a timestamp or independence unit.
    ``source_metadata=None`` preserves missing original source clocks. Meaningful
    base order must be attested by the producer; a native loader's lexical order
    cannot supply it implicitly. IDs/variants/sibling jobs retain source grouping
    outside this preparation; no independent-project inference runs here.
    """

    step: Annotated[int, Field(strict=True, ge=1, le=512)]
    member_id: Identifier
    source_metadata: TemporalObservation | None
    decided_at: Timestamp | None
    tests: Annotated[tuple[BaselineTest, ...], Field(min_length=1, max_length=10000)]
    original_order: Annotated[tuple[Identifier, ...], Field(max_length=10000)] | None

    @model_validator(mode="after")
    def _membership(self) -> Self:
        """Reject duplicate tests, incomplete base order and future feature declarations."""
        ids = [test.test_id for test in self.tests]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate logical baseline test")
        if self.original_order is not None and sorted(self.original_order) != sorted(ids):
            raise ValueError("Meaningful original order must contain the complete unique test universe")
        self._validate_feature_clocks()
        object.__setattr__(self, "tests", tuple(sorted(self.tests, key=lambda item: item.test_id)))
        return self

    def _validate_feature_clocks(self) -> None:
        """Reject features declared after this decision, preserving unknown clock declarations."""
        for test in self.tests:
            features = (*test.context, *((test.prior_cost,) if test.prior_cost is not None else ()))
            for feature in features:
                if feature.available_before_step >= self.step:
                    raise ValueError("Baseline feature is not declared available before its decision step")
                if (
                    self.decided_at is not None
                    and feature.available_at is not None
                    and feature.available_at > self.decided_at
                ):
                    raise ValueError("Baseline feature contains future information")


class BaselineSource(VersionedContract):
    """Bounded interpreted source with explicit provenance, clock assumptions and completeness.

    This DTO records declarations, never verifies a source receipt or grants
    scientific admission. Existing source-specific verifiers remain authoritative.
    Assumed-order post-hoc simulations retain null original clocks; they cannot
    masquerade as historical protected evidence. V1's native feedback is synchronous
    and budget-censored, so actually late labels make that source incompatible.
    """

    source_id: Identifier
    source_uri: Identifier
    source_fingerprint: Fingerprint
    license_id: Identifier
    evidence_basis: Literal["constructed_mechanism", "observed_ci_projection", "controlled_functional"]
    clock_basis: Literal["source_point_in_time", "assumed_order_only"]
    admission_fingerprint: Fingerprint | None
    builds: Annotated[tuple[BaselineBuild, ...], Field(min_length=1, max_length=512)]

    @model_validator(mode="after")
    def _source(self) -> Self:
        """Freeze a contiguous replay grid without silently reordering physical time or source groups."""
        ordered = tuple(sorted(self.builds, key=lambda item: item.step))
        if tuple(item.step for item in ordered) != tuple(range(1, len(ordered) + 1)):
            raise ValueError("Baseline source steps must be a complete contiguous grid")
        members = [item.member_id for item in ordered]
        if len(members) != len(set(members)):
            raise ValueError("Duplicate baseline source member")
        metadata = tuple(item.source_metadata for item in ordered if item.source_metadata is not None)
        if len({item.observation_id for item in metadata}) != len(metadata) or len(
            {item.group_id for item in metadata}
        ) != len(metadata):
            raise ValueError("Repeated revisions or sibling source groups require one logical baseline member")
        if sum(len(item.tests) for item in ordered) > 100000:
            raise ValueError("Baseline source exceeds 100,000 declared test cells")
        self._validate_source_clocks(ordered)
        object.__setattr__(self, "builds", ordered)
        return self

    def _validate_source_clocks(self, ordered: tuple[BaselineBuild, ...]) -> None:
        """Validate declared source clocks separately from membership and source-group identity."""
        if self.clock_basis == "source_point_in_time":
            if any(item.source_metadata is None or item.decided_at is None for item in ordered):
                raise ValueError("Point-in-time sources require original decision and outcome clocks")
            decisions = tuple(item.decided_at for item in ordered if item.decided_at is not None)
            if len(set(decisions)) != len(decisions) or decisions != tuple(sorted(decisions)):
                raise ValueError("Point-in-time source decisions must increase strictly")
            for item in ordered:
                if (
                    item.source_metadata is not None
                    and item.decided_at is not None
                    and item.source_metadata.available_at < item.decided_at
                ):
                    raise ValueError("Current target outcome was already available before the declared decision")


class BaselineLimits(Immutable):
    """Common cooperative limits for one native cell, checked between decision phases.

    Elapsed seconds cover the native simulation loop; host/bootstrap elapsed is
    reported separately. Memory is this executable's peak RSS in MiB. Output
    counts actual native/checkpoint/sidecar bytes, excluding shared frozen inputs.
    These trusted-code guards may overshoot during a policy, metric or checkpoint
    operation; they are not OS allocation caps or an arbitrary-code sandbox.
    """

    maximum_elapsed_seconds: Annotated[float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)] = 300.0
    maximum_peak_memory_mib: Annotated[float, Field(strict=True, ge=64, le=8192, allow_inf_nan=False)] = 2048.0
    maximum_output_bytes: Annotated[int, Field(strict=True, ge=1048576, le=1073741824)] = 134217728
    maximum_output_entries: Annotated[int, Field(strict=True, ge=64, le=65536)] = 8192


class BaselinePlan(VersionedContract):
    """Freeze one development fold's candidates, budgets, seeds and exact source before execution.

    V1 is development/model-comparison preparation, never protected confirmation.
    Hyperparameters are fixed by versioned method IDs: UCB c=1, Thompson Beta(1,1),
    discounted UCB gamma=.95/c=1, LinUCB alpha=.5 and Random/UCB portfolio c=1/W=20.
    All candidates get identical source membership, native-work budgets and seeds.
    A seed is a reproducibility label, never an independent project. No I/O occurs;
    unknown schema/method versions fail closed and legacy artifacts are not migrated.
    """

    fold_id: Identifier
    source: BaselineSource
    methods: Annotated[tuple[Method, ...], Field(min_length=1, max_length=len(METHODS))] = METHODS
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1, max_length=64)]
    budgets: Annotated[tuple[Quantity, ...], Field(min_length=1, max_length=16)]
    limits: BaselineLimits = BaselineLimits()
    phase: Literal["development"] = "development"
    feedback: Literal["native_synchronous_budget_censored_RNFail_v1"] = "native_synchronous_budget_censored_RNFail_v1"
    initial_order: Literal["native_seeded_cold_start_except_declared_heuristics_v1"] = (
        "native_seeded_cold_start_except_declared_heuristics_v1"
    )
    randomness: Literal["native_agent_local_policy_generator_v1"] = "native_agent_local_policy_generator_v1"
    cost_projection: Literal["binary32_costs_binary64_prefix_source_binary64_budget_v1"] = (
        "binary32_costs_binary64_prefix_source_binary64_budget_v1"
    )

    @model_validator(mode="after")
    def _design(self) -> Self:
        """Bound total native work and reject duplicate or incompatible repetition dimensions."""
        if len(set(self.methods)) != len(self.methods) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate baseline method or seed")
        if any(b.semantics != BUDGET_RATIO or not 0 < b.value <= 1 for b in self.budgets):
            raise ValueError("Baseline budgets require explicit native-work ratios in (0, 1]")
        values = tuple(b.value for b in self.budgets)
        if len(set(values)) != len(values):
            raise ValueError("Duplicate baseline budget")
        if len(self.methods) * len(self.seeds) * len(values) * len(self.source.builds) > 65536:
            raise ValueError("Baseline plan exceeds 65,536 declared native result cells")
        object.__setattr__(self, "methods", tuple(sorted(self.methods)))
        object.__setattr__(self, "seeds", tuple(sorted(self.seeds)))
        object.__setattr__(self, "budgets", tuple(sorted(self.budgets, key=lambda item: item.value)))
        return self
