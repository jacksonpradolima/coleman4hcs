"""Explain candidate incompatibility before running a native CI baseline matrix."""

from __future__ import annotations

import math
import struct
from typing import Annotated

from pydantic import Field

from coleman.corpus.baseline_contracts import BaselineBuild, BaselineFeature, BaselinePlan, Method
from coleman.results.domain import Identifier
from coleman.selection.contracts import Immutable


class BaselineEligibility(Immutable):
    """One method's native compatibility, with reasons instead of a fabricated worst score.

    Compatibility is for the declared development replay, not scientific source
    admission. Incompatible entries retain every seed/budget in the final matrix.
    The reasons are stable machine-readable codes; contextual features and original
    order must cover the complete universe, not a favorable subset of tests.
    """

    method: Method
    reasons: Annotated[tuple[Identifier, ...], Field(max_length=16)]

    @property
    def compatible(self) -> bool:
        """Return true only when all declared native prerequisites are met."""
        return not self.reasons


def _native_cost(value: float) -> float | None:
    """Check finite binary32 projection without interpreting it as exact source-decimal work."""
    try:
        projected = struct.unpack("!f", struct.pack("!f", value))[0]
        return projected if math.isfinite(projected) else None
    except (OverflowError, struct.error):
        return None


def _available(feature: BaselineFeature, decided_at, point_in_time: bool) -> bool:
    """Keep missing/unknown feature clocks distinct from assumed ordinal availability."""
    return feature.measurement.status == "observed" and (
        not point_in_time
        or (feature.available_at is not None and decided_at is not None and feature.available_at <= decided_at)
    )


def _source_reasons(plan: BaselinePlan) -> tuple[str, ...]:
    """Reject unsupported native truth/costs and label clocks contradicting immediate feedback."""
    reasons: set[str] = set()
    for build in plan.source.builds:
        reasons.update(_build_work_reasons(build))
    for previous, following in zip(plan.source.builds, plan.source.builds[1:], strict=False):
        if (
            previous.source_metadata is not None
            and following.decided_at is not None
            and previous.source_metadata.available_at > following.decided_at
        ):
            reasons.add("native_immediate_feedback_contradicts_known_late_labels")
    return tuple(sorted(reasons))


def _context_reasons(plan: BaselinePlan) -> tuple[str, ...]:
    """Require the same complete finite feature vector at each actual decision frontier."""
    definitions = set()
    for build in plan.source.builds:
        for test in build.tests:
            if not test.context:
                return ("context_features_missing",)
            for feature in test.context:
                if not _available(feature, build.decided_at, plan.source.clock_basis == "source_point_in_time"):
                    return ("context_features_unavailable_at_decision",)
            definitions.add(
                tuple(
                    (
                        f.measurement.name,
                        f.measurement.unit,
                        f.measurement.meaning,
                        f.measurement.direction,
                        f.measurement.minimum,
                    )
                    for f in test.context
                )
            )
    return () if len(definitions) == 1 else ("context_feature_semantics_differ_across_source",)


def baseline_eligibility(plan: BaselinePlan) -> tuple[BaselineEligibility, ...]:
    """Return every frozen candidate with its compatibility before any execution or I/O.

    V1 deliberately retains native seeded cold starts and RNFail's zero reward for
    unselected actions. Heuristic observed-frequency denominators instead count
    only genuinely observed feedback, never treating an unseen test as passing.
    Cost heuristics can learn selected prior costs; missing cold-start/new-test
    estimates use an explicit stable-ID fallback, not zero cost. The external
    DANTE component is retained as unqualified until its separate source/runtime,
    class-unit mapping and point-in-time relevance/cost inputs are bound. Its
    original metric summaries cannot be silently compared to native APFDc.
    """
    plan = BaselinePlan.model_validate(plan)
    common = _source_reasons(plan)
    answer = []
    for method in plan.methods:
        reasons = list(common)
        if method == "original_order_v1" and any(b.original_order is None for b in plan.source.builds):
            reasons.append("meaningful_original_order_missing")
        if method == "linucb_alpha05_v1":
            reasons.extend(_context_reasons(plan))
        if method == "external_dante_prioritization":
            reasons.append("external_component_runtime_and_same_universe_inputs_not_qualified")
        answer.append(BaselineEligibility(method=method, reasons=tuple(sorted(set(reasons)))))
    return tuple(answer)


def _build_work_reasons(build: BaselineBuild) -> set[str]:
    """Validate one complete observed cost universe without changing its declared source membership."""
    reasons: set[str] = set()
    total = 0.0
    projected_total = 0.0
    for test in build.tests:
        if test.duration.status != "observed" or test.failure.status != "observed":
            reasons.add("native_requires_complete_observed_duration_and_binary_outcome")
        if test.duration.value is not None:
            projected = _native_cost(test.duration.value)
            if projected is None:
                reasons.add("native_binary32_duration_overflow")
            else:
                projected_total += projected
            total += test.duration.value
    if total <= 0 or not math.isfinite(total):
        reasons.add("native_requires_finite_positive_full_suite_work")
    if projected_total <= 0 or not math.isfinite(projected_total):
        reasons.add("native_requires_finite_positive_projected_work")
    return reasons
