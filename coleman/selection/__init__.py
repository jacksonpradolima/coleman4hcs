"""Typed finite weighted coverage selection, exact certificates and native artifact adapters."""

from coleman.selection.artifacts import coverage_record, publish_coverage_result
from coleman.selection.contracts import (
    CoverageCandidate,
    CoverageConstraints,
    CoverageOutcome,
    CoverageProblem,
    Quantity,
    UnitSemantics,
)
from coleman.selection.resources import SearchLimits
from coleman.selection.results import CoverageCheckpoint, CoverageResult
from coleman.selection.solver import solve_coverage

__all__ = [
    "CoverageCandidate",
    "CoverageCheckpoint",
    "CoverageConstraints",
    "CoverageOutcome",
    "CoverageProblem",
    "CoverageResult",
    "Quantity",
    "SearchLimits",
    "UnitSemantics",
    "coverage_record",
    "publish_coverage_result",
    "solve_coverage",
]
