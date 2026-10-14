"""Installed-package synthetic resource coverage; no service, learning or financial claim."""

import json
import sys
from decimal import Decimal
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.results.domain import fingerprint
from coleman.selection import (
    CoverageCandidate,
    CoverageConstraints,
    CoverageOutcome,
    CoverageProblem,
    Quantity,
    SearchLimits,
    UnitSemantics,
    publish_coverage_result,
    solve_coverage,
)


def main(output):
    """Reserve synthetic runner slots to cover overlapping CI worker capabilities."""
    workspace = Path.cwd().resolve()
    root = (workspace / output).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("Example output must be within the current working directory")
    cost = UnitSemantics(unit="slot", meaning="synthetic-runner-slot-reservation-v1")
    weight = UnitSemantics(unit="1", meaning="synthetic-worker-capability-weight-v1")
    candidates = [("A", 2, ("x", "y")), ("B", 2, ("y", "z")), ("C", 1, ("z",))]
    request = CoverageProblem(
        schema_version=1,
        data_fingerprint=fingerprint(
            {
                "fixture": "synthetic-ci-runner-capabilities-v1",
                "candidates": [
                    {"id": name, "cost": str(cost), "covers": list(covers)} for name, cost, covers in candidates
                ],
                "weights": {"x": "0.5", "y": "0.3", "z": "0.2"},
            }
        ),
        objective_semantics=weight,
        candidates=tuple(
            CoverageCandidate(candidate_id=name, cost=Quantity(value=Decimal(slots), semantics=cost), covers=covers)
            for name, slots, covers in candidates
        ),
        outcomes=tuple(
            CoverageOutcome(outcome_id=name, weight=Quantity(value=Decimal(value), semantics=weight))
            for name, value in (("x", "0.5"), ("y", "0.3"), ("z", "0.2"))
        ),
        constraints=CoverageConstraints(budget=Quantity(value=Decimal(3), semantics=cost)),
    )
    result = solve_coverage(request, method="branch_bound_v1", limits=SearchLimits(deadline_seconds=None))
    path = publish_coverage_result(result, ArtifactWriter(root))
    print(json.dumps({"selected_ids": result.selected_ids, "status": result.status, "artifact": str(path)}))


if __name__ == "__main__":
    main(sys.argv[1])
