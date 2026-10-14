"""Library-first protected temporal evaluation and versioned paired reports."""

from coleman.evaluation.protected.audit import AccessReceipt, EvaluationAudit
from coleman.evaluation.protected.comparison import FamilyReport, evaluate
from coleman.evaluation.protected.contracts import (
    ComparisonHypothesis,
    EvaluationProtocol,
    FrozenCandidate,
    FrozenEvaluation,
    freeze,
    named_seed,
)
from coleman.evaluation.protected.inference import holm_adjust
from coleman.evaluation.protected.observations import ObservationReference, PairedObservation
from coleman.evaluation.protected.temporal import SplitDecision, TemporalObservation, TemporalPlan

__all__ = [
    "AccessReceipt",
    "ComparisonHypothesis",
    "EvaluationAudit",
    "EvaluationProtocol",
    "FamilyReport",
    "FrozenCandidate",
    "FrozenEvaluation",
    "ObservationReference",
    "PairedObservation",
    "SplitDecision",
    "TemporalObservation",
    "TemporalPlan",
    "evaluate",
    "freeze",
    "holm_adjust",
    "named_seed",
]
