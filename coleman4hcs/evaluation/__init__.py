"""
coleman4hcs.evaluation - Evaluation Metrics for COLEMAN.

This module provides classes and methods to evaluate the performance of the COLEMAN framework
in the context of test case prioritization. Various metrics such as NAPFD (Normalized Average
Percentage of Faults Detected) based on errors or verdicts can be utilized to measure the
effectiveness.

Classes
-------
EvaluationMetric
    Base class for all evaluation metrics.
    Defines basic attributes and methods used across all metrics.
NAPFDMetric
    Implements the NAPFD metric based on error counts.
NAPFDVerdictMetric
    Implements the NAPFD metric based on test verdicts (e.g., pass/fail).

Notes
-----
The ``evaluate`` method in ``EvaluationMetric`` is abstract and should be overridden in
child classes. Ensure that the ``reset`` method is called at the beginning of each
evaluation to reset metric values.
"""

from .base import EvaluationMetric
from .napfd import NAPFDMetric, NAPFDVerdictMetric

__all__ = ["EvaluationMetric", "NAPFDMetric", "NAPFDVerdictMetric"]
