"""Independent finite-model correctness gates and bounded observational performance reports."""

from coleman.benchmarks.artifacts import publish_benchmark_report
from coleman.benchmarks.bundled import load_suite
from coleman.benchmarks.contracts import AllocationCase, BenchmarkSuite, CoverageCase, ReferenceAnswer
from coleman.benchmarks.gates import GateSummary, check_allocation, check_coverage, reference, same_mathematics
from coleman.benchmarks.report import (
    BenchmarkCheckpoint,
    BenchmarkEnvironment,
    BenchmarkFailure,
    BenchmarkObservation,
    BenchmarkReport,
    collect_environment,
)
from coleman.benchmarks.runner import BenchmarkExecution, BenchmarkRunError, execute_native, run_benchmarks

__all__ = [
    "AllocationCase",
    "BenchmarkCheckpoint",
    "BenchmarkEnvironment",
    "BenchmarkExecution",
    "BenchmarkFailure",
    "BenchmarkObservation",
    "BenchmarkReport",
    "BenchmarkRunError",
    "BenchmarkSuite",
    "CoverageCase",
    "GateSummary",
    "ReferenceAnswer",
    "check_allocation",
    "check_coverage",
    "collect_environment",
    "execute_native",
    "load_suite",
    "publish_benchmark_report",
    "reference",
    "run_benchmarks",
    "same_mathematics",
]
