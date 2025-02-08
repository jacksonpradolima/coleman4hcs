"""
Unit tests for the coleman4hcs evaluation module.

This test suite validates the correctness and robustness of the evaluation metrics
implemented in the coleman4hcs library, including NAPFDMetric and NAPFDVerdictMetric.

Tests cover various scenarios, including:
- Standard test cases with faults and verdicts.
- Handling of edge cases like empty records and no failures.
- Common behaviors such as string representation and abstract class enforcement.
- Metrics computation with records having identical durations or results.

Fixtures:
- `sample_records`: Provides a set of sample test cases with varying durations, errors, and verdicts.
- `available_time`: Computes the total available time from the sample records.

Constants:
- `NAPFD_FITNESS_NON_NEGATIVE`: Ensures that the NAPFD fitness value is non-negative.
- `NAPFD_FITNESS_NOT_EXCEED_ONE`: Ensures that the NAPFD fitness value does not exceed 1.
- `NAPFD_COST_NON_NEGATIVE`: Ensures that the NAPFD cost value is non-negative.

Helper Functions:
- `_common_test_napfd`: A shared utility to test NAPFDMetric across multiple scenarios.

Coverage:
- Ensures all methods, including edge cases, are well-tested to maintain high reliability.

Usage:
Run the tests using pytest to verify the functionality of evaluation metrics.
"""
import pytest
from coleman4hcs.evaluation import NAPFDMetric, NAPFDVerdictMetric, EvaluationMetric

# Constants for error messages
NAPFD_FITNESS_NON_NEGATIVE = "NAPFD fitness should be non-negative."
NAPFD_FITNESS_NOT_EXCEED_ONE = "NAPFD fitness should not exceed 1."
NAPFD_COST_NON_NEGATIVE = "NAPFD cost should be non-negative."


@pytest.fixture
def sample_records():
    """
    Provide sample test records for evaluation metrics.
    """
    return [
        {'Name': 8, 'Duration': 0.001, 'NumRan': 1, 'NumErrors': 3, 'Verdict': 1},
        {'Name': 9, 'Duration': 0.497, 'NumRan': 1, 'NumErrors': 1, 'Verdict': 1},
        {'Name': 4, 'Duration': 0.188, 'NumRan': 3, 'NumErrors': 2, 'Verdict': 1},
        {'Name': 6, 'Duration': 0.006, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        {'Name': 3, 'Duration': 0.006, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        {'Name': 1, 'Duration': 0.235, 'NumRan': 2, 'NumErrors': 0, 'Verdict': 0},
        {'Name': 2, 'Duration': 5.704, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        {'Name': 5, 'Duration': 3.172, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0},
        {'Name': 7, 'Duration': 0.034, 'NumRan': 1, 'NumErrors': 5, 'Verdict': 1}
    ]


@pytest.fixture
def available_time(sample_records):
    """
    Calculate total available time from sample records.
    """
    return sum(item['Duration'] for item in sample_records)


def test_evaluation_metric_not_implemented():
    """
    Test that the abstract EvaluationMetric class raises NotImplementedError.
    """
    metric = EvaluationMetric()
    with pytest.raises(NotImplementedError):
        metric.evaluate([])


def test_evaluation_metric_str():
    """
    Test that the __str__ method is implemented for derived metric classes.
    """
    napfd = NAPFDMetric()
    napfd_v = NAPFDVerdictMetric()

    assert str(napfd) == "NAPFD", "NAPFDMetric __str__ method failed."
    assert str(napfd_v) == "NAPFDVerdict", "NAPFDVerdictMetric __str__ method failed."


def test_napfd_metric(sample_records, available_time):
    """
    Test NAPFDMetric with standard records and 50% available time.
    """
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate(sample_records)

    assert napfd.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd.cost >= 0, NAPFD_COST_NON_NEGATIVE


def test_napfd_verdict_metric(sample_records, available_time):
    """
    Test NAPFDVerdictMetric with standard records and 50% available time.
    """
    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)
    napfd_v.evaluate(sample_records)

    assert napfd_v.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd_v.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd_v.cost >= 0, NAPFD_COST_NON_NEGATIVE


def test_empty_records(available_time):
    """
    Test metrics with empty records to ensure proper handling.
    """
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate([])

    assert napfd.fitness == 1, "NAPFD fitness should be 1 for empty records."
    assert napfd.cost == 1, "NAPFD cost should be 1 for empty records."


def test_napfd_verdict_metric_no_failures(available_time):
    """
    Test NAPFDVerdictMetric with no failures to ensure default metrics are set.
    """
    no_failure_records = [
        {'Name': i, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 0} for i in range(1, 10)
    ]

    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)
    napfd_v.evaluate(no_failure_records)

    assert napfd_v.fitness == 1, "NAPFD-V fitness should be 1 when no failures are present."
    assert napfd_v.cost == 1, "NAPFD-V cost should be 1 when no failures are present."


def _common_test_napfd(records, available_time):
    """
    Common helper function to test NAPFDMetric.
    """
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate(records)

    assert napfd.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd.cost >= 0, NAPFD_COST_NON_NEGATIVE


def test_identical_durations(available_time):
    """
    Test metrics with records having identical durations.
    """
    identical_records = [
        {'Name': i, 'Duration': 1, 'NumRan': 1, 'NumErrors': i % 2, 'Verdict': i % 2} for i in range(1, 10)
    ]
    _common_test_napfd(identical_records, available_time)


def test_identical_cost_and_results(available_time):
    """
    Test metrics where all records have the same cost and results.
    """
    identical_cost_records = [
        {'Name': i, 'Duration': 1, 'NumRan': 1, 'NumErrors': 0, 'Verdict': 1} for i in range(1, 10)
    ]
    _common_test_napfd(identical_cost_records, available_time)


@pytest.mark.benchmark(group="evaluation")
def test_benchmark_napfd_metric(benchmark, sample_records, available_time):
    """
    Benchmark the performance of NAPFDMetric with a large dataset.
    """
    large_dataset = sample_records * 10_000  # Simulate a large dataset
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)

    # Benchmark the evaluation process
    benchmark(napfd.evaluate, large_dataset)


@pytest.mark.benchmark(group="evaluation")
def test_benchmark_napfd_verdict_metric(benchmark, sample_records, available_time):
    """
    Benchmark the performance of NAPFDVerdictMetric with a large dataset.
    """
    large_dataset = sample_records * 10_000  # Simulate a large dataset
    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)

    # Benchmark the evaluation process
    benchmark(napfd_v.evaluate, large_dataset)
