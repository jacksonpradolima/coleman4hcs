"""
Unit tests for reward functions in the coleman4hcs.reward module.

This module provides unit tests for the TimeRankReward and RNFailReward classes,
which are part of the multi-armed bandit framework for test case prioritization.
The tests cover the following aspects:

- Correctness of reward function outputs for varying scenarios.
- Handling of edge cases such as no detections.
- Proper representation and naming of reward classes
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from coleman4hcs.evaluation import EvaluationMetric
from coleman4hcs.reward import TimeRankReward, RNFailReward


@pytest.fixture
def mock_evaluation_metric():
    """
    Provides a mock evaluation metric for testing reward functions.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 3, 5]  # Failures detected at these ranks
    mock_metric.scheduled_testcases = ["Test1", "Test2", "Test3", "Test4", "Test5"]
    mock_metric.detected_failures = True
    return mock_metric


@pytest.fixture
def mock_empty_evaluation_metric():
    """
    Provides an empty mock evaluation metric for edge case testing.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = []
    mock_metric.scheduled_testcases = ["Test1", "Test2", "Test3", "Test4", "Test5"]
    mock_metric.detected_failures = False
    return mock_metric


@pytest.fixture
def sample_prioritization():
    """
    Provides a sample prioritization list.
    """
    return ["Test1", "Test2", "Test3", "Test4", "Test5"]


def test_time_rank_reward_name():
    """
    Test the string representation and name of TimeRankReward.
    """
    reward = TimeRankReward()
    assert str(reward) == "Time-ranked Reward"
    assert reward.get_name() == "timerank"


def test_time_rank_reward_evaluation(mock_evaluation_metric, sample_prioritization):
    """
    Test the evaluation method of TimeRankReward.
    """
    reward = TimeRankReward()
    results = reward.evaluate(mock_evaluation_metric, sample_prioritization)
    print("Scheduled Testcases:", mock_evaluation_metric.scheduled_testcases)
    print("Results:", results)
    # Updated expected results to reflect the cumulative logic
    assert np.allclose(results, [1.0, 0.3333333333, 1.0, 0.6666666667, 1.0])


@pytest.mark.parametrize("detection_ranks, expected", [
    ([1, 2], [1.0, 1.0, 1.0, 1.0, 1.0]),  # Updated expectation for cumulative logic
    ([3], [0.0, 0.0, 1.0, 1.0, 1.0]),  # Adjusted expectation
    ([], [0.0, 0.0, 0.0, 0.0, 0.0])  # No detections result in zero rewards
])
def test_time_rank_reward_varied_detections(detection_ranks, expected, sample_prioritization):
    """
    Test TimeRankReward with varied detection ranks.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = detection_ranks
    mock_metric.scheduled_testcases = ["Test1", "Test2", "Test3", "Test4", "Test5"]
    reward = TimeRankReward()
    results = reward.evaluate(mock_metric, sample_prioritization)
    assert np.allclose(results, expected)


def test_rn_fail_reward_name():
    """
    Test the string representation and name of RNFailReward.
    """
    reward = RNFailReward()
    assert str(reward) == "Reward Based on Failures"
    assert reward.get_name() == "RNFail"


def test_rn_fail_reward_evaluation(mock_evaluation_metric, sample_prioritization):
    """
    Test the evaluation method of RNFailReward.
    """
    reward = RNFailReward()
    results = reward.evaluate(mock_evaluation_metric, sample_prioritization)
    assert np.allclose(results, [1.0, 0.0, 1.0, 0.0, 1.0])


def test_rn_fail_reward_no_failures(mock_empty_evaluation_metric, sample_prioritization):
    """
    Test RNFailReward evaluation with no failures.
    """
    reward = RNFailReward()
    results = reward.evaluate(mock_empty_evaluation_metric, sample_prioritization)
    assert np.allclose(results, [0.0] * len(sample_prioritization))


@pytest.mark.benchmark(group="reward")
@pytest.mark.parametrize("num_testcases", [100, 1000, 10000])
def test_time_rank_reward_performance(benchmark, num_testcases):
    """
    Performance test for TimeRankReward evaluation method.
    This test evaluates the performance of the reward function for larger datasets.
    """
    # Mocking a large dataset
    detection_ranks = list(range(1, num_testcases + 1, 2))  # Failures at odd indices
    scheduled_testcases = [f"Test{i}" for i in range(1, num_testcases + 1)]
    sample_prioritization = scheduled_testcases  # Assume prioritization is the same as scheduling

    # Mocking EvaluationMetric
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = detection_ranks
    mock_metric.scheduled_testcases = scheduled_testcases

    reward = TimeRankReward()

    # Benchmark the evaluation method
    def run_evaluation():
        reward.evaluate(mock_metric, sample_prioritization)

    benchmark(run_evaluation)


@pytest.mark.benchmark(group="reward")
@pytest.mark.parametrize("num_testcases", [100, 1000, 10000])
def test_rn_fail_reward_performance(benchmark, num_testcases):
    """
    Performance test for RNFailReward evaluation method.
    This test evaluates the performance of the reward function for larger datasets.
    """
    # Mocking a large dataset
    detection_ranks = list(range(1, num_testcases + 1, 2))  # Failures at odd indices
    scheduled_testcases = [f"Test{i}" for i in range(1, num_testcases + 1)]
    sample_prioritization = scheduled_testcases  # Assume prioritization is the same as scheduling

    # Mocking EvaluationMetric
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = detection_ranks
    mock_metric.scheduled_testcases = scheduled_testcases
    mock_metric.detected_failures = True

    reward = RNFailReward()

    # Benchmark the evaluation method
    def run_evaluation():
        reward.evaluate(mock_metric, sample_prioritization)

    benchmark(run_evaluation)

