"""
coleman4hcs.reward
------------------

Defines reward functions for agents in a multi-armed bandit framework in the context of software testing.
These reward functions help agents to prioritize software test cases based on various strategies.

The module provides an abstract base class `Reward` that serves as a blueprint for all reward functions.
Derived classes implement specific reward strategies based on the number of failures and the order
of test cases.

Classes:
    - Reward: An abstract base class that defines the structure and interface of a reward function.
    - TimeRankReward: A reward function that considers the order of test cases and the number of failures.
      It rewards each test case based on its rank in the test schedule and its pass/fail status.
    - RNFailReward: A reward function that rewards based on the number of failures associated with test cases.

Notes:
    - Reward functions are essential components of the bandit-based test case prioritization framework.
      They guide agents to make better decisions about which test cases to prioritize.
    - Ensure that the evaluation metric provides necessary details like detection ranks for the
      reward functions to work correctly.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from coleman4hcs.evaluation import EvaluationMetric


class Reward(ABC):
    """
    A reward function is used by the agent in the observe method
    """

    def get_name(self):
        """
        Retrieve the name or identifier of the reward function.

        The name is used for identification purposes, such as logging or display in results.

        :return: The name or identifier of the reward function.
        :rtype: str
        """
        return NotImplementedError

    @abstractmethod
    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        """
        The reward function evaluate a bandit result and return a reward
        """
        return NotImplementedError


class TimeRankReward(Reward):
    """
    Time-ranked Reward (TimeRank)

    This reward function explicitly includes the order of test cases and rewards each test case
    based on its rank in the test schedule and whether it failed.
    As a good schedule executes failing test cases early,
    every passed test case reduces the schedule's quality if it precedes a failing test case.
    Each test cases is rewarded by the total number of failed test cases,
    for failed test cases it is the same as reward function 'RNFailReward'.
    For passed test cases, the reward is further decreased by the number of failed test cases ranked
    after the passed test case to penalize scheduling passing test cases early.
    """

    def __str__(self):
        return 'Time-ranked Reward'

    def get_name(self):
        return 'timerank'

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list):
        """
        Evaluate rewards based on the prioritization rank of test cases.

        :param reward: The evaluation metric containing detection ranks and scheduled test cases.
        :param last_prioritization: The list of test case names in the prioritization order.
        :return: A list of rewards for each test case in the prioritization.
        """
        # Total number of failing test cases
        num_failing_tests = len(reward.detection_ranks)
        if num_failing_tests == 0:
            # No failing test cases: All rewards are 0
            return [0.0] * len(last_prioritization)

        # Detection ranks (convert to 0-based index)
        failing_indices = np.array(reward.detection_ranks) - 1

        # Rewards for scheduled test cases
        rewards = np.zeros(len(reward.scheduled_testcases))
        rewards[failing_indices] = 1  # Assign reward for failing test cases
        rewards = np.cumsum(rewards)  # Accumulate rewards for passing test cases
        rewards[failing_indices] = num_failing_tests  # Assign full reward to failing test cases

        # Normalize the rewards to scale them between 0 and 1.
        # This ensures that the rewards are proportionate to the number of failing test cases,
        # aligning with the equation's intent to penalize non-failing test cases scheduled before failing ones.
        normalized_rewards = [
            rewards[reward.scheduled_testcases.index(tc)] / num_failing_tests
            if tc in reward.scheduled_testcases else 0.0
            for tc in last_prioritization
        ]

        return normalized_rewards


class RNFailReward(Reward):
    """
    Reward Based on Failures (RNFail)

    This reward function is based on the number of failures associated with test cases t' in T':
    1 if t' failed; 0 otherwise
    """

    def __str__(self):
        return 'Reward Based on Failures'

    def get_name(self):
        return 'RNFail'

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        if not reward.detected_failures:
            return [0.0] * len(last_prioritization)

        rank_idx = np.array(reward.detection_ranks) - 1
        rewards = np.zeros(len(reward.scheduled_testcases))
        rewards[rank_idx] = 1

        return [rewards[reward.scheduled_testcases.index(tc)] if tc in reward.scheduled_testcases else 0.0 for tc in
                last_prioritization]
