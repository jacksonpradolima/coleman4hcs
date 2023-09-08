from abc import ABC, abstractmethod
from typing import List

import numpy as np

from coleman4hcs.evaluation import EvaluationMetric


class Reward(ABC):
    """
    A reward function is used by the agent in the observe method
    """

    def get_name(self):
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

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        # number of test cases which failed
        detected_failures = len(reward.detection_ranks)

        if not detected_failures:
            return [0.0] * len(last_prioritization)

        rank_idx = np.array(reward.detection_ranks) - 1
        rewards = np.zeros(len(reward.scheduled_testcases))
        rewards[rank_idx] = 1
        rewards = np.cumsum(rewards)  # Rewards for passed testcases
        rewards[rank_idx] = detected_failures  # Rewards for failed testcases

        return [rewards[reward.scheduled_testcases.index(tc)] if tc in reward.scheduled_testcases else 0.0 for tc in
                last_prioritization]


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
