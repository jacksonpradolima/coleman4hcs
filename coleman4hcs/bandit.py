"""
coleman4hcs.bandit: Multi-Armed Bandit Models for Test Case Prioritization

This module provides implementations of different multi-armed bandit (MAB) models to be used in
the context of software testing.
Specifically, it provides a general Bandit class, a dynamic version that allows for the addition and removal of arms,
and an extension that incorporates evaluation metrics to provide feedback on the selected arms.

Key Classes:
- `Bandit`: Represents the classic multi-armed bandit model.
- `DynamicBandit`: An extension of the basic Bandit that supports dynamic management of its arms.
- `EvaluationMetricBandit`: A dynamic bandit that provides feedback based on a given evaluation metric.

The module facilitates the use of MAB models for test case prioritization in software testing scenarios.
By integrating feedback from evaluation metrics, it allows for adaptive prioritization strategies that
can react to changes in the testing environment or system under test.

Usage examples and further details can be found in the documentation of individual classes.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from coleman4hcs.evaluation import EvaluationMetric


class Bandit(ABC):
    """
    Represents a multi-armed bandit model.
    """

    def __init__(self, arms: List[Dict]):
        """
        Initialize a Bandit with its arms.

        :param arms: The arms of the bandit (Test Cases record). Required columns are `self.tc_fieldnames`
        """
        # ColName | Description
        # Name | Unique numeric identifier of the test case
        # Duration | Approximated runtime of the test case
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        # NumRan | Test runs
        # NumErrors | Test errors revealed
        # LastResults | List of previous test results(Failed: 1, Passed: 0), ordered by ascending age
        self.tc_fieldnames = ['Name',
                              'Duration',
                              'CalcPrio',
                              'LastRun',
                              'NumRan',
                              'NumErrors',
                              'Verdict',
                              'LastResults']

        self.reset()

        self.add_arms(arms)

        # Convert columns
        self.arms = self.arms.infer_objects()

    def reset(self):
        """
        Reset the arms.
        """

        self.arms = pd.DataFrame(columns=self.tc_fieldnames)

    def get_arms(self) -> List[str]:
        """
        Retrieve the list of arm names (Test Cases) currently associated with the bandit.

        :return: List of arm names.
        """
        return self.arms['Name'].tolist()

    def add_arms(self, arms: List[Dict]):
        """
        Add one or multiple arms to the bandit.

        :param arms: List of arms.
        """
        self.arms = pd.concat([self.arms, pd.DataFrame(arms, columns=self.tc_fieldnames)], ignore_index=True)

    @abstractmethod
    def pull(self, action):
        """
        Simulate pulling an arm. To be implemented by subclasses.
        """

        return NotImplementedError('You must to implemented this function')

    def update_priority(self, action):
        """
        Update the Priority column with the priorities
        :param action: List of test cases in order of prioritization
        """
        self.arms['CalcPrio'] = self.arms['Name'].apply(lambda x: action.index(x) + 1)


class DynamicBandit(Bandit):
    """
    A Bandit that allows dynamic management of its arms.
    """

    def update_arms(self, arms: List[Dict]):
        """
        Update the arms of the bandit.
        """

        # I can replace all arms because the bandit don't need to maintain a "history"
        # The agent needs to maintain the "history"
        self.reset()

        # Add new arms
        self.add_arms(arms)

        # Convert columns
        self.arms = self.arms.infer_objects()


class EvaluationMetricBandit(DynamicBandit):
    """
     A Dynamic Bandit that provides feedback based on an evaluation metric.
    """

    def __init__(self, arms: List[Dict], evaluation_metric: EvaluationMetric):
        """
        :param arms: The arms of the bandit (Test Cases record)
        :param evaluation_metric: Evaluation Metric
        """
        super().__init__(arms)
        self.evaluation_metric = evaluation_metric

    def __str__(self):
        return str(self.evaluation_metric)

    def pull(self, action):
        """
        Submit prioritized test set for evaluation and get new measurements.

        :param action: The Prioritized Test Suite List
        :return: The result ("state") of an evaluation by Evaluation Metric
        """
        super().update_priority(action)

        # After, we need to order the test cases based on the priorities
        # Sort tc by Prio ASC (for backwards scheduling)
        self.arms = self.arms.sort_values(by=['CalcPrio'])

        self.evaluation_metric.evaluate(self.arms.to_dict('records'))

        return self.evaluation_metric
