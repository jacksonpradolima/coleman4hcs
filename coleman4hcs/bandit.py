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

import numpy as np
import polars as pl

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

        self.arms = None
        self.reset()
        self.add_arms(arms)

    def reset(self):
        """
        Reset the arms.
        """
        schema = {
            'Name': pl.Utf8,
            'Duration': pl.Float32,
            'CalcPrio': pl.Int32,
            'LastRun': pl.Utf8,
            'NumRan': pl.Int64,
            'NumErrors': pl.Int64,
            'Verdict': pl.Int8,
            'LastResults': pl.Utf8
        }
        self.arms = pl.DataFrame(schema=schema)

    def get_arms(self) -> List[str]:
        """
        Retrieve the list of arm names (Test Cases) currently associated with the bandit.

        :return: List of arm names.
        """
        return self.arms['Name'].to_list()

    def add_arms(self, arms: List[Dict]):
        """
        Add one or multiple arms to the bandit.

        :param arms: List of arms.
        """
        if arms:
            # Convert list values to strings to match schema
            processed_arms = []
            for arm in arms:
                processed_arm = arm.copy()
                if 'LastResults' in processed_arm and isinstance(processed_arm['LastResults'], list):
                    processed_arm['LastResults'] = str(processed_arm['LastResults'])
                processed_arms.append(processed_arm)
            
            new_arms = pl.DataFrame(processed_arms, schema=self.arms.schema)
            self.arms = pl.concat([self.arms, new_arms], how="vertical")

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
        action_map = {name: priority for priority, name in enumerate(action, start=1)}
        priorities = np.vectorize(action_map.get)(self.arms['Name'].to_numpy())
        self.arms = self.arms.with_columns([
            pl.Series('CalcPrio', priorities, dtype=pl.Int32)
        ])


class DynamicBandit(Bandit, ABC):
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
        if not action:
            raise ValueError("Action list cannot be empty")

        super().update_priority(action)

        # After, we need to order the test cases based on the priorities
        # Sort tc by Prio ASC (for backwards scheduling)
        sorted_indices = self.arms['CalcPrio'].to_numpy().argsort(kind='stable')
        self.arms = self.arms[[int(i) for i in sorted_indices]]

        self.evaluation_metric.evaluate(self.arms.to_dicts())

        return self.evaluation_metric
