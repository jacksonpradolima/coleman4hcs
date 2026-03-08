"""
coleman4hcs.bandit - Multi-Armed Bandit Models for Test Case Prioritization.

This module provides implementations of different multi-armed bandit (MAB) models to be used in
the context of software testing. Specifically, it provides a general Bandit class, a dynamic
version that allows for the addition and removal of arms, and an extension that incorporates
evaluation metrics to provide feedback on the selected arms.

Classes
-------
Bandit
    Represents the classic multi-armed bandit model.
DynamicBandit
    An extension of the basic Bandit that supports dynamic management of its arms.
EvaluationMetricBandit
    A dynamic bandit that provides feedback based on a given evaluation metric.

Notes
-----
The module facilitates the use of MAB models for test case prioritization in software testing
scenarios. By integrating feedback from evaluation metrics, it allows for adaptive
prioritization strategies that can react to changes in the testing environment or system
under test.
"""

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from coleman4hcs.evaluation import EvaluationMetric

#: Schema for the bandit arms DataFrame.
#: Columns follow the standard test-case fieldnames used throughout the framework.
BANDIT_SCHEMA: dict = {
    "Name": pl.Utf8,
    "Duration": pl.Float32,
    "CalcPrio": pl.Int32,
    "LastRun": pl.Utf8,
    "NumRan": pl.Int64,
    "NumErrors": pl.Int64,
    "Verdict": pl.Int8,
    "LastResults": pl.Utf8,
}


class Bandit(ABC):
    """Represents a multi-armed bandit model.

    Parameters
    ----------
    arms : list of dict
        The arms of the bandit (test case records). Required columns are
        defined by ``tc_fieldnames``.

    Attributes
    ----------
    tc_fieldnames : list of str
        Column names expected in the arms data.
    arms : polars.DataFrame or None
        DataFrame containing the current arms of the bandit.
    """

    def __init__(self, arms: list[dict]):
        """Initialize a Bandit with its arms.

        Parameters
        ----------
        arms : list of dict
            The arms of the bandit (test case records). Required columns are
            defined by ``tc_fieldnames``.
        """
        # ColName | Description
        # Name | Unique numeric identifier of the test case
        # Duration | Approximated runtime of the test case
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        # NumRan | Test runs
        # NumErrors | Test errors revealed
        # LastResults | List of previous test results(Failed: 1, Passed: 0), ordered by ascending age
        self.tc_fieldnames = [
            "Name",
            "Duration",
            "CalcPrio",
            "LastRun",
            "NumRan",
            "NumErrors",
            "Verdict",
            "LastResults",
        ]

        self.arms: pl.DataFrame = pl.DataFrame(schema=BANDIT_SCHEMA)
        self.reset()
        self.add_arms(arms)

    def reset(self):
        """Reset the arms to an empty DataFrame."""
        self.arms = pl.DataFrame(schema=BANDIT_SCHEMA)

    def get_arms(self) -> list[str]:
        """Retrieve the list of arm names currently associated with the bandit.

        Returns
        -------
        list of str
            List of arm names (test cases).
        """
        return self.arms["Name"].to_list()

    def add_arms(self, arms: list[dict]):
        """Add one or multiple arms to the bandit.

        Parameters
        ----------
        arms : list of dict
            List of arms to add.
        """
        if arms:
            last_results = [
                str(arm.get("LastResults"))
                if isinstance(arm.get("LastResults"), list)
                else (arm.get("LastResults") or "")
                for arm in arms
            ]
            data = {field: [arm.get(field) for arm in arms] for field in BANDIT_SCHEMA if field != "LastResults"}
            data["LastResults"] = last_results
            new_arms = pl.DataFrame(data, schema=BANDIT_SCHEMA)
            self.arms = pl.concat([self.arms, new_arms], how="vertical")

    @abstractmethod
    def pull(self, action):
        """Simulate pulling an arm.

        To be implemented by subclasses.

        Parameters
        ----------
        action : list
            The action (prioritized test suite) to pull.
        """
        return NotImplementedError("You must to implemented this function")

    def update_priority(self, action):
        """Update the Priority column with the priorities.

        Parameters
        ----------
        action : list of str
            List of test cases in order of prioritization.
        """
        action_map = {name: priority for priority, name in enumerate(action, start=1)}
        priorities = np.vectorize(action_map.get)(self.arms["Name"].to_numpy())
        self.arms = self.arms.with_columns([pl.Series("CalcPrio", priorities, dtype=pl.Int32)])


class DynamicBandit(Bandit, ABC):
    """A Bandit that allows dynamic management of its arms.

    Extends the base Bandit to support updating the set of arms at runtime.
    """

    def update_arms(self, arms: list[dict]):
        """Update the arms of the bandit.

        Parameters
        ----------
        arms : list of dict
            The new set of arms to replace the current ones.
        """
        # I can replace all arms because the bandit don't need to maintain a "history"
        # The agent needs to maintain the "history"
        self.reset()

        # Add new arms
        self.add_arms(arms)


class EvaluationMetricBandit(DynamicBandit):
    """A Dynamic Bandit that provides feedback based on an evaluation metric.

    Parameters
    ----------
    arms : list of dict
        The arms of the bandit (test case records).
    evaluation_metric : EvaluationMetric
        The evaluation metric used to provide feedback.

    Attributes
    ----------
    evaluation_metric : EvaluationMetric
        The evaluation metric instance.
    """

    def __init__(self, arms: list[dict], evaluation_metric: EvaluationMetric):
        """Initialize the EvaluationMetricBandit.

        Parameters
        ----------
        arms : list of dict
            The arms of the bandit (test case records).
        evaluation_metric : EvaluationMetric
            The evaluation metric used to provide feedback.
        """
        super().__init__(arms)
        self.evaluation_metric = evaluation_metric

    def __str__(self):
        """Return a string representation of the bandit.

        Returns
        -------
        str
            String representation of the evaluation metric.
        """
        return str(self.evaluation_metric)

    def pull(self, action):
        """Submit prioritized test set for evaluation and get new measurements.

        Parameters
        ----------
        action : list of str
            The prioritized test suite list.

        Returns
        -------
        EvaluationMetric
            The result ("state") of an evaluation by the evaluation metric.

        Raises
        ------
        ValueError
            If the action list is empty.
        """
        if not action:
            raise ValueError("Action list cannot be empty")

        super().update_priority(action)

        # After, we need to order the test cases based on the priorities
        # Sort tc by Prio ASC (for backwards scheduling)
        sorted_indices = self.arms["CalcPrio"].to_numpy().argsort(kind="stable")
        self.arms = self.arms[[int(i) for i in sorted_indices]]

        self.evaluation_metric.evaluate(self.arms.to_dicts())

        return self.evaluation_metric
