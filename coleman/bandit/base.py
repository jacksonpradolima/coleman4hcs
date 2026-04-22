"""Base classes and schema for multi-armed bandits."""

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

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
        raise NotImplementedError("You must implement this function")

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
