"""FRRMAB policy."""

import numpy as np
import polars as pl

from coleman4hcs.agent import HISTORY_SCHEMA, Agent, RewardSlidingWindowAgent

from .base import Policy


class FRRMABPolicy(Policy):
    """Fitness-Rate-Rank based Multi-Armed Bandit (FRRMAB) policy.

    Parameters
    ----------
    c : float
        Exploration parameter.
    decayed_factor : float, optional
        Decay factor for ranking. Default is 1.

    Attributes
    ----------
    c : float
        Exploration parameter.
    decayed_factor : float
        Decay factor for ranking.
    history : polars.DataFrame
        History of actions and their outcomes.
    """

    def __init__(self, c, decayed_factor=1):
        """Initialize the FRRMABPolicy.

        Parameters
        ----------
        c : float
            Exploration parameter.
        decayed_factor : float, optional
            Decay factor for ranking. Default is 1.
        """
        self.c = c
        self.decayed_factor = decayed_factor
        self.history = pl.DataFrame(schema=HISTORY_SCHEMA)

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C and D values.
        """
        return f"FRRMAB (C={self.c}, D={self.decayed_factor}"

    def choose_all(self, agent: Agent):
        """Choose all actions based on Q values from the FRRMAB history.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        existing_names = set(self.history["Name"].to_list())
        agent_names = set(agent.actions["Name"].to_list())
        new_names = agent_names - existing_names

        if new_names:
            new_entries = pl.DataFrame(
                {
                    "Name": list(new_names),
                    "ActionAttempts": [0.0] * len(new_names),
                    "ValueEstimates": [0.0] * len(new_names),
                    "Q": [0.0] * len(new_names),
                    "T": [0] * len(new_names),
                },
                schema=HISTORY_SCHEMA,
            )
            self.history = pl.concat([self.history, new_entries], how="vertical")

        return self.history.sort("Q", descending=True)["Name"].to_list()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """Assign credit using the Fitness-Rate-Rank method.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent for which credit assignment is to be performed.
        """
        self.history = agent.history.group_by("Name").agg(
            [
                pl.col("ActionAttempts").sum(),
                pl.col("ValueEstimates").sum(),
                pl.col("T").count().cast(HISTORY_SCHEMA["T"]).alias("T"),
            ]
        )

        self.history = self.history.sort("ValueEstimates", descending=True)
        reward_arm = self.history["ValueEstimates"].to_numpy()
        ranking = np.arange(1, len(reward_arm) + 1)

        decay_values = np.power(self.decayed_factor, ranking) * reward_arm

        decay_total = decay_values.sum()
        frr = np.zeros_like(decay_values, dtype=float) if np.isclose(decay_total, 0.0) else decay_values / decay_total

        selected_times = self.history["T"].to_numpy()

        log_selected_times = np.log1p(selected_times.sum())

        exploration = np.sqrt((2 * log_selected_times) / selected_times)
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)
        self.history = self.history.with_columns(
            pl.Series("frr", frr),
            pl.Series("exploration", exploration),
        )
        self.history = (
            self.history.with_columns((pl.col("frr") + self.c * pl.col("exploration")).alias("Q"))
            .drop(["frr", "exploration"])
            .select(list(HISTORY_SCHEMA.keys()))
        )
