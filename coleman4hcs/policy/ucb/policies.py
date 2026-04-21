"""UCB policy variants."""

import numpy as np
import polars as pl

from coleman4hcs.agent import Agent

from ..base import Policy


class UCBPolicyBase(Policy):
    """Base class for Upper Confidence Bound (UCB) policies.

    Parameters
    ----------
    c : float
        Exploration parameter controlling the width of the confidence bound.

    Attributes
    ----------
    c : float
        Exploration parameter.
    """

    def __init__(self, c: float):
        """Initialize the UCBPolicyBase.

        Parameters
        ----------
        c : float
            Exploration parameter controlling the width of the confidence bound.
        """
        self.c = c

    def choose_all(self, agent: Agent) -> list:
        """Choose all actions sorted by Q value in descending order.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        return agent.actions.sort("Q", descending=True)["Name"].to_list()


class UCB1Policy(UCBPolicyBase):
    """Upper Confidence Bound algorithm (UCB1).

    Applies an exploration factor to the expected value of each arm which can
    influence a greedy selection strategy to more intelligently explore less
    confident options.
    """

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C value.
        """
        return f"UCB1 (C={self.c})"

    def credit_assignment(self, agent: Agent):
        """Assign credit using the UCB1 formula.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        super().credit_assignment(agent)

        action_attempts = agent.actions["ActionAttempts"].to_numpy()
        quality_estimates = agent.actions["Q"].to_numpy()

        exploration = np.log1p(agent.t) / action_attempts
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)
        exploration = np.power(exploration, 1 / self.c)

        q_values = quality_estimates + exploration
        agent.actions = agent.actions.with_columns([pl.Series("Q", q_values)])


class UCBPolicy(UCBPolicyBase):
    """Upper Confidence Bound algorithm (UCB) with scaling factor."""

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C value.
        """
        return f"UCB (C={self.c})"

    def credit_assignment(self, agent: Agent):
        """Assign credit using the UCB formula with scaling factor.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        super().credit_assignment(agent)

        action_attempts = agent.actions["ActionAttempts"].to_numpy()
        quality_estimates = agent.actions["Q"].to_numpy()

        log_sum_attempts = np.log1p(action_attempts.sum())

        exploration = np.sqrt((2 * log_sum_attempts) / action_attempts)
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)

        q_values = quality_estimates + self.c * exploration
        agent.actions = agent.actions.with_columns([pl.Series("Q", q_values)])
