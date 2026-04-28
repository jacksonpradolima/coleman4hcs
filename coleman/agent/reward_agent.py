"""RewardAgent - agent that uses a reward function."""

import numpy as np
import polars as pl

from coleman.evaluation import EvaluationMetric

from .base import Agent


class RewardAgent(Agent):
    """An agent that learns using a reward function.

    Parameters
    ----------
    policy : object
        The policy used by the agent to choose an action.
    reward_function : object
        The reward function used by the agent to evaluate outcomes.

    Attributes
    ----------
    reward_function : object
        The reward function used by the agent.
    last_reward : float
        The last reward received by the agent.
    """

    def __init__(self, policy, reward_function, seed: int | None = None):
        """Initialize the RewardAgent.

        Parameters
        ----------
        policy : object
            The policy used by the agent to choose an action.
        reward_function : object
            The reward function used by the agent to evaluate outcomes.
        seed : int, optional
            Seed forwarded to :class:`Agent` for reproducible initial shuffle.
        """
        super().__init__(policy, seed=seed)
        self.reward_function = reward_function
        self.last_reward = 0

    def get_reward_function(self):
        """Retrieve the reward function associated with the agent.

        Returns
        -------
        object
            The reward function of the agent.
        """
        return self.reward_function

    def observe(self, reward: EvaluationMetric):
        """Observe the reward and update value estimates.

        Parameters
        ----------
        reward : EvaluationMetric
            The reward (result) obtained by the evaluation metric.
        """
        self.update_action_attempts()

        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)

        n = len(self.last_prioritization)
        if n > 0:
            idx_buf = np.empty(n, dtype=np.intp)
            r_buf = np.empty(n, dtype=np.float64)
            used = 0
            for i, nm in enumerate(self.last_prioritization):
                idx = self._name_to_idx.get(nm)
                if idx is None:
                    continue
                idx_buf[used] = idx
                r_buf[used] = float(self.last_reward[i])
                used += 1

            if used == 0:
                self.t += 1
                self.policy.credit_assignment(self)
                return

            indices = idx_buf[:used]
            rewards = r_buf[:used]
            values = np.array(self.actions["ValueEstimates"].to_numpy(), dtype=np.float64, copy=True)
            np.add.at(values, indices, rewards)
            self.actions = self.actions.with_columns(pl.Series("ValueEstimates", values))

        self.t += 1

        self.policy.credit_assignment(self)
