"""RewardAgent - agent that uses a reward function."""

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

        reward_map = dict(zip(self.last_prioritization, self.last_reward, strict=False))

        current_estimates = self.actions["ValueEstimates"].to_list()
        name_list = self.actions["Name"].to_list()
        new_estimates = [current_estimates[i] + reward_map.get(name_list[i], 0.0) for i in range(len(name_list))]

        self.actions = self.actions.with_columns([pl.Series("ValueEstimates", new_estimates)])

        self.t += 1

        self.policy.credit_assignment(self)
