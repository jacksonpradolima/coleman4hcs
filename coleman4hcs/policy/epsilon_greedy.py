"""Epsilon-greedy policy variants."""

import polars as pl

from coleman4hcs.agent import Agent

from .base import Policy, _rng


class EpsilonGreedyPolicy(Policy):
    """Epsilon-Greedy policy for action selection.

    Chooses a random action with probability epsilon and takes the best
    apparent approach with probability 1-epsilon. If multiple actions are tied
    for best choice, then a random action from that subset is selected.

    Parameters
    ----------
    epsilon : float
        Probability of choosing a random action.

    Attributes
    ----------
    epsilon : float
        Probability of choosing a random action.
    """

    def __init__(self, epsilon):
        """Initialize the EpsilonGreedyPolicy.

        Parameters
        ----------
        epsilon : float
            Probability of choosing a random action.
        """
        self.epsilon = epsilon

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with epsilon value.
        """
        return f"\u03b5-greedy (\u03b5={self.epsilon})"

    def choose_all(self, agent: Agent):
        """Choose all actions using the epsilon-greedy strategy.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names ordered by the epsilon-greedy strategy.
        """
        actions = agent.actions.clone()
        actions = actions.with_columns([pl.Series("is_random", _rng.random(len(actions)) < self.epsilon)])

        actions = actions.sort(["is_random", "Q"], descending=[True, True])

        return actions["Name"].to_list()


class GreedyPolicy(EpsilonGreedyPolicy):
    """Greedy policy that always takes the best apparent action.

    Ties are broken by random selection. This is a special case of
    EpsilonGreedy where epsilon = 0 (always exploit).
    """

    def __init__(self):
        """Initialize the GreedyPolicy with epsilon = 0."""
        super().__init__(0)

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return "Greedy"


class RandomPolicy(Policy):
    """Random policy that randomly selects from all available actions.

    No consideration is given to which action is apparently best. This is a
    special case of EpsilonGreedy where epsilon = 1 (always explore).
    """

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return "Random"

    def choose_all(self, agent: Agent):
        """Choose all actions randomly.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be shuffled.

        Returns
        -------
        list of str
            Randomly ordered list of action names.
        """
        actions = agent.actions["Name"].to_numpy()
        _rng.shuffle(actions)
        return actions.tolist()
