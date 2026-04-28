"""Base Agent class."""

import numpy as np
import polars as pl

from coleman.bandit import Bandit

from .schemas import ACTIONS_SCHEMA


class Agent:
    """An agent that selects one of a set of actions at each time step.

    The action is chosen using a strategy based on the history of prior actions
    and outcome observations.

    Parameters
    ----------
    policy : object
        The policy used by the agent to choose an action. For instance, FRRMAB.
    bandit : Bandit, optional
        The bandit instance the agent interacts with.

    Attributes
    ----------
    policy : object
        The policy used by the agent to choose an action.
    bandit : Bandit or None
        The bandit instance the agent interacts with.
    last_prioritization : list or None
        The last action (test case ordering) chosen by the agent.
    t : int
        The number of steps the agent has taken.
    actions : polars.DataFrame
        A DataFrame tracking the agent's actions and their respective outcomes.
    """

    def __init__(self, policy, bandit: Bandit | None = None, seed: int | None = None):
        """Initialize the Agent.

        Parameters
        ----------
        policy : object
            The policy used by the agent to choose an action. For instance, FRRMAB.
        bandit : Bandit, optional
            The bandit instance the agent interacts with.
        seed : int, optional
            Seed for the internal RNG used for the random initial shuffle at t=0.
            When provided, repeated runs with the same seed produce identical
            prioritization sequences.  When ``None``, the shuffle is non-deterministic.
        """
        self.policy = policy
        self.bandit: Bandit | None = bandit
        self.last_prioritization: list[str] = []
        self.t = 0
        self.actions = pl.DataFrame(schema=ACTIONS_SCHEMA)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._action_names: set[str] = set()
        self._name_to_idx: dict[str, int] = {}
        self._attempt_weights_cache: dict[int, np.ndarray] = {}

        self.reset()

    def __str__(self):
        """Return a string representation of the agent.

        Returns
        -------
        str
            String representation of the agent's policy.
        """
        return str(self.policy)

    def reset(self):
        """Reset the agent's memory to an initial state."""
        self.actions = self.actions.with_columns(
            [pl.lit(0.0).alias("ValueEstimates"), pl.lit(0.0).alias("ActionAttempts"), pl.lit(0.0).alias("Q")]
        )
        names = self.actions["Name"].to_list()
        self._action_names = set(names)
        self._name_to_idx = {name: i for i, name in enumerate(names)}

        self.last_prioritization = []

        self.t = 0

    def _attempt_weights(self, state_size: int) -> np.ndarray:
        """Return cached attempt weights for a given prioritization size."""
        cached = self._attempt_weights_cache.get(state_size)
        if cached is None:
            cached = np.linspace(1.0, 1e-12, state_size)
            self._attempt_weights_cache[state_size] = cached
        return cached

    def add_action(self, action):
        """Add a new action if it does not already exist.

        Parameters
        ----------
        action : str
            The name of the action (test case) to add.
        """
        if action not in self._action_names:
            self._name_to_idx[action] = len(self.actions)
            new_row = pl.DataFrame(
                {"Name": [action], "ActionAttempts": [0.0], "ValueEstimates": [0.0], "Q": [0.0]},
                schema=ACTIONS_SCHEMA,
            )
            self.actions = pl.concat([self.actions, new_row], how="vertical")
            self._action_names.add(action)

    def update_actions(self, actions):
        """Update the agent's action set.

        This method performs several tasks:
        1. Removes actions that are no longer available.
        2. Identifies and adds new actions that were not previously in the agent's set.
        3. Notifies the agent's policy of the new actions.

        Parameters
        ----------
        actions : list of str
            List of available actions.
        """
        current_actions = self._action_names
        new_actions = set(actions) - current_actions
        obsolete_actions = current_actions - set(actions)

        if obsolete_actions:
            self.actions = self.actions.filter(~pl.col("Name").is_in(list(obsolete_actions)))
            self._action_names -= obsolete_actions

        if new_actions:
            new_actions_df = pl.DataFrame(
                {
                    "Name": sorted(new_actions),
                    "ActionAttempts": [0.0] * len(new_actions),
                    "ValueEstimates": [0.0] * len(new_actions),
                    "Q": [0.0] * len(new_actions),
                },
                schema=ACTIONS_SCHEMA,
            )
            self.actions = pl.concat([self.actions, new_actions_df], how="vertical")
            self._action_names |= new_actions

        if obsolete_actions or new_actions:
            self._name_to_idx = {name: i for i, name in enumerate(self.actions["Name"].to_list())}

    def update_bandit(self, bandit):
        """Update the agent's associated bandit.

        This method sets the agent's bandit to the provided instance and then
        updates the agent's action set based on the arms available in the new
        bandit.

        Parameters
        ----------
        bandit : Bandit
            The new bandit instance to be associated with the agent.
        """
        self.bandit = bandit
        self.update_actions(bandit.get_arms())

    def choose(self) -> list[str]:
        """Choose an action using the agent's policy.

        An action is the prioritized test suite.

        Returns
        -------
        list of str
            List of test cases in ascending order of priority.
        """
        if self.t == 0:
            polars_seed = int(self._rng.integers(0, 2**31))
            self.last_prioritization = self.actions["Name"].shuffle(seed=polars_seed).to_list()
        else:
            self.actions = self.actions.with_columns([pl.col("Q").fill_null(0.0)])
            self.last_prioritization = self.policy.choose_all(self)

        return self.last_prioritization

    def update_action_attempts(self):
        """Update action counter k -> k+1.

        A weight is given to counterbalance the order of choice, since
        all tests are selected.
        """
        state_size = len(self.last_prioritization)
        if state_size == 0:
            return

        weights = self._attempt_weights(state_size)
        indices_all = np.fromiter(
            (self._name_to_idx.get(nm, -1) for nm in self.last_prioritization), dtype=np.intp, count=state_size
        )
        valid = indices_all >= 0
        if not valid.any():
            return

        indices = indices_all[valid]
        used_weights = weights[valid]
        attempts = np.array(self.actions["ActionAttempts"].to_numpy(), dtype=np.float64, copy=True)
        np.add.at(attempts, indices, used_weights)
        self.actions = self.actions.with_columns(pl.Series("ActionAttempts", attempts))

    def observe(self, reward):
        """Update Q action-value estimates.

        Uses the update rule: Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))

        Parameters
        ----------
        reward : array-like
            The reward values for each action in the last prioritization.
        """
        self.update_action_attempts()

        reward_list = list(reward)
        reward_len = min(len(self.last_prioritization), len(reward_list))
        if reward_len == 0:
            self.t += 1
            return

        indices_all = np.fromiter(
            (self._name_to_idx.get(nm, -1) for nm in self.last_prioritization[:reward_len]),
            dtype=np.intp,
            count=reward_len,
        )
        rewards_all = np.asarray(reward_list[:reward_len], dtype=np.float64)
        valid = indices_all >= 0
        if not valid.any():
            self.t += 1
            return

        indices = indices_all[valid]
        rewards = rewards_all[valid]
        attempts = np.array(self.actions["ActionAttempts"].to_numpy(), dtype=np.float64, copy=True)
        values = np.array(self.actions["ValueEstimates"].to_numpy(), dtype=np.float64, copy=True)

        k = attempts[indices]
        mask = k > 0
        idx_valid = indices[mask]
        values[idx_valid] += (1.0 / k[mask]) * (rewards[mask] - values[idx_valid])

        self.actions = self.actions.with_columns(pl.Series("ValueEstimates", values))

        self.t += 1
