"""
coleman4hcs.policy
~~~~~~~~~~~~~~~~~~

This module provides a collection of policies that are designed to operate
with multi-armed bandits and contextual bandits. Each policy dictates how an
agent will select its actions based on prior knowledge, current context, or
exploration strategies.

Classes
-------
Policy
    Basic policy class that prescribes actions based on the memory of an agent.
EpsilonGreedyPolicy
    Chooses either the best apparent action or a random one based on a probability epsilon.
GreedyPolicy
    Always chooses the best apparent action.
RandomPolicy
    Always chooses a random action.
UCBPolicyBase
    Base class for Upper Confidence Bound policies.
UCB1Policy
    Implementation of the UCB1 algorithm.
UCBPolicy
    A variation of the UCB algorithm with a scaling factor.
FRRMABPolicy
    Fitness-Rate-Rank based Multi-Armed Bandit policy.
SlMABPolicy
    Sliding window-based Multi-Armed Bandit policy.
LinUCBPolicy
    Contextual bandit policy using linear upper confidence bounds.
SWLinUCBPolicy
    Variation of LinUCBPolicy using a sliding window approach.

Notes
-----
- UCB (Upper Confidence Bound) policies are designed to balance exploration and exploitation by
  considering both the estimated reward of an action and the uncertainty around that reward.
- EpsilonGreedy and its variations (Greedy, Random) are simpler strategies that either exploit
  the best-known action or explore random actions based on a fixed probability.
- LinUCB and SWLinUCB are contextual bandits. They choose actions not just based on past rewards,
  but also considering the current context. SWLinUCB adds a sliding window mechanism to LinUCB,
  giving more weight to recent actions.

References
----------
.. [1] Lihong Li, et al. "A Contextual-Bandit Approach to Personalized News Article
   Recommendation." In Proceedings of the 19th International Conference on World Wide
   Web (WWW), 2010.
.. [2] Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel. "Global Versus
   Individual Accuracy in Contextual Multi-Armed Bandit." In Proceedings of the 34th
   ACM/SIGAPP Symposium on Applied Computing (SAC '19), April 8-12, 2019, Limassol, Cyprus.
"""

import numpy as np
import polars as pl

from coleman4hcs.agent import (
    HISTORY_SCHEMA,
    Agent,
    ContextualAgent,
    RewardSlidingWindowAgent,
    SlidingWindowContextualAgent,
)
from coleman4hcs.exceptions import QException


class Policy:
    """A policy prescribes an action to be taken based on the memory of an agent."""

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return 'Untreat'

    def choose_all(self, agent: Agent):
        """Return all actions in their default (untreated) order.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be returned.

        Returns
        -------
        list of str
            List of action names.
        """
        return agent.actions['Name'].to_list()

    def credit_assignment(self, agent):
        """Assign credit to actions based on their outcomes.

        The credit assignment method calculates the value estimates for each
        action based on the rewards observed. The specific implementation of
        how credit is assigned depends on the policy in use.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.

        Notes
        -----
        This is a base method and should be overridden in derived classes to
        provide specific credit assignment logic. The method modifies the
        agent's state, updating the value estimates for each action based on
        the outcomes observed.
        """
        action_attempts = agent.actions['ActionAttempts'].to_numpy()
        value_estimates = agent.actions['ValueEstimates'].to_numpy()

        # Prevent division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Quality estimate: average of reward
            q_values = np.where(action_attempts > 0, value_estimates / action_attempts, 0)
            agent.actions = agent.actions.with_columns([
                pl.Series('Q', q_values)
            ])


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
        return f'\u03B5-greedy (\u03B5={self.epsilon})'

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
        # Copy the actions and add a random mask column
        actions = agent.actions.clone()
        actions = actions.with_columns([
            pl.Series('is_random', np.random.random(len(actions)) < self.epsilon)
        ])

        # Use sorting to prioritize best actions for exploitation (high Q) and exploration (random actions)
        actions = actions.sort(['is_random', 'Q'], descending=[True, True])

        # Return the ordered list of action names
        return actions['Name'].to_list()


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
        return 'Greedy'


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
        return 'Random'

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
        actions = agent.actions['Name'].to_numpy()
        np.random.shuffle(actions)
        return actions.tolist()


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
        return agent.actions.sort('Q', descending=True)['Name'].to_list()


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
        return f'UCB1 (C={self.c})'

    def credit_assignment(self, agent: Agent):
        """Assign credit using the UCB1 formula.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        # Compute the average of rewards
        super().credit_assignment(agent)

        action_attempts = agent.actions['ActionAttempts'].to_numpy()
        quality_estimates = agent.actions['Q'].to_numpy()

        # Exploration term with precomputed logarithm
        exploration = np.log1p(agent.t) / action_attempts
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)
        exploration = np.power(exploration, 1 / self.c)

        q_values = quality_estimates + exploration
        agent.actions = agent.actions.with_columns([
            pl.Series('Q', q_values)
        ])


class UCBPolicy(UCBPolicyBase):
    """Upper Confidence Bound algorithm (UCB) with scaling factor."""

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C value.
        """
        return f'UCB (C={self.c})'

    def credit_assignment(self, agent: Agent):
        """Assign credit using the UCB formula with scaling factor.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        # Compute the average of rewards
        super().credit_assignment(agent)

        action_attempts = agent.actions['ActionAttempts'].to_numpy()
        quality_estimates = agent.actions['Q'].to_numpy()

        # Precompute log(sum of action attempts)
        log_sum_attempts = np.log1p(action_attempts.sum())

        exploration = np.sqrt((2 * log_sum_attempts) / action_attempts)
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)

        # Update Q values directly
        q_values = quality_estimates + self.c * exploration
        agent.actions = agent.actions.with_columns([
            pl.Series('Q', q_values)
        ])


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

    def choose_all(self, agent: RewardSlidingWindowAgent):
        """Choose all actions based on Q values from the FRRMAB history.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        # Identify new test cases
        existing_names = set(self.history['Name'].to_list())
        agent_names = set(agent.actions['Name'].to_list())
        new_names = agent_names - existing_names

        if new_names:
            new_entries = pl.DataFrame(
                {
                    'Name': list(new_names),
                    'ActionAttempts': [0.0] * len(new_names),
                    'ValueEstimates': [0.0] * len(new_names),
                    'Q': [0.0] * len(new_names),
                    'T': [0] * len(new_names),
                },
                schema=HISTORY_SCHEMA,
            )
            self.history = pl.concat([self.history, new_entries], how="vertical")

        # Sort by Q values to determine priorities
        return self.history.sort('Q', descending=True)['Name'].to_list()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """Assign credit using the Fitness-Rate-Rank method.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent for which credit assignment is to be performed.
        """
        # We must calculate the sum of the rewards (FIRs, Fitness Improvement Rates) by each arm in the sliding window
        self.history = agent.history.group_by('Name').agg([
            pl.col('ActionAttempts').sum(),
            pl.col('ValueEstimates').sum(),
            pl.col('T').count().alias('T')
        ])

        # Find rank of each arm
        self.history = self.history.sort('ValueEstimates', descending=True)
        reward_arm = self.history['ValueEstimates'].to_numpy()
        ranking = np.arange(1, len(reward_arm) + 1)

        # Compute decay values
        decay_values = np.power(self.decayed_factor, ranking) * reward_arm

        # Compute FRR
        frr = decay_values / decay_values.sum()

        # End of Credit Assignment
        ################################################

        # Now, we will use the values to compute Q.
        # This is done once I can "remove" the test cases selected (multiple choose calls)
        # So, I compute Q once and pass to choose function select many times

        # Compute Q
        # T column contains the count of usage for each "arm"
        selected_times = self.history['T'].to_numpy()

        # Precompute log(sum of selected times)
        log_selected_times = np.log1p(selected_times.sum())

        exploration = np.sqrt((2 * log_selected_times) / selected_times)
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)
        self.history = self.history.with_columns(
            pl.Series('frr', frr),
            pl.Series('exploration', exploration),
        )
        self.history = (
            self.history
            .with_columns(
                (pl.col('frr') + self.c * pl.col('exploration')).alias('Q')
            )
            .drop(['frr', 'exploration'])
            .select(list(HISTORY_SCHEMA.keys()))
        )


class SlMABPolicy(Policy):
    """Sliding Multi-Armed Bandit policy.

    Attributes
    ----------
    history : polars.DataFrame
        History of actions and their outcomes.
    """

    def __init__(self):
        """Initialize the SlMABPolicy."""
        self.history = pl.DataFrame(schema=HISTORY_SCHEMA)

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return "SlMAB ("

    def choose_all(self, agent: RewardSlidingWindowAgent):
        """Choose all actions based on Q values from the SlMAB history.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        # Identify new test cases
        existing_names = set(self.history['Name'].to_list())
        agent_names = set(agent.actions['Name'].to_list())
        new_names = agent_names - existing_names

        if new_names:
            new_entries = pl.DataFrame(
                {
                    'Name': list(new_names),
                    'ActionAttempts': [0.0] * len(new_names),
                    'ValueEstimates': [0.0] * len(new_names),
                    'Q': [0.0] * len(new_names),
                    'T': [0] * len(new_names),
                },
                schema=HISTORY_SCHEMA,
            )
            self.history = pl.concat([self.history, new_entries], how="vertical")

        # Sort by Q values to determine priorities
        return self.history.sort('Q', descending=True)['Name'].to_list()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """Assign credit using the SlMAB method.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent for which credit assignment is to be performed.
        """
        # Compute the average of rewards (and save in Q column)
        super().credit_assignment(agent)

        # Group by Name and aggregate T column
        self.history = agent.history.group_by(['Name']).agg([
            pl.col('T').count().alias('T'),
            pl.col('T').max().alias('Ti')
        ])

        # Get Q and R values from agent.actions by joining on Name
        agent_data = agent.actions.select(['Name', 'Q']).rename({'Q': 'action_Q'})
        self.history = self.history.join(agent_data, on='Name', how='left')
        self.history = self.history.rename({'action_Q': 'Q'})

        # Add R column (assuming R is a column in agent.actions, or set to 0 if not exists)
        if 'R' in agent.actions.columns:
            agent_r = agent.actions.select(['Name', 'R']).rename({'R': 'action_R'})
            self.history = self.history.join(agent_r, on='Name', how='left')
            self.history = self.history.rename({'action_R': 'R'})
        else:
            self.history = self.history.with_columns([pl.lit(0.0).alias('R')])

        # Compute DiffSelection, T, and Q using with_columns
        self.history = self.history.with_columns([
            (pl.lit(agent.t) - pl.col('Ti')).alias('DiffSelection')
        ])

        self.history = self.history.with_columns([
            (pl.col('T') * (
                (agent.window_size / (agent.window_size + pl.col('DiffSelection')))
                + (1.0 / (pl.col('T') + 1))
            )).alias('T')
        ])

        self.history = self.history.with_columns([
            (pl.col('Q') * (
                (agent.window_size / (agent.window_size + pl.col('DiffSelection')))
                + pl.col('R') * (1.0 / (pl.col('T') + 1))
            )).alias('Q')
        ])


class LinUCBPolicy(Policy):
    """LinUCB with Disjoint Linear Models.

    Parameters
    ----------
    alpha : float, optional
        The constant that determines the width of the upper confidence bound.
        Default is 0.5.

    Attributes
    ----------
    alpha : float
        The exploration parameter.
    context : dict
        Dictionary containing A matrices, their inverses, and b vectors for
        each action.
    context_features : object or None
        Current context features.
    features : object or None
        Feature names.

    References
    ----------
    .. [1] Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
       News Article Recommendation." In Proceedings of the 19th
       International Conference on World Wide Web (WWW), 2010.
    """

    def __init__(self, alpha=0.5):
        """Initialize LinUCBPolicy.

        Parameters
        ----------
        alpha : float, optional
            The constant that determines the width of the upper confidence
            bound. Default is 0.5.
        """
        self.alpha = alpha

        # Initialize LinUCB Model Parameters
        self.context = {
            # dictionary - For any action "a" in actions,
            # A[a] = (DaT*Da + I) the ridge reg solution
            'A': {},
            # Inverse
            'A_inv': {},
            # dictionary - The cumulative return of action "a", given the
            # context xt.
            'b': {},
        }

        # The context is given by the agent (see choose_all function)
        self.context_features = self.features = None

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with alpha value.
        """
        return f"LinUCB (Alpha={self.alpha})"

    def add_action(self, action_id):
        """Add an action to the policy's context.

        Parameters
        ----------
        action_id : str
            The identifier of the action to add.
        """
        context_dimension = len(self.features)
        a = np.identity(context_dimension)

        self.context['A'][action_id] = a
        self.context['A_inv'][action_id] = np.linalg.inv(a)
        self.context['b'][action_id] = np.zeros((context_dimension, 1))

    def update_actions(self, agent: ContextualAgent, new_actions):
        """Update actions based on the agent's context.

        Parameters
        ----------
        agent : ContextualAgent
            The contextual agent providing context information.
        new_actions : list of str
            List of new action identifiers to add.
        """
        # Update the current context given by the agent
        self.context_features = agent.context_features.sort('Name')
        self.features = agent.features

        # Add new actions
        for a in new_actions:
            self.add_action(a)

    def choose_all(self, agent: Agent):
        """Choose all actions based on the LinUCB policy.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value in descending order.

        Raises
        ------
        QException
            If Q computation results in unexpected shape.
        """

        features = self.context_features.select(self.features).to_numpy()  # Shape: (num_actions, context_dim)
        actions = self.context_features['Name'].to_list()

        # Batch processing for theta and confidence bounds
        q_values = []
        for a, x in zip(actions, features):
            # Get the specific features from te current test case (action)
            x_i = x.reshape(-1, 1)

            A_inv = self.context['A_inv'][a]
            theta_a = A_inv.dot(self.context['b'][a])

            # Confidence bound for all actions
            p_t = theta_a.T.dot(x_i) + self.alpha * np.sqrt(x_i.T.dot(A_inv).dot(x_i))

            if len(p_t) > 1:
                raise QException("[LinUCB] q is more than 1: {q}")

            # Add a tuple containing the elements:
            # a: action name
            # q[0,0]: scalar Q-value (confidence-adjusted reward estimate) computed for the given action
            q_values.append((a, p_t[0, 0]))

        # Sort actions by Q value in descending order
        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]

    def credit_assignment(self, agent):
        """Assign credit based on the agent's actions and rewards.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        # We always have the same test case set
        assert len(set(agent.actions['Name'].to_list()) - set(self.context_features['Name'].to_list())) == 0

        actions = agent.actions.clone()
        actions = actions.sort('Name')

        features = self.context_features.select(self.features).to_numpy()
        actions_data = actions.select(['Name', 'ValueEstimates']).to_numpy()
        for a, x in zip(actions_data, features):
            x_i = x.reshape(-1, 1)
            act = a[0]
            reward = a[1]  # ValueEstimates

            self.context['A'][act] += x_i.dot(x_i.T)
            self.context['A_inv'][act] = np.linalg.inv(self.context['A'][act])
            self.context['b'][act] += reward * x_i


class SWLinUCBPolicy(LinUCBPolicy):
    """LinUCB with Disjoint Linear Models and Sliding Window.

    References
    ----------
    .. [1] Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel.
       "Global Versus Individual Accuracy in Contextual Multi-Armed Bandit."
       In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing
       (SAC '19), April 8-12, 2019, Limassol, Cyprus. ACM, 8 pages.
    """

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with alpha value.
        """
        return f"SWLinUCB (Alpha={self.alpha}"

    def choose_all(self, agent: SlidingWindowContextualAgent):
        """Choose all actions based on the sliding window policy.

        Parameters
        ----------
        agent : SlidingWindowContextualAgent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value in descending order.

        Raises
        ------
        QException
            If Q computation results in unexpected shape.
        """

        features = self.context_features.select(self.features).to_numpy()  # Shape: (num_actions, context_dim)
        actions = self.context_features['Name'].to_list()

        # Precompute sliding window factors
        history_names = set(agent.history['Name'].unique().to_list())  # Faster membership check
        history_counts = agent.history['Name'].value_counts().to_dicts()  # List of dicts
        history_counts_dict = {item['Name']: item['count'] for item in history_counts}  # Convert to dict

        q_values = []
        for a, x in zip(actions, features):
            # Get the specific features from te current test case (action)
            x_i = x.reshape(-1, 1)

            a_inv = self.context['A_inv'][a]
            theta_a = a_inv.dot(self.context['b'][a])
            q = theta_a.T.dot(x_i) + self.alpha * np.sqrt(x_i.T.dot(a_inv).dot(x_i))

            # This part is applied only when the iteration reaches the sliding window size
            # and if the test case is on the sliding window
            occ = 0
            if agent.t > agent.window_size and a in history_names:
                occ = history_counts_dict.get(a, 0)  # Get count or 0 if not present

            q *= (1 - occ / agent.window_size)

            if len(q) > 1:
                raise QException(f"[SWLinUCB] Q computation resulted in unexpected shape: {q.shape}")

            q_values.append((a, q[0, 0]))

        # Sort actions by Q value in descending order and return action names
        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]
