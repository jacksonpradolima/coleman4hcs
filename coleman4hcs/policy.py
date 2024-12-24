"""
coleman4hcs.policy
~~~~~~~~~~~~~~~~~~

This module provides a collection of policies that are designed to operate
with multi-armed bandits and contextual bandits. Each policy dictates how an
agent will select its actions based on prior knowledge, current context, or
exploration strategies.

Classes:
    - Policy: Basic policy class that prescribes actions based on the memory of an agent.
    - EpsilonGreedyPolicy: Chooses either the best apparent action or a random one based on a probability epsilon.
    - GreedyPolicy: Always chooses the best apparent action.
    - RandomPolicy: Always chooses a random action.
    - UCBPolicyBase: Base class for Upper Confidence Bound policies.
    - UCB1Policy: Implementation of the UCB1 algorithm.
    - UCBPolicy: A variation of the UCB algorithm with a scaling factor.
    - FRRMABPolicy: Fitness-Rate-Rank based Multi-Armed Bandit policy.
    - SlMABPolicy: Sliding window-based Multi-Armed Bandit policy.
    - LinUCBPolicy: Contextual bandit policy using linear upper confidence bounds.
    - SWLinUCBPolicy: Variation of LinUCBPolicy using a sliding window approach.

Notes:
    - UCB (Upper Confidence Bound) policies are designed to balance exploration and exploitation by considering
      both the estimated reward of an action and the uncertainty around that reward.
    - EpsilonGreedy and its variations (Greedy, Random) are simpler strategies that either exploit the best-known
      action or explore random actions based on a fixed probability.
    - LinUCB and SWLinUCB are contextual bandits. They choose actions not just based on past rewards, but also
      considering the current context. SWLinUCB adds a sliding window mechanism to LinUCB, giving more weight to
      recent actions.

References:
    - Lihong Li, et al. "A Contextual-Bandit Approach to Personalized News Article Recommendation."
      In Proceedings of the 19th International Conference on World Wide Web (WWW), 2010.
    - Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel. "Global Versus Individual Accuracy in
      Contextual Multi-Armed Bandit." In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing (SAC ’19),
      April 8–12, 2019, Limassol, Cyprus.
"""

import random

import numpy as np
import pandas as pd

from coleman4hcs.agent import Agent, RewardSlidingWindowAgent, ContextualAgent, SlidingWindowContextualAgent
from coleman4hcs.exceptions import QException


class Policy:
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """

    def __str__(self):
        return 'Untreat'

    def choose_all(self, agent: Agent):
        """
        By default, a policy returns untreated actions.
        """
        return agent.actions['Name'].tolist()

    def credit_assignment(self, agent):
        """
       Assigns credit to actions based on their outcomes.

       The credit assignment method calculates the value estimates for
       each action based on the rewards observed. The specific implementation
       of how credit is assigned depends on the policy in use.

       :param agent: The agent for which credit assignment is to be performed.
                     The agent can have a history of actions and their outcomes.
       :type agent: Agent

       .. note:: This is a base method and should be overridden in derived classes
                 to provide specific credit assignment logic.

       .. note:: The method modifies the agent's state, updating the value estimates
                 for each action based on the outcomes observed.
       """
        action_attempts = agent.actions['ActionAttempts'].to_numpy(copy=True)
        value_estimates = agent.actions['ValueEstimates'].to_numpy(copy=True)

        # Prevent division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Quality estimate: average of reward
            agent.actions['Q'] = np.where(action_attempts > 0, value_estimates / action_attempts, 0)


class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return f'\u03B5-greedy (\u03B5={self.epsilon})'

    def choose_all(self, agent: Agent):
        # Copy the actions and add a random mask column
        actions = agent.actions.copy()
        actions['is_random'] = np.random.random(len(actions)) < self.epsilon

        # Use sorting to prioritize best actions for exploitation (high Q) and exploration (random actions)
        actions = actions.sort_values(by=['is_random', 'Q'], ascending=[False, False])

        # Return the ordered list of action names
        return actions['Name'].tolist()


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """

    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return 'Greedy'


class RandomPolicy(Policy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """

    def __str__(self):
        return 'Random'

    def choose_all(self, agent: Agent):
        actions = agent.actions['Name'].to_numpy(copy=True)
        np.random.shuffle(actions)
        return actions.tolist()


class UCBPolicyBase(Policy):
    """
    Base class for Upper Confidence Bound (UCB) policies.
    """

    def __init__(self, c: float):
        self.c = c

    def choose_all(self, agent: Agent) -> list:
        return agent.actions.sort_values(by='Q', ascending=False)['Name'].tolist()


class UCB1Policy(UCBPolicyBase):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """

    def __str__(self):
        return f'UCB1 (C={self.c})'

    def credit_assignment(self, agent: Agent):
        # Compute the average of rewards
        super().credit_assignment(agent)

        action_attempts = agent.actions['ActionAttempts'].to_numpy(copy=True)
        quality_estimates = agent.actions['Q'].to_numpy(copy=True)

        # Exploration term with precomputed logarithm
        exploration = np.log1p(agent.t) / action_attempts
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)
        exploration = np.power(exploration, 1 / self.c)

        agent.actions['Q'] = quality_estimates + exploration


class UCBPolicy(UCBPolicyBase):
    """
    The Upper Confidence Bound algorithm (UCB) with scaling factor.
    """

    def __str__(self):
        return f'UCB (C={self.c})'

    def credit_assignment(self, agent: Agent):
        # Compute the average of rewards
        super().credit_assignment(agent)

        action_attempts = agent.actions['ActionAttempts'].to_numpy(copy=True)
        quality_estimates = agent.actions['Q'].to_numpy(copy=True)

        # Precompute log(sum of action attempts)
        log_sum_attempts = np.log1p(action_attempts.sum())

        exploration = np.sqrt((2 * log_sum_attempts) / action_attempts)
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)

        # Update Q values directly
        agent.actions['Q'] = quality_estimates + self.c * exploration


class FRRMABPolicy(Policy):
    """
    The Fitness-Rate-Rank based Multi-Armed Bandit (FRRMAB).
    """

    def __init__(self, c, decayed_factor=1):
        self.c = c
        self.decayed_factor = decayed_factor

        # Name | Action name
        # ActionAttempts | Number of times action was chosen
        # ValueEstimates | Reward values of an action
        # T | Time of usage
        # Q | Quality estimate
        self.hist_col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'T', 'Q']

        self.history = pd.DataFrame(columns=self.hist_col_names)

    def __str__(self):
        # return f"FRRMAB (C={self.c}, D={self.decayed_factor})"
        # leave without ")" to agent put the window size
        return f"FRRMAB (C={self.c}, D={self.decayed_factor}"

    def choose_all(self, agent: RewardSlidingWindowAgent):
        # Identify new test cases
        new_tcs = agent.actions.loc[~agent.actions.Name.isin(self.history['Name'])]

        if not new_tcs.empty:
            new_entries = new_tcs[['Name']].assign(ActionAttempts=0, ValueEstimates=0, T=0, Q=0)
            self.history = pd.concat([self.history, new_entries], ignore_index=True)

        # Sort by Q values to determine priorities
        return self.history.sort_values(by='Q', ascending=False)['Name'].tolist()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """
        Fitness-Rate-Rank Credit assignment
        :return: FRR, Selected Times, and Sum Applications for all arms
        """
        # We must calculate the sum of the rewards (FIRs, Fitness Improvement Rates) by each arm in the sliding window
        self.history = agent.history.groupby('Name', as_index=False).agg(
            ActionAttempts=('ActionAttempts', 'sum'),
            ValueEstimates=('ValueEstimates', 'sum'),
            T=('T', 'count')
        )

        # Find rank of each arm
        self.history.sort_values(by='ValueEstimates', ascending=False, inplace=True)
        reward_arm = self.history['ValueEstimates'].to_numpy(copy=True)
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
        selected_times = self.history['T'].to_numpy(copy=True)

        # Precompute log(sum of selected times)
        log_selected_times = np.log1p(selected_times.sum())

        exploration = np.sqrt((2 * log_selected_times) / selected_times)
        exploration = np.nan_to_num(exploration, nan=0.0, posinf=0.0, neginf=0.0)
        self.history['Q'] = frr + self.c * exploration


class SlMABPolicy(Policy):
    """
    The Sliding Multi-Armed Bandit.
    """

    def __init__(self):
        # Name | Action name
        # ActionAttempts | Number of times action was chosen
        # ValueEstimates | Reward values of an action
        # T | Time of usage
        self.hist_col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'T', 'Q']

        self.history = pd.DataFrame(columns=self.hist_col_names)

    def __str__(self):
        # return f"SLMAB ("
        # leave with out ")" to agent put the window size
        return "SlMAB ("

    def choose_all(self, agent: RewardSlidingWindowAgent):
        # Identify new test cases
        new_tcs = agent.actions.loc[~agent.actions.Name.isin(self.history['Name'])]

        if not new_tcs.empty:
            new_entries = new_tcs[['Name']].assign(ActionAttempts=0, ValueEstimates=0, T=0, Q=0)
            self.history = pd.concat([self.history, new_entries], ignore_index=True)

        # Sort by Q values to determine priorities
        return self.history.sort_values(by='Q', ascending=False)['Name'].tolist()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """
        Credit assignment for SlMAB
        :return: FRR, Selected Times, and Sum Applications for all arms
        """
        # Compute the average of rewards (and save in Q column)
        super().credit_assignment(agent)

        self.history = agent.history.groupby(['Name'], as_index=False).agg({'T': ['count', np.max]})
        self.history.columns = ['Name', 'T', 'Ti']

        self.history['Q'] = agent.actions['Q']
        self.history['R'] = agent.actions['R']

        # Check, at the time point t, the number of time points elapsed since
        # the previous time point ti in which the ith arm has been applied
        self.history['DiffSelection'] = self.history['Ti'].apply(lambda x: agent.t - x)

        # T column contains the count of usage for each "arm"
        self.history['T'] = self.history.apply(
            lambda x: x['T'] * ((agent.window_size / (agent.window_size + x['DiffSelection'])) + (1 / (x['T'] + 1))),
            axis=1)

        # Compute Q
        self.history['Q'] = self.history.apply(
            lambda x: x['Q'] * (
                (agent.window_size / (agent.window_size + x['DiffSelection'])) + x['R'] * (1 / (x['T'] + 1))),
            axis=1)


class LinUCBPolicy(Policy):
    """
    LinUCB with Disjoint Linear Models

    References
    ----------
    [1]  Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
         News Article Recommendation." In Proceedings of the 19th
         International Conference on World Wide Web (WWW), 2010.
    """

    def __init__(self, alpha=0.5):
        """
        Initialize LinUCBPolicy.

        :param alpha: The constant determines the width of the upper confidence bound.
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
        return f"LinUCB (Alpha={self.alpha})"

    def add_action(self, action_id):
        """
        Add an action to the policy's context.
        """
        context_dimension = len(self.features)
        a = np.identity(context_dimension)

        self.context['A'][action_id] = a
        self.context['A_inv'][action_id] = np.linalg.inv(a)
        self.context['b'][action_id] = np.zeros((context_dimension, 1))

    def update_actions(self, agent: ContextualAgent, new_actions):
        """
        Update actions based on the agent's context.
        """
        # Update the current context given by the agent
        self.context_features = agent.context_features.sort_values(by=['Name'])
        self.features = agent.features

        # Add new actions
        for a in new_actions:
            self.add_action(a)

    def choose_all(self, agent: Agent):
        """
        Choose all actions based on the policy.
        """

        # features = list(self.context_features[self.features].values)
        features = self.context_features[self.features].values  # Shape: (num_actions, context_dim)
        actions = self.context_features.Name.tolist()

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

        # arms = pd.DataFrame(q_values, columns=['Name', 'Q'])
        # arms.sort_values(by='Q', ascending=False, inplace=True)
        # return arms['Name'].tolist()
        # Sort actions by Q value
        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]

    def credit_assignment(self, agent):
        """
       Assign credit based on the agent's actions and rewards.
       """
        # We always have the same test case set
        assert len(set(agent.actions['Name']) - set(self.context_features['Name'])) == 0

        actions = agent.actions.copy()
        actions.sort_values(by=['Name'], inplace=True)

        features = self.context_features[self.features].values
        actions = list(actions[['Name', 'ValueEstimates']].values)
        for a, x in zip(actions, features):
            x_i = x.reshape(-1, 1)
            act = a[0]
            reward = a[1]  # ValueEstimates

            self.context['A'][act] += x_i.dot(x_i.T)
            self.context['A_inv'][act] = np.linalg.inv(self.context['A'][act])
            self.context['b'][act] += reward * x_i


class SWLinUCBPolicy(LinUCBPolicy):
    """
    LinUCB with Disjoint Linear Models and Sliding Window

    References
    ----------
    [1]  Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel. 2019.
         "Global Versus Individual Accuracy in Contextual Multi-Armed Bandit."
         In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing (SAC ’19),
         April 8–12, 2019, Limassol, Cyprus. ACM, Limassol, Cyprus, 8 pages.
         https://doi. org/10.1145/3297280.3297440
    """

    def __str__(self):
        # return f"SWLinUCB (Alpha={self.alpha}"
        # leave without ")" to agent put the window size
        return f"SWLinUCB (Alpha={self.alpha}"

    def choose_all(self, agent: SlidingWindowContextualAgent):
        """
        Choose all actions based on the sliding window policy.
        """

        features = self.context_features[self.features].values  # Shape: (num_actions, context_dim)
        actions = self.context_features.Name.tolist()

        # Precompute sliding window factors
        history_names = set(agent.history['Name'].unique())  # Faster membership check
        history_counts = agent.history['Name'].value_counts().to_dict()  # Fast dictionary lookup

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
                occ = history_counts.get(a, 0)  # Get count or 0 if not present

            q *= (1 - occ / agent.window_size)

            if len(q) > 1:
                raise QException(f"[SWLinUCB] Q computation resulted in unexpected shape: {q.shape}")

            q_values.append((a, q[0, 0]))

        # Sort actions by Q value in descending order and return action names
        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]
