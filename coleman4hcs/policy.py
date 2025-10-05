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

import numpy as np
import polars as pl

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
        return agent.actions['Name'].to_list()

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
        actions = agent.actions.clone()
        actions = actions.with_columns([
            pl.Series('is_random', np.random.random(len(actions)) < self.epsilon)
        ])

        # Use sorting to prioritize best actions for exploitation (high Q) and exploration (random actions)
        actions = actions.sort(['is_random', 'Q'], descending=[True, True])

        # Return the ordered list of action names
        return actions['Name'].to_list()


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
        return actions.tolist()  # numpy tolist() is correct


class UCBPolicyBase(Policy):
    """
    Base class for Upper Confidence Bound (UCB) policies.
    """

    def __init__(self, c: float):
        self.c = c

    def choose_all(self, agent: Agent) -> list:
        return agent.actions.sort('Q', descending=True)['Name'].to_list()


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

        schema = {
            'Name': pl.Utf8,
            'ActionAttempts': pl.Float64,
            'ValueEstimates': pl.Float64,
            'T': pl.Int64,
            'Q': pl.Float64
        }
        self.history = pl.DataFrame(schema=schema)

    def __str__(self):
        # return f"FRRMAB (C={self.c}, D={self.decayed_factor})"
        # leave without ")" to agent put the window size
        return f"FRRMAB (C={self.c}, D={self.decayed_factor}"

    def choose_all(self, agent: RewardSlidingWindowAgent):
        # Identify new test cases
        existing_names = set(self.history['Name'].to_list())
        agent_names = set(agent.actions['Name'].to_list())
        new_names = agent_names - existing_names
        
        if new_names:
            new_entries = pl.DataFrame({
                'Name': list(new_names),
                'ActionAttempts': [0.0] * len(new_names),
                'ValueEstimates': [0.0] * len(new_names),
                'T': [0] * len(new_names),
                'Q': [0.0] * len(new_names)
            })
            self.history = pl.concat([self.history, new_entries], how="vertical")

        # Sort by Q values to determine priorities
        return self.history.sort('Q', descending=True)['Name'].to_list()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """
        Fitness-Rate-Rank Credit assignment
        :return: FRR, Selected Times, and Sum Applications for all arms
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
        self.history = self.history.with_columns([
            (pl.lit(frr) + self.c * pl.lit(exploration)).alias('Q')
        ])


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

        schema = {
            'Name': pl.Utf8,
            'ActionAttempts': pl.Float64,
            'ValueEstimates': pl.Float64,
            'T': pl.Int64,
            'Q': pl.Float64
        }
        self.history = pl.DataFrame(schema=schema)

    def __str__(self):
        # return f"SLMAB ("
        # leave with out ")" to agent put the window size
        return "SlMAB ("

    def choose_all(self, agent: RewardSlidingWindowAgent):
        # Identify new test cases
        existing_names = set(self.history['Name'].to_list())
        agent_names = set(agent.actions['Name'].to_list())
        new_names = agent_names - existing_names
        
        if new_names:
            new_entries = pl.DataFrame({
                'Name': list(new_names),
                'ActionAttempts': [0.0] * len(new_names),
                'ValueEstimates': [0.0] * len(new_names),
                'T': [0] * len(new_names),
                'Q': [0.0] * len(new_names)
            })
            self.history = pl.concat([self.history, new_entries], how="vertical")

        # Sort by Q values to determine priorities
        return self.history.sort('Q', descending=True)['Name'].to_list()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """
        Credit assignment for SlMAB
        :return: FRR, Selected Times, and Sum Applications for all arms
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
            (pl.col('T') * ((agent.window_size / (agent.window_size + pl.col('DiffSelection'))) + (1.0 / (pl.col('T') + 1)))).alias('T')
        ])
        
        self.history = self.history.with_columns([
            (pl.col('Q') * ((agent.window_size / (agent.window_size + pl.col('DiffSelection'))) + pl.col('R') * (1.0 / (pl.col('T') + 1)))).alias('Q')
        ])


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
        self.context_features = agent.context_features.sort('Name')
        self.features = agent.features

        # Add new actions
        for a in new_actions:
            self.add_action(a)

    def choose_all(self, agent: Agent):
        """
        Choose all actions based on the policy.
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
        """
       Assign credit based on the agent's actions and rewards.
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
