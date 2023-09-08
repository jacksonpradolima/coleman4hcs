import numpy as np
import pandas as pd
import random

from coleman4hcs.agent import Agent


class Policy(object):
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
        action_attempts = np.array(agent.actions['ActionAttempts'].values.tolist())
        value_estimates = np.array(agent.actions['ValueEstimates'].values.tolist())

        # Quality estimate: average of reward
        agent.actions['Q'] = value_estimates / action_attempts


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
        # Copy the actions (we will remove the actions for policy choose them)        
        # and order by best values (Quality estimate)
        temp_actions = agent.actions.copy().sort_values(by='Q', ascending=False)

        # How much are been selected by "best" value
        qnt_actions = sum([1 for _ in range(len(temp_actions)) if np.random.random() > self.epsilon])

        actions = temp_actions.head(qnt_actions)['Name'].tolist() if qnt_actions > 0 else []
        t_actions = temp_actions[~temp_actions.Name.isin(actions)]['Name'].tolist()
        random.shuffle(t_actions)
        actions.extend(t_actions)

        return actions


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


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """

    def __init__(self):
        super().__init__(1)

    def __str__(self):
        return 'Random'


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
        return 'UCB1 (C={})'.format(self.c)

    def credit_assignment(self, agent: Agent):
        # Compute the average of rewards
        super().credit_assignment(agent)

        action_attempts = agent.actions['ActionAttempts'].values.tolist()
        quality_estimates = agent.actions['Q'].values.tolist()

        exploration = np.log(agent.t + 1) / action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)

        agent.actions['Q'] = quality_estimates + exploration


class UCBPolicy(UCBPolicyBase):
    """
    The Upper Confidence Bound algorithm (UCB) with scaling factor.
    """

    def __str__(self):
        return 'UCB (C={})'.format(self.c)

    def credit_assignment(self, agent: Agent):
        # Compute the average of rewards
        super().credit_assignment(agent)

        action_attempts = agent.actions['ActionAttempts'].values.tolist()
        quality_estimates = np.array(agent.actions['Q'].values.tolist())

        exploration = np.sqrt((2 * np.log(sum(action_attempts))) / action_attempts)
        exploration[np.isnan(exploration)] = 0
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
        self.hist_col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'T']

        self.history = pd.DataFrame(columns=self.hist_col_names)

    def __str__(self):
        # return f"FRRMAB (C={self.c}, D={self.decayed_factor})"
        # leave with out ")" to agent put the window size
        return f"FRRMAB (C={self.c}, D={self.decayed_factor}"

    def choose_all(self, agent: Agent):
        # New test cases
        new_tcs = agent.actions[~agent.actions.Name.isin(self.history['Name'].tolist())]
        new_tcs = new_tcs['Name'].tolist()

        if len(new_tcs) > 0:
            # random.shuffle(new_tcs)
            for tc in new_tcs:
                self.history = pd.concat([self.history, pd.DataFrame([[tc, 0, 0, 0, 0]],
                                                                     columns=self.history.columns)],
                                         ignore_index=True)

        actions = self.history.sort_values(by='Q', ascending=False)
        return actions['Name'].tolist()

    def credit_assignment(self, agent):
        """
        Fitness-Rate-Rank Credit assignment
        :return: FRR, Selected Times, and Sum Applications for all arms
        """
        # I must calculate the sum of the rewards (FIRs, Fitness Improvement Rates) by each arm in the sliding window
        self.history = agent.history.groupby(['Name'], as_index=False).agg(
            {'ActionAttempts': np.sum, 'ValueEstimates': np.sum, 'T': 'count'})

        # Find rank of each arm
        self.history = self.history.sort_values(by='ValueEstimates', ascending=False)
        reward_arm = np.array(self.history['ValueEstimates'].values.tolist())
        ranking = np.array(list(range(1, len(reward_arm) + 1)))

        # Compute decay values
        decay_values = np.power(self.decayed_factor, ranking) * reward_arm
        # decay_values = np.power(self.decayed_factor, ranking * reward_arm)

        # Compute FRR
        frr = decay_values / sum(decay_values)

        # End of Credit Assignment
        ################################################

        # Now, we will use the values to compute Q.
        # This is done once I can "remove" the test cases selected (multiple choose calls)
        # So, I compute Q once and pass to choose function select many times

        # Compute Q
        # T column contains the count of usage for each "arm"
        selected_times = np.array(self.history['T'].values.tolist())
        exploration = np.sqrt((2 * np.log(sum(selected_times))) / selected_times)
        # selected_times = np.array(self.history['ActionAttempts'].values.tolist())
        # exploration = np.sqrt((2 * np.log(reward_arm)) / selected_times)
        exploration[np.isnan(exploration)] = 0
        self.history['Q'] = frr + self.c * exploration
