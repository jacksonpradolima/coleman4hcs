import random
from typing import List

import numpy as np
import pandas as pd

from coleman4hcs.bandit import Bandit
from coleman4hcs.evaluation import EvaluationMetric


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step.
    The action is chosen using a strategy based on the history of prior actions and outcome observations.
    """

    def __init__(self, policy, bandit: Bandit = None):
        """

        :param policy: The policy used by the agent to choose an action. For instance, FRRMAB
        :param bandit: Bandit
        """
        self.policy = policy
        self.bandit = bandit

        # Name | Action name
        # ActionAttempts | Number of times action was chosen
        # ValueEstimates | Reward values of an action
        # Q | Q value used in the Policy (updated in the credit assignment)
        self.col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'Q']

        self.actions = pd.DataFrame(columns=self.col_names).infer_objects()

        self.reset()

    def __str__(self):
        return str(self.policy)

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self.actions[['ValueEstimates', 'ActionAttempts', 'Q']] = 0

        # Last action (TC) choosen
        self.last_prioritization = None

        # Time of usage
        self.t = 0

    def add_action(self, action):
        """
        Add an action
        :param action:
        :return:
        """
        self.actions = self.actions.append(pd.DataFrame([[action, 0, 0, 0]], columns=self.col_names), ignore_index=True)

    def update_actions(self, actions):
        # Preserve the actual actions and remove the unnecessary
        self.actions = self.actions[self.actions.Name.isin(actions)]

        # Find the new actions (they are not in the actions that already exists)
        new_actions = [action for action in actions if action not in self.actions['Name'].tolist()]

        # Add new actions        
        new_actions_df = pd.DataFrame(new_actions, columns=['Name'])
        new_actions_df[['ValueEstimates', 'ActionAttempts', 'Q']] = 0

        self.actions = pd.concat([self.actions, new_actions_df], ignore_index=True)

    def update_bandit(self, bandit):
        self.bandit = bandit
        self.update_actions(self.bandit.get_arms())

    def choose(self) -> List[str]:
        """
        The policy choose an action.
        An action is the Priorized Test Suite
        :return: List of Test Cases in ascendent order of priority
        """
        # If is the first time that the agent has been used, we don't have a "history" (rewards).
        # So, we can choose randomly
        if self.t == 0:
            actions = self.actions['Name'].tolist()
            random.shuffle(actions)
            self.last_prioritization = actions
        else:
            # To avoid arms non-applied yet
            self.actions['Q'] = self.actions['Q'].fillna(value=0)
            self.last_prioritization = self.policy.choose_all(self)

        # Return the Prioritized Test Suite
        return self.last_prioritization

    def update_action_attempts(self):
        """
        Update action counter k -> k+1
        :return:
        """
        # We have a list, so all the tests were select.
        # to counterbalance the order of choice, a weight is given

        state_size = len(self.last_prioritization)
        weights = np.arange(1e-12, 1.0, (1. / state_size))[::-1]

        self.actions['ActionAttempts'] = self.actions.apply(
            lambda x: weights[self.last_prioritization.index(x['Name'])] + x['ActionAttempts'], axis=1)

    def observe(self, reward):
        """
        Update Q action-value using:
        Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
        :param reward:
        :return:
        """

        self.update_action_attempts()

        for test_case, r in zip(self.last_prioritization, reward):
            # Update Q action-value, in our case ValueEstimates column is Q
            k = self.actions.loc[self.actions.Name == test_case, 'ActionAttempts'].values[0]
            q = self.actions.loc[self.actions.Name == test_case, 'ValueEstimates'].values[0]

            alpha = 1. / k

            # Update Q value by keeping running average of rewards for each action
            self.actions.loc[self.actions.Name == test_case, 'ValueEstimates'] += alpha * (r - q)

        self.t += 1


class RewardAgent(Agent):
    """
    An agent that learns using a reward function.
    """

    def __init__(self, policy, reward_function):
        super().__init__(policy)
        self.reward_function = reward_function
        self.last_reward = 0

    def get_reward_function(self):
        return self.reward_function

    def observe(self, reward: EvaluationMetric):
        """

        :param reward: The reward (result) obtained by the evaluation metric
        """
        self.update_action_attempts()

        # Get rewards for each test case
        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)

        # Update value estimates (accumulative reward)
        self.actions['ValueEstimates'] += self.actions['Name'].apply(
            lambda x: self.last_reward[self.last_prioritization.index(x)])

        self.t += 1

        # Apply credit assignment
        self.policy.credit_assignment(self)

    def reset(self):
        super().reset()


class RewardSlidingWindowAgent(RewardAgent):
    """
    An agent that learns using a sliding window and a reward function.
    """

    def __init__(self, policy, reward_function, window_size):
        super().__init__(policy, reward_function)
        self.window_size = window_size

        # Name | Action name
        # ActionAttempts | Number of times action was chosen
        # ValueEstimates | Reward values of an action
        # T | Time of usage
        self.hist_col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'Q', 'T']

        self.history = pd.DataFrame(columns=self.hist_col_names)

    def __str__(self):
        return f'{str(self.policy)}, SW={self.window_size})'

    def observe(self, reward: EvaluationMetric):
        """

        :param reward: The reward (result) obtained by the evaluation metric
        """        
        self.update_action_attempts()

        # Get rewards for each test case
        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)
        # Update value estimates
        self.actions['ValueEstimates'] = self.actions['Name'].apply(
            lambda x: self.last_reward[self.last_prioritization.index(x)])

        self.t += 1
        self.update_history()
        self.policy.credit_assignment(self)

    def update_history(self):        
        temp_hist = self.actions.copy()
        temp_hist['T'] = self.t

        self.history = self.history.append(temp_hist)
        self.history = self.history.infer_objects()

        # Truncate
        unique_t = self.history['T'].unique()

        if len(unique_t) > self.window_size:
            # Remove older
            min_t = max(unique_t) - self.window_size
            self.history = self.history[self.history['T'] > min_t]
