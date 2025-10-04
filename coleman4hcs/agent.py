"""
coleman4hcs.agent
-----------------

This module provides an abstract representation of an agent in the Coleman4HCS framework.

An `Agent` represents an entity that interacts with the environment to perform
test case prioritization. The agent uses a policy to decide on an action (i.e.,
a prioritized list of test cases) and then observes the environment to receive a reward.
The agent updates its internal state or knowledge based on the reward, allowing it to
improve its decisions over time.

Classes
-------
- `Agent`: Base class for agents. Defines common methods and properties all agents should have.
- `RewardAgent`: An agent that learns using a reward function. Inherits from `Agent`.
- `ContextualAgent`: Extends the `RewardAgent` to incorporate contextual information for decision-making.
- `RewardSlidingWindowAgent`: An agent that learns using a sliding window mechanism and a reward function.
   Inherits from `RewardAgent`.
- `SlidingWindowContextualAgent`: Combines the sliding window mechanism with contextual information.
   Inherits from `RewardAgent`.

Attributes
----------
- `policy`: The policy used by the agent to choose an action.
- `bandit`: An instance of the Bandit class that the agent interacts with.
- `actions`: A DataFrame that tracks the agent's actions and their respective outcomes.
- `last_prioritization`: Stores the last action chosen by the agent.
- `t`: Represents the time or the number of steps the agent has taken.
- `context_features`: (For contextual agents) Contains the features of the context.
- `history`: (For sliding window agents) Maintains a history of actions taken by the agent.
- `window_size`: (For sliding window agents) Determines the size of the sliding window.
"""
import random
from typing import List

import numpy as np
import polars as pl

from coleman4hcs.bandit import Bandit
from coleman4hcs.evaluation import EvaluationMetric


class Agent:
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
        self.last_prioritization = None  # Last action (TC) chosen
        self.t = 0

        # Name | Action name
        # ActionAttempts | Number of times action was chosen
        # ValueEstimates | Reward values of an action
        # Q | Q value used in the Policy (updated in the credit assignment)
        self.col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'Q']

        # Define schema for the DataFrame
        schema = {
            'Name': pl.Utf8,
            'ActionAttempts': pl.Float64,
            'ValueEstimates': pl.Float64,
            'Q': pl.Float64
        }
        self.actions = pl.DataFrame(schema=schema)

        self.reset()

    def __str__(self):
        return str(self.policy)

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        if self.actions.height > 0:
            self.actions = self.actions.with_columns([
                pl.lit(0.0).alias('ValueEstimates'),
                pl.lit(0.0).alias('ActionAttempts'),
                pl.lit(0.0).alias('Q')
            ])

        # Last action (TC) chosen
        self.last_prioritization = None

        # Time of usage
        self.t = 0

    def add_action(self, action):
        """
        Efficiently add a new action if it doesn't exist.
        """
        if action not in self.actions['Name'].to_list():
            new_row = pl.DataFrame({
                'Name': [action],
                'ActionAttempts': [0.0],
                'ValueEstimates': [0.0],
                'Q': [0.0]
            })
            self.actions = pl.concat([self.actions, new_row], how="vertical")

    def update_actions(self, actions):
        """
        Update the agent's action set.

        This method performs several tasks:
        1. Removes actions that are no longer available.
        2. Identifies and adds new actions that were not previously in the agent's set.
        3. Notifies the agent's policy of the new actions.

        :param actions: List of available actions.
        :type actions: list[str]
        """
        current_actions = set(self.actions['Name'].to_list())
        new_actions = set(actions) - current_actions
        obsolete_actions = current_actions - set(actions)

        # Remove obsolete actions
        if obsolete_actions:
            self.actions = self.actions.filter(~pl.col('Name').is_in(list(obsolete_actions)))

        # Add new actions
        if new_actions:
            new_actions_df = pl.DataFrame({
                'Name': list(new_actions),
                'ActionAttempts': [0.0] * len(new_actions),
                'ValueEstimates': [0.0] * len(new_actions),
                'Q': [0.0] * len(new_actions)
            })
            self.actions = pl.concat([self.actions, new_actions_df], how="vertical")

    def update_bandit(self, bandit):
        """
        Update the agent's associated bandit.

        This method sets the agent's bandit to the provided instance and then updates the agent's action set
        based on the arms available in the new bandit.

        :param bandit: The new bandit instance to be associated with the agent.
        :type bandit: Bandit
        """
        self.bandit = bandit
        self.update_actions(self.bandit.get_arms())

    def choose(self) -> List[str]:
        """
        The policy choose an action.
        An action is the Prioritized Test Suite
        :return: List of Test Cases in ascendant order of priority
        """
        # If is the first time that the agent has been used, we don't have a "history" (rewards).
        # So, we can choose randomly
        if self.t == 0:
            self.last_prioritization = random.sample(self.actions['Name'].to_list(), self.actions.height)
        else:
            # To avoid arms non-applied yet
            self.actions = self.actions.with_columns([
                pl.col('Q').fill_null(0.0)
            ])
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
        weights = np.linspace(1.0, 1e-12, state_size)
        index_map = {name: idx for idx, name in enumerate(self.last_prioritization)}
        
        # Create weight mapping
        weight_map = {name: weights[idx] for name, idx in index_map.items()}
        
        # Update using replace_strict for newer polars versions
        self.actions = self.actions.with_columns([
            (pl.col('ActionAttempts') + pl.col('Name').replace_strict(weight_map, default=0.0, return_dtype=pl.Float64)).alias('ActionAttempts')
        ])

    def observe(self, reward):
        """
        Update Q action-value using:
        Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
        :param reward:
        :return:
        """
        self.update_action_attempts()

        for test_case, r in zip(self.last_prioritization, reward):
            # Get current values using filter
            row_data = self.actions.filter(pl.col('Name') == test_case)
            
            if row_data.height > 0:
                k = row_data['ActionAttempts'][0]
                q = row_data['ValueEstimates'][0]
                
                alpha = 1.0 / k
                
                # Update Q value by keeping running average of rewards for each action
                new_value = q + alpha * (r - q)
                self.actions = self.actions.with_columns([
                    pl.when(pl.col('Name') == test_case)
                      .then(new_value)
                      .otherwise(pl.col('ValueEstimates'))
                      .alias('ValueEstimates')
                ])

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
        """
        Retrieve the reward function associated with the agent.

        This method returns the reward function instance that the agent uses to evaluate and update its policies.

        :return: The reward function of the agent.
        :rtype: RewardFunction (or the specific type of your reward function class)
        """
        return self.reward_function

    def observe(self, reward: EvaluationMetric):
        """

        :param reward: The reward (result) obtained by the evaluation metric
        """
        self.update_action_attempts()

        # Get rewards for each test case
        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)

        # Update value estimates (accumulative reward) - create mapping
        reward_map = {name: self.last_reward[self.last_prioritization.index(name)]
                     for name in self.actions['Name'].to_list() if name in self.last_prioritization}
        
        # Update using with_columns
        current_estimates = self.actions['ValueEstimates'].to_list()
        name_list = self.actions['Name'].to_list()
        new_estimates = [current_estimates[i] + reward_map.get(name_list[i], 0.0) 
                        for i in range(len(name_list))]
        
        self.actions = self.actions.with_columns([
            pl.Series('ValueEstimates', new_estimates)
        ])

        self.t += 1

        # Apply credit assignment
        self.policy.credit_assignment(self)


class ContextualAgent(RewardAgent):
    """
    The Reward Window Agent learns using a reward function and contextual information that the user can choose
    """

    def __init__(self, policy, reward_function):
        super().__init__(policy, reward_function)

        # List of features
        self.context_features = self.features = None

    def __str__(self):
        return f'{str(self.policy)}'

    def choose(self) -> List[str]:
        """
        The policy choose an action.
        An action is the Prioritized Test Suite
        :return: List of Test Cases in ascendant order of priority
        """
        self.last_prioritization = self.policy.choose_all(self)
        return self.last_prioritization

    def update_actions(self, actions):
        """
        Update the set of available actions based on the current context.

        This method adjusts the agent's possible actions based on the current context. In some situations,
        the available actions might change based on the state of the environment or other contextual information.
        This method ensures that the agent always has an up-to-date set of actions to choose from.
        """
        # Preserve the actual actions and remove the unnecessary
        self.actions = self.actions.filter(pl.col('Name').is_in(actions))

        # Find the new actions (they are not in the actions that already exists)
        new_actions = [action for action in actions if action not in self.actions['Name'].to_list()]

        # Update the information about the arms in the policy
        self.policy.update_actions(self, new_actions)

        if new_actions:
            new_actions_df = pl.DataFrame({
                'Name': new_actions,
                'ValueEstimates': [0.0] * len(new_actions),
                'ActionAttempts': [0.0] * len(new_actions),
                'Q': [0.0] * len(new_actions)
            })
            self.actions = pl.concat([self.actions, new_actions_df], how="vertical")

    def update_bandit(self, bandit):
        """
        Update the internal bandit instance used by the agent.

        This method updates the agent's internal bandit to the provided instance. This can be useful
        when the agent needs to adapt to changes in the environment or when the bandit's state changes
        over time.

        :param bandit: The new bandit instance to be used by the agent.
        :type bandit: coleman4hcs.bandit.Bandit
        """
        self.bandit = bandit
        self.update_actions(self.bandit.get_arms())

    def update_context(self, context_features):
        """
        Update the agent's current context information.

        The context provides additional information that can help the agent in making decisions.
        This might include external factors or environmental states that could influence the agent's strategy.

        :param context_features: A collection or dataframe containing the contextual information.
        """
        self.context_features = context_features

    def update_features(self, features):
        """
        Update the features used by the agent for decision-making.

        Features represent specific characteristics or properties of data that the agent uses
        to make its decisions.

        :param features: A list or collection of features.
        """
        self.features = features


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

        schema = {
            'Name': pl.Utf8,
            'ActionAttempts': pl.Float64,
            'ValueEstimates': pl.Float64,
            'Q': pl.Float64,
            'T': pl.Int64
        }
        self.history = pl.DataFrame(schema=schema)

    def __str__(self):
        return f'{str(self.policy)}, SW={self.window_size})'

    def observe(self, reward: EvaluationMetric):
        """

        :param reward: The reward (result) obtained by the evaluation metric
        """
        self.update_action_attempts()

        # Get rewards for each test case
        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)
        
        # Update value estimates - create mapping
        reward_map = {name: self.last_reward[self.last_prioritization.index(name)] 
                     for name in self.actions['Name'].to_list() if name in self.last_prioritization}
        
        # Update using with_columns
        name_list = self.actions['Name'].to_list()
        new_estimates = [reward_map.get(name, 0.0) for name in name_list]
        
        self.actions = self.actions.with_columns([
            pl.Series('ValueEstimates', new_estimates)
        ])

        self.t += 1
        self.update_history()
        self.policy.credit_assignment(self)

    def update_history(self):
        """
        Update the agent's history of actions and outcomes.

        This method adds the current action and its outcome to the agent's history.
        If the length of the history exceeds the window size,
        the oldest entries are removed to maintain the specified window size.
        """
        temp_hist = self.actions.clone()
        temp_hist = temp_hist.with_columns([
            pl.lit(self.t).alias('T')
        ])

        self.history = pl.concat([self.history, temp_hist], how="vertical")

        # Truncate
        unique_t = self.history['T'].unique().to_list()

        if len(unique_t) > self.window_size:
            # Remove older
            min_t = max(unique_t) - self.window_size
            self.history = self.history.filter(pl.col('T') > min_t)


class SlidingWindowContextualAgent(ContextualAgent):
    """
    The Reward Window Agent learns using a reward function and contextual information that the user can choose
    Additionally, it uses a sliding window
    """

    def __init__(self, policy, reward_function, window_size):
        super().__init__(policy, reward_function)

        self.window_size = window_size

        # List of features
        self.context_features = self.features = None

        # Name | Action name
        # ActionAttempts | Number of times action was chosen
        # ValueEstimates | Reward values of an action
        # T | Time of usage
        self.hist_col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'Q', 'T']

        schema = {
            'Name': pl.Utf8,
            'ActionAttempts': pl.Float64,
            'ValueEstimates': pl.Float64,
            'Q': pl.Float64,
            'T': pl.Int64
        }
        self.history = pl.DataFrame(schema=schema)

    def __str__(self):
        return f'{str(self.policy)}, SW={self.window_size})'

    def observe(self, reward: EvaluationMetric):
        """

        :param reward: The reward (result) obtained by the evaluation metric
        """
        self.update_action_attempts()

        # Get rewards for each test case
        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)

        # Update value estimates - create mapping
        reward_map = {name: self.last_reward[self.last_prioritization.index(name)] 
                     for name in self.actions['Name'].to_list() if name in self.last_prioritization}
        
        self.actions = self.actions.with_columns([
            pl.col('Name').replace(reward_map).alias('ValueEstimates')
        ])

        self.t += 1
        self.update_history()
        self.policy.credit_assignment(self)

    def update_history(self):
        """
        Update the agent's history of actions and outcomes.

        This method adds the current action and its outcome to the agent's history.
        If the length of the history exceeds the window size,
        the oldest entries are removed to maintain the specified window size.
        """
        temp_hist = self.actions.clone()
        temp_hist = temp_hist.with_columns([
            pl.lit(self.t).alias('T')
        ])

        self.history = pl.concat([self.history, temp_hist], how="vertical")

        # Truncate
        unique_t = self.history['T'].unique().to_list()

        if len(unique_t) > self.window_size:
            # Remove older
            min_t = max(unique_t) - self.window_size
            self.history = self.history.filter(pl.col('T') > min_t)
