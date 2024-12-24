"""
coleman4hcs.agent.tests
-----------------------

This module contains unit and performance tests for the `Agent` module in the Coleman4HCS framework.

Tests ensure the functionality and efficiency of the following classes:
- `Agent`: Base class for agents, including action management, selection, and reward observation.
- `RewardAgent`: Extends `Agent` to incorporate reward-based learning.
- `ContextualAgent`: Extends `RewardAgent` with context-aware decision-making.
- `RewardSlidingWindowAgent`: Combines reward-based learning with a sliding window for historical data.
- `SlidingWindowContextualAgent`: Combines the sliding window mechanism with context-aware decision-making.

Features Tested
---------------
- Initialization, action management, and reward observation methods for all agent types.
- Contextual updates and historical data management for contextual and sliding window agents.
- Performance benchmarks for scaling with a large number of test cases.

Performance Benchmark Results
-----------------------------
- `Agent` execution time reduced by ~69% from 92s to 28s in large-scale experiments.
- Significant improvements in reward-based decision-making speed.
"""

import random
from unittest.mock import MagicMock

import pandas as pd
import pytest

from coleman4hcs.agent import (
    Agent,
    RewardAgent,
    ContextualAgent,
    RewardSlidingWindowAgent,
    SlidingWindowContextualAgent
)
from coleman4hcs.bandit import Bandit
from coleman4hcs.evaluation import EvaluationMetric


@pytest.fixture
def mock_bandit():
    """
    Provides a mock Bandit instance.
    """
    bandit = MagicMock(spec=Bandit)
    bandit.get_arms.return_value = ["Test1", "Test2", "Test3"]
    return bandit


@pytest.fixture
def mock_policy():
    """
    Provides a mock policy for the agents.
    """
    policy = MagicMock()
    policy.choose_all.return_value = ["Test1", "Test2", "Test3"]
    policy.credit_assignment = MagicMock()
    return policy


@pytest.fixture
def mock_evaluation_metric():
    """
    Provides a mock evaluation metric for testing.
    """
    metric = MagicMock(spec=EvaluationMetric)
    metric.detection_ranks = [1, 2]
    metric.scheduled_testcases = ["Test1", "Test2", "Test3"]
    return metric


def test_agent_initialization(mock_policy, mock_bandit):
    """
    Test the initialization of the base Agent class.
    """
    agent = Agent(mock_policy, mock_bandit)
    assert agent.policy == mock_policy
    assert agent.bandit == mock_bandit
    assert agent.t == 0
    assert isinstance(agent.actions, pd.DataFrame)


def test_agent_add_action():
    """
    Test adding an action to the agent.
    """
    agent = Agent(MagicMock())
    agent.add_action("Test1")
    assert "Test1" in agent.actions["Name"].values


def test_agent_update_actions():
    """
    Test updating the action set of the agent.
    """
    agent = Agent(MagicMock())
    agent.update_actions(["Test1", "Test2"])
    assert all(name in agent.actions["Name"].values for name in ["Test1", "Test2"])


def test_agent_choose(mock_policy, mock_bandit):
    """
    Test the action selection logic of the agent.
    """
    agent = Agent(mock_policy, mock_bandit)
    agent.update_actions(["Test1", "Test2", "Test3"])
    chosen = agent.choose()
    if agent.t == 0:
        # Check if the initial random choice contains all actions
        assert set(chosen) == set(["Test1", "Test2", "Test3"])
    else:
        assert chosen == mock_policy.choose_all.return_value


def test_reward_agent_observe(mock_policy, mock_bandit, mock_evaluation_metric):
    """
    Test the observe method of RewardAgent.
    """
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [1.0, 0.5, 0.0]

    agent = RewardAgent(mock_policy, reward_function)
    agent.update_bandit(mock_bandit)
    agent.update_actions(["Test1", "Test2", "Test3"])
    agent.choose()
    agent.observe(mock_evaluation_metric)

    assert agent.last_reward == [1.0, 0.5, 0.0]


def test_contextual_agent_context_update():
    """
    Test the context update functionality of ContextualAgent.
    """
    agent = ContextualAgent(MagicMock(), MagicMock())
    context_features = {"feature1": 1, "feature2": 2}
    agent.update_context(context_features)
    assert agent.context_features == context_features


def test_reward_sliding_window_agent_observe(mock_policy, mock_bandit, mock_evaluation_metric):
    """
    Test the observe method of RewardSlidingWindowAgent.
    """
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [1.0, 0.5, 0.0]

    agent = RewardSlidingWindowAgent(mock_policy, reward_function, window_size=2)
    agent.update_bandit(mock_bandit)
    agent.choose()
    agent.observe(mock_evaluation_metric)

    assert len(agent.history) > 0
    assert "T" in agent.history.columns


def test_sliding_window_contextual_agent_history_truncation():
    """
    Test the history truncation logic of SlidingWindowContextualAgent.
    """
    agent = SlidingWindowContextualAgent(MagicMock(), MagicMock(), window_size=2)
    agent.history = pd.DataFrame({
        "Name": ["Test1", "Test2", "Test3"],
        "ActionAttempts": [1, 1, 1],
        "ValueEstimates": [0.1, 0.2, 0.3],
        "Q": [0.1, 0.2, 0.3],
        "T": [1, 2, 3],
    })

    agent.update_history()
    assert len(agent.history) == 2
    assert agent.history["T"].min() == 2


@pytest.mark.benchmark(group="agent")
@pytest.mark.parametrize("action_count", [100, 1000, 10000])
def test_agent_performance(benchmark, action_count):
    """
    Benchmark the performance of the Agent class with a large number of actions.
    """
    agent = Agent(MagicMock())
    actions = [f"Test{i}" for i in range(action_count)]

    def update_actions():
        agent.update_actions(actions)

    benchmark(update_actions)
    assert len(agent.actions) == action_count


@pytest.mark.benchmark(group="agent")
@pytest.mark.parametrize("action_count", [100, 1000, 10000])
def test_reward_agent_performance(benchmark, mock_policy, action_count):
    """
    Benchmark the performance of the RewardAgent class with a large number of actions.
    """
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [random.random() for _ in range(action_count)]

    agent = RewardAgent(mock_policy, reward_function)
    actions = [f"Test{i}" for i in range(action_count)]

    agent.update_actions(actions)
    agent.choose()

    def observe():
        agent.observe(MagicMock())

    benchmark(observe)
    assert len(agent.actions) == action_count
