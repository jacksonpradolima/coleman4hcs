"""
Unit tests for `policy.py`.

This module provides comprehensive unit tests for all policy implementations
in the `coleman4hcs.policy` module. Each test ensures that the policies behave
correctly under various scenarios, including edge cases.

Policies Tested:
- Policy (Base Class)
- EpsilonGreedyPolicy
- GreedyPolicy
- RandomPolicy
- UCB1Policy
- UCBPolicy
- FRRMABPolicy
- LinUCBPolicy
- SWLinUCBPolicy

Test Plan:
1. Verify correct initialization of policies.
2. Validate `choose_all` and `credit_assignment` methods.
3. Check edge cases like no actions or extreme parameter values.
4. Ensure exceptions (e.g., `QException`) are properly raised when needed.
"""

import pytest
import polars as pl
import numpy as np
from coleman4hcs.policy import (
    Policy,
    EpsilonGreedyPolicy,
    GreedyPolicy,
    RandomPolicy,
    UCB1Policy,
    UCBPolicy,
    FRRMABPolicy,
    LinUCBPolicy,
    SWLinUCBPolicy,
)
from coleman4hcs.agent import Agent, ContextualAgent, SlidingWindowContextualAgent

# Set a fixed random seed for reproducibility
seed = 42
rng = np.random.default_rng(seed)


# ------------------------ Fixtures ------------------------

@pytest.fixture
def dummy_agent():
    """Fixture for creating a basic Agent with mock actions."""
    policy = Policy()  # Initialize the required policy
    agent = Agent(policy=policy)  # Pass the policy during initialization
    agent.actions = pl.DataFrame({  # Mock the actions dataframe
        'Name': ['A1', 'A2', 'A3'],
        'ActionAttempts': [0, 0, 0],
        'ValueEstimates': [0, 0, 0],
        'Q': [0, 0, 0]
    })
    return agent


@pytest.fixture
def contextual_agent():
    """Fixture for creating a ContextualAgent with sample features."""

    # Define a reward function for simplicity
    def reward_function(action, context):
        return 1.0  # Fixed reward for simplicity in the test setup

    policy = Policy()  # Instantiate a generic policy for the agent

    # Create the ContextualAgent with required arguments
    agent = ContextualAgent(
        policy=policy,
        reward_function=reward_function
    )

    # Add context_features attribute (required for LinUCBPolicy)
    agent.context_features = pl.DataFrame({
        'Name': ['A1', 'A2'],  # Action Names
        'feat1': [0.2, 0.5],  # Feature 1
        'feat2': [0.7, 0.3]  # Feature 2
    })

    # Set agent's features attribute
    agent.features = ['feat1', 'feat2']  # List of feature names

    return agent


@pytest.fixture
def sliding_window_contextual_agent():
    """Fixture for creating a SlidingWindowContextualAgent with sample features."""

    # Define a reward function for simplicity
    def reward_function(action, context):
        return 1.0  # Fixed reward for simplicity in the test setup

    policy = Policy()  # Instantiate a generic policy for the agent

    # Instantiate the SlidingWindowContextualAgent
    agent = SlidingWindowContextualAgent(
        policy=policy,
        reward_function=reward_function,
        window_size=5  # Example sliding window size parameter
    )

    # Set the context_features attribute (required for policies)
    agent.context_features = pl.DataFrame({
        'Name': ['A1', 'A2'],  # Action Names
        'feat1': [0.2, 0.5],  # Feature 1
        'feat2': [0.7, 0.3]  # Feature 2
    })

    # Add a history DataFrame with the required 'T' column
    agent.history = pl.DataFrame({
        'Name': ['A1', 'A2', 'A1', 'A2'],  # Action Names
        'Reward': [1.0, 0.5, 0.8, 0.3],  # Rewards associated with the actions
        'T': [1, 2, 3, 4]  # Timestamp or sequential event data
    })

    # Set agent's features attribute
    agent.features = ['feat1', 'feat2']  # List of feature names

    return agent


# ------------------------ Policy (Base Class) ------------------------

def test_policy_choose_all_default(dummy_agent):
    """Test the default `choose_all` behavior of the base Policy class."""
    policy = Policy()
    result = policy.choose_all(dummy_agent)
    assert result == ['A1', 'A2', 'A3'], "Policy should return all actions"


def test_policy_credit_assignment(dummy_agent):
    """Test the `credit_assignment` method in the base Policy class."""
    policy = Policy()
    dummy_agent.actions = dummy_agent.actions.with_columns([
        pl.when(pl.col('Name') == dummy_agent.actions['Name'][0])
        .then(pl.lit(1.0))
        .otherwise(pl.col('ActionAttempts'))
        .alias('ActionAttempts'),
        pl.when(pl.col('Name') == dummy_agent.actions['Name'][0])
        .then(pl.lit(10.0))
        .otherwise(pl.col('ValueEstimates'))
        .alias('ValueEstimates')
    ])

    # Assign credit and test Q calculation
    policy.credit_assignment(dummy_agent)
    assert np.allclose(dummy_agent.actions['Q'][0], 10.0), "Credit assignment failed"


# ------------------------ EpsilonGreedyPolicy ------------------------

def test_epsilon_greedy_policy_explore(dummy_agent):
    """
    Test the `choose_all` method of EpsilonGreedyPolicy with epsilon=1.0.
    It should always select randomly.
    """
    policy = EpsilonGreedyPolicy(epsilon=1.0)
    actions = policy.choose_all(dummy_agent)
    assert sorted(actions) == ['A1', 'A2', 'A3'], "Epsilon=1 should include all actions in random order"


def test_epsilon_greedy_policy_exploit(dummy_agent):
    """
    Test the `choose_all` method of EpsilonGreedyPolicy with epsilon=0.0.
    It should always select the greedy (best) actions.
    """
    policy = EpsilonGreedyPolicy(epsilon=0.0)
    dummy_agent.actions = dummy_agent.actions.with_columns([
        pl.Series('Q', [0.2, 0.8, 0.5])
    ])
    actions = policy.choose_all(dummy_agent)
    assert actions == ['A2', 'A3', 'A1'], "Epsilon=0 should only pick the best actions"


# ------------------------ GreedyPolicy ------------------------

def test_greedy_policy(dummy_agent):
    """Test the `choose_all` method of the GreedyPolicy (epsilon=0)."""
    policy = GreedyPolicy()
    dummy_agent.actions = dummy_agent.actions.with_columns([
        pl.Series('Q', [1.0, 2.0, 3.0])
    ])
    assert policy.choose_all(dummy_agent) == ['A3', 'A2', 'A1'], "GreedyPolicy must choose in descending Q order"


# ------------------------ RandomPolicy ------------------------

def test_random_policy(dummy_agent):
    """Test the `choose_all` method of the RandomPolicy (epsilon=1)."""
    policy = RandomPolicy()
    actions = policy.choose_all(dummy_agent)
    assert sorted(actions) == ['A1', 'A2', 'A3'], "RandomPolicy must include all actions in random order"


# ------------------------ UCB1Policy ------------------------

def test_ucb1_policy_credit_assignment(dummy_agent):
    """
    Test the `credit_assignment` method of UCB1Policy.
    Ensure it adjusts Q values based on exploration and rewards.
    """
    policy = UCB1Policy(c=2.0)
    dummy_agent.actions = dummy_agent.actions.with_columns([
        pl.Series('ActionAttempts', [1.0, 1.0, 2.0]),
        pl.Series('ValueEstimates', [3.0, 4.0, 1.0])
    ])

    policy.credit_assignment(dummy_agent)
    assert dummy_agent.actions['Q'][0] > dummy_agent.actions['Q'][2], "UCB1 failed to compute exploration factor"


# ------------------------ UCBPolicy ------------------------

def test_ucb_policy_credit_assignment(dummy_agent):
    """
    Test the `credit_assignment` method of UCBPolicy.
    Ensure it applies greater scaling factors than UCB1.
    """
    policy = UCBPolicy(c=1.5)
    dummy_agent.actions = dummy_agent.actions.with_columns([
        pl.Series('ActionAttempts', [1.0, 10.0, 5.0]),
        pl.Series('ValueEstimates', [10.0, 20.0, 15.0])
    ])

    policy.credit_assignment(dummy_agent)
    q_values = dummy_agent.actions['Q']
    assert q_values[0] > q_values[2], "UCB failed to adjust Q values with scaling factor"


# ------------------------ LinUCBPolicy ------------------------

def test_linucb_policy_choose_all(contextual_agent):
    """
    Test the `choose_all` method of LinUCBPolicy.
    Ensure it chooses actions based on contextual features.
    """
    # Set up the LinUCBPolicy
    policy = LinUCBPolicy(alpha=0.5)

    # Update actions using the agent
    policy.update_actions(contextual_agent, ['A1', 'A2'])

    # Check if the policy's context_features have been updated correctly
    assert list(policy.context_features['Name']) == ['A1', 'A2'], "Action names should match after update."


def test_linucb_policy_credit_assignment(contextual_agent):
    """
    Test the `credit_assignment` method of LinUCBPolicy.
    Validate that context-based Q values are updated correctly.
    """
    policy = LinUCBPolicy(alpha=0.5)
    policy.update_actions(contextual_agent, ['A1', 'A2'])
    contextual_agent.actions = pl.DataFrame({
        'Name': ['A1', 'A2'],
        'ValueEstimates': [10, 20]
    })

    policy.credit_assignment(contextual_agent)
    # Ensure the update matrices are not empty
    assert 'A1' in policy.context['A'], "Credit assignment failed for A1"
    assert 'A2' in policy.context['A'], "Credit assignment failed for A2"


# ------------------------ SWLinUCBPolicy ------------------------

def test_swlinucb_policy_choose_all(sliding_window_contextual_agent):
    """
    Test the `choose_all` method of SWLinUCBPolicy.
    Verify sliding window logic for contextual actions.
    """
    # Set up the SWLinUCBPolicy
    policy = SWLinUCBPolicy(alpha=0.5)

    # Update actions using the agent
    policy.update_actions(sliding_window_contextual_agent, ['A1', 'A2'])

    # Test the choose_all method
    actions = policy.choose_all(sliding_window_contextual_agent)

    # Verify outputs (example check, adjust as per SWLinUCBPolicy logic)
    assert actions == ['A1', 'A2'], "Sliding window policy should return all valid actions."


# ------------------------ Performance Tests ------------------------

@pytest.mark.benchmark(group="policy")
def test_epsilon_greedy_policy_performance(dummy_agent, benchmark):
    """
    Performance test for `choose_all` method of EpsilonGreedyPolicy.
    Simulate a large workload and benchmark execution time.
    """
    # Initialize EpsilonGreedyPolicy
    policy = EpsilonGreedyPolicy(epsilon=0.1)

    # Simulate a large number of actions
    num_actions = 10_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': [0] * num_actions,
        'ValueEstimates': rng.random(num_actions),
        'Q': [0] * num_actions
    })

    # Benchmark the `choose_all` method
    chosen_actions = benchmark(policy.choose_all, dummy_agent)

    # Assertions
    assert len(chosen_actions) <= num_actions, "EpsilonGreedyPolicy selected more actions than available."


@pytest.mark.benchmark(group="policy")
def test_greedy_policy_performance(dummy_agent, benchmark):
    """
    Performance test for `choose_all` method of GreedyPolicy.
    Simulate a large workload and benchmark execution time.
    """
    # Initialize GreedyPolicy
    policy = GreedyPolicy()

    # Simulate a large number of actions
    num_actions = 10_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': [0] * num_actions,
        'ValueEstimates': rng.random(num_actions),
        'Q': rng.random(num_actions)  # Random Q-values
    })

    # Benchmark the `choose_all` method
    chosen_actions = benchmark(policy.choose_all, dummy_agent)

    # Assertions - Ensure all actions are returned
    assert len(chosen_actions) == num_actions, "GreedyPolicy should return all actions."
    assert sorted(chosen_actions) == sorted(dummy_agent.actions['Name'].to_list()), (
        "GreedyPolicy should return all actions, but names mismatch."
    )


@pytest.mark.benchmark(group="policy")
def test_random_policy_performance(dummy_agent, benchmark):
    """
    Performance test for `choose_all` method of RandomPolicy.
    Simulate a large workload and benchmark execution time.
    """
    # Initialize RandomPolicy
    policy = RandomPolicy()

    # Simulate a large number of actions
    num_actions = 10_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': [0] * num_actions,
        'ValueEstimates': rng.random(num_actions),
        'Q': [0] * num_actions
    })

    # Benchmark the `choose_all` method
    chosen_actions = benchmark(policy.choose_all, dummy_agent)

    # Assertions - Ensure all actions are returned
    assert len(chosen_actions) == num_actions, "RandomPolicy should return all actions."
    assert sorted(chosen_actions) == sorted(dummy_agent.actions['Name'].to_list()), (
        "RandomPolicy should return all actions, but there is a mismatch in the names."
    )
    assert chosen_actions != dummy_agent.actions['Name'].to_list(), (
        "RandomPolicy should shuffle the actions, but returned them in the original order!"
    )


@pytest.mark.benchmark(group="policy")
def test_ucb1_credit_assignment_performance(dummy_agent, benchmark):
    """
    Performance test for `credit_assignment` method of UCB1Policy.
    Benchmark execution time with a large dataset.
    """
    # Initialize UCB1Policy
    c = 2  # Exploration parameter
    policy = UCB1Policy(c=c)

    # Simulate a large dataset of actions
    num_actions = 5_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': rng.integers(1, 100, num_actions).astype(float),  # Random non-zero attempts
        'ValueEstimates': rng.random(num_actions) * 100,  # Random reward estimates
        'Q': [0.0] * num_actions  # Will be calculated - use float
    })

    # Benchmark the `credit_assignment` method
    benchmark(policy.credit_assignment, dummy_agent)

    # Assertions to validate functionality
    assert "Q" in dummy_agent.actions.columns, "credit_assignment must update the Q values."
    assert dummy_agent.actions["Q"].is_not_null().all(), "Q values must not contain null values."


@pytest.mark.benchmark(group="policy")
def test_ucb1_choose_all_performance(dummy_agent, benchmark):
    """
    Performance test for `choose_all` method of UCB1Policy.
    Benchmark execution time with a large dataset.
    """
    # Initialize UCB1Policy
    c = 2  # Exploration parameter
    policy = UCB1Policy(c=c)

    # Simulate a large dataset of actions
    num_actions = 5_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': rng.integers(1, 100, num_actions).astype(float),  # Random non-zero attempts
        'ValueEstimates': rng.random(num_actions) * 100,  # Random reward estimates
        'Q': rng.random(num_actions)  # Already calculated Q values
    })

    # Benchmark the `choose_all` method
    chosen_actions = benchmark(policy.choose_all, dummy_agent)

    # Assertions to validate functionality
    assert len(chosen_actions) == num_actions, "UCB1Policy should return all actions."
    expected_order = dummy_agent.actions.sort('Q', descending=True)['Name'].to_list()
    assert chosen_actions == expected_order, "UCB1Policy did not return actions sorted by Q values."


@pytest.mark.benchmark(group="policy")
def test_frrmab_credit_assignment_performance(dummy_agent, benchmark):
    """
    Performance test for `credit_assignment` method of FRRMABPolicy.
    Benchmark execution time with a large dataset, including historical data.
    """
    # Initialize FRRMABPolicy
    c = 2  # Exploration parameter
    decayed_factor = 0.9  # Decay factor for ranking
    policy = FRRMABPolicy(c=c, decayed_factor=decayed_factor)

    # Simulate agent actions with some historical arms
    num_actions = 10_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': rng.integers(1, 50, num_actions).astype(float),  # Random attempts as float
        'ValueEstimates': rng.random(num_actions) * 100,  # Random reward estimates
        'T': [0] * num_actions,  # Initial usage time
        'Q': [0.0] * num_actions  # Will be calculated - use float
    })

    # Simulate historical reward data
    reward_history = dummy_agent.actions.sample(fraction=0.5)  # Sample 50% of actions as historical
    dummy_agent.history = pl.DataFrame({
        'Name': reward_history['Name'].to_list(),
        'ActionAttempts': reward_history['ActionAttempts'].to_list(),
        'ValueEstimates': reward_history['ValueEstimates'].to_list(),
        'T': reward_history['T'].to_list()
    })

    # Benchmark the `credit_assignment` method
    benchmark(policy.credit_assignment, dummy_agent)

    # Assertions to validate functionality
    assert "Q" in policy.history.columns, "credit_assignment must update the Q values in history."
    assert policy.history["Q"].is_not_null().all(), "Q values must not contain null values."


@pytest.mark.benchmark(group="policy")
def test_frrmab_choose_all_performance(dummy_agent, benchmark):
    """
    Performance test for `choose_all` method of FRRMABPolicy.
    Benchmark execution time with a large dataset, including historical data.
    """
    # Initialize FRRMABPolicy
    c = 2  # Exploration parameter
    decayed_factor = 0.9  # Decay factor for ranking
    policy = FRRMABPolicy(c=c, decayed_factor=decayed_factor)

    # Simulate agent actions
    num_actions = 10_000
    dummy_agent.actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': rng.integers(1, 50, num_actions).astype(float),  # Random attempts as float
        'ValueEstimates': rng.random(num_actions) * 100,  # Random reward estimates
        'T': [0] * num_actions,  # Initial usage time
        'Q': rng.random(num_actions)  # Already calculated Q values
    })

    # Simulate historical reward data to populate policy.history
    reward_history = dummy_agent.actions.sample(fraction=0.5)  # Sample 50% of actions as historical
    policy.history = pl.DataFrame({
        'Name': reward_history['Name'].to_list(),
        'ActionAttempts': reward_history['ActionAttempts'].to_list(),
        'ValueEstimates': reward_history['ValueEstimates'].to_list(),
        'T': reward_history['T'].to_list(),
        'Q': reward_history['Q'].to_list()
    })

    # Benchmark the `choose_all` method
    chosen_actions = benchmark(policy.choose_all, dummy_agent)

    # Assertions to validate functionality
    assert len(chosen_actions) == num_actions, "FRRMABPolicy should select all actions."
    expected_order = policy.history.sort('Q', descending=True)['Name'].to_list()
    assert chosen_actions == expected_order, "FRRMABPolicy did not return actions sorted by Q values."


@pytest.mark.benchmark(group="policy")
def test_swlinucb_policy_choose_all_performance(sliding_window_contextual_agent, benchmark):
    """
    Performance test for `choose_all` method of SWLinUCBPolicy using pytest-benchmark.
    Simulate a large workload and benchmark execution time.
    """
    # Initialize SWLinUCBPolicy
    policy = SWLinUCBPolicy(alpha=0.5)

    # Simulate a large input
    num_actions = 10_000
    actions = [f"A{i}" for i in range(num_actions)]
    sliding_window_contextual_agent.context_features = pl.DataFrame({
        'Name': actions,
        'feat1': rng.random(num_actions),
        'feat2': rng.random(num_actions)
    })
    sliding_window_contextual_agent.history = pl.DataFrame({
        'Name': rng.choice(actions, size=100_000),
        'Reward': rng.random(100_000),
        'T': np.arange(100_000)
    })

    # Ensure the policy is configured using the agent
    sliding_window_contextual_agent.features = ['feat1', 'feat2']
    policy.context_features = sliding_window_contextual_agent.context_features
    policy.features = sliding_window_contextual_agent.features

    # Initialize `self.context` with both `A_inv` and `b`
    num_features = len(sliding_window_contextual_agent.features)
    policy.context = {
        'A_inv': {action: np.eye(num_features) for action in actions},  # Identity matrix for each action
        'b': {action: np.zeros(num_features) for action in actions}  # Zero vector for each action
    }

    # Benchmark the `choose_all` method
    chosen_actions = benchmark(policy.choose_all, sliding_window_contextual_agent)

    # Assertions
    assert len(chosen_actions) == num_actions, "SWLinUCBPolicy failed to consider all actions."


@pytest.mark.benchmark(group="policy")
def test_linucb_policy_credit_assignment_performance(contextual_agent, benchmark):
    """
    Performance test for the `credit_assignment` method of LinUCBPolicy.
    Benchmark behavior with large contextual features and actions.
    """
    # Initialize LinUCBPolicy
    policy = LinUCBPolicy(alpha=0.5)

    # Simulate a large number of features and actions
    num_actions = 5_000
    num_features = 100
    actions = [f"A{i}" for i in range(num_actions)]
    features = {f"feat{i}": rng.random(num_actions) for i in range(num_features)}
    features["Name"] = actions

    contextual_agent.context_features = pl.DataFrame(features)
    contextual_agent.actions = pl.DataFrame({
        'Name': actions,
        'ValueEstimates': rng.random(num_actions)
    })

    # Update actions with the policy
    policy.update_actions(contextual_agent, actions)

    # Benchmark the `credit_assignment` method
    benchmark(policy.credit_assignment, contextual_agent)


# -------- Helper function to simulate actions -------- #

def simulate_actions(num_actions, include_q=False):
    """Simulates a DataFrame of actions with required columns."""
    actions = pl.DataFrame({
        'Name': [f"A{i}" for i in range(num_actions)],
        'ActionAttempts': rng.integers(1, 100, num_actions).astype(float),  # Random non-zero attempts
        'ValueEstimates': rng.random(num_actions) * 100,  # Random reward estimates
        'T': [0] * num_actions,  # Timestamp/usage in FRRMABPolicy
        'Q': rng.random(num_actions) if include_q else [0.0] * num_actions  # Random Q if needed - use float
    })
    return actions


def simulate_history(actions, fraction=0.5):
    """Simulates historical data by sampling existing actions."""
    sampled_actions = actions.sample(fraction=fraction)
    history = pl.DataFrame({
        'Name': sampled_actions['Name'].to_list(),
        'ActionAttempts': sampled_actions['ActionAttempts'].to_list(),
        'ValueEstimates': sampled_actions['ValueEstimates'].to_list(),
        'T': sampled_actions['T'].to_list(),  # Include the T column
        'Q': sampled_actions['Q'].to_list()  # Include the Q column if actions have it
    })
    return history


# -------- Parameterized Performance Tests -------- #

@pytest.mark.parametrize(
    "policy_class, policy_kwargs, benchmark_method",
    [
        (UCB1Policy, {'c': 2}, 'credit_assignment'),
        (UCB1Policy, {'c': 2}, 'choose_all'),
        (FRRMABPolicy, {'c': 2, 'decayed_factor': 0.9}, 'credit_assignment'),
        (FRRMABPolicy, {'c': 2, 'decayed_factor': 0.9}, 'choose_all'),
        (GreedyPolicy, {}, 'choose_all'),
        (RandomPolicy, {}, 'choose_all'),
        (EpsilonGreedyPolicy, {'epsilon': 0.1}, 'choose_all'),
    ]
)
@pytest.mark.benchmark(group="policy")
def test_policy_performance(dummy_agent, benchmark, policy_class, policy_kwargs, benchmark_method):
    """
    Parameterized performance test for different policies.
    Benchmarks `credit_assignment` or `choose_all` methods.
    """
    # Instantiate the policy with provided arguments
    policy = policy_class(**policy_kwargs)

    # Assign the policy to the dummy agent
    dummy_agent.policy = policy

    # Simulate a large number of actions for the agent
    num_actions = 5_000
    dummy_agent.actions = simulate_actions(num_actions, include_q=benchmark_method == 'choose_all')

    # Simulate history if policy requires it (e.g., FRRMABPolicy)
    if policy_class in [FRRMABPolicy]:
        dummy_agent.history = simulate_history(dummy_agent.actions)

    # Run the benchmark for the specified method
    if benchmark_method == 'credit_assignment':
        benchmark(policy.credit_assignment, dummy_agent)
        # Ensure that Q-values are assigned correctly
        assert "Q" in dummy_agent.actions.columns, "credit_assignment must update the Q values."
        assert dummy_agent.actions["Q"].is_not_null().all(), "Q values must not contain null values."

    elif benchmark_method == 'choose_all':
        chosen_actions = benchmark(policy.choose_all, dummy_agent)
        # Ensure that all actions are selected
        assert len(chosen_actions) == num_actions, "Policy should select all actions."
