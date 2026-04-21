"""
Unit and integration tests for the experiment runner logic.

This module contains unit and integration tests to verify the functionality
of various core functions present in the ``coleman4hcs.runner`` module. The
tests cover a wide range of behaviors, including logging setup, managing
agents, loading classes dynamically, working with experiments, and scenario
providers.

Purpose:
- Ensure the robustness, reliability, and correct behavior of the runner
  module's core logic.
- Validate the integrations with other components like agents, policies,
  and scenario providers.

Tested Functionalities:
1. **Logging**:
   - `create_logger`: Verifies correct logger creation and configuration.

2. **Experiment Execution**:
   - `exp_run_industrial_dataset`: Tests running experiments with mock environments.

3. **Dynamic Class Loading**:
   - `load_class_from_module`: Ensures classes are correctly loaded dynamically from modules.

4. **Agent Creation**:
   - `create_agents`: Verifies the creation of different agent types
     (e.g., `RewardSlidingWindowAgent`, `ContextualAgent`) based on the
     provided policies and window sizes.

5. **Scenario Setup**:
   - `get_scenario_provider`: Tests the initialization of different scenario providers.

6. **End-to-End Workflow**:
   - Ensures integration of various components with mock objects to validate behavior across larger workflows.
"""

import logging
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

import coleman4hcs.policy
from coleman4hcs.agent import ContextualAgent, RewardSlidingWindowAgent
from coleman4hcs.environment import Environment
from coleman4hcs.policy import FRRMABPolicy, SWLinUCBPolicy
from coleman4hcs.runner import (
    EnvironmentBuildConfig,
    ExecutionPlan,
    _effective_parallel_pool_size,
    build_runtime_metadata,
    create_agents,
    create_logger,
    exp_run_industrial_dataset,
    get_scenario_provider,
    load_class_from_module,
    run_parallel_executions,
)
from coleman4hcs.scenarios import IndustrialDatasetScenarioProvider

# ------------------------
# Unit Tests
# ------------------------


def test_create_logger():
    """Test that a logger is created with the correct level and handler."""
    # Cleanup global handlers to isolate the logger for this test
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create a logger
    logger = create_logger(level=logging.DEBUG)

    # Ensure the logger's level is set correctly
    assert logger.level == logging.DEBUG

    # Verify there's exactly one handler attached
    assert len(logger.handlers) == 1

    # Cleanup: Remove all handlers from the logger to avoid reuse issues
    logger.handlers.clear()


@patch("coleman4hcs.runner.create_logger")
@patch("coleman4hcs.runner.Environment")
def test_exp_run_industrial_dataset(mock_environment, mock_create_logger, tmpdir):
    """Test that a single experiment run executes the expected environment methods."""
    mock_env = mock_environment.return_value
    mock_logger = mock_create_logger.return_value
    mock_env.scenario_provider = Mock()
    mock_env.logger = mock_logger

    exp_run_industrial_dataset(
        iteration=1,
        trials=10,
        env=cast(Environment, mock_env),
        level=20,
    )

    mock_env.set_runtime_metadata.assert_called_once()
    mock_env.run_single.assert_called_once()
    mock_env.store_experiment.assert_called_once()


def test_build_runtime_metadata_is_unique_per_execution():
    """Execution metadata should uniquely identify independent runs."""
    metadata_a = build_runtime_metadata("fakedata", 0.5, 1, "process")
    metadata_b = build_runtime_metadata("fakedata", 0.5, 1, "process")

    assert metadata_a["worker_id"] == "1"
    assert metadata_a["parallel_mode"] == "process"
    assert metadata_a["execution_id"] != metadata_b["execution_id"]


def test_effective_parallel_pool_size_keeps_parallel_when_not_profiled():
    """Without profiler instrumentation, pool size should be preserved."""
    with patch.dict("os.environ", {}, clear=True):
        assert _effective_parallel_pool_size(4) == 4


def test_effective_parallel_pool_size_forces_sequential_under_scalene():
    """Scalene instrumentation should force sequential execution for stability."""
    with patch.dict("os.environ", {"SCALENE_ALLOCATION_SAMPLING_WINDOW": "1024"}, clear=True):
        assert _effective_parallel_pool_size(4) == 1


def test_effective_parallel_pool_size_allows_scalene_parallel_override():
    """Users may opt out of Scalene fallback via execution config."""
    with patch.dict("os.environ", {"SCALENE_ALLOCATION_SAMPLING_WINDOW": "1024"}, clear=True):
        assert _effective_parallel_pool_size(4, force_sequential_under_scalene=False) == 4


def test_run_parallel_executions_dispatches_unique_execution_plans():
    """Parallel dispatch should preserve per-worker isolation metadata."""
    captured: dict[str, Any] = {}

    class FakeAsyncResult:
        def get(self, timeout=None):  # noqa: ARG002
            return None

    class FakePool:
        def __init__(self, size, maxtasksperchild=None):
            captured["pool_size"] = size
            if maxtasksperchild is not None:
                captured["maxtasksperchild"] = maxtasksperchild

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        def starmap_async(self, func, args_list):
            captured["args"] = list(args_list)
            return FakeAsyncResult()

        def terminate(self):
            return None

        def join(self):
            return None

    class FakeContext:
        def Pool(self, size, maxtasksperchild=None):  # noqa: N802
            return FakePool(size, maxtasksperchild=maxtasksperchild)

    build_config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plans = [
        ExecutionPlan(
            iteration=1,
            trials=10,
            level=20,
            execution_id="exec-1",
            worker_id="1",
            parallel_mode="process",
        ),
        ExecutionPlan(
            iteration=2,
            trials=10,
            level=20,
            execution_id="exec-2",
            worker_id="2",
            parallel_mode="process",
        ),
    ]

    with patch("coleman4hcs.runner.get_context", return_value=FakeContext()):
        run_parallel_executions(2, build_config, plans)

    assert captured["pool_size"] == 2
    assert captured["maxtasksperchild"] == 1
    assert len(captured["args"]) == 2

    dispatched_first_plan = captured["args"][0][1]
    dispatched_second_plan = captured["args"][1][1]

    assert isinstance(dispatched_first_plan, ExecutionPlan)
    assert isinstance(dispatched_second_plan, ExecutionPlan)
    assert dispatched_first_plan.execution_id == "exec-1"
    assert dispatched_second_plan.execution_id == "exec-2"
    assert dispatched_first_plan.execution_id != dispatched_second_plan.execution_id


def test_load_class_from_module_valid():
    """Test that a valid class is loaded correctly from a module."""
    policy_class = load_class_from_module(coleman4hcs.policy, "FRRMABPolicy")
    assert policy_class.__name__ == "FRRMABPolicy"


def test_load_class_from_module_invalid():
    """Test that a ValueError is raised when an invalid class name is provided."""
    with pytest.raises(ValueError, match="Class 'InvalidPolicy' not found"):
        load_class_from_module(coleman4hcs.policy, "InvalidPolicy")


def test_create_agents_frrmab():
    """Test that create_agents returns RewardSlidingWindowAgent instances for FRRMABPolicy."""
    policy = Mock(spec=FRRMABPolicy)
    reward_fun = Mock()
    window_sizes = [5, 10]

    agents = create_agents(policy, reward_fun, window_sizes)
    assert len(agents) == 2
    assert all(isinstance(agent, RewardSlidingWindowAgent) for agent in agents)


def test_create_agents_swlinucb():
    """Test that create_agents returns ContextualAgent instances for SWLinUCBPolicy."""
    policy = Mock(spec=SWLinUCBPolicy)
    reward_fun = Mock()
    window_sizes = [5, 10]

    agents = create_agents(policy, reward_fun, window_sizes)
    assert len(agents) == 2
    assert all(isinstance(agent, ContextualAgent) for agent in agents)


def test_get_scenario_provider_basic():
    """Test that get_scenario_provider returns an IndustrialDatasetScenarioProvider instance."""
    datasets_dir = "examples"
    dataset = "fakedata"
    sched_time_ratio = 0.5
    use_hcs = False
    use_context = False
    context_config = {}
    feature_groups = {}

    scenario_provider = get_scenario_provider(
        datasets_dir, dataset, sched_time_ratio, use_hcs, use_context, context_config, feature_groups
    )

    assert isinstance(scenario_provider, IndustrialDatasetScenarioProvider)


# ------------------------
# Integration Tests
# ------------------------


@patch("coleman4hcs.runner.create_logger")
@patch("coleman4hcs.runner.Environment")
def test_end_to_end_execution(mock_environment, mock_create_logger, tmpdir):
    """Test end-to-end execution using mocked environment and logger."""
    mock_env = mock_environment.return_value
    mock_env.scenario_provider = Mock()
    mock_env.max_builds = 10
    mock_env.logger = mock_create_logger.return_value

    independent_executions = 3
    parameters = [(i + 1, 10, cast(Environment, mock_env), 20) for i in range(independent_executions)]

    for iteration, trials, env, level in parameters:
        exp_run_industrial_dataset(iteration, trials, env, level)

    assert mock_env.run_single.call_count == independent_executions
    assert mock_env.store_experiment.call_count == independent_executions
