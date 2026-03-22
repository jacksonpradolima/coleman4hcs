"""
Unit tests for the Environment class and related components.

This module contains a suite of unit tests to validate the functionality of
the `Environment` class as well as the associated functionalities in the
`coleman4hcs` package. These tests cover the behaviors, interactions, and
edge cases for:

- Initialization of `Environment` objects.
- Reset and run behaviors for the Environment with various scenarios and agents.
- Integration with other components such as Scenario Providers, Agents, Bandits, and Monitor Collectors.
- Checkpoint handling and experiment recovery.

Fixtures:
- `mock_agent`: A fixture for a mocked `ContextualAgent`.
- `mock_scenario_provider`: A fixture for a mocked `IndustrialDatasetHCSScenarioProvider`.
- `mock_evaluation_metric`: A fixture for a mocked evaluation metric.
- `environment`: A fixture for an initialized instance of the `Environment` class.

Tested Functionalities:
1. Initialization and reset operations.
2. Simulation workflows, including single runs (`run_single`) and prioritization.
3. Checkpoint-based saving (`save_experiment`) and loading (`load_experiment`).
4. Exception handling during file operations.
5. Monitoring and metric updates during simulations.
6. Execution and prioritization of test cases and variants.
"""

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from coleman4hcs.agent import ContextualAgent
from coleman4hcs.bandit import EvaluationMetricBandit
from coleman4hcs.checkpoint.checkpoint_store import NullCheckpointStore
from coleman4hcs.checkpoint.state import CheckpointPayload
from coleman4hcs.environment import Environment
from coleman4hcs.results.sink_base import NullSink
from coleman4hcs.scenarios import IndustrialDatasetHCSScenarioProvider
from coleman4hcs.utils.monitor import MonitorCollector


# Fixtures for common dependencies
@pytest.fixture
def mock_agent():
    """Fixture for creating a mock agent."""
    return MagicMock(spec=ContextualAgent)


@pytest.fixture
def mock_scenario_provider():
    """Fixture for creating a mock scenario provider."""
    mock_provider = MagicMock(spec=IndustrialDatasetHCSScenarioProvider)
    mock_provider.get_total_variants.return_value = 0
    mock_provider.get_avail_time_ratio.return_value = 0.5
    return mock_provider


@pytest.fixture
def mock_evaluation_metric():
    """Fixture for creating a mock evaluation metric."""
    return MagicMock()


@pytest.fixture
def environment(mock_agent, mock_scenario_provider, mock_evaluation_metric):
    """Fixture for creating an instance of Environment."""
    return Environment(
        agents=[mock_agent],
        scenario_provider=mock_scenario_provider,
        evaluation_metric=mock_evaluation_metric,
    )


# Test cases
def test_initialization(environment, mock_agent, mock_scenario_provider, mock_evaluation_metric):
    """Test initialization of the Environment class."""
    assert environment.agents == [mock_agent]
    assert environment.scenario_provider == mock_scenario_provider
    assert environment.evaluation_metric == mock_evaluation_metric
    assert isinstance(environment.monitor, MonitorCollector)
    assert isinstance(environment.checkpoint_store, NullCheckpointStore)


def test_reset(environment, mock_agent, mock_scenario_provider):
    """Test the reset method to ensure monitors and variant monitors are correctly initialized/reset."""
    mock_scenario_provider.get_total_variants.return_value = 2
    mock_scenario_provider.get_all_variants.return_value = ["variant_1", "variant_2"]

    environment.reset()

    assert mock_agent.reset.call_count == 2  # Init uses reset
    assert isinstance(environment.monitor, MonitorCollector)
    assert len(environment.variant_monitors) == 2


def test_run_single(environment, mock_scenario_provider, mock_agent):
    """Test the run_single method for executing a single simulation."""
    # Mock scenario
    mock_virtual_scenario = MagicMock()
    mock_virtual_scenario.get_available_time.return_value = 100
    mock_virtual_scenario.get_testcases.return_value = ["testcase1", "testcase2"]
    mock_virtual_scenario.get_context_features.return_value = {"feature1": 0.5}
    mock_virtual_scenario.get_features.return_value = [0.1, 0.2, 0.3]
    mock_virtual_scenario.get_feature_group.return_value = "group1"
    mock_scenario_provider.__iter__.return_value = iter([mock_virtual_scenario])

    mock_scenario_provider.total_build_duration = 150
    mock_scenario_provider.name = "MockScenarioProvider"
    mock_scenario_provider.avail_time_ratio = 0.8

    # Mock bandit
    mock_bandit = MagicMock(spec=EvaluationMetricBandit)
    mock_bandit.pull.return_value = MagicMock(fitness=0.95, cost=0.85)

    # Mock agent
    mock_agent.bandit = mock_bandit
    mock_agent.choose.return_value = ["testcase1"]
    mock_agent.get_reward_function.return_value = "reward_function"
    mock_agent.last_reward = [0.9]

    # Mock checkpoint store to return a checkpoint
    environment.checkpoint_store = MagicMock()
    environment.checkpoint_store.load.return_value = CheckpointPayload(
        run_id="MockScenarioProvider",
        experiment=1,
        step=1,
        agents=[mock_agent],
        bandit=mock_bandit,
    )

    # Run a single simulation with restore=True
    environment.run_single("experiment_1", trials=5, restore=True)

    # Assertions
    assert mock_agent.reset.call_count == 2
    assert mock_virtual_scenario.get_available_time.call_count == 2
    environment.evaluation_metric.update_available_time.assert_called_with(100)
    mock_bandit.pull.assert_called_once_with(["testcase1"])


def test_run_single_resets_scenario_provider_and_run_lifecycle(environment, mock_scenario_provider):
    """Each independent experiment must restart the scenario stream and close telemetry lifecycle."""
    mock_virtual_scenario = MagicMock()
    mock_virtual_scenario.get_available_time.return_value = 100
    mock_virtual_scenario.get_testcases.return_value = [
        {"Name": "testcase1", "Duration": 1.0, "CalcPrio": 0, "LastRun": "0", "LastResults": ""},
        {"Name": "testcase2", "Duration": 2.0, "CalcPrio": 0, "LastRun": "0", "LastResults": ""},
    ]
    mock_scenario_provider.__iter__.side_effect = lambda: iter([mock_virtual_scenario])
    mock_scenario_provider.total_build_duration = 150

    environment.telemetry = MagicMock()
    environment.run_prioritization = MagicMock(return_value=([], 0.0, "policy", 0.0))
    environment.save_periodically = MagicMock()

    environment.run_single("experiment_1", trials=1, restore=False)
    environment.run_single("experiment_2", trials=1, restore=False)

    assert mock_scenario_provider.last_build.call_args_list[0].args == (0,)
    assert mock_scenario_provider.last_build.call_args_list[1].args == (0,)
    assert environment.telemetry.mark_run_started.call_count == 2
    assert environment.telemetry.mark_run_finished.call_count == 2
    assert environment.telemetry.flush.call_count == 2


def test_run_single_resumes_from_checkpoint_step(environment, mock_scenario_provider, mock_agent):
    """Resume must continue from the step after the checkpoint instead of replaying prior builds."""
    mock_virtual_scenario = MagicMock()
    mock_virtual_scenario.get_available_time.return_value = 100
    mock_virtual_scenario.get_testcases.return_value = [
        {"Name": "testcase1", "Duration": 1.0, "CalcPrio": 0, "LastRun": "0", "LastResults": ""},
        {"Name": "testcase2", "Duration": 2.0, "CalcPrio": 0, "LastRun": "0", "LastResults": ""},
    ]
    mock_scenario_provider.__iter__.side_effect = lambda: iter([mock_virtual_scenario])
    mock_scenario_provider.total_build_duration = 150

    mock_bandit = MagicMock(spec=EvaluationMetricBandit)
    mock_agent.bandit = mock_bandit
    mock_agent.get_reward_function.return_value = "reward_function"
    mock_agent.last_reward = [0.4]

    environment.checkpoint_store = MagicMock()
    environment.checkpoint_store.load.return_value = CheckpointPayload(
        run_id=str(mock_scenario_provider),
        experiment=1,
        step=5,
        agents=[mock_agent],
        bandit=mock_bandit,
    )
    environment.run_prioritization = MagicMock(return_value=([], 0.0, "policy", 0.0))
    environment.save_periodically = MagicMock()

    environment.run_single(1, trials=10, restore=True)

    assert mock_scenario_provider.last_build.call_args_list[0].args == (0,)
    assert mock_scenario_provider.last_build.call_args_list[1].args == (5,)
    assert environment.run_prioritization.call_args.args[4] == 6
    mock_bandit.update_arms.assert_called_once()


def test_save_periodically(environment):
    """Test the save_periodically method to ensure saving happens only at specified intervals."""
    environment.save_experiment = MagicMock()

    # Test case: step is at a save interval (e.g., 50000)
    environment.save_periodically(restore=True, t=50000, experiment="exp1", bandit=None)
    environment.save_experiment.assert_called_once()

    # Test case: step is not at a save interval
    environment.save_experiment.reset_mock()
    environment.save_periodically(restore=True, t=100, experiment="exp1", bandit=None)
    environment.save_experiment.assert_not_called()


def test_run_prioritization(environment):
    """Test run_prioritization behavior including agent prioritization and monitor collection."""
    mock_agent = MagicMock()
    mock_bandit = MagicMock()
    mock_virtual_scenario = MagicMock()
    mock_metric = MagicMock()

    mock_metric.fitness = 0.95
    mock_metric.cost = 0.85
    mock_agent.bandit.pull.return_value = mock_metric

    environment.scenario_provider.total_build_duration = 10
    environment.scenario_provider.name = "mock_scenario_provider"
    environment.scenario_provider.avail_time_ratio = 0.8

    environment.monitor = MagicMock()
    environment.monitor.collect = MagicMock()

    _, _, _, _ = environment.run_prioritization(
        agent=mock_agent,
        bandit=mock_bandit,
        bandit_duration=1.5,
        experiment="exp1",
        t=1,
        virtual_scenario=mock_virtual_scenario,
    )

    assert mock_agent.update_context.call_count == 0
    mock_agent.choose.assert_called_once()
    environment.monitor.collect.assert_called_once()


def test_exception_handling_in_save(environment):
    """Test exception handling for the save_experiment method."""
    environment.checkpoint_store = MagicMock()
    environment.checkpoint_store.save.side_effect = Exception("Save error")

    with pytest.raises(Exception, match="Save error"):
        environment.save_experiment("exp1", 1, None)


def test_run_with_multiple_experiments(environment, mocker):
    """Test the run method to ensure multiple experiments are executed correctly."""
    mock_run_single = mocker.patch.object(environment, "run_single")

    environment.run(experiments=3, trials=50, restore=False)

    assert mock_run_single.call_count == 3


def test_load_experiment_returns_none_when_no_checkpoint(environment):
    """Test load_experiment returns None when no checkpoint exists."""
    result = environment.load_experiment(1)
    assert result is None


def test_load_experiment_returns_payload(environment, tmp_path):
    """Test load_experiment returns a CheckpointPayload when checkpoint exists."""
    from coleman4hcs.checkpoint.checkpoint_store import LocalCheckpointStore

    store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
    environment.checkpoint_store = store

    payload = CheckpointPayload(
        run_id=str(environment.scenario_provider),
        experiment=1,
        step=100,
        agents=["agent1"],
    )
    store.save(payload)

    loaded = environment.load_experiment(1)
    assert loaded is not None
    assert loaded.step == 100
    assert loaded.agents == ["agent1"]


def test_store_experiment_flushes(environment):
    """Test store_experiment flushes the monitor."""
    environment.monitor = MagicMock()
    environment.variant_monitors = {"v1": MagicMock()}

    environment.store_experiment()

    environment.monitor.flush.assert_called_once()
    environment.variant_monitors["v1"].flush.assert_called_once()


def test_run_prioritization_hcs(environment):
    """Test run_prioritization_hcs calls the correct methods and processes variants."""
    mock_agent = MagicMock()
    mock_agent.get_reward_function.return_value = "mock_reward_function"

    mock_virtual_scenario = MagicMock()

    mock_variants = pl.DataFrame(
        {"Variant": ["v1", "v1", "v2"], "Name": ["tc1", "tc2", "tc3"], "Duration": [1.0, 2.0, 3.0]}
    )
    mock_virtual_scenario.get_variants.return_value = mock_variants
    environment.scenario_provider = MagicMock()

    mock_action = ["tc1", "tc2", "tc3"]

    environment.variant_monitors = {"v1": MagicMock(), "v2": MagicMock()}
    environment.evaluation_metric.evaluate = MagicMock()
    environment.evaluation_metric.update_available_time = MagicMock()

    avail_time_ratio = 0.5

    environment.run_prioritization_hcs(
        agent=mock_agent,
        action=mock_action,
        avail_time_ratio=avail_time_ratio,
        bandit_duration=1.5,
        end=0,
        exp_name="mock_exp",
        experiment="test_experiment",
        start=0,
        t=1,
        virtual_scenario=mock_virtual_scenario,
    )

    mock_virtual_scenario.get_variants.assert_called_once()
    assert environment.variant_monitors["v1"].collect.call_count == 1
    assert environment.variant_monitors["v2"].collect.call_count == 1
    environment.evaluation_metric.update_available_time.assert_any_call(np.float64(1.5))
    environment.evaluation_metric.evaluate.assert_called()


def test_environment_with_results_config(mock_agent, mock_scenario_provider, mock_evaluation_metric, tmp_path):
    """Test Environment creation with results config produces a ParquetSink."""
    from coleman4hcs.results.parquet_sink import ParquetSink

    env = Environment(
        agents=[mock_agent],
        scenario_provider=mock_scenario_provider,
        evaluation_metric=mock_evaluation_metric,
        results_config={"enabled": True, "sink": "parquet", "out_dir": str(tmp_path / "runs"), "batch_size": 50},
    )
    assert isinstance(env.monitor.sink, ParquetSink)


def test_environment_disabled_results(mock_agent, mock_scenario_provider, mock_evaluation_metric):
    """Test Environment with disabled results uses NullSink."""
    env = Environment(
        agents=[mock_agent],
        scenario_provider=mock_scenario_provider,
        evaluation_metric=mock_evaluation_metric,
        results_config={"enabled": False},
    )
    assert isinstance(env.monitor.sink, NullSink)
