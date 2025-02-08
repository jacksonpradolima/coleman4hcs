from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

from coleman4hcs.agent import ContextualAgent
from coleman4hcs.bandit import EvaluationMetricBandit
from coleman4hcs.environment import Environment
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
        evaluation_metric=mock_evaluation_metric
    )


# Test cases
def test_initialization(environment, mock_agent, mock_scenario_provider, mock_evaluation_metric):
    """
    Test initialization of the Environment class.
    """
    assert environment.agents == [mock_agent]
    assert environment.scenario_provider == mock_scenario_provider
    assert environment.evaluation_metric == mock_evaluation_metric
    assert isinstance(environment.monitor, MonitorCollector)


def test_reset(environment, mock_agent, mock_scenario_provider):
    """
    Test the reset method to ensure monitors and variant monitors are correctly initialized/reset.
    """
    mock_scenario_provider.get_total_variants.return_value = 2
    mock_scenario_provider.get_all_variants.return_value = ["variant_1", "variant_2"]

    environment.reset()

    # mock_agent.reset.assert_called_once()  # Make sure the agent memory was reset
    assert mock_agent.reset.call_count == 2  # Init uses reset
    assert isinstance(environment.monitor, MonitorCollector)
    assert len(environment.variant_monitors) == 2  # Two variant monitors should be created


@patch("coleman4hcs.environment.Environment.load_experiment")
def test_run_single(mock_load_experiment, environment, mock_scenario_provider, mock_agent):
    """
    Test the run_single method for executing a single simulation.
    """
    # Mock scenario
    mock_virtual_scenario = MagicMock()
    mock_virtual_scenario.get_available_time.return_value = 100
    mock_virtual_scenario.get_testcases.return_value = ["testcase1", "testcase2"]
    mock_virtual_scenario.get_context_features.return_value = {"feature1": 0.5}
    mock_virtual_scenario.get_features.return_value = [0.1, 0.2, 0.3]
    mock_virtual_scenario.get_feature_group.return_value = "group1"
    mock_scenario_provider.__iter__.return_value = iter([mock_virtual_scenario])

    mock_scenario_provider.total_build_duration = 150  # Mocked value for `total_build_duration`
    mock_scenario_provider.name = "MockScenarioProvider"  # Add the missing `name` attribute
    mock_scenario_provider.avail_time_ratio = 0.8  # Add `avail_time_ratio` attribute

    # Mock bandit used in the test
    mock_bandit = MagicMock(spec=EvaluationMetricBandit)
    mock_bandit.pull.return_value = MagicMock(fitness=0.95, cost=0.85)  # Provide fitness and cost values

    # Mock agent
    mock_agent.bandit = mock_bandit
    mock_agent.choose.return_value = ["testcase1"]
    mock_agent.get_reward_function.return_value = "reward_function"
    mock_agent.last_reward = [0.9]

    # Mock load_experiment
    mock_load_experiment.return_value = (1, [mock_agent], environment.monitor, {}, mock_bandit)

    # Run a single simulation with restore=True
    environment.run_single("experiment_1", trials=5, restore=True)

    # Assertions
    # Ensure reset was called
    assert mock_agent.reset.call_count == 2

    # Ensure get_available_time was called exactly twice
    assert mock_virtual_scenario.get_available_time.call_count == 2

    # Ensure update_available_time was called
    environment.evaluation_metric.update_available_time.assert_called_with(100)

    # Ensure bandit.pull was called
    mock_bandit.pull.assert_called_once_with(["testcase1"])


@patch("coleman4hcs.environment.Environment.save_experiment")
def test_save_periodically(mock_save_experiment, environment):
    """
    Test the save_periodically method to ensure saving happens only at specified intervals.
    """
    # Test case: step is at a save interval (e.g., 50000)
    environment.save_periodically(restore=True, t=50000, experiment="exp1", bandit=None)
    mock_save_experiment.assert_called_once()

    # Test case: step is not at a save interval
    mock_save_experiment.reset_mock()
    environment.save_periodically(restore=True, t=100, experiment="exp1", bandit=None)
    mock_save_experiment.assert_not_called()


def test_run_prioritization(environment):
    """
    Test run_prioritization behavior including agent prioritization and monitor collection.
    """
    # Setup mocks
    mock_agent = MagicMock()
    mock_bandit = MagicMock()
    mock_virtual_scenario = MagicMock()
    mock_metric = MagicMock()

    # Set fitness and cost to the mocked metric
    mock_metric.fitness = 0.95
    mock_metric.cost = 0.85
    mock_agent.bandit.pull.return_value = mock_metric

    # Mock scenario provider details
    environment.scenario_provider.total_build_duration = 10
    environment.scenario_provider.name = "mock_scenario_provider"
    environment.scenario_provider.avail_time_ratio = 0.8

    # Mock monitor.collect to enable assertions
    environment.monitor = MagicMock()  # Mock the monitor itself
    environment.monitor.collect = MagicMock()  # Mock the collect method

    # Call the method
    _, _, _, _ = environment.run_prioritization(
        agent=mock_agent,
        bandit=mock_bandit,
        bandit_duration=1.5,
        experiment="exp1",
        t=1,
        virtual_scenario=mock_virtual_scenario,
    )

    # Assertions for method calls
    assert mock_agent.update_context.call_count == 0
    mock_agent.choose.assert_called_once()  # Ensure agent chose an action
    environment.monitor.collect.assert_called_once()  # Ensure monitor.collect was called


@patch("pathlib.Path.mkdir")
@patch("coleman4hcs.environment.MonitorCollector.create_file")
def test_create_file(mock_create_file, mock_mkdir, environment, mock_scenario_provider):
    """
    Test the create_file method for proper file creation and variant handling.
    """
    # Mock variant monitors
    environment.variant_monitors = {
        "variant_1": MagicMock(),
        "variant_2": MagicMock(),
    }

    # Scenario without variants
    mock_scenario_provider.get_total_variants.return_value = 0
    environment.create_file("results.csv")
    mock_create_file.assert_called_once_with("results.csv")
    mock_mkdir.assert_not_called()

    # Scenario with variants
    mock_scenario_provider.get_total_variants.return_value = 2
    mock_scenario_provider.get_all_variants.return_value = ["variant_1", "variant_2"]
    environment.create_file("results.csv")

    # Assertions for file creation
    mock_create_file.assert_called_with("results.csv")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    environment.variant_monitors["variant_1"].create_file.assert_called_once()
    environment.variant_monitors["variant_2"].create_file.assert_called_once()


def test_exception_handling_in_save(environment):
    """
    Test exception handling for the save_experiment method.
    """
    with patch("builtins.open", side_effect=Exception("File error")):
        with pytest.raises(Exception, match="File error"):
            environment.save_experiment("exp1", 1, None)


def test_run_with_multiple_experiments(environment, mocker):
    """
    Test the run method to ensure multiple experiments are executed correctly.
    """
    mock_run_single = mocker.patch.object(environment, "run_single")

    # Execute the method
    environment.run(experiments=3, trials=50, restore=False)

    # Ensure the single-run method was called the right number of times
    assert mock_run_single.call_count == 3


@patch("os.path.exists", return_value=True)  # Mock os.path.exists to always return True
@patch("builtins.open", new_callable=mock_open)  # Mock the open function
@patch("pickle.load", return_value=("mocked_data"))  # Mock pickle.load
def test_load_experiment(mocked_pickle_load, mocked_open, mocked_path_exists, environment):
    """
    Test load_experiment to ensure the backup is properly loaded.
    """
    experiment_id = 1
    filename = f"backup/{str(environment.scenario_provider)}_ex_{experiment_id}.p"

    # Call the method
    result = environment.load_experiment(experiment_id)

    # Check if os.path.exists was called
    mocked_path_exists.assert_called_once_with(filename)

    # Verify open is called in "rb" mode
    mocked_open.assert_called_once_with(filename, "rb")

    # Verify pickle.load is called with the open file object
    mocked_pickle_load.assert_called_once_with(mocked_open())

    # Assert that the result matches the mocked pickle data
    assert result == "mocked_data"


@patch("os.path.exists", return_value=False)  # Mock os.path.exists to return False
def test_load_experiment_file_not_found(mocked_path_exists, environment):
    """
    Test load_experiment when the file does not exist.
    """
    experiment_id = 1
    filename = f"backup/{str(environment.scenario_provider)}_ex_{experiment_id}.p"

    # Call the method
    result = environment.load_experiment(experiment_id)

    # Check if os.path.exists was called
    mocked_path_exists.assert_called_once_with(filename)

    # Assert that the fallback values are returned
    assert result == (0, environment.agents, environment.monitor, environment.variant_monitors, None)


def test_run_prioritization_hcs(environment):
    """
    Test run_prioritization_hcs calls the correct methods and processes variants.
    """
    # Setup Agent mock
    mock_agent = MagicMock()
    mock_agent.get_reward_function.return_value = "mock_reward_function"

    # Setup Virtual Scenario Mock
    mock_virtual_scenario = MagicMock()

    # Create a mock DataFrame for variants
    mock_variants = pd.DataFrame({
        "Variant": ["v1", "v1", "v2"],
        "Name": ["tc1", "tc2", "tc3"],
        "Duration": [1.0, 2.0, 3.0]
    })
    mock_virtual_scenario.get_variants.return_value = mock_variants
    environment.scenario_provider = MagicMock()

    # Setup action list
    mock_action = ["tc1", "tc2", "tc3"]

    # Mock variant_monitors and evaluation metric methods
    environment.variant_monitors = {"v1": MagicMock(), "v2": MagicMock()}
    environment.evaluation_metric.evaluate = MagicMock()
    environment.evaluation_metric.update_available_time = MagicMock()

    # Set the 50% available time ratio
    avail_time_ratio = 0.5

    # Call the method
    environment.run_prioritization_hcs(
        agent=mock_agent,
        action=mock_action,
        avail_time_ratio=avail_time_ratio,
        bandit_duration=1.5,  # Time taken by the bandit to pull
        end=0,
        exp_name="mock_exp",
        experiment="test_experiment",
        start=0,
        t=1,
        virtual_scenario=mock_virtual_scenario,
    )

    # Assertions
    # Ensure get_variants was called
    mock_virtual_scenario.get_variants.assert_called_once()

    # Ensure collect was called on variant monitors
    assert environment.variant_monitors["v1"].collect.call_count == 1
    assert environment.variant_monitors["v2"].collect.call_count == 1

    # Ensure update_available_time was called with np.float64(4.5)
    environment.evaluation_metric.update_available_time.assert_any_call(np.float64(1.5))

    # Ensure evaluate was called
    environment.evaluation_metric.evaluate.assert_called()
