"""
Unit and integration tests for the main application logic.

This module contains unit and integration tests to verify the functionality
of various core functions present in the project's `main` module. The tests
cover a wide range of behaviors, including logging setup, managing agents,
loading classes dynamically, working with experiments, scenario providers,
and CSV processing.

Purpose:
- Ensure the robustness, reliability, and correct behavior of the `main` module's core logic.
- Validate the integrations with other components like agents, policies, and scenario providers.

Tested Functionalities:
1. **Logging**:
   - `create_logger`: Verifies correct logger creation and configuration.

2. **Experiment Execution**:
   - `exp_run_industrial_dataset`: Tests running experiments with mock environments.

3. **Dynamic Class Loading**:
   - `load_class_from_module`: Ensures classes are correctly loaded dynamically from modules.

4. **Agent Creation**:
   - `create_agents`: Verifies the creation of different agent types (e.g., `RewardSlidingWindowAgent`, `ContextualAgent`) based on the provided policies and window sizes.

5. **Scenario Setup**:
   - `get_scenario_provider`: Tests the initialization of different scenario providers.

6. **File Management**:
   - `merge_csv`: Verifies merging of multiple CSV files into one while ensuring cleanup of temporary files.
   - `store_experiments`: Tests the storage of experiment results into the database.

7. **End-to-End Workflow**:
   - Ensures integration of various components with mock objects to validate behavior across larger workflows.

Dependencies:
- `pytest` for test management and fixtures.
- `unittest.mock` for mocking dependencies like functions, modules, and objects.
- Core components from the `coleman4hcs` package that integrate with the tested functions.

Test Structure:
- **Unit Tests**:
  Focused on validating individual functions in the `main` module.
  Examples: `test_create_logger`, `test_create_agents_frrmab`.

- **Integration Tests**:
  Comprehensive tests to ensure modules and components work together as expected.
  Examples: `test_end_to_end_execution`.

"""
import csv
import logging
from unittest.mock import Mock, patch, call

import pytest

from coleman4hcs.agent import RewardSlidingWindowAgent, ContextualAgent
from coleman4hcs.policy import FRRMABPolicy, SWLinUCBPolicy
from coleman4hcs.scenarios import IndustrialDatasetHCSScenarioProvider, IndustrialDatasetScenarioProvider
from main import (
    create_logger,
    exp_run_industrial_dataset,
    load_class_from_module,
    create_agents,
    get_scenario_provider,
    merge_csv,
    store_experiments
)


# ------------------------
# Unit Tests
# ------------------------

def test_create_logger():
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

@patch("main.create_logger")
@patch("main.Environment")
def test_exp_run_industrial_dataset(mock_environment, mock_create_logger, tmpdir):
    mock_env = mock_environment.return_value
    mock_logger = mock_create_logger.return_value
    mock_env.scenario_provider = Mock()
    mock_env.logger = mock_logger

    exp_run_industrial_dataset(
        iteration=1,
        trials=10,
        env=mock_env,
        experiment_directory=str(tmpdir),
        level=20
    )

    mock_env.create_file.assert_called_once()
    mock_env.run_single.assert_called_once()
    mock_env.store_experiment.assert_called_once()


def test_load_class_from_module_valid():
    import coleman4hcs.policy
    policy_class = load_class_from_module(coleman4hcs.policy, "FRRMABPolicy")
    assert policy_class.__name__ == "FRRMABPolicy"


def test_load_class_from_module_invalid():
    import coleman4hcs.policy
    with pytest.raises(ValueError, match="Class 'InvalidPolicy' not found"):
        load_class_from_module(coleman4hcs.policy, "InvalidPolicy")


def test_create_agents_frrmab():
    policy = Mock(spec=FRRMABPolicy)
    reward_fun = Mock()
    window_sizes = [5, 10]

    agents = create_agents(policy, reward_fun, window_sizes)
    assert len(agents) == 2
    assert all(isinstance(agent, RewardSlidingWindowAgent) for agent in agents)


def test_create_agents_swlinucb():
    policy = Mock(spec=SWLinUCBPolicy)
    reward_fun = Mock()
    window_sizes = [5, 10]

    agents = create_agents(policy, reward_fun, window_sizes)
    assert len(agents) == 2
    assert all(isinstance(agent, ContextualAgent) for agent in agents)


def test_get_scenario_provider_basic():
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


@patch("main.os.remove")
@patch("main.pl.read_csv")
@patch("main.pl.concat")
def test_merge_csv(mock_concat, mock_read_csv, mock_remove, tmpdir):
    # Mocked return values
    df_merged = Mock()
    mock_concat.return_value = df_merged

    # Temporary CSV files
    temp_files = [tmpdir.join(f"file{i}.csv") for i in range(3)]

    # Call merge_csv
    output_file = tmpdir.join("output.csv")
    merge_csv(temp_files, output_file)

    # Ensure files were merged
    mock_read_csv.assert_has_calls([call(file, separator=";") for file in temp_files])
    mock_concat.assert_called_once()
    df_merged.write_csv.assert_called_once_with(output_file, separator=";", quote_style="never")

    # Ensure temp files were deleted
    mock_remove.assert_has_calls([call(file) for file in temp_files])


@patch("main.duckdb.connect")
@patch("main.pl.read_csv")
def test_store_experiments(mock_read_csv, mock_duckdb_connect, tmpdir):
    # Mock database connection and DataFrame
    mock_conn = mock_duckdb_connect.return_value
    mock_df = Mock()
    mock_read_csv.return_value = mock_df

    # CSV file and dummy scenario
    csv_file = tmpdir.join("results.csv")
    scenario = Mock(spec=IndustrialDatasetHCSScenarioProvider)
    scenario.get_total_variants.return_value = 0  # Explicitly define behavior for mock

    # Call store_experiments
    store_experiments(csv_file, scenario)

    # Verify database commands
    mock_conn.execute.assert_called()
    mock_conn.execute.assert_any_call("INSERT INTO experiments SELECT * FROM df;")


# ------------------------
# Integration Tests
# ------------------------

@patch("main.create_logger")
@patch("main.Environment")
def test_end_to_end_execution(mock_environment, mock_create_logger, tmpdir):
    # Mock environment and logger creation
    mock_env = mock_environment.return_value
    mock_env.scenario_provider = Mock()
    mock_env.max_builds = 10
    mock_env.logger = mock_create_logger.return_value

    # Temporary experiment directory
    experiment_directory = tmpdir.join("experiment/")

    # Parameters
    independent_executions = 3
    parameters = [
        (i + 1, 10, mock_env, experiment_directory, 20)
        for i in range(independent_executions)
    ]

    # Parallel pool size test (single-threaded for simplicity)
    for param in parameters:
        exp_run_industrial_dataset(*param)

    # Check that required environment methods were called
    assert mock_env.create_file.call_count == independent_executions
    assert mock_env.run_single.call_count == independent_executions
    assert mock_env.store_experiment.call_count == independent_executions
