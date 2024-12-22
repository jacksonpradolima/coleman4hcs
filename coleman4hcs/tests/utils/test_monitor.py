"""
Test cases for the MonitorCollector utility in the coleman4hcs package.

These tests cover functionality including data collection, handling of temporary
buffers, and performance benchmarks.
"""
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from coleman4hcs.evaluation import EvaluationMetric
from coleman4hcs.utils.monitor import MonitorCollector


@pytest.fixture
def mock_metric():
    """
    Create a mock for the EvaluationMetric class.
    """
    metric = MagicMock(spec=EvaluationMetric)
    metric.detected_failures = 5
    metric.undetected_failures = 3
    metric.scheduled_testcases = ['test1', 'test2', 'test3']
    metric.unscheduled_testcases = ['test4', 'test5']
    metric.ttf = 2
    metric.ttf_duration = 10
    metric.fitness = 0.8
    metric.cost = 0.6
    metric.avg_precision = 0.75
    return metric


@pytest.fixture
def mock_scenario_provider():
    """
    Create a mock for a scenario provider.
    """
    scenario_provider = MagicMock()
    scenario_provider.name = "TestScenario"
    scenario_provider.avail_time_ratio = 0.5
    return scenario_provider


@pytest.fixture
def monitor_collector():
    """
    Create an instance of MonitorCollector.
    """
    return MonitorCollector()


def test_collect_single_record(monitor_collector, mock_scenario_provider, mock_metric):
    """
    Test collecting a single record.
    """
    monitor_collector.collect(
        scenario_provider=mock_scenario_provider,
        available_time=50,
        experiment=1,
        t=1,
        policy="TestPolicy",
        reward_function="TestReward",
        metric=mock_metric,
        total_build_duration=100,
        prioritization_time=10,
        rewards=0.9,
        prioritization_order=['test1', 'test2']
    )

    assert len(monitor_collector.temp_rows) == 1
    record = monitor_collector.temp_rows[0]
    assert record['scenario'] == "TestScenario"
    assert record['experiment'] == 1
    assert record['policy'] == "TestPolicy"
    # Use pytest.approx for floating-point comparisons
    assert record['fitness'] == pytest.approx(0.8, rel=1e-6)
    assert record['cost'] == pytest.approx(0.6, rel=1e-6)


def test_collect_from_temp(monitor_collector, mock_scenario_provider, mock_metric):
    """
    Test transferring data from temp_rows to df.
    """
    for i in range(5):
        monitor_collector.collect(
            scenario_provider=mock_scenario_provider,
            available_time=50,
            experiment=i,
            t=1,
            policy="TestPolicy",
            reward_function="TestReward",
            metric=mock_metric,
            total_build_duration=100,
            prioritization_time=10,
            rewards=0.9,
            prioritization_order=['test1', 'test2']
        )

    monitor_collector.collect_from_temp()

    assert len(monitor_collector.df) == 5
    assert len(monitor_collector.temp_rows) == 0


def test_create_file(tmp_path, monitor_collector):
    """
    Test creating a CSV file with headers.
    """
    file_path = tmp_path / "test_file.csv"
    monitor_collector.create_file(file_path)

    assert os.path.exists(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        assert header == ";".join(monitor_collector.col_names)


def test_save_to_file(tmp_path, monitor_collector, mock_scenario_provider, mock_metric):
    """
    Test saving collected data to a CSV file.
    """
    file_path = tmp_path / "test_save.csv"

    for i in range(3):
        monitor_collector.collect(
            scenario_provider=mock_scenario_provider,
            available_time=50,
            experiment=i,
            t=1,
            policy="TestPolicy",
            reward_function="TestReward",
            metric=mock_metric,
            total_build_duration=100,
            prioritization_time=10,
            rewards=0.9,
            prioritization_order=['test1', 'test2']
        )

    monitor_collector.save(file_path)

    assert os.path.exists(file_path)
    saved_data = pd.read_csv(file_path, sep=';')
    assert len(saved_data) == 3, f"Expected 3 records, found {len(saved_data)}"
    assert 'scenario' in saved_data.columns
    assert saved_data['scenario'].iloc[0] == "TestScenario"


def test_clear(monitor_collector):
    """
    Test clearing the dataframe.
    """
    monitor_collector.df = pd.DataFrame({'a': [1, 2, 3]})
    monitor_collector.temp_rows = [{'b': 4}, {'b': 5}, {'b': 6}]

    monitor_collector.clear()

    assert monitor_collector.df.empty
    assert len(monitor_collector.temp_rows) == 0


def test_temp_limit_trigger(monitor_collector, mock_scenario_provider, mock_metric):
    """
    Test that collect_from_temp is triggered when temp_rows exceeds the temp_limit.
    """
    monitor_collector.temp_limit = 5

    for i in range(6):  # Exceed the limit by 1 record
        monitor_collector.collect(
            scenario_provider=mock_scenario_provider,
            available_time=50,
            experiment=i,
            t=1,
            policy="TestPolicy",
            reward_function="TestReward",
            metric=mock_metric,
            total_build_duration=100,
            prioritization_time=10,
            rewards=0.9,
            prioritization_order=['test1', 'test2']
        )

    # After exceeding the limit, temp_rows should only contain the last record
    assert len(
        monitor_collector.temp_rows) == 1, f"Expected 1 record in temp_rows, found {len(monitor_collector.temp_rows)}"

    # The main df should have the flushed records
    assert len(monitor_collector.df) == 5, f"Expected 5 records in df, found {len(monitor_collector.df)}"


def test_temp_limit_exceeded():
    """
    Test the behavior of MonitorCollector when the temp_limit is exceeded.
    Ensures temp_rows is flushed into df upon exceeding temp_limit.
    """
    # Create a mock EvaluationMetric
    mock_metric = MagicMock()
    mock_metric.detected_failures = 5
    mock_metric.undetected_failures = 3
    mock_metric.scheduled_testcases = [1, 2, 3]
    mock_metric.unscheduled_testcases = [4, 5]
    mock_metric.ttf = 1
    mock_metric.ttf_duration = 10
    mock_metric.fitness = 0.9
    mock_metric.cost = 0.8
    mock_metric.avg_precision = 0.85

    # Create a mock scenario provider
    mock_scenario_provider = MagicMock()
    mock_scenario_provider.name = "TestScenario"
    mock_scenario_provider.avail_time_ratio = 0.5

    # Instantiate the MonitorCollector
    collector = MonitorCollector()
    collector.temp_limit = 1000  # Ensure this matches the actual limit

    # Generate records to exceed temp_limit
    for i in range(1100):  # Exceed the temp_limit by 100 records
        collector.collect(
            scenario_provider=mock_scenario_provider,
            available_time=0.5,
            experiment=1,
            t=i,
            policy="PolicyA",
            reward_function="RewardFuncA",
            metric=mock_metric,
            total_build_duration=100,
            prioritization_time=10,
            rewards=0.95,
            prioritization_order=[1, 2, 3]
        )

    # Check that temp_rows has only the remaining 100 records
    assert len(collector.temp_rows) == 100, "Temp rows should contain the remaining records after flush."

    # Check that df has exactly 1000 records flushed from temp_rows
    assert len(collector.df) == 1000, "Main DataFrame should contain 1000 records after flushing temp DataFrame."

    # Ensure all data is preserved
    total_records = len(collector.df) + len(collector.temp_rows)
    assert total_records == 1100, "Total records should match the number of records added."

    # Validate the structure of the DataFrame
    assert all(col in collector.df.columns for col in collector.col_names), "DataFrame structure is incorrect."


def test_collect_from_temp_empty_temp_rows(monitor_collector):
    """
    Test that collect_from_temp does nothing when temp_rows is empty.
    """
    monitor_collector.temp_rows = []  # Ensure temp_rows is empty
    monitor_collector.collect_from_temp()

    # Assert that df remains empty
    assert monitor_collector.df.empty, "Expected df to remain empty when temp_rows is empty."


def test_collect_from_temp_empty_batch_df(monitor_collector):
    """
    Test that collect_from_temp does nothing when batch_df is empty.
    """
    # Create a temp_rows list that results in an empty DataFrame
    monitor_collector.temp_rows = [{}]  # Add a record that lacks proper data
    monitor_collector.collect_from_temp()

    # Assert that df remains empty
    assert monitor_collector.df.empty, (
        f"Expected df to remain empty when batch_df is empty. Found: {monitor_collector.df}"
    )
    # Assert that temp_rows is cleared
    assert not monitor_collector.temp_rows, (
        f"Expected temp_rows to be cleared but found: {monitor_collector.temp_rows}"
    )


def test_create_file_existing_file(tmp_path, monitor_collector):
    """
    Test that create_file does not overwrite an existing file.
    """
    file_path = tmp_path / "test_existing_file.csv"

    # Create a file manually
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("existing_header\n")

    # Call create_file and verify it does not overwrite
    monitor_collector.create_file(file_path)

    # Read back the file and ensure it retains the original content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert content == "existing_header\n", "File content was unexpectedly overwritten."


def test_save_with_empty_temp_rows(tmp_path, monitor_collector):
    """
    Test saving data to a CSV file when temp_rows is empty.
    """
    file_path = tmp_path / "test_empty_temp_rows_save.csv"

    # Ensure temp_rows is empty
    monitor_collector.temp_rows = []

    # Add some data to df directly
    monitor_collector.df = pd.DataFrame(
        [{'scenario': 'TestScenario', 'experiment': 1, 'step': 1, 'policy': 'PolicyA'}],
        columns=monitor_collector.col_names
    )

    # Call save and ensure no errors occur
    monitor_collector.save(file_path)

    # Verify that the file is created and has the data from df
    assert os.path.exists(file_path), "File was not created."
    saved_data = pd.read_csv(file_path, sep=';')
    assert len(saved_data) == 1, f"Expected 1 record, found {len(saved_data)}."
    assert saved_data['scenario'].iloc[0] == "TestScenario"


def test_save_with_non_empty_temp_rows(tmp_path, monitor_collector, mock_scenario_provider, mock_metric):
    """
    Test saving data to a CSV file when temp_rows has data.
    """
    file_path = tmp_path / "test_non_empty_temp_rows_save.csv"

    # Collect some data into temp_rows
    for i in range(3):
        monitor_collector.collect(
            scenario_provider=mock_scenario_provider,
            available_time=50,
            experiment=i,
            t=1,
            policy="TestPolicy",
            reward_function="TestReward",
            metric=mock_metric,
            total_build_duration=100,
            prioritization_time=10,
            rewards=0.9,
            prioritization_order=['test1', 'test2']
        )

    # Call save and ensure temp_rows is flushed
    monitor_collector.save(file_path)

    # Verify that the file is created and has the data from temp_rows
    assert os.path.exists(file_path), "File was not created."
    saved_data = pd.read_csv(file_path, sep=';')
    assert len(saved_data) == 3, f"Expected 3 records, found {len(saved_data)}."

    # Verify that temp_rows is empty
    assert not monitor_collector.temp_rows, "temp_rows should be empty after save."


@pytest.mark.performance
@pytest.mark.parametrize("num_records", [1000, 10_000, 100_000])
def test_temp_limit_performance(benchmark, num_records):
    """
    Test the performance of the MonitorCollector when collecting large numbers of records.

    This test benchmarks the time taken to add records to the MonitorCollector and ensures
    that all records are correctly handled, whether stored in `temp_rows` or flushed to `df`.

    Args:
        benchmark: Pytest benchmark fixture to measure performance.
        num_records (int): Number of records to test performance for.
    """
    mock_metric = MagicMock()
    mock_metric.detected_failures = 5
    mock_metric.undetected_failures = 3
    mock_metric.scheduled_testcases = [1, 2, 3]
    mock_metric.unscheduled_testcases = [4, 5]
    mock_metric.ttf = 1
    mock_metric.ttf_duration = 10
    mock_metric.fitness = 0.9
    mock_metric.cost = 0.8
    mock_metric.avg_precision = 0.85

    mock_scenario_provider = MagicMock()
    mock_scenario_provider.name = "BenchmarkScenario"
    mock_scenario_provider.avail_time_ratio = 0.5

    def add_records():
        collector = MonitorCollector()  # Re-instantiate for each round
        collector.temp_limit = 1000
        for i in range(num_records):
            collector.collect(
                scenario_provider=mock_scenario_provider,
                available_time=0.5,
                experiment=1,
                t=i,
                policy="PolicyBenchmark",
                reward_function="RewardBenchmark",
                metric=mock_metric,
                total_build_duration=100,
                prioritization_time=10,
                rewards=0.95,
                prioritization_order=[1, 2, 3]
            )
        collector.collect_from_temp()
        return collector

    # Benchmark and verify results
    collector = benchmark(add_records)
    total_records = len(collector.df) + len(collector.temp_rows)
    assert total_records == num_records, f"Expected {num_records} records, found {total_records}."
