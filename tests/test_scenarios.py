import pandas as pd
import pytest
from coleman4hcs.scenarios import (
    VirtualScenario,
    VirtualHCSScenario,
    VirtualContextScenario,
    IndustrialDatasetScenarioProvider,
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetContextScenarioProvider,
)


# ------------------------ Fixtures for Mock Data ------------------------

@pytest.fixture
def mock_testcases():
    """Fixture to generate a mock DataFrame of test cases."""
    return [
        {"Name": "TC1", "Duration": "1,000", "CalcPrio": 0, "LastRun": "2023-01-01", "Verdict": 1},
        {"Name": "TC2", "Duration": 2, "CalcPrio": 1, "LastRun": "2023-01-02", "Verdict": 0},
        {"Name": "TC3", "Duration": 3.5, "CalcPrio": 2, "LastRun": "2023-01-03", "Verdict": 1},
    ]


@pytest.fixture
def mock_variants():
    """Fixture to generate a mock DataFrame of variants."""
    return pd.DataFrame({
        "BuildId": [1, 1, 2],
        "Variant": ["A", "B", "A"],
        "LastRun": ["2023-01-01", "2023-01-02", "2023-01-03"]
    })


@pytest.fixture
def mock_csv_data(mock_testcases):
    """Fixture to generate a mock CSV DataFrame."""
    data = pd.DataFrame(mock_testcases)
    data["BuildId"] = [1, 1, 2]
    return data


# ------------------------ Fixtures for Large Mock Data ------------------------

@pytest.fixture
def large_testcases():
    """Fixture to generate a large number of test cases."""
    n = 10000  # Large number of test cases
    return [{"Name": f"TC{i}", "Duration": i % 10 + 1, "CalcPrio": i % 5, "LastRun": "2023-01-01", "Verdict": 1} for i
            in range(n)]


@pytest.fixture
def large_variants():
    """Fixture to generate a large dataset of variants."""
    return pd.DataFrame({
        "BuildId": [i % 100 for i in range(10000)],  # Simulate 100 builds
        "Variant": [f"Variant_{i % 5}" for i in range(10000)],
        "LastRun": ["2023-01-01"] * 10000
    })


@pytest.fixture
def large_csv_data(large_testcases):
    """Fixture to generate a large CSV-like DataFrame."""
    data = pd.DataFrame(large_testcases)
    data["BuildId"] = [i % 100 for i in range(len(data))]  # 100 builds
    return data


# ------------------------ Unit Tests ------------------------

# VirtualScenario
def test_virtual_scenario_initialization(mock_testcases):
    """Test initialization of VirtualScenario."""
    scenario = VirtualScenario(
        available_time=10,
        testcases=mock_testcases,
        build_id=1,
        total_build_duration=6.5
    )
    assert scenario.get_available_time() == 10
    assert scenario.get_testcases() == mock_testcases
    assert scenario.build_id == 1
    assert scenario.total_build_duration == 6.5


def test_virtual_scenario_reset(mock_testcases):
    """Test `reset` method of VirtualScenario."""
    scenario = VirtualScenario(10, mock_testcases, 1, 6.5)
    scenario.reset()
    for case in scenario.get_testcases():
        assert case["CalcPrio"] == 0, "CalcPrio should be reset to 0."


# VirtualHCSScenario
def test_virtual_hcs_scenario(mock_testcases, mock_variants):
    """Test VirtualHCSScenario attributes and methods."""
    scenario = VirtualHCSScenario(
        available_time=10,
        testcases=mock_testcases,
        build_id=1,
        total_build_duration=6.5,
        variants=mock_variants
    )
    assert scenario.get_variants().equals(mock_variants), "Variants should match the input DataFrame."


# VirtualContextScenario
def test_virtual_context_scenario(mock_testcases):
    """Test VirtualContextScenario initialization and getters."""
    context_features = pd.DataFrame({"Name": ["TC1", "TC2"], "feat1": [0.5, 1], "feat2": [1, 0.2]})
    scenario = VirtualContextScenario(
        available_time=10,
        testcases=mock_testcases,
        build_id=1,
        total_build_duration=6.5,
        feature_group="test_group",
        features=["feat1", "feat2"],
        context_features=context_features
    )
    assert scenario.get_feature_group() == "test_group", "Feature group name should match."
    assert scenario.get_features() == ["feat1", "feat2"], "Features should match the input."
    assert scenario.get_context_features().equals(context_features), "Context features should match."


# IndustrialDatasetScenarioProvider
def test_industrial_dataset_scenario_provider(mock_csv_data, tmp_path):
    """Test IndustrialDatasetScenarioProvider functionality, ensuring available_time is computed correctly."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.to_csv(csv_file, sep=";", index=False)

    # Initialize provider with sched_time_ratio = 0.5
    provider = IndustrialDatasetScenarioProvider(str(csv_file), sched_time_ratio=0.5)
    scenario = next(provider)  # Retrieve the first scenario

    # Debugging output
    print("Total Build Duration:", scenario.total_build_duration)
    print("Expected Available Time:", 0.5 * scenario.total_build_duration)
    print("Scenario Available Time:", scenario.get_available_time())

    # Verify total duration and available time
    assert scenario.get_available_time() == 0.5 * scenario.total_build_duration, (
        f"Available time should be half of total duration. "
        f"Expected: {0.5 * scenario.total_build_duration}, Got: {scenario.get_available_time()}"
    )
    assert len(scenario.get_testcases()) == 2, "Should retrieve test cases for BuildId=1 only."


def test_provider_stop_iteration(mock_csv_data, tmp_path):
    """Test StopIteration after all scenarios are retrieved."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.to_csv(csv_file, sep=";", index=False)

    provider = IndustrialDatasetScenarioProvider(str(csv_file))
    scenarios = list(provider)  # Consume all scenarios
    assert len(scenarios) == 2, "Should return scenarios for unique BuildIds."
    with pytest.raises(StopIteration):
        next(provider)


# IndustrialDatasetHCSScenarioProvider
def test_industrial_dataset_hcs_scenario_provider(mock_csv_data, mock_variants, tmp_path):
    """Test IndustrialDatasetHCSScenarioProvider with variants, ensuring available_time is computed correctly."""
    csv_file = tmp_path / "testcases.csv"
    var_file = tmp_path / "variants.csv"
    mock_csv_data.to_csv(csv_file, sep=";", index=False)
    mock_variants.to_csv(var_file, sep=";", index=False)

    # Initialize provider with sched_time_ratio = 0.5
    provider = IndustrialDatasetHCSScenarioProvider(str(csv_file), str(var_file), sched_time_ratio=0.5)
    scenario = next(provider)  # Retrieve the first scenario

    # Debugging output
    print("Total Build Duration:", scenario.total_build_duration)
    print("Expected Available Time:", 0.5 * scenario.total_build_duration)
    print("Scenario Available Time:", scenario.get_available_time())

    # Verify total duration and available time
    assert scenario.get_available_time() == 0.5 * scenario.total_build_duration, (
        f"Available time should be half of total duration. "
        f"Expected: {0.5 * scenario.total_build_duration}, Got: {scenario.get_available_time()}"
    )
    assert not scenario.get_variants().empty, "Variants should be non-empty."


# IndustrialDatasetContextScenarioProvider
def test_industrial_dataset_context_scenario_provider(mock_csv_data, tmp_path):
    """Test IndustrialDatasetContextScenarioProvider."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.to_csv(csv_file, sep=";", index=False)

    provider = IndustrialDatasetContextScenarioProvider(
        str(csv_file),
        feature_group_name="context",
        feature_group_values=["CalcPrio", "Duration"],
        previous_build=["Duration"]
    )
    scenario = next(provider)

    assert scenario.get_features() == ["CalcPrio", "Duration"], "Features should match."
    assert not scenario.get_context_features().empty, "Context features should not be empty."


# ------------------------ Benchmark Tests ------------------------

@pytest.mark.benchmark(group="scenarios")
def test_scenario_provider_benchmark(mock_csv_data, tmp_path, benchmark):
    """Benchmark for scenario provider performance."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.to_csv(csv_file, sep=";", index=False)

    provider = IndustrialDatasetScenarioProvider(str(csv_file))

    def retrieve_all_scenarios():
        return list(provider)

    benchmark(retrieve_all_scenarios)


@pytest.mark.benchmark(group="scenarios")
def test_virtual_scenario_reset_benchmark(mock_testcases, benchmark):
    """Benchmark for resetting a VirtualScenario."""
    scenario = VirtualScenario(10, mock_testcases, 1, 6.5)

    benchmark(scenario.reset)


# ------------------------ Benchmark Tests ------------------------

# VirtualScenario
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_virtual_scenario_reset(large_testcases, benchmark):
    """Benchmark resetting a large number of test cases in VirtualScenario."""
    scenario = VirtualScenario(
        available_time=100,
        testcases=large_testcases,
        build_id=1,
        total_build_duration=50000
    )
    benchmark(scenario.reset)


# VirtualHCSScenario
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_virtual_hcs_scenario_get_variants(large_testcases, large_variants, benchmark):
    """Benchmark getting variants in VirtualHCSScenario."""
    scenario = VirtualHCSScenario(
        available_time=100,
        testcases=large_testcases,
        build_id=1,
        total_build_duration=50000,
        variants=large_variants
    )
    benchmark(scenario.get_variants)


# VirtualContextScenario
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_virtual_context_scenario_get_features(large_testcases, benchmark):
    """Benchmark retrieving features from a VirtualContextScenario."""
    context_features = pd.DataFrame({
        "Name": [f"TC{i}" for i in range(10000)],
        "feat1": [i % 3 for i in range(10000)],
        "feat2": [i % 5 for i in range(10000)]
    })
    scenario = VirtualContextScenario(
        available_time=100,
        testcases=large_testcases,
        build_id=1,
        total_build_duration=50000,
        feature_group="mock_group",
        features=["feat1", "feat2"],
        context_features=context_features
    )
    benchmark(scenario.get_context_features)


# IndustrialDatasetScenarioProvider
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_industrial_scenario_provider_iteration(large_csv_data, tmp_path, benchmark):
    """Benchmark iterating over scenarios with a large dataset."""
    csv_file = tmp_path / "testcases.csv"
    large_csv_data.to_csv(csv_file, sep=";", index=False)

    provider = IndustrialDatasetScenarioProvider(str(csv_file), sched_time_ratio=0.5)

    def iterate_provider():
        return list(provider)

    benchmark(iterate_provider)


# IndustrialDatasetHCSScenarioProvider
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_industrial_hcs_scenario_provider(large_csv_data, large_variants, tmp_path, benchmark):
    """Benchmark initialization and iteration of IndustrialDatasetHCSScenarioProvider with large data."""
    csv_file = tmp_path / "testcases.csv"
    var_file = tmp_path / "variants.csv"
    large_csv_data.to_csv(csv_file, sep=";", index=False)
    large_variants.to_csv(var_file, sep=";", index=False)

    provider = IndustrialDatasetHCSScenarioProvider(str(csv_file), str(var_file), sched_time_ratio=0.5)

    def iterate_provider():
        return list(provider)

    benchmark(iterate_provider)


# IndustrialDatasetContextScenarioProvider
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_industrial_context_scenario_provider(large_csv_data, tmp_path, benchmark):
    """Benchmark context scenario provider initialization and iteration with large data."""
    csv_file = tmp_path / "testcases.csv"
    large_csv_data.to_csv(csv_file, sep=";", index=False)

    provider = IndustrialDatasetContextScenarioProvider(
        str(csv_file),
        feature_group_name="context",
        feature_group_values=["CalcPrio", "Duration"],
        previous_build=["Duration"]
    )

    def iterate_provider():
        return list(provider)

    benchmark(iterate_provider)
