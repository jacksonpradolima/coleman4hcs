"""
`coleman4hcs.scenarios` - Scenario Management for the Coleman4HCS Framework

This module provides utilities for managing and processing different scenarios in the context of the
Coleman4HCS framework. This includes virtual scenarios for commits, scenarios specific to HCS context,
and scenarios that consider context information for each commit. The module also provides utilities to process CSV
files for experimental evaluations.

Classes:
    - VirtualScenario: Basic virtual scenario to manipulate data for each commit.
    - VirtualHCSScenario: Extends VirtualScenario to handle HCS context.
    - VirtualContextScenario: Extends VirtualScenario to handle context information.
    - IndustrialDatasetScenarioProvider: Provider to process CSV files for experiments.
    - IndustrialDatasetHCSScenarioProvider: Extends IndustrialDatasetScenarioProvider to handle HCS scenarios.
    - IndustrialDatasetContextScenarioProvider: Extends IndustrialDatasetScenarioProvider to handle context scenarios.
"""
import os
from typing import List, Dict, Optional

import polars as pl


class VirtualScenario:
    """
    Virtual Scenario, used to manipulate the data for each commit
    """

    def __init__(self, available_time: float, testcases: List[Dict], build_id: int, total_build_duration: float):
        self.available_time = available_time
        self.testcases = testcases
        self.build_id = build_id
        self.total_build_duration = total_build_duration
        self.reset()

    def reset(self) -> None:
        """Resets the priorities for all test cases in the scenario."""
        # Reset the priorities
        for testcase in self.testcases:
            testcase['CalcPrio'] = 0

    def get_available_time(self) -> float:
        """Returns the available time to execute the tests."""
        return self.available_time

    def get_testcases(self) -> List[Dict]:
        """Returns the test cases for the scenario."""
        return self.testcases


class VirtualHCSScenario(VirtualScenario):
    """
    Extends VirtualScenario to handle data in an HCS-specific context.
    """

    def __init__(self, *args, variants: pl.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self.variants = variants

    def get_variants(self):
        """Returns the variants associated with the system."""
        return self.variants


class VirtualContextScenario(VirtualScenario):
    """
    Extends VirtualScenario to include context-specific features for each commit.
    """

    def __init__(
        self,
        *args,
        feature_group: str,
        features: List[str],
        context_features: pl.DataFrame,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.feature_group = feature_group
        self.features = features
        self.context_features = context_features

    def get_feature_group(self) -> str:
        """Returns the feature group."""
        return self.feature_group

    def get_features(self) -> List[str]:
        """Returns the features associated."""
        return self.features

    def get_context_features(self) -> pl.DataFrame:
        """Returns the context features associated."""
        return self.context_features


class IndustrialDatasetScenarioProvider:
    """
    Base class for scenario providers that process test case data from CSV files.
    """

    # ColName | Description
    # Name | Unique numeric identifier of the test case
    # Duration | Approximated runtime of the test case
    # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
    # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
    # LastResults | List of previous test results (Failed: 1, Passed: 0), ordered by ascending age
    # Verdict | Test Case result (Failed: 1, Passed: 0)
    REQUIRED_COLUMNS = ['Name', 'Duration', 'CalcPrio', 'LastRun', 'Verdict']

    def __init__(self, tcfile: str, sched_time_ratio: float = 0.5) -> None:
        self.name = os.path.split(os.path.dirname(tcfile))[1]
        self.avail_time_ratio = sched_time_ratio
        self.current_build = 0
        self.total_build_duration = 0
        self.scenario: Optional[VirtualScenario] = None

        self.tcdf = self._read_testcases(tcfile)
        self.max_builds = self.tcdf["BuildId"].max()

    def _read_testcases(self, tcfile: str) -> pl.DataFrame:
        """Reads the test cases from a provided CSV file."""
        # We use ';' separated values to avoid issues with thousands
        df = pl.read_csv(tcfile, separator=';', try_parse_dates=True)
        
        # Handle Duration column - convert to numeric and fill nulls
        df = df.with_columns([
            pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0)
        ])
        
        return df

    def get_avail_time_ratio(self) -> float:
        """Returns the available time ratio."""
        return self.avail_time_ratio

    def last_build(self, build: int) -> None:
        """Sets the last build."""
        self.current_build = build

    def get(self) -> Optional[VirtualScenario]:
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        self.current_build += 1

        # Stop when reaches the max build
        if self.current_build > self.max_builds:
            return None

        # Select the data for the current build
        build_df = self.tcdf.filter(pl.col("BuildId") == self.current_build)

        # Convert the solutions to a list of dict
        testcases = build_df.select(self.REQUIRED_COLUMNS).to_dicts()

        self.total_build_duration = build_df['Duration'].sum()
        available_time = self.total_build_duration * self.avail_time_ratio

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualScenario(
            available_time=available_time,
            testcases=testcases,
            build_id=self.current_build,
            total_build_duration=self.total_build_duration)

        return self.scenario

    def __str__(self) -> str:
        """Returns the name of the scenario provider."""
        return self.name

    def __iter__(self):
        """Returns the iterator for the scenario provider."""
        return self

    def __next__(self) -> VirtualScenario:
        """Returns the next scenario. Raises StopIteration if no more scenarios."""
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


class IndustrialDatasetHCSScenarioProvider(IndustrialDatasetScenarioProvider):
    """
    Scenario provider for HCS-specific data, extending the base scenario provider.
    """

    def __init__(self, tcfile: str, variantsfile: str, sched_time_ratio=0.5) -> None:
        super().__init__(tcfile, sched_time_ratio)

        self.variants = self._read_variants(variantsfile)

    def _read_variants(self, variantsfile: str) -> pl.DataFrame:
        # Read the variants (additional file)
        df = pl.read_csv(variantsfile, separator=';', try_parse_dates=True)

        # We remove weird characters
        df = df.with_columns([
            pl.col("Variant").str.replace_all(r'[!#$%^&*()\[\]{};:,.<>?|`~=+]', '_')
        ])

        return df

    def get_total_variants(self):
        return self.variants['Variant'].n_unique()

    def get_all_variants(self):
        return self.variants['Variant'].unique().to_list()

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        base_scenario = super().get()

        if not base_scenario:
            return None

        # Match variants to the current build
        variants = self.variants.filter(pl.col("BuildId") == self.current_build)

        self.scenario = VirtualHCSScenario(
            **base_scenario.__dict__,
            variants=variants
        )

        return self.scenario


class IndustrialDatasetContextScenarioProvider(IndustrialDatasetScenarioProvider):
    """
    Scenario provider for context-aware data.
    """

    def __init__(
        self,
        tcfile: str,
        feature_group_name: str,
        feature_group_values: List[str],
        previous_build: List[str],
        sched_time_ratio: float = 0.5,
    ):
        """
        Initializes the context-aware scenario provider.
        :param tcfile: Path to the test case file.
        :param feature_group_name: Name of the feature group.
        :param feature_group_values: List of features.
        :param previous_build: List of features from the previous build.
        :param sched_time_ratio: Ratio of the total build time to be used for scheduling.
        """
        super().__init__(tcfile, sched_time_ratio)
        self.feature_group = feature_group_name
        # List of columns that are features
        self.features = feature_group_values
        self.previous_build = previous_build

    def __str__(self):
        return self.name

    def _merge_context_features(self, build_df):
        if self.current_build == 1:
            return self._initialize_first_build_features(build_df)

        # Compute intersections of features for current and previous builds
        current_features = list(set(self.features).difference(self.previous_build))
        # Previous features are those shared between `self.features` and `previous_build`
        previous_features = list(set(self.previous_build).intersection(self.features))

        # Extract data for the previous build
        previous_build_df = self.tcdf.filter(pl.col("BuildId") == self.current_build - 1)

        # Start with the current build's features
        merged_df = build_df.select(["Name"] + list(current_features))

        # Merge features from the previous build if any are relevant
        if previous_features:
            previous_data = previous_build_df.select(["Name"] + list(previous_features))
            merged_df = merged_df.join(previous_data, on="Name", how="left")

        # Fill missing values for previous features using their mean values from the previous build
        # Precompute mean values for efficiency
        if previous_features:
            feature_means_df = previous_build_df.select(previous_features).mean()
            fill_exprs = []
            for feature in previous_features:
                mean_val = feature_means_df[feature][0] if feature_means_df.height > 0 else 0
                fill_exprs.append(
                    pl.col(feature).fill_null(mean_val).alias(feature)
                )
            if fill_exprs:
                merged_df = merged_df.with_columns(fill_exprs)

        return merged_df

    def _initialize_first_build_features(self, build_df: pl.DataFrame) -> pl.DataFrame:
        """
        Initializes default feature values for the first build.
        - Creates a DataFrame with default values for all features, as the first build has no prior context.
        - Each feature is assigned a default value of 1.
        :param build_df: DataFrame for the current build.
        :return: DataFrame with default feature values for the first build.
        """
        data_dict = {
            "Name": build_df["Name"].to_list(),
            **{feature: [1] * build_df.height for feature in self.features}  # Default value of 1 for all features
        }
        return pl.DataFrame(data_dict)

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        base_scenario = super().get()

        if not base_scenario:
            return None

        build_df = self.tcdf.filter(pl.col("BuildId") == self.current_build)
        context_features = self._merge_context_features(build_df)

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualContextScenario(
            **base_scenario.__dict__,
            feature_group=self.feature_group,
            features=self.features,
            context_features=context_features
        )

        return self.scenario
