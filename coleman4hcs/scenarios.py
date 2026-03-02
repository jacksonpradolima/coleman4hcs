"""
coleman4hcs.scenarios - Scenario Management for the Coleman4HCS Framework.

This module provides utilities for managing and processing different scenarios in the context
of the Coleman4HCS framework. This includes virtual scenarios for commits, scenarios specific
to HCS context, and scenarios that consider context information for each commit. The module
also provides utilities to process CSV files for experimental evaluations.

Classes
-------
VirtualScenario
    Basic virtual scenario to manipulate data for each commit.
VirtualHCSScenario
    Extends VirtualScenario to handle HCS context.
VirtualContextScenario
    Extends VirtualScenario to handle context information.
IndustrialDatasetScenarioProvider
    Provider to process CSV files for experiments.
IndustrialDatasetHCSScenarioProvider
    Extends IndustrialDatasetScenarioProvider to handle HCS scenarios.
IndustrialDatasetContextScenarioProvider
    Extends IndustrialDatasetScenarioProvider to handle context scenarios.
"""
import os
from typing import List, Dict, Optional

import polars as pl


class VirtualScenario:
    """Virtual scenario used to manipulate data for each commit.

    Parameters
    ----------
    available_time : float
        The time available to execute tests.
    testcases : list of dict
        The test cases for the scenario.
    build_id : int
        The build identifier.
    total_build_duration : float
        The total duration of the build.

    Attributes
    ----------
    available_time : float
        The time available to execute tests.
    testcases : list of dict
        The test cases for the scenario.
    build_id : int
        The build identifier.
    total_build_duration : float
        The total duration of the build.
    """

    def __init__(self, available_time: float, testcases: List[Dict], build_id: int, total_build_duration: float):
        """Initialize the VirtualScenario.

        Parameters
        ----------
        available_time : float
            The time available to execute tests.
        testcases : list of dict
            The test cases for the scenario.
        build_id : int
            The build identifier.
        total_build_duration : float
            The total duration of the build.
        """
        self.available_time = available_time
        self.testcases = testcases
        self.build_id = build_id
        self.total_build_duration = total_build_duration
        self.reset()

    def reset(self) -> None:
        """Reset the priorities for all test cases in the scenario."""
        # Reset the priorities
        for testcase in self.testcases:
            testcase['CalcPrio'] = 0

    def get_available_time(self) -> float:
        """Return the available time to execute the tests.

        Returns
        -------
        float
            The available time.
        """
        return self.available_time

    def get_testcases(self) -> List[Dict]:
        """Return the test cases for the scenario.

        Returns
        -------
        list of dict
            The test cases.
        """
        return self.testcases


class VirtualHCSScenario(VirtualScenario):
    """Extends VirtualScenario to handle data in an HCS-specific context.

    Parameters
    ----------
    *args
        Positional arguments passed to ``VirtualScenario``.
    variants : polars.DataFrame
        DataFrame containing variant information.
    **kwargs
        Keyword arguments passed to ``VirtualScenario``.

    Attributes
    ----------
    variants : polars.DataFrame
        DataFrame containing variant information.
    """

    def __init__(self, *args, variants: pl.DataFrame, **kwargs):
        """Initialize the VirtualHCSScenario.

        Parameters
        ----------
        *args
            Positional arguments passed to ``VirtualScenario``.
        variants : polars.DataFrame
            DataFrame containing variant information.
        **kwargs
            Keyword arguments passed to ``VirtualScenario``.
        """
        super().__init__(*args, **kwargs)
        self.variants = variants

    def get_variants(self):
        """Return the variants associated with the system.

        Returns
        -------
        polars.DataFrame
            DataFrame containing variant information.
        """
        return self.variants


class VirtualContextScenario(VirtualScenario):
    """Extends VirtualScenario to include context-specific features for each commit.

    Parameters
    ----------
    *args
        Positional arguments passed to ``VirtualScenario``.
    feature_group : str
        The name of the feature group.
    features : list of str
        The feature names.
    context_features : polars.DataFrame
        DataFrame containing context features.
    **kwargs
        Keyword arguments passed to ``VirtualScenario``.

    Attributes
    ----------
    feature_group : str
        The name of the feature group.
    features : list of str
        The feature names.
    context_features : polars.DataFrame
        DataFrame containing context features.
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
        """Return the feature group.

        Returns
        -------
        str
            The feature group name.
        """
        return self.feature_group

    def get_features(self) -> List[str]:
        """Return the features associated with the scenario.

        Returns
        -------
        list of str
            The feature names.
        """
        return self.features

    def get_context_features(self) -> pl.DataFrame:
        """Return the context features associated with the scenario.

        Returns
        -------
        polars.DataFrame
            DataFrame containing context features.
        """
        return self.context_features


class IndustrialDatasetScenarioProvider:
    """Base class for scenario providers that process test case data from CSV files.

    Parameters
    ----------
    tcfile : str
        Path to the test case CSV file.
    sched_time_ratio : float, optional
        Ratio of scheduled time to total build time. Default is 0.5.

    Attributes
    ----------
    name : str
        Name derived from the dataset directory.
    avail_time_ratio : float
        The available time ratio for scheduling.
    current_build : int
        The current build number being processed.
    total_build_duration : float
        The total duration of the current build.
    scenario : VirtualScenario or None
        The current virtual scenario.
    tcdf : polars.DataFrame
        DataFrame containing test case data.
    max_builds : int
        The maximum number of builds in the dataset.
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
        """Initialize the IndustrialDatasetScenarioProvider.

        Parameters
        ----------
        tcfile : str
            Path to the test case CSV file.
        sched_time_ratio : float, optional
            Ratio of scheduled time to total build time. Default is 0.5.
        """
        self.name = os.path.split(os.path.dirname(tcfile))[1]
        self.avail_time_ratio = sched_time_ratio
        self.current_build = 0
        self.total_build_duration = 0
        self.scenario: Optional[VirtualScenario] = None

        self.tcdf = self._read_testcases(tcfile)
        self.max_builds = self.tcdf["BuildId"].max()

    def _read_testcases(self, tcfile: str) -> pl.DataFrame:
        """Read the test cases from a provided CSV file.

        Parameters
        ----------
        tcfile : str
            Path to the CSV file.

        Returns
        -------
        polars.DataFrame
            DataFrame containing the test case data.
        """
        # We use ';' separated values to avoid issues with thousands
        df = pl.read_csv(tcfile, separator=';', try_parse_dates=True)

        # Handle Duration column - convert to numeric and fill nulls
        df = df.with_columns([
            pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0)
        ])

        return df

    def get_avail_time_ratio(self) -> float:
        """Return the available time ratio.

        Returns
        -------
        float
            The available time ratio.
        """
        return self.avail_time_ratio

    def last_build(self, build: int) -> None:
        """Set the last build number.

        Parameters
        ----------
        build : int
            The build number to set.
        """
        self.current_build = build

    def get(self) -> Optional[VirtualScenario]:
        """Get the next virtual scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as a scenario.

        Returns
        -------
        VirtualScenario or None
            The next scenario, or None if no more builds remain.
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
        """Return the name of the scenario provider.

        Returns
        -------
        str
            The scenario provider name.
        """
        return self.name

    def __iter__(self):
        """Return the iterator for the scenario provider.

        Returns
        -------
        IndustrialDatasetScenarioProvider
            The iterator instance.
        """
        return self

    def __next__(self) -> VirtualScenario:
        """Return the next scenario.

        Returns
        -------
        VirtualScenario
            The next virtual scenario.

        Raises
        ------
        StopIteration
            If no more scenarios are available.
        """
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


class IndustrialDatasetHCSScenarioProvider(IndustrialDatasetScenarioProvider):
    """Scenario provider for HCS-specific data.

    Extends the base scenario provider to support variant-specific data.

    Parameters
    ----------
    tcfile : str
        Path to the test case CSV file.
    variantsfile : str
        Path to the variants CSV file.
    sched_time_ratio : float, optional
        Ratio of scheduled time to total build time. Default is 0.5.

    Attributes
    ----------
    variants : polars.DataFrame
        DataFrame containing variant data.
    """

    def __init__(self, tcfile: str, variantsfile: str, sched_time_ratio=0.5) -> None:
        """Initialize the IndustrialDatasetHCSScenarioProvider.

        Parameters
        ----------
        tcfile : str
            Path to the test case CSV file.
        variantsfile : str
            Path to the variants CSV file.
        sched_time_ratio : float, optional
            Ratio of scheduled time to total build time. Default is 0.5.
        """
        super().__init__(tcfile, sched_time_ratio)

        self.variants = self._read_variants(variantsfile)

    def _read_variants(self, variantsfile: str) -> pl.DataFrame:
        """Read the variants from a provided CSV file.

        Parameters
        ----------
        variantsfile : str
            Path to the variants CSV file.

        Returns
        -------
        polars.DataFrame
            DataFrame containing variant data.
        """
        # Read the variants (additional file)
        df = pl.read_csv(variantsfile, separator=';', try_parse_dates=True)

        # We remove weird characters
        df = df.with_columns([
            pl.col("Variant").str.replace_all(r'[!#$%^&*()\[\]{};:,.<>?|`~=+]', '_')
        ])

        return df

    def get_total_variants(self):
        """Return the number of unique variants.

        Returns
        -------
        int
            The number of unique variants.
        """
        return self.variants['Variant'].n_unique()

    def get_all_variants(self):
        """Return all unique variant names as a list.

        Returns
        -------
        list of str
            List of unique variant names.
        """
        return self.variants['Variant'].unique().to_list()

    def get(self):
        """Get the next virtual HCS scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as an HCS scenario with variant data.

        Returns
        -------
        VirtualHCSScenario or None
            The next HCS scenario, or None if no more builds remain.
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
    """Scenario provider for context-aware data.

    Parameters
    ----------
    tcfile : str
        Path to the test case file.
    feature_group_name : str
        Name of the feature group.
    feature_group_values : list of str
        List of features.
    previous_build : list of str
        List of features from the previous build.
    sched_time_ratio : float, optional
        Ratio of the total build time to be used for scheduling. Default is 0.5.

    Attributes
    ----------
    feature_group : str
        The feature group name.
    features : list of str
        List of feature column names.
    previous_build : list of str
        List of features from the previous build.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        tcfile: str,
        feature_group_name: str,
        feature_group_values: List[str],
        previous_build: List[str],
        sched_time_ratio: float = 0.5,
    ):
        """Initialize the context-aware scenario provider.

        Parameters
        ----------
        tcfile : str
            Path to the test case file.
        feature_group_name : str
            Name of the feature group.
        feature_group_values : list of str
            List of features.
        previous_build : list of str
            List of features from the previous build.
        sched_time_ratio : float, optional
            Ratio of the total build time to be used for scheduling.
            Default is 0.5.
        """
        super().__init__(tcfile, sched_time_ratio)
        self.feature_group = feature_group_name
        # List of columns that are features
        self.features = feature_group_values
        self.previous_build = previous_build

    def __str__(self):
        """Return the name of the scenario provider.

        Returns
        -------
        str
            The scenario provider name.
        """
        return self.name

    def _merge_context_features(self, build_df):
        """Merge context features from current and previous builds.

        Parameters
        ----------
        build_df : polars.DataFrame
            DataFrame for the current build.

        Returns
        -------
        polars.DataFrame
            DataFrame with merged context features.
        """
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
                mean_val = feature_means_df[feature][0] if feature_means_df.height > 0 else 0.0
                fill_exprs.append(
                    pl.col(feature).fill_null(mean_val).alias(feature)
                )
            if fill_exprs:
                merged_df = merged_df.with_columns(fill_exprs)

        return merged_df

    def _initialize_first_build_features(self, build_df: pl.DataFrame) -> pl.DataFrame:
        """Initialize default feature values for the first build.

        Creates a DataFrame with default values for all features, as the first
        build has no prior context. Each feature is assigned a default value of 1.

        Parameters
        ----------
        build_df : polars.DataFrame
            DataFrame for the current build.

        Returns
        -------
        polars.DataFrame
            DataFrame with default feature values for the first build.
        """
        result = build_df.select(['Name'])
        for feature in self.features:
            result = result.with_columns([pl.lit(1).alias(feature)])
        return result

    def get(self):
        """Get the next virtual context scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as a context scenario with feature data.

        Returns
        -------
        VirtualContextScenario or None
            The next context scenario, or None if no more builds remain.
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
