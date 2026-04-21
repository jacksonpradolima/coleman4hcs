"""Industrial dataset scenario providers."""

import os
import warnings
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl

from ..virtual import VirtualContextScenario, VirtualHCSScenario, VirtualScenario


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

    REQUIRED_COLUMNS = ["Name", "Duration", "CalcPrio", "LastRun", "Verdict"]

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
        self.total_build_duration = 0.0
        self.scenario: VirtualScenario | None = None

        self._testcases_lazy = self._read_testcases(tcfile)
        max_builds = cast(
            int | None, self._testcases_lazy.select(pl.col("BuildId").max().cast(pl.Int64)).collect().item()
        )
        self.max_builds = max_builds if max_builds is not None else 0

    def _read_testcases(self, tcfile: str) -> pl.LazyFrame:
        """Read the test cases from a provided dataset file.

        Parameters
        ----------
        tcfile : str
            Path to the test case file (Parquet preferred, CSV supported).

        Returns
        -------
        polars.LazyFrame
            LazyFrame containing normalized test case data.
        """
        suffix = Path(tcfile).suffix.lower()
        if suffix == ".parquet":
            df = pl.scan_parquet(tcfile)
        elif suffix == ".csv":
            warnings.warn(
                "CSV scenario files are deprecated and will be removed in a future release. "
                "Please migrate to Parquet files.",
                DeprecationWarning,
                stacklevel=2,
            )
            df = pl.scan_csv(tcfile, separator=";", try_parse_dates=True)
        else:
            msg = f"Unsupported scenario file format: {tcfile!r}. Supported formats: .parquet, .csv"
            raise ValueError(msg)

        expressions = [
            pl.col("Name").cast(pl.Utf8, strict=False),
            pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0),
        ]

        schema_names = df.collect_schema().names()

        if "LastRun" in schema_names:
            expressions.append(pl.col("LastRun").cast(pl.Utf8, strict=False).fill_null(""))

        if "LastResults" in schema_names:
            expressions.append(pl.col("LastResults").cast(pl.Utf8, strict=False).fill_null(""))

        return df.with_columns(expressions)

    def _collect_build(self, build_id: int, columns: list[str] | None = None) -> pl.DataFrame:
        """Collect one build slice using lazy filtering/projection."""
        query = self._testcases_lazy.filter(pl.col("BuildId") == build_id)
        if columns:
            query = query.select(columns)
        return query.collect()

    @property
    def tcdf(self) -> pl.DataFrame:
        """Legacy eager view of scenario data.

        Returns
        -------
        polars.DataFrame
            Materialized DataFrame. Prefer lazy internal access for scalability.
        """
        warnings.warn(
            "The `tcdf` eager DataFrame attribute is deprecated. "
            "Use provider iteration APIs instead (e.g., iterate the provider with `for scenario in provider`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._testcases_lazy.collect()

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

    def get(self) -> VirtualScenario | None:
        """Get the next virtual scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as a scenario.

        Returns
        -------
        VirtualScenario or None
            The next scenario, or None if no more builds remain.
        """
        self.current_build += 1

        if self.current_build > self.max_builds:
            return None

        build_df = self._collect_build(self.current_build)

        testcases = build_df.select(self.REQUIRED_COLUMNS).to_dicts()

        total_build_duration = cast(int | float | Decimal | None, build_df["Duration"].sum())
        self.total_build_duration = float(total_build_duration or 0.0)
        available_time = self.total_build_duration * self.avail_time_ratio

        self.scenario = VirtualScenario(
            available_time=available_time,
            testcases=testcases,
            build_id=self.current_build,
            total_build_duration=self.total_build_duration,
        )

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

        self._variants_lazy = self._read_variants(variantsfile)
        self._variant_names = cast(
            list[str], self._variants_lazy.select(pl.col("Variant").unique()).collect().to_series().to_list()
        )

    def _read_variants(self, variantsfile: str) -> pl.LazyFrame:
        """Read the variants from a provided dataset file.

        Parameters
        ----------
        variantsfile : str
            Path to the variants CSV file.

        Returns
        -------
        polars.LazyFrame
            LazyFrame containing variant data.
        """
        suffix = Path(variantsfile).suffix.lower()
        if suffix == ".parquet":
            df = pl.scan_parquet(variantsfile)
        elif suffix == ".csv":
            warnings.warn(
                "CSV variants files are deprecated and will be removed in a future release. "
                "Please migrate to Parquet files.",
                DeprecationWarning,
                stacklevel=2,
            )
            df = pl.scan_csv(variantsfile, separator=";", try_parse_dates=True)
        else:
            msg = f"Unsupported variants file format: {variantsfile!r}. Supported formats: .parquet, .csv"
            raise ValueError(msg)

        return df.with_columns([pl.col("Variant").str.replace_all(r"[!#$%^&*()\[\]{};:,.<>?|`~=+]", "_")])

    @property
    def variants(self) -> pl.DataFrame:
        """Legacy eager view of variants data."""
        warnings.warn(
            "The `variants` eager DataFrame attribute is deprecated. "
            "Use scenario-provider APIs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._variants_lazy.collect()

    def get_total_variants(self):
        """Return the number of unique variants.

        Returns
        -------
        int
            The number of unique variants.
        """
        return len(self._variant_names)

    def get_all_variants(self):
        """Return all unique variant names as a list.

        Returns
        -------
        list of str
            List of unique variant names.
        """
        return list(self._variant_names)

    def get(self) -> VirtualHCSScenario | None:
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

        variants = self._variants_lazy.filter(pl.col("BuildId") == self.current_build).collect()

        self.scenario = VirtualHCSScenario(**base_scenario.__dict__, variants=variants)

        return self.scenario

    def __next__(self) -> VirtualHCSScenario:
        """Return the next HCS scenario."""
        sc = self.get()
        if sc is None:
            raise StopIteration()
        return sc


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
        feature_group_values: list[str],
        previous_build: list[str],
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

        current_features = list(set(self.features).difference(self.previous_build))
        previous_features = list(set(self.previous_build).intersection(self.features))

        previous_build_df = self._collect_build(self.current_build - 1)

        merged_df = build_df.select(["Name"] + list(current_features))

        if previous_features:
            previous_data = previous_build_df.select(["Name"] + list(previous_features))
            merged_df = merged_df.join(previous_data, on="Name", how="left")

        if previous_features:
            feature_means_df = previous_build_df.select(previous_features).mean()
            fill_exprs = []
            for feature in previous_features:
                mean_val = feature_means_df[feature][0] if feature_means_df.height > 0 else 0.0
                fill_exprs.append(pl.col(feature).fill_null(mean_val).alias(feature))
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
        result = build_df.select(["Name"])
        for feature in self.features:
            result = result.with_columns([pl.lit(1).alias(feature)])
        return result

    def get(self) -> VirtualContextScenario | None:
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

        build_df = self._collect_build(self.current_build)
        context_features = self._merge_context_features(build_df)

        self.scenario = VirtualContextScenario(
            **base_scenario.__dict__,
            feature_group=self.feature_group,
            features=self.features,
            context_features=context_features,
        )

        return self.scenario

    def __next__(self) -> VirtualContextScenario:
        """Return the next contextual scenario."""
        sc = self.get()
        if sc is None:
            raise StopIteration()
        return sc
