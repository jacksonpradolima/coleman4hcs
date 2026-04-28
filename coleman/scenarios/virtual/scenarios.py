"""Virtual scenario classes."""

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

    def __init__(self, available_time: float, testcases: list[dict], build_id: int, total_build_duration: float):
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
        for testcase in self.testcases:
            testcase["CalcPrio"] = 0

    def get_available_time(self) -> float:
        """Return the available time to execute the tests.

        Returns
        -------
        float
            The available time.
        """
        return self.available_time

    def get_testcases(self) -> list[dict]:
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

    def __init__(self, *args, feature_group: str, features: list[str], context_features: pl.DataFrame, **kwargs):
        """Initialise the VirtualContextScenario.

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
        """
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

    def get_features(self) -> list[str]:
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
