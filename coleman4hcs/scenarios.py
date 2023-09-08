import os

import pandas as pd


class VirtualScenario(object):
    """
    Virtual Scenario, used to manipulate the data for each commit
    """

    def __init__(self, available_time, testcases, build_id, total_build_duration):
        self.available_time = available_time
        self.testcases = testcases
        self.build_id = build_id
        self.total_build_duration = total_build_duration
        self.reset()

    def reset(self):
        # Reset the priorities
        for row in self.get_testcases():
            row['CalcPrio'] = 0

    def get_available_time(self):
        return self.available_time

    def get_testcases(self):
        return self.testcases

    def get_testcases_names(self):
        return [row['Name'] for row in self.get_testcases()]


class VirtualHCSScenario(VirtualScenario):
    """
    Virtual Scenario HCS, used to manipulate the data for each commit in the HCS context.
    """

    def __init__(self, available_time, testcases, build_id, total_build_duration, variants):
        super().__init__(available_time, testcases, build_id, total_build_duration)

        self.variants = variants

    def get_variants(self):
        return self.variants


class VirtualContextScenario(VirtualScenario):
    """
    Virtual Context Scenario, used to manipulate the data for each commit 
    considering context information from each commit
    """

    def __init__(self, available_time, testcases, build_id, total_build_duration,
                 feature_group, features, context_features):
        super().__init__(available_time, testcases, build_id, total_build_duration)
        self.feature_group = feature_group
        self.features = features
        self.context_features = context_features

    def get_feature_group(self):
        return self.feature_group

    def get_features(self):
        return self.features

    def get_context_features(self):
        return self.context_features


class IndustrialDatasetScenarioProvider:
    """
    Scenario provider to process CSV files for experimental evaluation.
    Required columns are `self.tc_fieldnames`
    """

    def __init__(self, tcfile, sched_time_ratio=0.5):
        self.name = os.path.split(os.path.dirname(tcfile))[1]

        self.read_testcases(tcfile)

        self.build = 0
        self.max_builds = max(self.tcdf.BuildId)
        self.scenario = None
        self.avail_time_ratio = sched_time_ratio
        self.total_build_duration = 0

        # ColName | Description
        # Name | Unique numeric identifier of the test case
        # Duration | Approximated runtime of the test case
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        # LastResults | List of previous test results (Failed: 1, Passed: 0), ordered by ascending age
        # Verdict | Test Case result (Failed: 1, Passed: 0)
        self.tc_fieldnames = ['Name',
                              'Duration',
                              'CalcPrio',
                              'LastRun',
                              'Verdict']

    def read_testcases(self, tcfile):
        # We use ';' separated values to avoid issues with thousands
        self.tcdf = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'])
        self.tcdf["Duration"] = self.tcdf["Duration"].apply(
            lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)

    def __str__(self):
        return self.name

    def get_avail_time_ratio(self):
        return self.avail_time_ratio

    def last_build(self, build):
        self.build = build

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        self.build += 1

        # Stop when reaches the max build
        if self.build > self.max_builds:
            self.scenario = None
            return None

        # Select the data for the current build
        builddf = self.tcdf.loc[self.tcdf.BuildId == self.build]

        # Convert the solutions to a list of dict
        seltc = builddf[self.tc_fieldnames].to_dict('records')

        self.total_build_duration = builddf['Duration'].sum()
        total_time = self.total_build_duration * self.avail_time_ratio

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualScenario(testcases=seltc,
                                        available_time=total_time,
                                        build_id=self.build,
                                        total_build_duration=self.total_build_duration)

        return self.scenario

    # Generator functions
    def __iter__(self):
        return self

    def __next__(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


class IndustrialDatasetHCSScenarioProvider(IndustrialDatasetScenarioProvider):
    def __init__(self, tcfile, variantsfile, sched_time_ratio=0.5):
        super().__init__(tcfile, sched_time_ratio)

        self.read_variants(variantsfile)

    def read_variants(self, variantsfile):
        # Read the variants (additional file)
        self.variants = pd.read_csv(
            variantsfile, sep=';', parse_dates=['LastRun'])

        # We remove weird characters
        self.variants['Variant'] = self.variants['Variant'].apply(
            lambda x: x.translate({ord(c): "_" for c in "!#$%^&*()[]{};:,.<>?|`~=+"}))

    def get_total_variants(self):
        return self.variants['Variant'].nunique()

    def get_all_variants(self):
        return self.variants['Variant'].unique()

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        base_scenario = super().get()

        if not base_scenario:
            return None

        variants = self.variants.loc[self.variants.BuildId == self.build]

        self.scenario = VirtualHCSScenario(testcases=base_scenario.get_testcases(),
                                           available_time=base_scenario.get_available_time(),
                                           build_id=self.build,
                                           total_build_duration=self.total_build_duration,
                                           variants=variants)

        return self.scenario


class IndustrialDatasetContextScenarioProvider(IndustrialDatasetScenarioProvider):
    """
    Scenario provider to process CSV files for experimental evaluation.
    Required columns are `self.tc_fieldnames`
    """

    def __init__(self, tcfile, feature_group_name, feature_group_values, previous_build, sched_time_ratio=0.5):
        super().__init__(tcfile, sched_time_ratio)

        self.feature_group = feature_group_name

        # List of columns used as features
        self.features = feature_group_values
        
        self.previous_build = previous_build

    def __str__(self):
        return self.name

    def get_context(self, builddf):
        # Now, we will get the features from previous build
        if self.build == 1:
            # If we are in the first build, we do not have previous build
            # We create an empty dataframe
            df = pd.DataFrame(columns=self.features)

            # Get the test cases from current build
            df['Name'] = builddf['Name']

            # So we fill all features to start with a default value equal 1
            df[self.features] = 1

            return df
        else:
            # Features from the current build
            current_build_features = list(set(self.features) - set(self.previous_build))
            df = builddf[['Name'] + current_build_features]

            # We have previous build, lets to get all the previous features for each test case
            previous_build = self.tcdf.loc[self.tcdf.BuildId == self.build - 1, ['Name'] + self.previous_build]

            # Merge the data from current build with the data which we obtain from previous build
            # (only with features chosen)
            previous_build_features = [x for x in self.previous_build if x in self.features]

            if len(previous_build_features) > 0:
                # In the case if we do not have an of then chosen
                df = df.merge(previous_build[['Name'] + previous_build_features], on=['Name'], how='left')

            # Fill the NA values for each feature
            # This is done when we have a new test case and the tests were not execute yet
            # (If the feature chosen is one of the features that we have information only after the tests are executed)
            for feature in previous_build_features:
                # We fill it with the mean from previous build
                df[feature].fillna((previous_build[feature].mean()), inplace=True)

            return df

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        base_scenario = super().get()

        if not base_scenario:
            return None

        builddf = self.tcdf.loc[self.tcdf.BuildId == self.build]
        context_features = self.get_context(builddf)

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualContextScenario(testcases=base_scenario.get_testcases(),
                                               available_time=base_scenario.get_available_time(),
                                               build_id=self.build,
                                               total_build_duration=self.total_build_duration,
                                               feature_group=self.feature_group,
                                               features=self.features,
                                               context_features=context_features)
        
        return self.scenario
